"""
V2 AgentRunner - Executes a single agent with validation and learning

Extracted from SingleAgentOrchestrator for reuse in Orchestrator.

AgentScope-inspired lifecycle hooks (Gao et al., 2025):
    Pluggable hooks at 6 lifecycle points let you add tracing, profiling,
    rate limiting, or custom logic without editing core code.

    runner.add_hook('pre_execute', my_logger)
    runner.add_hook('post_run', my_metrics_recorder)
"""

import asyncio
import logging
import time as _time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig, SwarmLearningConfig, EpisodeResult
from Jotty.core.infrastructure.utils.async_utils import StatusReporter
from Jotty.core.infrastructure.foundation.exceptions import (
    AgentExecutionError,
    ToolExecutionError,
    LLMError,
    DSPyError,
    MemoryRetrievalError,
    MemoryStorageError,
    ConsolidationError,
    LearningError,
)
from Jotty.core.modes.agent.tools.inspector import ValidatorAgent, MultiRoundValidator
from Jotty.core.intelligence.memory.cortex import SwarmMemory
from Jotty.core.infrastructure.utils.prompt_selector import get_prompt_selector, PromptSelector
from Jotty.core.intelligence.orchestration.prompts import (
    get_swarm_architect_prompt,
    get_swarm_auditor_prompt,
    get_generic_auditor_prompt,
)
from Jotty.core.intelligence.learning.learning import (
    TDLambdaLearner, AdaptiveLearningRate,
)
from Jotty.core.intelligence.learning.shaped_rewards import ShapedRewardManager
from Jotty.core.intelligence.orchestration.validation_gate import (
    ValidationGate, ValidationMode, GateDecision, get_validation_gate,
)

logger = logging.getLogger(__name__)


# =========================================================================
# LIFECYCLE HOOK TYPES (AgentScope-inspired)
# =========================================================================

HOOK_TYPES = (
    'pre_run',        # Before entire run() — modify goal, inject context
    'post_run',       # After entire run() — record metrics, log results
    'pre_architect',  # Before architect validation
    'post_architect', # After architect — inspect/override proceed decision
    'pre_execute',    # Before agent execution — last chance to modify goal
    'post_execute',   # After agent execution — inspect/transform output
)


@dataclass
class AgentRunnerConfig:
    """Configuration for AgentRunner"""
    architect_prompts: List[str]
    auditor_prompts: List[str]
    config: SwarmConfig
    agent_name: str = "agent"
    enable_learning: bool = True
    enable_memory: bool = True
    enable_terminal: bool = True  # Enable SwarmTerminal for auto-fix


@dataclass
class ExecutionContext:
    """Mutable state passed through the agent execution pipeline.

    Each stage of AgentRunner.run() receives this context, mutates
    specific fields, and returns it. Replaces scattered locals() checks
    with explicit, defaulted fields.
    """
    goal: str
    kwargs: Dict[str, Any]
    start_time: float = 0.0
    status_callback: Optional[Callable] = None
    _status: Optional[Any] = None  # StatusReporter
    gate_decision: Optional[Any] = None  # GateDecision
    skip_architect: bool = False
    skip_auditor: bool = False
    learning_context_parts: List[str] = field(default_factory=list)
    enriched_goal: str = ""
    architect_results: List[Any] = field(default_factory=list)
    proceed: bool = True
    architect_shaped_reward: float = 0.0
    agent_output: Any = None
    trajectory: List[Dict] = field(default_factory=list)
    inner_success: bool = False
    auditor_results: List[Any] = field(default_factory=list)
    success: bool = False
    auditor_reasoning: str = ""
    auditor_confidence: float = 0.0
    learning_data: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    task_progress: Optional[Any] = None  # TaskProgress
    ws_checkpoint_id: Optional[str] = None


class TaskProgress:
    """
    Visible task progress tracker (Cline's Focus Chain pattern).

    A simple checklist that updates as the agent works. Each step
    can be pending, in_progress, done, or failed. The checklist
    is exposed via the status callback so UI/CLI can display it.

    Usage:
        progress = TaskProgress("Research AI trends")
        progress.add_step("Search for recent papers")
        progress.add_step("Analyze findings")
        progress.add_step("Write summary")
        progress.start_step(0)  # "Search for recent papers" → in_progress
        progress.complete_step(0)  # → done
        print(progress.render())
    """

    def __init__(self, goal: str = '') -> None:
        self.goal = goal
        self.steps: List[Dict[str, Any]] = []  # {name, status, started_at, finished_at}
        self.created_at = _time.time()

    def add_step(self, name: str) -> int:
        """Add a step. Returns its index."""
        idx = len(self.steps)
        self.steps.append({
            'name': name, 'status': 'pending',
            'started_at': None, 'finished_at': None,
        })
        return idx

    def start_step(self, idx: int) -> None:
        if 0 <= idx < len(self.steps):
            self.steps[idx]['status'] = 'in_progress'
            self.steps[idx]['started_at'] = _time.time()

    def complete_step(self, idx: int) -> None:
        if 0 <= idx < len(self.steps):
            self.steps[idx]['status'] = 'done'
            self.steps[idx]['finished_at'] = _time.time()

    def fail_step(self, idx: int) -> None:
        if 0 <= idx < len(self.steps):
            self.steps[idx]['status'] = 'failed'
            self.steps[idx]['finished_at'] = _time.time()

    def render(self) -> str:
        """Render checklist as text (suitable for status callback or logging)."""
        icons = {'pending': '[ ]', 'in_progress': '[>]', 'done': '[x]', 'failed': '[!]'}
        lines = [f"Task: {self.goal}"] if self.goal else []
        for i, s in enumerate(self.steps):
            icon = icons.get(s['status'], '[ ]')
            elapsed = ""
            if s['status'] == 'done' and s['started_at'] and s['finished_at']:
                elapsed = f" ({s['finished_at'] - s['started_at']:.1f}s)"
            lines.append(f"  {icon} {s['name']}{elapsed}")
        done = sum(1 for s in self.steps if s['status'] == 'done')
        total = len(self.steps)
        if total:
            lines.append(f"  Progress: {done}/{total} ({done/total:.0%})")
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Machine-readable summary."""
        done = sum(1 for s in self.steps if s['status'] == 'done')
        failed = sum(1 for s in self.steps if s['status'] == 'failed')
        return {
            'total': len(self.steps), 'done': done, 'failed': failed,
            'completion_pct': done / len(self.steps) if self.steps else 0,
            'steps': [{'name': s['name'], 'status': s['status']} for s in self.steps],
        }


class AgentRunner:
    """
    Executes ONE agent with validation and learning.
    
    V2 User-Friendly Component:
    - Wraps agent execution
    - Provides Architect (pre-execution)
    - Provides Auditor (post-execution)
    - Handles learning and memory
    """
    
    def __init__(self, agent: Any, config: AgentRunnerConfig, task_planner: Any = None, task_board: Any = None, swarm_memory: Any = None, swarm_state_manager: Any = None, learning_manager: Any = None, transfer_learning: Any = None, swarm_terminal: Any = None, swarm_intelligence: Any = None) -> None:
        """
        Initialize AgentRunner.

        Args:
            agent: The agent to execute (AutoAgent or DSPy module)
            config: AgentRunner configuration
            task_planner: Shared TaskPlanner (optional)
            task_board: Shared TaskBoard (optional)
            swarm_memory: Shared SwarmMemory (optional)
            swarm_state_manager: SwarmStateManager for state tracking (optional)
            swarm_intelligence: SwarmIntelligence for curriculum feedback (optional)
        """
        self.agent = agent
        self.config = config
        self.agent_name = config.agent_name

        # Shared components (V2)
        self.task_planner = task_planner
        self.task_board = task_board
        self.swarm_memory = swarm_memory
        self.swarm_state_manager = swarm_state_manager
        self.learning_manager = learning_manager
        self.transfer_learning = transfer_learning
        self.swarm_intelligence = swarm_intelligence  # Agent0: Curriculum feedback

        # SwarmTerminal for intelligent command execution and auto-fix
        self.swarm_terminal = swarm_terminal
        if swarm_terminal is None and config.enable_terminal:
            try:
                from .swarm_terminal import SwarmTerminal
                self.swarm_terminal = SwarmTerminal(
                    config=config.config,
                    auto_fix=True,
                    max_fix_attempts=3
                )
                logger.info(f" SwarmTerminal enabled for agent '{self.agent_name}'")
            except Exception as e:
                logger.debug(f"SwarmTerminal not available for {self.agent_name}: {e}")

        # Get agent state tracker (creates if doesn't exist)
        if self.swarm_state_manager:
            self.agent_tracker = self.swarm_state_manager.get_agent_tracker(self.agent_name)
            logger.info(f" AgentStateTracker initialized for '{self.agent_name}'")
        
        from pathlib import Path
        
        from Jotty.core.infrastructure.foundation.data_structures import SharedScratchpad
        
        # Shared scratchpad for agent communication
        scratchpad = SharedScratchpad()
        
        # Architect (pre-execution planning)
        architect_agents = [
            ValidatorAgent(
                md_path=Path(prompt),
                is_architect=True,
                tools=[],
                config=config.config,
                scratchpad=scratchpad
            )
            for prompt in config.architect_prompts
        ]
        
        # Auditor (post-execution validation)
        auditor_agents = [
            ValidatorAgent(
                md_path=Path(prompt),
                is_architect=False,
                tools=[],
                config=config.config,
                scratchpad=scratchpad
            )
            for prompt in config.auditor_prompts
        ]
        
        # Multi-round validators
        self.architect_validator = MultiRoundValidator(architect_agents, config.config)
        self.auditor_validator = MultiRoundValidator(auditor_agents, config.config)
        
        # Per-agent memory: use the shared swarm memory if available,
        # otherwise fall back to creating a standalone instance.
        # This ensures all agents share the same memory store (with
        # agent_name used for namespacing in store/retrieve calls).
        self.agent_memory: Optional[SwarmMemory] = None
        if config.enable_memory:
            if swarm_memory is not None:
                # Shared memory — all agents see each other's experiences
                self.agent_memory = swarm_memory
                logger.debug(f"Agent '{self.agent_name}' using shared swarm memory")
            else:
                # Fallback: standalone memory (no sharing)
                self.agent_memory = SwarmMemory(
                    config=config.config,
                    agent_name=self.agent_name
                )
                logger.debug(f"Agent '{self.agent_name}' using standalone memory (no swarm memory)")
        
        # Per-agent learning: prefer the swarm-level learning_manager if available,
        # so all agents contribute to a single shared learner.
        # Falls back to standalone TDLambdaLearner only when no swarm context exists.
        from Jotty.core.intelligence.learning.learning import AdaptiveLearningRate
        self.agent_learner: Optional[TDLambdaLearner] = None
        self._using_shared_learner = False
        if config.enable_learning:
            if learning_manager is not None and hasattr(learning_manager, 'td_learner'):
                # Shared learner from SwarmLearningPipeline
                self.agent_learner = learning_manager.td_learner
                self._using_shared_learner = True
                logger.debug(f"Agent '{self.agent_name}' using shared TD-Lambda learner")
            else:
                # Standalone fallback
                adaptive_lr = AdaptiveLearningRate(config.config)
                self.agent_learner = TDLambdaLearner(
                    config=config.config,
                    adaptive_lr=adaptive_lr
                )
                logger.debug(f"Agent '{self.agent_name}' using standalone TD-Lambda learner")

        # Shaped rewards for dense learning signal
        self.shaped_reward_manager: Optional[ShapedRewardManager] = None
        if config.enable_learning:
            self.shaped_reward_manager = ShapedRewardManager()

        # Consecutive mistake counter (Cline pattern):
        # Tracks sequential failures. When high, injects a "change approach"
        # hint into the agent's context. Resets on any success.
        self._consecutive_failures: int = 0
        self._max_consecutive_before_hint: int = 2  # inject hint after 2 failures

        # Tool guard (Cline ToolExecutor patterns):
        # - Plan/Act mode: restrict side-effect tools during planning
        # - One side-effect per turn: prevent cascading mutations
        # - Path access control: block sensitive file operations
        from Jotty.core.capabilities.registry.tool_validation import ToolGuard
        self.tool_guard = ToolGuard()

        # Host provider (Cline HostProvider pattern):
        # Core never imports CLI/Web directly — uses HostProvider.
        from Jotty.core.interface.interfaces.host_provider import HostProvider
        self._host = HostProvider.get()

        # Task progress tracker (Cline Focus Chain pattern):
        # Visible checklist of steps, updated via status callback.
        self.task_progress: Optional[TaskProgress] = None

        logger.info(f"AgentRunner initialized: {self.agent_name}")

        # Prompt selector for dynamic template selection
        self._prompt_selector: Optional[PromptSelector] = None
        self._current_task_type: str = 'default'
        self._scratchpad = scratchpad  # Store for validator recreation

        # Swarm-level prompts cache
        self._swarm_prompts_loaded = False

        # Intelligent validation gate (Haiku-powered)
        # Replaces boolean skip_validation with per-task LLM classification:
        #   DIRECT     → actor only  (simple Q&A, lookups)
        #   AUDIT_ONLY → actor + auditor  (medium: summaries, analysis)
        #   FULL       → architect + actor + auditor  (complex: code, multi-step)
        self._validation_gate: Optional[ValidationGate] = None

        # AgentScope-inspired lifecycle hooks (pluggable middleware)
        self._hooks: Dict[str, List[Callable]] = {ht: [] for ht in HOOK_TYPES}

        # Bridge: auto-import hooks from the underlying agent (BaseAgent)
        # so users only need one registration point. BaseAgent._pre_hooks
        # map to 'pre_execute' and _post_hooks map to 'post_execute'.
        self._bridge_agent_hooks()

    # =========================================================================
    # AGENT HOOK BRIDGE
    # =========================================================================

    def _bridge_agent_hooks(self) -> None:
        """Bridge hooks from the underlying BaseAgent into the AgentRunner
        lifecycle, so users only need to register hooks in one place.

        BaseAgent._pre_hooks  -> AgentRunner 'pre_execute'
        BaseAgent._post_hooks -> AgentRunner 'post_execute'
        """
        agent = self.agent
        if not hasattr(agent, '_pre_hooks') and not hasattr(agent, '_post_hooks'):
            return

        # Wrap BaseAgent hooks to match AgentRunner's hook signature (**context)
        for pre_hook in getattr(agent, '_pre_hooks', []):
            def _wrap_pre(fn: Any = pre_hook, **ctx: Any) -> None:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        # Can't await in sync hook runner — skip async agent hooks
                        logger.debug(f"Skipping async agent pre-hook (not supported in AgentRunner)")
                        return
                    fn(agent, **{k: v for k, v in ctx.items() if k in ('goal', 'kwargs')})
                except Exception as e:
                    logger.debug(f"Bridged agent pre-hook failed: {e}")
            self.add_hook('pre_execute', _wrap_pre, name=f"agent_bridge_pre_{id(pre_hook)}")

        for post_hook in getattr(agent, '_post_hooks', []):
            def _wrap_post(fn: Any = post_hook, **ctx: Any) -> None:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        return
                    result = ctx.get('agent_output')
                    fn(agent, result, **{k: v for k, v in ctx.items() if k == 'goal'})
                except Exception as e:
                    logger.debug(f"Bridged agent post-hook failed: {e}")
            self.add_hook('post_execute', _wrap_post, name=f"agent_bridge_post_{id(post_hook)}")

    # =========================================================================
    # LIFECYCLE HOOKS (AgentScope-inspired)
    # =========================================================================

    def add_hook(self, hook_type: str, fn: Callable, name: str = None) -> str:
        """
        Register a hook at a lifecycle point.

        AgentScope insight: Hooks make learning, tracing, profiling,
        and rate limiting pluggable instead of hardcoded.

        Args:
            hook_type: One of HOOK_TYPES (pre_run, post_run, etc.)
            fn: Callable(**context) -> Optional[dict]. If it returns a dict,
                the context is updated (e.g. modify goal in pre_run).
            name: Optional name for later removal.

        Returns:
            Hook name (auto-generated if not provided)

        Example:
            def log_timing(**ctx):
                print(f"Agent {ctx.get('agent_name')} took {ctx.get('elapsed'):.1f}s")

            runner.add_hook('post_run', log_timing)
        """
        if hook_type not in self._hooks:
            raise ValueError(
                f"Unknown hook type '{hook_type}'. "
                f"Valid: {', '.join(HOOK_TYPES)}"
            )
        hook_name = name or f"{hook_type}_{len(self._hooks[hook_type])}"
        fn._hook_name = hook_name  # tag for removal
        self._hooks[hook_type].append(fn)
        logger.info(f" Hook registered: {hook_type}/{hook_name}")
        return hook_name

    def remove_hook(self, hook_type: str, name: str) -> bool:
        """Remove a named hook. Returns True if found."""
        hooks = self._hooks.get(hook_type, [])
        for i, fn in enumerate(hooks):
            if getattr(fn, '_hook_name', None) == name:
                hooks.pop(i)
                return True
        return False

    def _run_hooks(self, hook_type: str, **context: Any) -> dict:
        """
        Run all hooks for a lifecycle point.

        KISS: Hooks receive context as kwargs and optionally return
        a dict to update it. No complex middleware chain.

        Args:
            hook_type: Which lifecycle point
            **context: Current execution context (goal, result, etc.)

        Returns:
            Updated context dict
        """
        for fn in self._hooks.get(hook_type, []):
            try:
                result = fn(**context)
                if isinstance(result, dict):
                    context.update(result)
            except Exception as e:
                hook_name = getattr(fn, '_hook_name', '?')
                logger.warning(f" Hook {hook_type}/{hook_name} failed: {e}")
        return context

    def _update_validators_for_task(self, goal: str) -> str:
        """
        Update validators with task-specific prompts based on goal analysis.

        Returns the detected task type.
        """
        from pathlib import Path

        # Initialize prompt selector on first use
        if self._prompt_selector is None:
            try:
                self._prompt_selector = get_prompt_selector()
            except Exception as e:
                logger.warning(f"Prompt selector not available: {e}")
                return 'default'

        # Detect task type from goal
        task_type = self._prompt_selector.detect_task_type(goal)

        # Skip if same task type (validators already configured)
        if task_type == self._current_task_type:
            return task_type

        # Get task-specific prompts
        architect_path, auditor_path = self._prompt_selector.select_prompts(goal)

        # Recreate validators with new prompts
        try:
            # Task-specific validation (primary layer)
            architect_agents = [
                ValidatorAgent(
                    md_path=Path(architect_path),
                    is_architect=True,
                    tools=[],
                    config=self.config.config,
                    scratchpad=self._scratchpad
                )
            ]

            auditor_agents = [
                ValidatorAgent(
                    md_path=Path(auditor_path),
                    is_architect=False,
                    tools=[],
                    config=self.config.config,
                    scratchpad=self._scratchpad
                )
            ]

            # Add swarm-level validation (orchestration/coordination layer)
            # Only load once, then reuse across task types
            if not self._swarm_prompts_loaded:
                try:
                    import tempfile

                    # Create temp files for swarm prompts (ValidatorAgent needs file paths)
                    swarm_arch_file = tempfile.NamedTemporaryFile(mode='w', suffix='_swarm_architect.md', delete=False)
                    swarm_arch_file.write(get_swarm_architect_prompt())
                    swarm_arch_file.close()

                    swarm_aud_file = tempfile.NamedTemporaryFile(mode='w', suffix='_swarm_auditor.md', delete=False)
                    swarm_aud_file.write(get_swarm_auditor_prompt('coordination'))
                    swarm_aud_file.close()

                    # Add swarm architect (orchestration readiness)
                    architect_agents.append(
                        ValidatorAgent(
                            md_path=Path(swarm_arch_file.name),
                            is_architect=True,
                            tools=[],
                            config=self.config.config,
                            scratchpad=self._scratchpad
                        )
                    )

                    # Add swarm auditor (coordination & goal alignment)
                    auditor_agents.append(
                        ValidatorAgent(
                            md_path=Path(swarm_aud_file.name),
                            is_architect=False,
                            tools=[],
                            config=self.config.config,
                            scratchpad=self._scratchpad
                        )
                    )

                    self._swarm_prompts_loaded = True
                    logger.info(" Swarm-level validators added (orchestration + coordination)")

                except Exception as e:
                    logger.warning(f"Failed to load swarm prompts: {e}")

            self.architect_validator = MultiRoundValidator(architect_agents, self.config.config)
            self.auditor_validator = MultiRoundValidator(auditor_agents, self.config.config)

            self._current_task_type = task_type
            logger.info(f" Validators updated for task type: {task_type}")

        except Exception as e:
            logger.warning(f"Failed to update validators for {task_type}: {e}")
            # Fallback to generic auditor if task-specific prompts fail
            try:
                import tempfile
                generic_aud_file = tempfile.NamedTemporaryFile(mode='w', suffix='_generic_auditor.md', delete=False)
                generic_aud_file.write(get_generic_auditor_prompt())
                generic_aud_file.close()

                auditor_agents = [
                    ValidatorAgent(
                        md_path=Path(generic_aud_file.name),
                        is_architect=False,
                        tools=[],
                        config=self.config.config,
                        scratchpad=self._scratchpad
                    )
                ]
                self.auditor_validator = MultiRoundValidator(auditor_agents, self.config.config)
                logger.info(" Fallback: Using generic auditor for validation")
            except Exception as fallback_err:
                logger.error(f"Fallback validator also failed: {fallback_err}")

        return task_type

    # =========================================================================
    # PIPELINE STAGES (extracted from run() for readability)
    # =========================================================================

    def _gather_learning_context(self, goal: str) -> List[str]:
        """Stage 1: Gather learning context from memory, Q-learning, transfer
        learning, and swarm intelligence. Returns list of context strings.

        Separated from run() so the method stays focused on orchestration.

        Budget guard (Cline-inspired): total injected context is capped at
        max_learning_context_chars (default ~8000 chars ≈ 2000 tokens).
        Prevents learning context from eating up the model's context window.
        """
        parts = []
        _max_chars = getattr(
            self.config.config, 'max_learning_context_chars', 8000
        )

        # 1. Memory retrieval
        if self.agent_memory:
            try:
                from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
                memory_budget = getattr(
                    self.config.config, 'memory_retrieval_budget', 3000
                )
                # Use fast retrieval (keyword + recency + value, NO LLM call).
                # The full LLM-scored retrieve() costs 1+ extra LLM calls per
                # execution; for pre-execution context injection, fast retrieval
                # gives 80% of the recall at 0% of the latency cost.
                _retrieve_fn = getattr(
                    self.agent_memory, 'retrieve_fast',
                    self.agent_memory.retrieve,  # fallback if not available
                )
                relevant_memories = _retrieve_fn(
                    query=goal,
                    goal=goal,
                    budget_tokens=memory_budget,
                    levels=[MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL, MemoryLevel.META]
                )
                if relevant_memories:
                    context = "\n".join([m.content for m in relevant_memories[:5]])
                    parts.append(f"Relevant past experience:\n{context}")
                    logger.info(f"Memory retrieval: {len(relevant_memories)} memories injected as context")
            except (MemoryRetrievalError, KeyError, TypeError) as e:
                logger.debug(f"Memory retrieval skipped: {e}")
            except Exception as e:
                logger.warning(f"Memory retrieval unexpected error: {type(e).__name__}: {e}")

        # 2. Q-learning context from swarm-level learner
        #    Include task_type so Q-learner matches relevant lessons only
        if self.learning_manager:
            try:
                _task_type = ''
                if self.transfer_learning:
                    _task_type = self.transfer_learning.extractor.extract_task_type(goal)
                state = {'query': goal, 'agent': self.agent_name, 'task_type': _task_type}
                q_context = self.learning_manager.get_learned_context(state)
                if q_context:
                    parts.append(f"Learned Insights:\n{q_context}")
            except (LearningError, KeyError, AttributeError) as e:
                logger.debug(f"Q-learning context injection skipped: {e}")

        # 3. Transferable learning context (cross-swarm, cross-goal)
        if self.transfer_learning:
            try:
                transfer_context = self.transfer_learning.format_context_for_agent(goal, self.agent_name)
                if transfer_context and 'Transferable Learnings' in transfer_context:
                    parts.append(transfer_context)
            except (LearningError, KeyError, AttributeError) as e:
                logger.debug(f"Transfer learning context injection skipped: {e}")

        # 4. Swarm intelligence hints (stigmergy + agent profile + condensed history)
        if self.swarm_intelligence:
            try:
                _si = self.swarm_intelligence
                profile = _si.agent_profiles.get(self.agent_name)
                if profile and profile.total_tasks > 0:
                    _success_pct = int(profile.trust_score * 100)
                    _spec = profile.specialization.value
                    parts.append(
                        f"Your track record: {_success_pct}% trust, "
                        f"specialization={_spec}, {profile.total_tasks} tasks completed."
                    )

                if hasattr(_si, 'stigmergy'):
                    task_type = self.transfer_learning.extractor.extract_task_type(goal) if self.transfer_learning else None
                    if task_type:
                        route_signals = _si.stigmergy.get_route_signals(task_type)
                        if route_signals:
                            best_agent = max(route_signals, key=route_signals.get)
                            if best_agent == self.agent_name:
                                parts.append(
                                    f"Stigmergy hint: you are the top-performing agent "
                                    f"for '{task_type}' tasks. Lean into your strengths."
                                )

                # Condensed collective history (Cline condense pattern):
                # Instead of raw episode data, give the agent a statistical
                # summary of old episodes. Recent ones are kept verbatim in
                # the collective_memory deque for Q-learning / routing.
                condensed = _si.condense_collective_memory(keep_recent=20)
                if condensed:
                    parts.append(condensed)
            except Exception as e:
                logger.debug(f"Warm-start context skipped: {e}")

        # Consecutive failure hint — tell the agent to change approach
        if self._consecutive_failures >= self._max_consecutive_before_hint:
            parts.append(
                f"WARNING: {self._consecutive_failures} consecutive failures. "
                f"Your previous approach is not working. Try a fundamentally "
                f"different strategy — different tools, simpler steps, or "
                f"break the problem into smaller parts."
            )

        # Budget guard: compress (not drop) when total exceeds budget.
        # Pattern from SmartContextManager: keep start + end of each chunk,
        # compress the middle. This preserves signal while fitting budget.
        total_chars = sum(len(p) for p in parts)
        if total_chars > _max_chars and parts:
            # Calculate how much each part can have (fair share)
            per_part_budget = _max_chars // len(parts)
            compressed_parts = []
            for p in parts:
                if len(p) <= per_part_budget:
                    compressed_parts.append(p)
                else:
                    # Keep start + end, mark middle as compressed
                    keep = max(100, per_part_budget)
                    half = keep // 2
                    compressed_parts.append(
                        p[:half] + "\n[...compressed...]\n" + p[-half:]
                    )
            parts = compressed_parts
            logger.debug(
                f"Context budget: compressed {total_chars} → "
                f"{sum(len(p) for p in parts)} chars "
                f"(budget: {_max_chars})"
            )

        for i, p in enumerate(parts, 1):
            logger.info(f" Context {i}: {p[:500]}")
        return parts

    async def _record_post_execution_learning(self, goal: str, agent_output: Any, success: bool, trajectory: list, architect_results: list, auditor_results: list, architect_shaped_reward: float, duration: float, kwargs: dict) -> dict:
        """Stage 3: Record learning data (memory, TD-lambda, Q-learning, feedback).

        Returns dict with episode_memory_entry, tagged_outputs, agent_contributions.
        """
        import time

        # Memory storage
        episode_memory_entry = None
        if self.agent_memory:
            from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
            episode_memory_entry = self.agent_memory.store(
                content=f"Goal: {goal}\nOutput: {str(agent_output)[:500]}",
                level=MemoryLevel.EPISODIC,
                context={'agent': self.agent_name, 'goal': goal},
                goal=goal
            )

        # Shaped rewards after auditor validation
        auditor_shaped_reward = 0.0
        if self.shaped_reward_manager:
            auditor_shaped_reward = self.shaped_reward_manager.check_rewards(
                event_type="validation",
                state={
                    'auditor_results': auditor_results,
                    'passed': success,
                    'goal': goal,
                    'output': str(agent_output)[:500]
                },
                trajectory=trajectory
            )
            self.shaped_reward_manager.check_rewards(
                event_type="actor_complete",
                state={
                    'output': str(agent_output)[:500],
                    'success': success,
                    'goal': goal
                },
                trajectory=trajectory
            )

        # TD(lambda) learning update
        final_reward = 0.0
        if self.agent_learner:
            step_reward = architect_shaped_reward + auditor_shaped_reward
            if episode_memory_entry:
                self.agent_learner.record_access(episode_memory_entry, step_reward=step_reward)

            terminal_reward = 1.0 if success else -0.5
            shaped_total = self.shaped_reward_manager.get_total_reward() if self.shaped_reward_manager else 0.0
            final_reward = terminal_reward + shaped_total

            memories_dict = {}
            if episode_memory_entry:
                memories_dict[episode_memory_entry.key] = episode_memory_entry

            updates = self.agent_learner.end_episode(
                final_reward=final_reward,
                memories=memories_dict
            )
            if updates:
                logger.debug(f"Learning: Updated {len(updates)} memory values (shaped={shaped_total:.3f})")

        # Q-learning record
        if self.learning_manager:
            try:
                q_state = {'query': goal, 'agent': self.agent_name, 'success': success}
                q_action = {'actor': self.agent_name, 'task': goal[:100]}
                q_reward = final_reward if final_reward else (1.0 if success else -0.5)
                self.learning_manager.record_outcome(q_state, q_action, q_reward, done=True)
            except (LearningError, KeyError, AttributeError) as e:
                logger.debug(f"Swarm Q-learning record skipped: {e}")

        # Memory consolidation
        if self.agent_memory:
            try:
                await self.agent_memory.consolidate()
            except Exception as e:
                logger.debug(f"Memory consolidation skipped: {type(e).__name__}: {e}")

        # Executor feedback to SwarmIntelligence
        if self.swarm_intelligence:
            try:
                detected_task_type = self._current_task_type if hasattr(self, '_current_task_type') else None
                # Extract skills actually used from agent output
                _tools_used = []
                if isinstance(agent_output, dict):
                    _tools_used = list(agent_output.get('skills_used', []))
                if not _tools_used and hasattr(agent_output, 'skills_used'):
                    _tools_used = list(getattr(agent_output, 'skills_used', []))
                self.swarm_intelligence.receive_executor_feedback(
                    task_id=f"{self.agent_name}_{int(time.time())}",
                    success=success,
                    tools_used=_tools_used,
                    execution_time=duration,
                    error_type=None,
                    task_type=detected_task_type
                )
            except Exception as fb_err:
                logger.debug(f"Executor feedback skipped: {fb_err}")

        # Build tagged outputs
        from Jotty.core.infrastructure.foundation.types.learning_types import TaggedOutput
        tagged_outputs = []
        if auditor_results:
            for result in auditor_results:
                if result.output_tag:
                    tagged_outputs.append(TaggedOutput(
                        name=self.agent_name,
                        tag=result.output_tag,
                        why_useful=result.why_useful or result.reasoning,
                        content=agent_output
                    ))

        # Build agent contributions
        agent_contributions = {}
        if architect_results:
            from Jotty.core.infrastructure.foundation.types.agent_types import AgentContribution
            for result in architect_results:
                agent_contributions[result.agent_name] = AgentContribution(
                    agent_name=result.agent_name,
                    contribution_score=result.confidence if result.should_proceed else -result.confidence,
                    decision="approve" if result.should_proceed else "reject",
                    decision_correct=success,
                    counterfactual_impact=0.5,
                    reasoning_quality=result.confidence,
                    evidence_used=[],
                    tools_used=result.tool_calls or [],
                    decision_timing=0.5,
                    temporal_weight=1.0
                )

        return {
            'episode_memory_entry': episode_memory_entry,
            'tagged_outputs': tagged_outputs,
            'agent_contributions': agent_contributions,
        }

    async def run(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Run agent execution with validation and learning.

        Orchestrates a pipeline of stages via ExecutionContext:
        setup → gather context → architect → execute → auditor → record result.

        Judge intervention (MALLM-inspired): when auditor_confidence < 0.6 and
        feedback is not "No feedback", retries with judge_goal incorporating
        auditor reasoning. Uses _judge_retried flag (checks not _judge_retried)
        to prevent infinite retry loops.

        Args:
            goal: Task goal/description
            skip_validation: If True, skip architect validation (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            **kwargs: Additional arguments for agent

        Returns:
            EpisodeResult with output and metadata
        """
        ctx = None
        try:
            ctx = await self._setup_context(goal, **kwargs)
            ctx = await self._gather_context(ctx)
            ctx = await self._validate_architect(ctx)
            ctx = await self._execute_agent(ctx)
            ctx = await self._validate_auditor_with_retry(ctx)
            return await self._record_and_build_result(ctx)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if ctx is None:
                import time as _t
                ctx = ExecutionContext(goal=goal, kwargs=kwargs, start_time=_t.time())
            return await self._handle_execution_error(ctx, e)

    # =========================================================================
    # PIPELINE STAGES (extracted from run() for testability)
    # =========================================================================

    async def _setup_context(self, goal: str, **kwargs: Any) -> ExecutionContext:
        """Parse kwargs, validation gate, task progress, pre_run hook, TD-lambda start.

        Returns a populated ExecutionContext ready for subsequent stages.
        """
        import time
        ctx = ExecutionContext(
            goal=goal,
            kwargs=kwargs,
            start_time=time.time(),
        )

        # Extract flags
        skip_validation = ctx.kwargs.pop('skip_validation', False)
        validation_mode_override = ctx.kwargs.pop('validation_mode', None)
        ctx.status_callback = ctx.kwargs.pop('status_callback', None)

        ctx._status = StatusReporter(ctx.status_callback, logger, emoji=" ")

        # ── Intelligent Validation Gate ───────────────────────────────
        if skip_validation:
            force_mode = ValidationMode.DIRECT
        elif validation_mode_override:
            mode_map = {"direct": ValidationMode.DIRECT, "audit": ValidationMode.AUDIT_ONLY, "full": ValidationMode.FULL}
            force_mode = mode_map.get(str(validation_mode_override).lower(), None)
        else:
            force_mode = None

        # Lazy-init the gate (singleton, shared across runs)
        if self._validation_gate is None:
            self._validation_gate = get_validation_gate()

        ctx.gate_decision = await self._validation_gate.decide(
            goal=goal,
            agent_name=self.agent_name,
            force_mode=force_mode,
        )

        # Derive skip flags from gate decision
        ctx.skip_architect = ctx.gate_decision.mode in (ValidationMode.DIRECT, ValidationMode.AUDIT_ONLY)
        ctx.skip_auditor = ctx.gate_decision.mode == ValidationMode.DIRECT

        mode_labels = {
            ValidationMode.DIRECT: "DIRECT (actor only)",
            ValidationMode.AUDIT_ONLY: "AUDIT (actor + auditor)",
            ValidationMode.FULL: "FULL (architect + actor + auditor)",
        }
        mode_label = mode_labels[ctx.gate_decision.mode]
        logger.info(
            f"AgentRunner.run: {self.agent_name} - {goal[:50]}... "
            f"[{mode_label}] gate={ctx.gate_decision.confidence:.0%} "
            f"({ctx.gate_decision.reason}) {ctx.gate_decision.latency_ms:.0f}ms"
        )
        ctx._status("ValidationGate", f"{mode_label} — {ctx.gate_decision.reason}")

        # Reset tool guard per turn
        self.tool_guard.reset_turn()

        # Initialize task progress tracker (Cline Focus Chain)
        ctx.task_progress = TaskProgress(goal=goal)
        ctx.task_progress.add_step("Gather context")
        ctx.task_progress.add_step("Validate approach")
        ctx.task_progress.add_step("Execute task")
        ctx.task_progress.add_step("Verify output")
        self.task_progress = ctx.task_progress

        # Hook: pre_run — modify goal, inject context, log start
        hook_ctx = self._run_hooks(
            'pre_run',
            goal=goal, agent_name=self.agent_name,
            skip_validation=ctx.skip_architect and ctx.skip_auditor,
            gate_decision=ctx.gate_decision, kwargs=ctx.kwargs,
        )
        ctx.goal = hook_ctx.get('goal', goal)

        # Dynamic prompt selection based on task type
        if not ctx.skip_architect:
            task_type = self._update_validators_for_task(ctx.goal)
            logger.debug(f"Task type detected: {task_type}")

        # Start episode for TD(λ) learning (if enabled)
        if self.agent_learner:
            self.agent_learner.start_episode(ctx.goal)

        # Reset shaped rewards for new episode
        if self.shaped_reward_manager:
            self.shaped_reward_manager.reset()

        return ctx

    async def _gather_context(self, ctx: ExecutionContext) -> ExecutionContext:
        """Gather learning context from memory, Q-learning, transfer learning."""
        ctx.task_progress.start_step(0)  # Gather context
        ctx._status("Preparing", "retrieving context")

        ctx.learning_context_parts = self._gather_learning_context(ctx.goal)

        if ctx.learning_context_parts:
            ctx.kwargs['learning_context'] = "\n\n".join(ctx.learning_context_parts)
            logger.info(
                f"Learning context: {len(ctx.learning_context_parts)} sections "
                f"({sum(len(p) for p in ctx.learning_context_parts)} chars)"
            )

        # Pass workspace_dir so project rules get loaded
        if 'workspace_dir' not in ctx.kwargs:
            import os
            ctx.kwargs['workspace_dir'] = os.getcwd()

        ctx.enriched_goal = ctx.goal
        ctx.task_progress.complete_step(0)  # Context gathered
        return ctx

    async def _validate_architect(self, ctx: ExecutionContext) -> ExecutionContext:
        """Architect (pre-execution) validation + shaped reward."""
        if not ctx.skip_architect:
            # Hook: pre_architect
            self._run_hooks('pre_architect', goal=ctx.goal, agent_name=self.agent_name)

            ctx.task_progress.start_step(1)  # Validate approach
            ctx._status("Architect", "validating approach")
            ctx.architect_results, ctx.proceed = await self.architect_validator.validate(
                goal=ctx.goal,
                inputs={'goal': ctx.goal, **ctx.kwargs},
                trajectory=[],
                is_architect=True
            )

            # Track architect validation in state
            if self.swarm_state_manager:
                avg_confidence = (
                    sum(r.confidence for r in ctx.architect_results) / len(ctx.architect_results)
                    if ctx.architect_results else 0.0
                )
                self.agent_tracker.record_validation(
                    validation_type='architect',
                    passed=ctx.proceed,
                    confidence=avg_confidence,
                    feedback=ctx.architect_results[0].reasoning if ctx.architect_results else None
                )
                self.swarm_state_manager.record_swarm_step({
                    'agent': self.agent_name,
                    'step': 'architect',
                    'proceed': ctx.proceed,
                    'confidence': avg_confidence,
                    'architect_confidence': avg_confidence
                })

            if ctx.architect_results:
                avg_confidence = sum(r.confidence for r in ctx.architect_results) / len(ctx.architect_results)
                logger.debug(
                    f"Architect confidence: {avg_confidence:.2f} "
                    f"(Decision: {'PROCEED' if ctx.proceed else 'BLOCKED'})"
                )

            # Hook: post_architect — inspect/override proceed decision
            arch_ctx = self._run_hooks(
                'post_architect', goal=ctx.goal, agent_name=self.agent_name,
                architect_results=ctx.architect_results, proceed=ctx.proceed,
            )
            ctx.proceed = arch_ctx.get('proceed', ctx.proceed)

            # Shaped reward: architect validation
            if self.shaped_reward_manager and ctx.architect_results:
                ctx.architect_shaped_reward = self.shaped_reward_manager.check_rewards(
                    event_type="actor_start",
                    state={'architect_results': ctx.architect_results, 'proceed': ctx.proceed, 'goal': ctx.goal},
                    trajectory=[]
                )
        else:
            logger.info(f" Skipping architect: gate={ctx.gate_decision.mode.value}")

        ctx.task_progress.complete_step(1)  # Approach validated
        return ctx

    async def _execute_agent(self, ctx: ExecutionContext) -> ExecutionContext:
        """Pre-execute hook, workspace checkpoint, agent execution, trajectory building."""
        # Hook: pre_execute — last chance to modify goal
        exec_ctx = self._run_hooks(
            'pre_execute', goal=ctx.enriched_goal, agent_name=self.agent_name,
            architect_results=ctx.architect_results,
        )
        ctx.enriched_goal = exec_ctx.get('goal', ctx.enriched_goal)

        ctx.task_progress.start_step(2)  # Execute task

        # Workspace checkpoint before execution (Cline checkpoint pattern)
        try:
            from .workspace_checkpoint import WorkspaceCheckpoint
            _ws_cp = WorkspaceCheckpoint()
            if _ws_cp._git_available:
                ctx.ws_checkpoint_id = _ws_cp.save(f"pre-exec:{ctx.goal[:50]}")
                logger.debug(f"Workspace checkpoint: {ctx.ws_checkpoint_id}")
        except Exception as _cp_err:
            logger.debug(f"Workspace checkpoint skipped: {_cp_err}")

        ctx._status("Agent", "executing task (this may take a while)")

        # Execute agent (handles AutoAgent, DSPy, callable)
        if hasattr(self.agent, 'execute'):
            if ctx.status_callback:
                ctx.kwargs['status_callback'] = ctx.status_callback
            ctx.agent_output = await self.agent.execute(ctx.enriched_goal, **ctx.kwargs)
        elif hasattr(self.agent, 'forward'):
            ctx.agent_output = self.agent(goal=ctx.goal, **ctx.kwargs)
        else:
            ctx.agent_output = (
                await self.agent(ctx.goal, **ctx.kwargs)
                if asyncio.iscoroutinefunction(self.agent)
                else self.agent(ctx.goal, **ctx.kwargs)
            )

        # Determine inner success from agent output
        ctx.inner_success = True
        if hasattr(ctx.agent_output, 'success'):
            ctx.inner_success = bool(ctx.agent_output.success)
        elif isinstance(ctx.agent_output, dict):
            ctx.inner_success = bool(ctx.agent_output.get('success', True))

        # Build rich trajectory from ExecutionResult
        _skills_used = []
        if hasattr(ctx.agent_output, 'skills_used') and ctx.agent_output.skills_used:
            _skills_used = list(ctx.agent_output.skills_used)
        elif isinstance(ctx.agent_output, dict):
            _skills_used = list(ctx.agent_output.get('skills_used', []))

        _outputs = {}
        if hasattr(ctx.agent_output, 'outputs') and isinstance(ctx.agent_output.outputs, dict):
            _outputs = ctx.agent_output.outputs
        elif isinstance(ctx.agent_output, dict):
            _outputs = ctx.agent_output.get('outputs', {})

        # Add one trajectory entry per executed step (skill)
        ctx.trajectory = []
        if _outputs:
            for i, (step_name, step_data) in enumerate(_outputs.items(), 1):
                step_success = True
                step_skill = step_name
                if isinstance(step_data, dict):
                    step_success = step_data.get('success', True)
                    step_skill = step_data.get('skill', step_name)
                ctx.trajectory.append({
                    'step': i,
                    'skill': step_skill,
                    'action': step_name,
                    'success': step_success,
                    'output': str(step_data)[:200] if step_data else None,
                })
        else:
            ctx.trajectory.append({
                'step': 1,
                'action': 'execute',
                'skills_used': _skills_used,
                'output': str(ctx.agent_output)[:500] if ctx.agent_output else None,
                'success': ctx.inner_success,
            })

        # Hook: post_execute — inspect/transform output before auditor
        self._run_hooks(
            'post_execute', goal=ctx.enriched_goal, agent_name=self.agent_name,
            agent_output=ctx.agent_output, inner_success=ctx.inner_success,
        )

        ctx.task_progress.complete_step(2)  # Task executed
        return ctx

    async def _validate_auditor_with_retry(self, ctx: ExecutionContext) -> ExecutionContext:
        """Auditor validation + MALLM judge retry logic."""
        ctx.task_progress.start_step(3)  # Verify output

        if not ctx.skip_auditor:
            ctx.auditor_results, passed = await self.auditor_validator.validate(
                goal=ctx.goal,
                inputs={'goal': ctx.goal, 'output': str(ctx.agent_output)},
                trajectory=ctx.trajectory,
                is_architect=False
            )

            ctx.success = passed and ctx.inner_success
            ctx.auditor_reasoning = ctx.auditor_results[0].reasoning if ctx.auditor_results else "No feedback"
            ctx.auditor_confidence = ctx.auditor_results[0].confidence if ctx.auditor_results else 0.0

            # JUDGE INTERVENTION (MALLM-inspired turn regeneration)
            _judge_retried = ctx.kwargs.get('_judge_retried', False)
            if (
                not ctx.success
                and not _judge_retried
                and ctx.auditor_reasoning
                and ctx.auditor_reasoning != "No feedback"
                and len(ctx.auditor_reasoning) > 20
            ):
                logger.info(
                    f" Judge intervention: auditor rejected "
                    f"(confidence={ctx.auditor_confidence:.2f}), retrying with feedback"
                )
                ctx._status("Judge intervention", f"retrying with auditor feedback")

                # Hook: allow external observers to see intervention
                self._run_hooks(
                    'post_execute', goal=ctx.enriched_goal,
                    agent_name=self.agent_name,
                    agent_output=ctx.agent_output, inner_success=False,
                    judge_intervention=True,
                    auditor_reasoning=ctx.auditor_reasoning,
                )

                judge_feedback = (
                    f"[Judge feedback — your previous attempt was rejected]:\n"
                    f"{ctx.auditor_reasoning}\n"
                    f"Please address the feedback and try again."
                )

                # Re-run execution with feedback
                if hasattr(self.agent, 'execute'):
                    retry_kwargs = dict(ctx.kwargs)
                    retry_kwargs['_judge_retried'] = True
                    existing_lc = retry_kwargs.get('learning_context', '') or ''
                    retry_kwargs['learning_context'] = (
                        existing_lc + '\n\n' + judge_feedback
                    ).strip()
                    if ctx.status_callback:
                        retry_kwargs['status_callback'] = ctx.status_callback
                    ctx.agent_output = await self.agent.execute(ctx.enriched_goal, **retry_kwargs)
                elif hasattr(self.agent, 'forward'):
                    ctx.agent_output = self.agent(goal=ctx.goal, **ctx.kwargs)
                else:
                    ctx.agent_output = (
                        await self.agent(ctx.goal, **ctx.kwargs)
                        if asyncio.iscoroutinefunction(self.agent)
                        else self.agent(ctx.goal, **ctx.kwargs)
                    )

                # Re-validate after retry
                ctx.inner_success = True
                if hasattr(ctx.agent_output, 'success'):
                    ctx.inner_success = bool(ctx.agent_output.success)
                elif isinstance(ctx.agent_output, dict):
                    ctx.inner_success = bool(ctx.agent_output.get('success', True))

                ctx.auditor_results, passed = await self.auditor_validator.validate(
                    goal=ctx.goal,
                    inputs={'goal': ctx.goal, 'output': str(ctx.agent_output)},
                    trajectory=ctx.trajectory,
                    is_architect=False
                )
                ctx.success = passed and ctx.inner_success
                ctx.auditor_reasoning = ctx.auditor_results[0].reasoning if ctx.auditor_results else "No feedback"
                ctx.auditor_confidence = ctx.auditor_results[0].confidence if ctx.auditor_results else 0.0

                logger.info(
                    f" Judge retry result: "
                    f"{' passed' if ctx.success else ' still failed'} "
                    f"(confidence={ctx.auditor_confidence:.2f})"
                )

            # Track auditor validation in state
            if self.swarm_state_manager:
                self.agent_tracker.record_validation(
                    validation_type='auditor',
                    passed=passed,
                    confidence=ctx.auditor_confidence,
                    feedback=ctx.auditor_reasoning
                )
                output_type = type(ctx.agent_output).__name__
                self.agent_tracker.record_output(ctx.agent_output, output_type)
                self.swarm_state_manager.record_swarm_step({
                    'agent': self.agent_name,
                    'step': 'auditor',
                    'success': ctx.success,
                    'validation_passed': passed,
                    'auditor_result': ctx.auditor_reasoning[:100],
                    'auditor_confidence': ctx.auditor_confidence
                })
        else:
            # DIRECT mode: skip auditor, but respect inner execution result
            logger.info(f" Skipping auditor: gate={ctx.gate_decision.mode.value}")
            ctx.success = ctx.inner_success
            ctx.auditor_reasoning = f"Gate={ctx.gate_decision.mode.value}: auditor skipped"
            ctx.auditor_confidence = 1.0
            ctx.auditor_results = []
            passed = True

        # Update trajectory with validation result
        if ctx.trajectory:
            ctx.trajectory[0]['success'] = ctx.success
            ctx.trajectory[0]['validation'] = {
                'passed': passed,
                'confidence': ctx.auditor_confidence,
                'tag': (
                    ctx.auditor_results[0].output_tag.value
                    if ctx.auditor_results and ctx.auditor_results[0].output_tag
                    else None
                )
            }

        return ctx

    async def _record_and_build_result(self, ctx: ExecutionContext) -> EpisodeResult:
        """Record learning, build EpisodeResult, post_run hook."""
        import time
        ctx.duration = time.time() - ctx.start_time
        ctx.learning_data = await self._record_post_execution_learning(
            goal=ctx.goal, agent_output=ctx.agent_output, success=ctx.success,
            trajectory=ctx.trajectory, architect_results=ctx.architect_results,
            auditor_results=ctx.auditor_results or [],
            architect_shaped_reward=ctx.architect_shaped_reward,
            duration=ctx.duration, kwargs=ctx.kwargs
        )

        episode_result = EpisodeResult(
            output=ctx.agent_output,
            success=ctx.success,
            trajectory=ctx.trajectory,
            tagged_outputs=ctx.learning_data['tagged_outputs'],
            episode=0,
            execution_time=ctx.duration,
            architect_results=ctx.architect_results or [],
            auditor_results=ctx.auditor_results or [],
            agent_contributions=ctx.learning_data['agent_contributions']
        )

        # Expose workspace checkpoint for rollback by caller
        if ctx.ws_checkpoint_id:
            episode_result.trajectory.append(
                {"type": "workspace_checkpoint", "id": ctx.ws_checkpoint_id}
            )

        # Update task progress + consecutive failure counter
        if ctx.success:
            ctx.task_progress.complete_step(3)  # Verified OK
            self._consecutive_failures = 0
        else:
            ctx.task_progress.fail_step(3)  # Verification failed
            self._consecutive_failures += 1

        # Record gate outcome for drift detection
        if self._validation_gate:
            self._validation_gate.record_outcome(ctx.gate_decision.mode, ctx.success)

        # Log task progress (Cline Focus Chain visibility)
        if ctx.task_progress:
            logger.info(f" {ctx.task_progress.render()}")

        # Hook: post_run — record metrics, log results, send notifications
        self._run_hooks(
            'post_run', goal=ctx.goal, agent_name=self.agent_name,
            result=episode_result, success=ctx.success, elapsed=ctx.duration,
            gate_decision=ctx.gate_decision,
            task_progress=ctx.task_progress.summary() if ctx.task_progress else None,
        )

        return episode_result

    async def _handle_execution_error(self, ctx: ExecutionContext, error: Exception) -> EpisodeResult:
        """Exception classification, auto-fix via SwarmTerminal, error recording."""
        import time

        if isinstance(error, (AgentExecutionError, ToolExecutionError)):
            logger.error(f" Agent execution error: {error}")
            error_str = str(error)
            error_type = type(error).__name__
            fix_applied = False
            fix_description = ""

        elif isinstance(error, (LLMError, DSPyError)):
            logger.error(f" LLM/DSPy error during execution: {error}")
            error_str = str(error)
            error_type = type(error).__name__
            fix_applied = False
            fix_description = ""

        elif isinstance(error, asyncio.TimeoutError):
            logger.error(f" Agent timed out: {error}")
            error_str = f"Agent timed out: {error}"
            error_type = "TimeoutError"
            fix_applied = False
            fix_description = ""

        else:
            # Unexpected errors — log full traceback, attempt auto-fix
            logger.error(f" Agent execution failed (unexpected {type(error).__name__}): {error}", exc_info=True)
            error_str = str(error)
            error_type = type(error).__name__
            fix_applied = False
            fix_description = ""

            # Try auto-fix using SwarmTerminal (only for unexpected errors)
            if self.swarm_terminal:
                try:
                    logger.info(f" Attempting auto-fix via SwarmTerminal...")
                    error_keywords = ['command', 'module', 'import', 'pip', 'npm', 'permission', 'not found']
                    if any(kw in error_str.lower() for kw in error_keywords):
                        diagnostics = await self.swarm_terminal.diagnose_system()
                        solution = await self.swarm_terminal._find_solution(ctx.goal, error_str)
                        if solution and solution.commands:
                            logger.info(f"   Found solution: {solution.solution[:100]}")
                            for cmd in solution.commands[:3]:
                                result = await self.swarm_terminal.execute(cmd, auto_fix=False)
                                if result.success:
                                    fix_applied = True
                                    fix_description = f"Applied: {cmd}"
                                    logger.info(f" Fix applied: {cmd}")

                            # Retry the original execution if fix was applied
                            if fix_applied:
                                logger.info(f" Retrying execution after fix...")
                                ctx._status("Retrying", "after auto-fix")
                                if not ctx.kwargs.get('_retry_after_fix'):
                                    ctx.kwargs['_retry_after_fix'] = True
                                    return await self.run(ctx.goal, **ctx.kwargs)

                except Exception as fix_error:
                    logger.debug(f"Auto-fix attempt failed: {fix_error}")

        # ── ERROR RECORDING (shared by all branches) ───────────
        if self.swarm_state_manager:
            self.agent_tracker.record_error(
                error=error_str,
                error_type=error_type,
                context={'goal': ctx.goal, 'kwargs': ctx.kwargs, 'fix_applied': fix_applied}
            )
            self.swarm_state_manager.record_swarm_step({
                'agent': self.agent_name,
                'step': 'error',
                'error': error_str,
                'error_type': error_type,
                'success': False,
                'fix_applied': fix_applied,
                'fix_description': fix_description
            })

        duration = time.time() - ctx.start_time

        # Agent0: Send executor feedback for failed execution
        if self.swarm_intelligence:
            try:
                detected_task_type = self._current_task_type if hasattr(self, '_current_task_type') else None
                self.swarm_intelligence.receive_executor_feedback(
                    task_id=f"{self.agent_name}_{int(time.time())}",
                    success=False,
                    tools_used=[],
                    execution_time=duration,
                    error_type=error_type,
                    task_type=detected_task_type
                )
                logger.debug(f"Executor feedback sent: success=False, error_type={error_type}")
            except Exception as fb_err:
                logger.debug(f"Executor feedback skipped: {fb_err}")

        # Record gate outcome for failure
        if self._validation_gate and ctx.gate_decision:
            self._validation_gate.record_outcome(ctx.gate_decision.mode, False)

        self._consecutive_failures += 1

        return EpisodeResult(
            output=None,
            success=False,
            trajectory=[{'step': 0, 'action': 'error', 'error': error_str, 'error_type': error_type, 'fix_applied': fix_applied}],
            tagged_outputs=[],
            episode=0,
            execution_time=duration,
            architect_results=ctx.architect_results,
            auditor_results=[],
            agent_contributions={},
            alerts=[f"{error_type}: {error_str[:100]}" + (f" (fix applied: {fix_description})" if fix_applied else "")]
        )

    @property
    def gate_stats(self) -> dict:
        """Get validation gate statistics for introspection."""
        if self._validation_gate:
            return self._validation_gate.stats()
        return {"status": "not initialized"}
