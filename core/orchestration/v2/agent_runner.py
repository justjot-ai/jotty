"""
V2 AgentRunner - Executes a single agent with validation and learning

Extracted from SingleAgentOrchestrator for reuse in SwarmManager.

AgentScope-inspired lifecycle hooks (Gao et al., 2025):
    Pluggable hooks at 6 lifecycle points let you add tracing, profiling,
    rate limiting, or custom logic without editing core code.

    runner.add_hook('pre_execute', my_logger)
    runner.add_hook('post_run', my_metrics_recorder)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from Jotty.core.foundation.data_structures import JottyConfig, EpisodeResult
from Jotty.core.foundation.exceptions import (
    AgentExecutionError,
    ToolExecutionError,
    LLMError,
    DSPyError,
    MemoryRetrievalError,
    MemoryStorageError,
    ConsolidationError,
    LearningError,
)
from Jotty.core.agents.inspector import InspectorAgent, MultiRoundValidator
from Jotty.core.memory.cortex import HierarchicalMemory
from Jotty.core.utils.prompt_selector import get_prompt_selector, PromptSelector
from Jotty.core.learning.learning import (
    TDLambdaLearner, AdaptiveLearningRate,
)
from Jotty.core.learning.shaped_rewards import ShapedRewardManager
from Jotty.core.orchestration.v2.validation_gate import (
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
    config: JottyConfig
    agent_name: str = "agent"
    enable_learning: bool = True
    enable_memory: bool = True
    enable_terminal: bool = True  # Enable SwarmTerminal for auto-fix


class AgentRunner:
    """
    Executes ONE agent with validation and learning.
    
    V2 User-Friendly Component:
    - Wraps agent execution
    - Provides Architect (pre-execution)
    - Provides Auditor (post-execution)
    - Handles learning and memory
    """
    
    def __init__(
        self,
        agent: Any,  # AutoAgent or DSPy agent
        config: AgentRunnerConfig,
        task_planner=None,  # Shared TaskPlanner (V2)
        task_board=None,  # Shared TaskBoard (V2)
        swarm_memory=None,  # Shared SwarmMemory (V2)
        swarm_state_manager=None,  # SwarmStateManager for state tracking (V2)
        learning_manager=None,  # Swarm-level LearningManager (V1 pipeline)
        transfer_learning=None,  # TransferableLearningStore for cross-swarm learning
        swarm_terminal=None,  # SwarmTerminal for intelligent command execution
        swarm_intelligence=None,  # SwarmIntelligence for curriculum feedback (Agent0)
    ):
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
                logger.info(f"🖥️  SwarmTerminal enabled for agent '{self.agent_name}'")
            except Exception as e:
                logger.debug(f"SwarmTerminal not available for {self.agent_name}: {e}")

        # Get agent state tracker (creates if doesn't exist)
        if self.swarm_state_manager:
            self.agent_tracker = self.swarm_state_manager.get_agent_tracker(self.agent_name)
            logger.info(f"📊 AgentStateTracker initialized for '{self.agent_name}'")
        
        from pathlib import Path
        
        from Jotty.core.foundation.data_structures import SharedScratchpad
        
        # Shared scratchpad for agent communication
        scratchpad = SharedScratchpad()
        
        # Architect (pre-execution planning)
        architect_agents = [
            InspectorAgent(
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
            InspectorAgent(
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
        self.agent_memory: Optional[HierarchicalMemory] = None
        if config.enable_memory:
            if swarm_memory is not None:
                # Shared memory — all agents see each other's experiences
                self.agent_memory = swarm_memory
                logger.debug(f"Agent '{self.agent_name}' using shared swarm memory")
            else:
                # Fallback: standalone memory (no sharing)
                self.agent_memory = HierarchicalMemory(
                    config=config.config,
                    agent_name=self.agent_name
                )
                logger.debug(f"Agent '{self.agent_name}' using standalone memory (no swarm memory)")
        
        # Per-agent learning: prefer the swarm-level learning_manager if available,
        # so all agents contribute to a single shared learner.
        # Falls back to standalone TDLambdaLearner only when no swarm context exists.
        from Jotty.core.learning.learning import AdaptiveLearningRate
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

        logger.info(f"AgentRunner initialized: {self.agent_name}")

        # Prompt selector for dynamic template selection
        self._prompt_selector: Optional[PromptSelector] = None
        self._current_task_type: str = 'default'
        self._scratchpad = scratchpad  # Store for validator recreation

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

    def _bridge_agent_hooks(self):
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
            def _wrap_pre(fn=pre_hook, **ctx):
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
            def _wrap_post(fn=post_hook, **ctx):
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
        logger.info(f"🪝 Hook registered: {hook_type}/{hook_name}")
        return hook_name

    def remove_hook(self, hook_type: str, name: str) -> bool:
        """Remove a named hook. Returns True if found."""
        hooks = self._hooks.get(hook_type, [])
        for i, fn in enumerate(hooks):
            if getattr(fn, '_hook_name', None) == name:
                hooks.pop(i)
                return True
        return False

    def _run_hooks(self, hook_type: str, **context) -> dict:
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
                logger.warning(f"🪝 Hook {hook_type}/{hook_name} failed: {e}")
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
            architect_agents = [
                InspectorAgent(
                    md_path=Path(architect_path),
                    is_architect=True,
                    tools=[],
                    config=self.config.config,
                    scratchpad=self._scratchpad
                )
            ]

            auditor_agents = [
                InspectorAgent(
                    md_path=Path(auditor_path),
                    is_architect=False,
                    tools=[],
                    config=self.config.config,
                    scratchpad=self._scratchpad
                )
            ]

            self.architect_validator = MultiRoundValidator(architect_agents, self.config.config)
            self.auditor_validator = MultiRoundValidator(auditor_agents, self.config.config)

            self._current_task_type = task_type
            logger.info(f"📋 Validators updated for task type: {task_type}")

        except Exception as e:
            logger.warning(f"Failed to update validators for {task_type}: {e}")

        return task_type

    # =========================================================================
    # PIPELINE STAGES (extracted from run() for readability)
    # =========================================================================

    def _gather_learning_context(self, goal: str) -> List[str]:
        """Stage 1: Gather learning context from memory, Q-learning, transfer
        learning, and swarm intelligence. Returns list of context strings.

        Separated from run() so the method stays focused on orchestration.
        """
        parts = []

        # 1. Memory retrieval
        if self.agent_memory:
            try:
                from Jotty.core.foundation.data_structures import MemoryLevel
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

        # 4. Swarm intelligence hints (stigmergy + agent profile)
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
            except Exception as e:
                logger.debug(f"Warm-start context skipped: {e}")

        for i, p in enumerate(parts, 1):
            logger.info(f"  📖 Context {i}: {p[:200]}")
        return parts

    async def _record_post_execution_learning(
        self, goal: str, agent_output, success: bool, trajectory: list,
        architect_results: list, auditor_results: list,
        architect_shaped_reward: float, duration: float, kwargs: dict
    ) -> dict:
        """Stage 3: Record learning data (memory, TD-lambda, Q-learning, feedback).

        Returns dict with episode_memory_entry, tagged_outputs, agent_contributions.
        """
        import time

        # Memory storage
        episode_memory_entry = None
        if self.agent_memory:
            from Jotty.core.foundation.data_structures import MemoryLevel
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
        from Jotty.core.foundation.types.learning_types import TaggedOutput
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
            from Jotty.core.foundation.types.agent_types import AgentContribution
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

    async def run(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Run agent execution with validation and learning.

        Args:
            goal: Task goal/description
            skip_validation: If True, skip architect validation (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            **kwargs: Additional arguments for agent

        Returns:
            EpisodeResult with output and metadata
        """
        import time
        start_time = time.time()

        # Extract flags
        skip_validation = kwargs.pop('skip_validation', False)
        validation_mode_override = kwargs.pop('validation_mode', None)
        status_callback = kwargs.pop('status_callback', None)

        def _status(stage: str, detail: str = ""):
            """Report progress if callback provided."""
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"  📍 {stage}" + (f": {detail}" if detail else ""))

        # ── Intelligent Validation Gate ───────────────────────────────
        # Instead of a boolean skip_validation, use a cheap Haiku call
        # to classify task complexity into DIRECT / AUDIT_ONLY / FULL.
        #
        # Backward compat: skip_validation=True → force DIRECT mode.
        # New: validation_mode="direct"|"audit"|"full" for explicit control.
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

        gate_decision: GateDecision = await self._validation_gate.decide(
            goal=goal,
            agent_name=self.agent_name,
            force_mode=force_mode,
        )

        # Derive skip flags from gate decision
        skip_architect = gate_decision.mode in (ValidationMode.DIRECT, ValidationMode.AUDIT_ONLY)
        skip_auditor = gate_decision.mode == ValidationMode.DIRECT

        mode_labels = {
            ValidationMode.DIRECT: "DIRECT (actor only)",
            ValidationMode.AUDIT_ONLY: "AUDIT (actor + auditor)",
            ValidationMode.FULL: "FULL (architect + actor + auditor)",
        }
        mode_label = mode_labels[gate_decision.mode]
        logger.info(
            f"AgentRunner.run: {self.agent_name} - {goal[:50]}... "
            f"[{mode_label}] gate={gate_decision.confidence:.0%} "
            f"({gate_decision.reason}) {gate_decision.latency_ms:.0f}ms"
        )
        _status("ValidationGate", f"{mode_label} — {gate_decision.reason}")

        # Hook: pre_run — modify goal, inject context, log start
        hook_ctx = self._run_hooks(
            'pre_run',
            goal=goal, agent_name=self.agent_name,
            skip_validation=skip_architect and skip_auditor,
            gate_decision=gate_decision, kwargs=kwargs,
        )
        goal = hook_ctx.get('goal', goal)  # allow hooks to modify goal

        # Dynamic prompt selection based on task type
        if not skip_architect:
            task_type = self._update_validators_for_task(goal)
            logger.debug(f"Task type detected: {task_type}")

        # Start episode for TD(λ) learning (if enabled)
        if self.agent_learner:
            self.agent_learner.start_episode(goal)

        # Reset shaped rewards for new episode
        if self.shaped_reward_manager:
            self.shaped_reward_manager.reset()

        _status("Preparing", "retrieving context")

        # Stage 1: Gather learning context (extracted to _gather_learning_context)
        learning_context_parts = self._gather_learning_context(goal)

        # Pass learning context as separate kwarg — NOT in the task string
        if learning_context_parts:
            kwargs['learning_context'] = "\n\n".join(learning_context_parts)
            logger.info(f"Learning context: {len(learning_context_parts)} sections ({sum(len(p) for p in learning_context_parts)} chars)")

        # Keep enriched_goal = goal (NO concatenation — task string stays clean)
        enriched_goal = goal

        # 1. Architect (pre-execution planning) - skip if gate says DIRECT or AUDIT_ONLY
        architect_results = []
        proceed = True
        architect_shaped_reward = 0.0

        if not skip_architect:
            # Hook: pre_architect
            self._run_hooks('pre_architect', goal=goal, agent_name=self.agent_name)

            _status("Architect", "validating approach")
            architect_results, proceed = await self.architect_validator.validate(
                goal=goal,
                inputs={'goal': goal, **kwargs},
                trajectory=[],
                is_architect=True
            )

            # Track architect validation in state
            if self.swarm_state_manager:
                avg_confidence = sum(r.confidence for r in architect_results) / len(architect_results) if architect_results else 0.0
                self.agent_tracker.record_validation(
                    validation_type='architect',
                    passed=proceed,
                    confidence=avg_confidence,
                    feedback=architect_results[0].reasoning if architect_results else None
                )
                # Record swarm-level step
                self.swarm_state_manager.record_swarm_step({
                    'agent': self.agent_name,
                    'step': 'architect',
                    'proceed': proceed,
                    'confidence': avg_confidence,
                    'architect_confidence': avg_confidence
                })

            # Architect doesn't block (exploration only) - log confidence to debug (not user-facing)
            if architect_results:
                avg_confidence = sum(r.confidence for r in architect_results) / len(architect_results)
                # Log to debug only - not user-facing
                logger.debug(
                    f"Architect confidence: {avg_confidence:.2f} "
                    f"(Decision: {'PROCEED' if proceed else 'BLOCKED'})"
                )

            # Hook: post_architect — inspect/override proceed decision
            arch_ctx = self._run_hooks(
                'post_architect', goal=goal, agent_name=self.agent_name,
                architect_results=architect_results, proceed=proceed,
            )
            proceed = arch_ctx.get('proceed', proceed)

            # Shaped reward: architect validation
            if self.shaped_reward_manager and architect_results:
                architect_shaped_reward = self.shaped_reward_manager.check_rewards(
                    event_type="actor_start",
                    state={'architect_results': architect_results, 'proceed': proceed, 'goal': goal},
                    trajectory=[]
                )
        else:
            logger.info(f"⚡ Skipping architect: gate={gate_decision.mode.value}")

        # 2. Agent execution (use enriched goal with memory context)
        # Hook: pre_execute — last chance to modify goal before agent runs
        exec_ctx = self._run_hooks(
            'pre_execute', goal=enriched_goal, agent_name=self.agent_name,
            architect_results=architect_results,
        )
        enriched_goal = exec_ctx.get('goal', enriched_goal)

        _status("Agent", "executing task (this may take a while)")
        try:
            # Always use AutoAgent for skill execution (skip_validation only affects architect/auditor)
            if hasattr(self.agent, 'execute'):
                # AutoAgent - pass status_callback if agent supports it
                if status_callback:
                    kwargs['status_callback'] = status_callback
                agent_output = await self.agent.execute(enriched_goal, **kwargs)
            elif hasattr(self.agent, 'forward'):
                # DSPy module
                agent_output = self.agent(goal=goal, **kwargs)
            else:
                # Callable
                agent_output = await self.agent(goal, **kwargs) if asyncio.iscoroutinefunction(self.agent) else self.agent(goal, **kwargs)
            
            # Build trajectory BEFORE auditor validation (so auditor can see execution history)
            trajectory = []
            # Determine inner success from agent output (ExecutionResult or dict)
            inner_success = True
            if hasattr(agent_output, 'success'):
                inner_success = bool(agent_output.success)
            elif isinstance(agent_output, dict):
                inner_success = bool(agent_output.get('success', True))

            # Build rich trajectory from ExecutionResult (if available)
            # so learning pipeline gets actual skill names, not just 'execute'
            _skills_used = []
            if hasattr(agent_output, 'skills_used') and agent_output.skills_used:
                _skills_used = list(agent_output.skills_used)
            elif isinstance(agent_output, dict):
                _skills_used = list(agent_output.get('skills_used', []))

            _outputs = {}
            if hasattr(agent_output, 'outputs') and isinstance(agent_output.outputs, dict):
                _outputs = agent_output.outputs
            elif isinstance(agent_output, dict):
                _outputs = agent_output.get('outputs', {})

            # Add one trajectory entry per executed step (skill) for rich learning
            if _outputs:
                for i, (step_name, step_data) in enumerate(_outputs.items(), 1):
                    step_success = True
                    step_skill = step_name
                    if isinstance(step_data, dict):
                        step_success = step_data.get('success', True)
                        step_skill = step_data.get('skill', step_name)
                    trajectory.append({
                        'step': i,
                        'skill': step_skill,
                        'action': step_name,
                        'success': step_success,
                        'output': str(step_data)[:200] if step_data else None,
                    })
            else:
                # Fallback: single trajectory entry
                trajectory.append({
                    'step': 1,
                    'action': 'execute',
                    'skills_used': _skills_used,
                    'output': str(agent_output)[:500] if agent_output else None,
                    'success': inner_success,
                })
            
            # Hook: post_execute — inspect/transform output before auditor
            self._run_hooks(
                'post_execute', goal=enriched_goal, agent_name=self.agent_name,
                agent_output=agent_output, inner_success=inner_success,
            )

            # 3. Auditor (post-execution validation) - skip if gate says DIRECT
            #    MALLM-inspired judge intervention: if auditor rejects,
            #    retry agent once with auditor feedback injected (Becker et al. 2025).
            if not skip_auditor:
                auditor_results, passed = await self.auditor_validator.validate(
                    goal=goal,
                    inputs={'goal': goal, 'output': str(agent_output)},
                    trajectory=trajectory,  # Pass trajectory so auditor can see execution history
                    is_architect=False
                )

                # If inner execution already failed, don't let auditor override
                success = passed and inner_success
                # ValidationResult has 'reasoning', not 'feedback'
                auditor_reasoning = auditor_results[0].reasoning if auditor_results else "No feedback"
                auditor_confidence = auditor_results[0].confidence if auditor_results else 0.0

                # ----------------------------------------------------------
                # JUDGE INTERVENTION (MALLM-inspired turn regeneration)
                # If auditor rejects with concrete feedback and we haven't
                # retried yet, re-run the agent with auditor critique.
                # KISS: max 1 retry, reuse existing execute path.
                #
                # Retry when:
                #   - Auditor rejected (not success)
                #   - Haven't retried yet
                #   - Auditor gave actionable reasoning
                # High-confidence rejections are BEST for retry because
                # the feedback is most specific and actionable.
                # ----------------------------------------------------------
                _judge_retried = kwargs.get('_judge_retried', False)
                if (
                    not success
                    and not _judge_retried
                    and auditor_reasoning
                    and auditor_reasoning != "No feedback"
                    and len(auditor_reasoning) > 20  # Must be substantive
                ):
                    logger.info(
                        f"🧑‍⚖️ Judge intervention: auditor rejected "
                        f"(confidence={auditor_confidence:.2f}), retrying with feedback"
                    )
                    _status("Judge intervention", f"retrying with auditor feedback")

                    # Hook: allow external observers to see intervention
                    self._run_hooks(
                        'post_execute', goal=enriched_goal,
                        agent_name=self.agent_name,
                        agent_output=agent_output, inner_success=False,
                        judge_intervention=True,
                        auditor_reasoning=auditor_reasoning,
                    )

                    # Build enriched goal with auditor critique
                    judge_goal = (
                        f"{enriched_goal}\n\n"
                        f"[Judge feedback — your previous attempt was rejected]:\n"
                        f"{auditor_reasoning}\n\n"
                        f"Please address the feedback and try again."
                    )

                    # Re-run execution with feedback (mark as retried)
                    if hasattr(self.agent, 'execute'):
                        retry_kwargs = dict(kwargs)
                        retry_kwargs['_judge_retried'] = True
                        if status_callback:
                            retry_kwargs['status_callback'] = status_callback
                        agent_output = await self.agent.execute(judge_goal, **retry_kwargs)
                    elif hasattr(self.agent, 'forward'):
                        agent_output = self.agent(goal=judge_goal, **kwargs)
                    else:
                        agent_output = (
                            await self.agent(judge_goal, **kwargs)
                            if asyncio.iscoroutinefunction(self.agent)
                            else self.agent(judge_goal, **kwargs)
                        )

                    # Re-validate after retry
                    inner_success = True
                    if hasattr(agent_output, 'success'):
                        inner_success = bool(agent_output.success)
                    elif isinstance(agent_output, dict):
                        inner_success = bool(agent_output.get('success', True))

                    auditor_results, passed = await self.auditor_validator.validate(
                        goal=goal,
                        inputs={'goal': goal, 'output': str(agent_output)},
                        trajectory=trajectory,
                        is_architect=False
                    )
                    success = passed and inner_success
                    auditor_reasoning = auditor_results[0].reasoning if auditor_results else "No feedback"
                    auditor_confidence = auditor_results[0].confidence if auditor_results else 0.0

                    logger.info(
                        f"🧑‍⚖️ Judge retry result: "
                        f"{'✅ passed' if success else '❌ still failed'} "
                        f"(confidence={auditor_confidence:.2f})"
                    )

                # Track auditor validation in state
                if self.swarm_state_manager:
                    self.agent_tracker.record_validation(
                        validation_type='auditor',
                        passed=passed,
                        confidence=auditor_confidence,
                        feedback=auditor_reasoning
                    )
                    # Record agent output
                    output_type = type(agent_output).__name__
                    self.agent_tracker.record_output(agent_output, output_type)
                    # Record swarm-level step
                    self.swarm_state_manager.record_swarm_step({
                        'agent': self.agent_name,
                        'step': 'auditor',
                        'success': success,
                        'validation_passed': passed,
                        'auditor_result': auditor_reasoning[:100],
                        'auditor_confidence': auditor_confidence
                    })
            else:
                # DIRECT mode: skip auditor, but respect inner execution result
                logger.info(f"⚡ Skipping auditor: gate={gate_decision.mode.value}")
                success = inner_success
                auditor_reasoning = f"Gate={gate_decision.mode.value}: auditor skipped"
                auditor_confidence = 1.0
                auditor_results = []
                passed = True
            
            # Update trajectory with validation result
            if trajectory:
                trajectory[0]['success'] = success
                trajectory[0]['validation'] = {
                    'passed': passed,
                    'confidence': auditor_confidence,
                    'tag': auditor_results[0].output_tag.value if auditor_results and auditor_results[0].output_tag else None
                }
            
            # Stage 3: Record learning (memory, TD-lambda, Q-learning, feedback)
            duration = time.time() - start_time
            learning_data = await self._record_post_execution_learning(
                goal=goal, agent_output=agent_output, success=success,
                trajectory=trajectory, architect_results=architect_results,
                auditor_results=auditor_results or [],
                architect_shaped_reward=architect_shaped_reward,
                duration=duration, kwargs=kwargs
            )

            episode_result = EpisodeResult(
                output=agent_output,
                success=success,
                trajectory=trajectory,
                tagged_outputs=learning_data['tagged_outputs'],
                episode=0,
                execution_time=duration,
                architect_results=architect_results or [],
                auditor_results=auditor_results or [],
                agent_contributions=learning_data['agent_contributions']
            )

            # Update consecutive failure counter
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            # Record gate outcome for drift detection
            # This lets the gate learn whether DIRECT/AUDIT routing
            # leads to successful outcomes over time.
            if self._validation_gate:
                self._validation_gate.record_outcome(gate_decision.mode, success)

            # Hook: post_run — record metrics, log results, send notifications
            self._run_hooks(
                'post_run', goal=goal, agent_name=self.agent_name,
                result=episode_result, success=success, elapsed=duration,
                gate_decision=gate_decision,
            )

            return episode_result
            
        except (AgentExecutionError, ToolExecutionError) as e:
            # Known execution errors — log cleanly, skip auto-fix (structural issue)
            logger.error(f"❌ Agent execution error: {e}")
            error_str = str(e)
            error_type = type(e).__name__
            fix_applied = False
            fix_description = ""

        except (LLMError, DSPyError) as e:
            # LLM provider or DSPy failures — potentially transient
            logger.error(f"❌ LLM/DSPy error during execution: {e}")
            error_str = str(e)
            error_type = type(e).__name__
            fix_applied = False
            fix_description = ""

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Agent timed out: {e}")
            error_str = f"Agent timed out: {e}"
            error_type = "TimeoutError"
            fix_applied = False
            fix_description = ""

        except (KeyboardInterrupt, SystemExit):
            raise  # Never swallow these

        except Exception as e:
            # Unexpected errors — log full traceback, attempt auto-fix
            logger.error(f"❌ Agent execution failed (unexpected {type(e).__name__}): {e}", exc_info=True)

            error_str = str(e)
            error_type = type(e).__name__
            fix_applied = False
            fix_description = ""

            # Try auto-fix using SwarmTerminal (only for unexpected errors)
            if self.swarm_terminal:
                try:
                    logger.info(f"🔧 Attempting auto-fix via SwarmTerminal...")
                    # Check if error is command/package related
                    error_keywords = ['command', 'module', 'import', 'pip', 'npm', 'permission', 'not found']
                    if any(kw in error_str.lower() for kw in error_keywords):
                        # Try to diagnose and fix
                        diagnostics = await self.swarm_terminal.diagnose_system()

                        # Search for solution
                        solution = await self.swarm_terminal._find_solution(goal, error_str)
                        if solution and solution.commands:
                            logger.info(f"   Found solution: {solution.solution[:100]}")
                            for cmd in solution.commands[:3]:
                                result = await self.swarm_terminal.execute(cmd, auto_fix=False)
                                if result.success:
                                    fix_applied = True
                                    fix_description = f"Applied: {cmd}"
                                    logger.info(f"   ✅ Fix applied: {cmd}")

                            # Retry the original execution if fix was applied
                            if fix_applied:
                                logger.info(f"   🔄 Retrying execution after fix...")
                                _status("Retrying", "after auto-fix")
                                # Recursive retry (limited to avoid infinite loop)
                                if not kwargs.get('_retry_after_fix'):
                                    kwargs['_retry_after_fix'] = True
                                    return await self.run(goal, **kwargs)

                except Exception as fix_error:
                    logger.debug(f"Auto-fix attempt failed: {fix_error}")

        # ── ERROR RECORDING (shared by all except branches) ───────────
        # If we reach here, execution failed (successful path returns above).
        # Track error in state
        if self.swarm_state_manager:
            self.agent_tracker.record_error(
                error=error_str,
                error_type=error_type,
                context={'goal': goal, 'kwargs': kwargs, 'fix_applied': fix_applied}
            )
            # Record swarm-level error step
            self.swarm_state_manager.record_swarm_step({
                'agent': self.agent_name,
                'step': 'error',
                'error': error_str,
                'error_type': error_type,
                'success': False,
                'fix_applied': fix_applied,
                'fix_description': fix_description
            })

        # Return failed EpisodeResult with correct structure
        duration = time.time() - start_time

        # Agent0: Send executor feedback for failed execution
        if self.swarm_intelligence:
            try:
                detected_task_type = self._current_task_type if hasattr(self, '_current_task_type') else None
                self.swarm_intelligence.receive_executor_feedback(
                    task_id=f"{self.agent_name}_{int(time.time())}",
                    success=False,
                    tools_used=[],  # No tools tracked in error case
                    execution_time=duration,
                    error_type=error_type,
                    task_type=detected_task_type
                )
                logger.debug(f"Executor feedback sent: success=False, error_type={error_type}")
            except Exception as fb_err:
                logger.debug(f"Executor feedback skipped: {fb_err}")

        # Record gate outcome for failure
        if self._validation_gate and 'gate_decision' in locals():
            self._validation_gate.record_outcome(gate_decision.mode, False)

        self._consecutive_failures += 1

        return EpisodeResult(
            output=None,
            success=False,
            trajectory=[{'step': 0, 'action': 'error', 'error': error_str, 'error_type': error_type, 'fix_applied': fix_applied}],
            tagged_outputs=[],
            episode=0,
            execution_time=duration,
            architect_results=architect_results if 'architect_results' in locals() else [],
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
