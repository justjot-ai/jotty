"""
Orchestrator - Composable Swarm Orchestrator
=============================================

Lazy-initialized, composable swarm orchestration.
Uses composition (has-a) instead of mixin inheritance (is-a) for all
cross-cutting concerns: providers, ensemble, learning, MAS-ZERO.

All components are lazy-loaded via descriptors — only created when first
accessed. Orchestrator.__init__ completes in < 50ms.

Architecture:
    Orchestrator (flat class, no mixins)
    ├── Core: config, agents, mode, runners
    ├── Composed: _providers, _ensemble, _learning_ops, _mas_zero
    ├── Planning: swarm_planner, swarm_task_board, swarm_intent_parser
    ├── Memory: swarm_memory, swarm_state_manager
    ├── Learning: learning (SwarmLearningPipeline), mas_learning
    └── Autonomous: swarm_researcher, swarm_installer, swarm_terminal, etc.

    Learning sub-components (accessed via sm.learning.xxx or sm.xxx):
        learning_manager, transfer_learning, swarm_intelligence,
        trajectory_predictor, divergence_memory, cooperative_credit,
        brain_state, agent_abstractor, swarm_learner,
        agent_slack, feedback_channel, credit_weights

Usage:
    sm = Orchestrator()  # Fast: ~10ms
    result = await sm.run("Research AI trends")  # Components init on demand
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union, Callable

if TYPE_CHECKING:
    from Jotty.core.orchestration.swarm_roadmap import SwarmTaskBoard
    from Jotty.core.agents.agentic_planner import TaskPlanner
    from Jotty.core.autonomous.intent_parser import IntentParser
    from Jotty.core.memory.cortex import SwarmMemory
    from Jotty.core.orchestration.swarm_provider_gateway import SwarmProviderGateway
    from Jotty.core.orchestration.swarm_researcher import SwarmResearcher
    from Jotty.core.orchestration.swarm_installer import SwarmInstaller
    from Jotty.core.orchestration.swarm_configurator import SwarmConfigurator
    from Jotty.core.orchestration.swarm_code_generator import SwarmCodeGenerator
    from Jotty.core.orchestration.swarm_workflow_learner import SwarmWorkflowLearner
    from Jotty.core.orchestration.swarm_integrator import SwarmIntegrator
    from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
    from Jotty.core.registry.tool_validation import ToolValidator
    from Jotty.core.orchestration.swarm_state_manager import SwarmStateManager
    from Jotty.core.persistence.shared_context import SharedContext
    from Jotty.core.data.io_manager import IOManager
    from Jotty.core.data.data_registry import DataRegistry
    from Jotty.core.context.context_guard import LLMContextManager
    from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
    from Jotty.core.orchestration.mas_learning import MASLearning
    from Jotty.core.monitoring.profiler import PerformanceProfiler

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.exceptions import (
    AgentExecutionError, LLMError, ConfigurationError, LearningError,
)

from ._lazy import LazyComponent
from Jotty.core.utils.async_utils import safe_status, StatusReporter

# Composed managers (has-a, not is-a) — replaces mixin inheritance
from .provider_manager import ProviderManager
from .ensemble_manager import EnsembleManager
from .learning_delegate import LearningDelegate
from .mas_zero_controller import MASZeroController
from .model_tier_router import ModelTierRouter
from .swarm_router import SwarmRouter
from .paradigm_executor import ParadigmExecutor
from .training_daemon import TrainingDaemon

logger = logging.getLogger(__name__)


# Suppress noisy LiteLLM CancelledError on asyncio loop shutdown.
# This is a known issue: LiteLLM's background LoggingWorker gets cancelled
# when asyncio.run() tears down, producing harmless but alarming tracebacks.
class _LiteLLMCancelledFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return 'CancelledError' not in msg and 'LoggingWorker cancelled' not in msg

logging.getLogger("LiteLLM").addFilter(_LiteLLMCancelledFilter())


# Skill Provider System - Lazy loaded via cache dict (no globals)
_provider_cache: Dict[str, Any] = {}


def _load_providers() -> bool:
    """
    Lazy load skill providers to avoid circular imports.

    Caches result in _provider_cache so the import cost is paid once.
    Returns True if providers are available.
    """
    if _provider_cache.get('available'):
        return True

    try:
        from Jotty.core.skills.providers import ProviderRegistry, SkillCategory
        from Jotty.core.skills.providers.browser_use_provider import BrowserUseProvider
        from Jotty.core.skills.providers.openhands_provider import OpenHandsProvider
        from Jotty.core.skills.providers.agent_s_provider import AgentSProvider
        from Jotty.core.skills.providers.open_interpreter_provider import OpenInterpreterProvider
        from Jotty.core.skills.providers.composite_provider import (
            ResearchAndAnalyzeProvider,
            AutomateWorkflowProvider,
            FullStackAgentProvider,
        )

        _provider_cache.update({
            'available': True,
            'ProviderRegistry': ProviderRegistry,
            'SkillCategory': SkillCategory,
            'BrowserUseProvider': BrowserUseProvider,
            'OpenHandsProvider': OpenHandsProvider,
            'AgentSProvider': AgentSProvider,
            'OpenInterpreterProvider': OpenInterpreterProvider,
            'ResearchAndAnalyzeProvider': ResearchAndAnalyzeProvider,
            'AutomateWorkflowProvider': AutomateWorkflowProvider,
            'FullStackAgentProvider': FullStackAgentProvider,
        })
        return True

    except ImportError as e:
        logger.debug(f"Skill providers not available: {e}")
        return False


# =========================================================================
# LAZY FACTORY FUNCTIONS - Called by LazyComponent descriptors
# =========================================================================

def _create_task_board() -> "SwarmTaskBoard":
    from Jotty.core.orchestration.swarm_roadmap import SwarmTaskBoard
    return SwarmTaskBoard()

def _create_planner() -> "TaskPlanner":
    from Jotty.core.agents.agentic_planner import TaskPlanner
    return TaskPlanner()

def _create_intent_parser(planner: "TaskPlanner") -> "IntentParser":
    from Jotty.core.autonomous.intent_parser import IntentParser
    return IntentParser(planner=planner)

def _create_memory(config: SwarmConfig) -> "SwarmMemory":
    from Jotty.core.memory.cortex import SwarmMemory
    return SwarmMemory(config=config, agent_name="SwarmShared")

def _create_provider_gateway(config: SwarmConfig) -> "SwarmProviderGateway":
    from Jotty.core.orchestration.swarm_provider_gateway import SwarmProviderGateway
    provider_preference = getattr(config, 'provider', None)
    return SwarmProviderGateway(config=config, provider=provider_preference)

def _create_researcher(config: SwarmConfig) -> "SwarmResearcher":
    from Jotty.core.orchestration.swarm_researcher import SwarmResearcher
    return SwarmResearcher(config=config)

def _create_installer(config: SwarmConfig) -> "SwarmInstaller":
    from Jotty.core.orchestration.swarm_installer import SwarmInstaller
    return SwarmInstaller(config=config)

def _create_configurator(config: SwarmConfig) -> "SwarmConfigurator":
    from Jotty.core.orchestration.swarm_configurator import SwarmConfigurator
    return SwarmConfigurator(config=config)

def _create_code_generator(config: SwarmConfig) -> "SwarmCodeGenerator":
    from Jotty.core.orchestration.swarm_code_generator import SwarmCodeGenerator
    return SwarmCodeGenerator(config=config)

def _create_workflow_learner(memory: "SwarmMemory") -> "SwarmWorkflowLearner":
    from Jotty.core.orchestration.swarm_workflow_learner import SwarmWorkflowLearner
    return SwarmWorkflowLearner(swarm_memory=memory)

def _create_integrator(config: SwarmConfig) -> "SwarmIntegrator":
    from Jotty.core.orchestration.swarm_integrator import SwarmIntegrator
    return SwarmIntegrator(config=config)

def _create_terminal(config: SwarmConfig) -> "SwarmTerminal":
    from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
    return SwarmTerminal(config=config, auto_fix=True, max_fix_attempts=3)

def _create_ui_registry() -> Any:
    from Jotty.core.registry.agui_component_registry import get_agui_registry
    return get_agui_registry()

def _create_tool_validator() -> "ToolValidator":
    from Jotty.core.registry.tool_validation import ToolValidator
    return ToolValidator()

def _create_tool_registry() -> Any:
    from Jotty.core.registry.tools_registry import get_tools_registry
    return get_tools_registry()

def _create_profiler(config: SwarmConfig) -> Optional["PerformanceProfiler"]:
    enable = getattr(config, 'enable_profiling', False)
    if not enable:
        return None
    from Jotty.core.monitoring.profiler import PerformanceProfiler
    return PerformanceProfiler(enable_cprofile=True)

def _create_state_manager(sm: "Orchestrator") -> "SwarmStateManager":
    from Jotty.core.orchestration.swarm_state_manager import SwarmStateManager
    agents_dict = {a.name: a for a in sm.agents}
    return SwarmStateManager(
        swarm_task_board=sm.swarm_task_board,
        swarm_memory=sm.swarm_memory,
        io_manager=sm.io_manager,
        data_registry=sm.data_registry,
        shared_context=sm.shared_context,
        context_guard=sm.context_guard,
        config=sm.config,
        agents=agents_dict,
        agent_signatures={},
    )

def _create_shared_context() -> "SharedContext":
    from Jotty.core.persistence.shared_context import SharedContext
    return SharedContext()

def _create_io_manager() -> "IOManager":
    from Jotty.core.data.io_manager import IOManager
    return IOManager()

def _create_data_registry() -> "DataRegistry":
    from Jotty.core.data.data_registry import DataRegistry
    return DataRegistry()

def _create_context_guard() -> "LLMContextManager":
    from Jotty.core.context.context_guard import LLMContextManager
    return LLMContextManager()

def _create_learning_pipeline(config: SwarmConfig) -> "SwarmLearningPipeline":
    from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
    return SwarmLearningPipeline(config)

def _create_mas_learning(sm: "Orchestrator") -> "MASLearning":
    from Jotty.core.orchestration.mas_learning import MASLearning
    workspace_path = getattr(sm.config, 'base_path', None)
    return MASLearning(
        config=sm.config,
        workspace_path=workspace_path,
        swarm_intelligence=sm.swarm_intelligence,
        learning_manager=sm.learning_manager,
        transfer_learning=sm.transfer_learning,
    )


class AgentFactory:
    """Creates and manages AgentRunners and LOTUS optimization."""

    def __init__(self, manager: 'Orchestrator'):
        self._manager = manager

    def ensure_runners(self) -> None:
        """Build AgentRunners on first run() call (not in __init__)."""
        sm = self._manager
        if sm._runners_built:
            return

        from Jotty.core.orchestration.agent_runner import AgentRunner, AgentRunnerConfig

        for agent_config in sm.agents:
            if agent_config.name in sm.runners:
                continue

            runner_config = AgentRunnerConfig(
                architect_prompts=sm.architect_prompts,
                auditor_prompts=sm.auditor_prompts,
                config=sm.config,
                agent_name=agent_config.name,
                enable_learning=True,
                enable_memory=True,
            )

            # Propagate SwarmConfig to agent so lazy-loaded components
            # (memory, context, etc.) use the same config as Orchestrator.
            agent = agent_config.agent
            if hasattr(agent, 'set_jotty_config'):
                agent.set_jotty_config(sm.config)

            runner = AgentRunner(
                agent=agent,
                config=runner_config,
                task_planner=sm.swarm_planner,
                task_board=sm.swarm_task_board,
                swarm_memory=sm.swarm_memory,
                swarm_state_manager=sm.swarm_state_manager,
                learning_manager=sm.learning_manager,
                transfer_learning=sm.transfer_learning,
                swarm_terminal=sm.swarm_terminal,
                swarm_intelligence=sm.swarm_intelligence,
            )
            sm.runners[agent_config.name] = runner

        # Register agents with Axon for inter-agent communication
        self.register_agents_with_axon()

        # LOTUS optimization
        if sm.enable_lotus:
            self.init_lotus_optimization()

        # Auto-load previous learnings + integrate MAS terminal in background.
        # Uses asyncio task instead of raw thread so run() can await readiness
        # via sm._learning_ready event before executing with partial state.
        async def _bg_learning_init():
            try:
                sm.learning.auto_load()
                sm.mas_learning.integrate_with_terminal(sm.swarm_terminal)
            except LearningError as e:
                logger.warning(f"Background learning init failed (learning): {e}")
            except Exception as e:
                logger.warning(f"Background learning init failed (unexpected): {e}", exc_info=True)
            finally:
                sm._learning_ready.set()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_bg_learning_init())
        except RuntimeError:
            # No running event loop — fall back to synchronous init
            try:
                sm.learning.auto_load()
                sm.mas_learning.integrate_with_terminal(sm.swarm_terminal)
            except LearningError as e:
                logger.warning(f"Synchronous learning init failed (learning): {e}")
            except Exception as e:
                logger.warning(f"Synchronous learning init failed (unexpected): {e}", exc_info=True)
            sm._learning_ready.set()

        # Init providers if available
        if _load_providers():
            sm._providers.init_provider_registry()

        sm._runners_built = True
        logger.info(f"Runners built: {list(sm.runners.keys())}")

    def register_agents_with_axon(self) -> None:
        """Register all agents with SmartAgentSlack for inter-agent messaging."""
        sm = self._manager
        from Jotty.core.agents.feedback_channel import FeedbackMessage, FeedbackType

        def _make_slack_callback(target_actor_name: str):
            def _callback(message):
                try:
                    fb = FeedbackMessage(
                        source_actor=message.from_agent,
                        target_actor=target_actor_name,
                        feedback_type=FeedbackType.RESPONSE,
                        content=str(message.data),
                        context={
                            'format': getattr(message, 'format', 'unknown'),
                            'size_bytes': getattr(message, 'size_bytes', None),
                            'metadata': getattr(message, 'metadata', {}) or {},
                            'timestamp': getattr(message, 'timestamp', None),
                        },
                        requires_response=False,
                        priority=2,
                    )
                    sm.feedback_channel.send(fb)
                except Exception as e:
                    logger.warning(f"Slack callback failed for {target_actor_name}: {e}")
            return _callback

        for agent_config in sm.agents:
            try:
                agent_obj = agent_config.agent
                signature_obj = getattr(agent_obj, 'signature', None)
                sm.agent_slack.register_agent(
                    agent_name=agent_config.name,
                    signature=signature_obj if hasattr(signature_obj, 'input_fields') else None,
                    callback=_make_slack_callback(agent_config.name),
                    max_context=getattr(sm.config, 'max_context_tokens', 16000),
                )
            except Exception as e:
                logger.warning(f"Could not register {agent_config.name} with SmartAgentSlack: {e}")

    def create_zero_config_agents(self, task: str, status_callback: Optional[Callable] = None) -> List[AgentConfig]:
        """Delegate to ZeroConfigAgentFactory."""
        sm = self._manager
        if not hasattr(sm, '_zero_config_factory') or sm._zero_config_factory is None:
            from Jotty.core.orchestration.zero_config_factory import ZeroConfigAgentFactory
            sm._zero_config_factory = ZeroConfigAgentFactory()
        return sm._zero_config_factory.create_agents(task, status_callback)

    def init_lotus_optimization(self) -> None:
        """
        Initialize LOTUS optimization layer.

        LOTUS-inspired optimizations:
        - Model Cascade: Use cheap models (Haiku) first, escalate to expensive (Opus) only when needed
        - Semantic Cache: Memoize semantic operations with content fingerprinting
        - Batch Executor: Batch LLM calls for throughput optimization
        - Adaptive Validator: Learn when to skip validation based on historical success

        DRY: Uses centralized LotusConfig for all optimization settings.
        """
        sm = self._manager
        try:
            from Jotty.core.lotus.integration import LotusEnhancement, _enhance_agent_runner

            # Create LOTUS enhancement with default config
            sm.lotus = LotusEnhancement(
                enable_cascade=True,
                enable_cache=True,
                enable_adaptive_validation=True,
            )
            sm.lotus_optimizer = sm.lotus.lotus_optimizer

            # Enhance all agent runners with adaptive validation
            for name, runner in sm.runners.items():
                _enhance_agent_runner(runner, sm.lotus)

                # Pre-warm the adaptive validator with initial trust
                # This allows validation skipping from the start
                # (simulates 15 successful validations per agent)
                for _ in range(15):
                    sm.lotus.adaptive_validator.record_result(name, "architect", success=True)
                    sm.lotus.adaptive_validator.record_result(name, "auditor", success=True)
                logger.debug(f"Pre-warmed LOTUS validator for agent: {name}")

            logger.info("LOTUS optimization layer initialized (pre-warmed validators)")

        except ImportError as e:
            logger.warning(f"LOTUS optimization not available: {e}")
            sm.lotus = None
            sm.lotus_optimizer = None

    def get_lotus_stats(self) -> Dict[str, Any]:
        """Get LOTUS optimization statistics."""
        sm = self._manager
        if sm.lotus:
            return sm.lotus.get_stats()
        return {}

    def get_lotus_savings(self) -> Dict[str, float]:
        """Get estimated cost savings from LOTUS optimization."""
        sm = self._manager
        if sm.lotus:
            return sm.lotus.get_savings()
        return {}


class ExecutionEngine:
    """Executes tasks via single/multi-agent paradigms."""

    def __init__(self, manager: 'Orchestrator'):
        self._manager = manager
        self._paradigms = ParadigmExecutor(manager)

    async def run(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Run task execution with full autonomy.

        Supports zero-config: natural language goal -> autonomous execution.
        For simple tool-calling tasks, use ChatExecutor directly instead.

        Args:
            goal: Task goal/description (natural language supported)
            context: Optional ExecutionContext from ModeRouter
            skip_autonomous_setup: If True, skip research/install/configure (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling
            **kwargs: Additional arguments

        Returns:
            EpisodeResult with output and metadata
        """
        sm = self._manager
        import time as _time
        run_start_time = _time.time()

        # Observability: Start trace and root span
        try:
            from Jotty.core.observability import get_tracer, get_metrics
            _tracer = get_tracer()
            _metrics = get_metrics()
            _tracer.new_trace(metadata={'goal': goal[:200], 'mode': sm.mode})
        except ImportError:
            _tracer = None
            _metrics = None

        # Lazy init: Build runners on first run
        sm._ensure_runners()

        # Wait for background learning init to complete (max 5s)
        # Prevents operating with partially-loaded learning state.
        try:
            await asyncio.wait_for(sm._learning_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Learning init timed out after 5s, proceeding without full learning state")

        # MAS-ZERO: Reset per-problem experience library
        sm._reset_experience()
        sm._efficiency_stats = {}  # Reset per-run
        sm._execution._efficiency_stats = {}  # Reset orchestrator stats too

        # Extract ExecutionContext if provided by ModeRouter
        exec_context = kwargs.pop('context', None)

        # Extract special kwargs (pop ALL so they don't leak into **kwargs downstream)
        skip_autonomous_setup = kwargs.pop('skip_autonomous_setup', False)
        skip_validation = kwargs.pop('skip_validation', None)  # Explicit override
        status_callback = kwargs.pop('status_callback', None)
        ensemble = kwargs.pop('ensemble', None)  # None = auto-detect, True/False = explicit
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')

        # If ExecutionContext provided, extract callbacks from it
        if exec_context is not None:
            if status_callback is None and hasattr(exec_context, 'status_callback'):
                status_callback = exec_context.status_callback
            # Emit start event
            if hasattr(exec_context, 'emit_event'):
                from Jotty.core.foundation.types.sdk_types import SDKEventType
                exec_context.emit_event(SDKEventType.PLANNING, {
                    "goal": goal,
                    "mode": sm.mode,
                    "agents": len(sm.agents),
                })

        # Auto-detect ensemble for certain task types (if not explicitly set)
        # Optima-inspired: adaptive sizing returns (should_ensemble, max_perspectives)
        max_perspectives = 4  # default
        if ensemble is None:
            ensemble, max_perspectives = sm._should_auto_ensemble(goal)
            if ensemble:
                logger.info(f"Auto-enabled ensemble: {max_perspectives} perspectives (use ensemble=False to override)")

        # Ensure DSPy LM is configured (critical for all agent operations)
        import dspy
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            lm = sm.swarm_provider_gateway.get_lm()
            if lm:
                dspy.configure(lm=lm)
                logger.info(f" DSPy LM configured: {getattr(lm, 'model', 'unknown')}")

        _status = StatusReporter(status_callback, logger, emoji="")

        # ── FAST PATH: Simple tasks bypass the entire agent pipeline ──
        # ValidationGate classifies task complexity using a cheap LLM (or heuristic).
        # For DIRECT tasks (Q&A, lookups, lists): skip zero-config, ensemble, and
        # AutonomousAgent overhead. Call the LLM directly. Target: <10s.
        # OPTIMIZATION: Skip gate for multi-agent mode — fast path only works
        # for single-agent, so running the gate wastes an LLM call.
        from Jotty.core.orchestration.validation_gate import (
            ValidationGate, ValidationMode, get_validation_gate,
        )

        _gate_decision = None
        if sm.mode == "single":
            _fast_gate = get_validation_gate()

            # Determine if user explicitly asked for skip/full validation
            _force_mode = None
            if skip_validation is True:
                _force_mode = ValidationMode.DIRECT
            elif skip_validation is False:
                _force_mode = ValidationMode.FULL

            _gate_decision = await _fast_gate.decide(
                goal=goal,
                agent_name=sm.agents[0].name if sm.agents else "auto",
                force_mode=_force_mode,
            )

        if _gate_decision and _gate_decision.mode == ValidationMode.DIRECT and sm.mode == "single":
            _status("Fast path", f"DIRECT mode — bypassing agent pipeline ({_gate_decision.reason})")

            # Model tier routing: use cheapest LM for DIRECT tasks
            try:
                if sm._model_tier_router is None:
                    sm._model_tier_router = ModelTierRouter()
                tier_decision = sm._model_tier_router.get_model_for_mode(ValidationMode.DIRECT)
                tier_lm = sm._model_tier_router.get_lm_for_mode(ValidationMode.DIRECT)
                if tier_lm:
                    import dspy as _dspy
                    lm = tier_lm
                    _status("Model tier", f"{tier_decision.tier.value} ({tier_decision.model}) — cost ratio {tier_decision.estimated_cost_ratio:.1f}x")
                else:
                    import dspy as _dspy
                    lm = _dspy.settings.lm
                    if lm is None:
                        raise AgentExecutionError("No LM configured")

                _fast_start = _time.time()

                # Try multiple calling conventions (DSPy 3.x is finicky)
                # Fast path: NO aggressive retry on rate limits.
                # If rate limited, fall through to full pipeline immediately.
                # Fast path must be FAST — wasting 56s on retries defeats the purpose.
                response = None
                _rate_limited = False
                for call_fn in [
                    lambda: lm(messages=[{"role": "user", "content": goal}]),
                    lambda: lm(prompt=goal),
                    lambda: lm(goal),
                ]:
                    try:
                        response = call_fn()
                        if response:
                            break
                    except LLMError as e:
                        logger.info(f"Fast path LLM error (recoverable): {e}")
                        err_str = str(e)
                        is_rate_limit = ('429' in err_str or 'RateLimit' in err_str or
                                         'rate limit' in err_str.lower())
                        if is_rate_limit:
                            logger.info("Fast path rate limited — falling through to full pipeline (no retry)")
                            _rate_limited = True
                            break
                        continue
                    except Exception as e:
                        err_str = str(e)
                        is_rate_limit = ('429' in err_str or 'RateLimit' in err_str or
                                         'rate limit' in err_str.lower())
                        if is_rate_limit:
                            logger.info("Fast path rate limited — falling through to full pipeline (no retry)")
                            _rate_limited = True
                            break
                        continue

                if response is None:
                    raise AgentExecutionError("All LM calling conventions failed" +
                                              (" (rate limited)" if _rate_limited else ""))

                if isinstance(response, list):
                    response = response[0] if response else ""
                elif hasattr(response, 'text'):
                    response = response.text
                response = str(response).strip()

                _fast_elapsed = _time.time() - _fast_start
                _status("Fast path complete", f"{_fast_elapsed:.1f}s")

                # Record outcome for gate learning
                _fast_gate.record_outcome(ValidationMode.DIRECT, bool(response))

                # Build minimal EpisodeResult
                fast_result = EpisodeResult(
                    output=response,
                    success=bool(response),
                    trajectory=[{'step': 1, 'action': 'direct_llm', 'output': response[:200]}],
                    tagged_outputs=[],
                    episode=sm.episode_count,
                    execution_time=_fast_elapsed,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                )
                sm.episode_count += 1

                # Save learning (lightweight)
                try:
                    sm._save_learnings()
                except LearningError as e:
                    logger.debug(f"Fast-path learning save failed (learning): {e}")
                except Exception as e:
                    logger.debug(f"Fast-path learning save failed (unexpected): {e}", exc_info=True)

                total_elapsed = _time.time() - run_start_time
                _status("Complete", f"fast path success ({total_elapsed:.1f}s)")
                return fast_result

            except (AgentExecutionError, LLMError) as e:
                logger.info(f"Fast path failed (recoverable: {type(e).__name__}): {e}, falling back to full pipeline")
                # Fall through to normal pipeline
            except Exception as e:
                logger.info(f"Fast path failed (unexpected: {type(e).__name__}): {e}, falling back to full pipeline")
                # Fall through to normal pipeline

        # Store gate decision for downstream use (AgentRunner will also gate architect/auditor)
        kwargs['_swarm_gate_decision'] = _gate_decision

        # ── MODEL TIER ROUTING: Select LM quality based on task complexity ──
        # DIRECT → cheap (Haiku), AUDIT_ONLY → balanced (Sonnet), FULL → quality (Opus/Sonnet)
        if _gate_decision and _gate_decision.mode != ValidationMode.DIRECT:
            try:
                if sm._model_tier_router is None:
                    sm._model_tier_router = ModelTierRouter()
                tier_decision = sm._model_tier_router.get_model_for_mode(_gate_decision.mode)
                tier_lm = sm._model_tier_router.get_lm_for_mode(_gate_decision.mode)
                if tier_lm:
                    import dspy
                    dspy.configure(lm=tier_lm)
                    _status("Model tier", f"{tier_decision.tier.value} ({tier_decision.model})")
            except (ConfigurationError, LLMError) as e:
                logger.debug(f"Model tier routing skipped (recoverable): {e}")
            except Exception as e:
                logger.debug(f"Model tier routing skipped (unexpected): {e}", exc_info=True)

        # Zero-config: LLM decides single vs multi-agent at RUN TIME (when goal is available)
        # SKIP when gate already classified as DIRECT — it's a simple task, no need
        # to burn an LLM call deciding single vs multi-agent.
        _gate_is_direct = (_gate_decision and _gate_decision.mode == ValidationMode.DIRECT)
        if sm.enable_zero_config and sm.mode == "single" and not _gate_is_direct:
            _status("Analyzing task", "deciding single vs multi-agent")
            new_agents = sm._create_zero_config_agents(goal, status_callback)
            if len(new_agents) > 1:
                # LLM detected parallel sub-goals - upgrade to multi-agent
                sm.agents = new_agents
                sm.mode = "multi"
                logger.info(f" Zero-config: Upgraded to {len(sm.agents)} agents for parallel execution")

                # Create runners for new agents
                from Jotty.core.orchestration.agent_runner import AgentRunner, AgentRunnerConfig
                for agent_config in sm.agents:
                    if agent_config.name not in sm.runners:
                        runner_config = AgentRunnerConfig(
                            architect_prompts=sm.architect_prompts,
                            auditor_prompts=sm.auditor_prompts,
                            config=sm.config,
                            agent_name=agent_config.name,
                            enable_learning=True,
                            enable_memory=True
                        )
                        runner = AgentRunner(
                            agent=agent_config.agent,
                            config=runner_config,
                            task_planner=sm.swarm_planner,
                            task_board=sm.swarm_task_board,
                            swarm_memory=sm.swarm_memory,
                            swarm_state_manager=sm.swarm_state_manager,
                            learning_manager=sm.learning_manager,
                            transfer_learning=sm.transfer_learning,
                            swarm_terminal=sm.swarm_terminal  # Shared intelligent terminal
                        )
                        sm.runners[agent_config.name] = runner

        agent_info = f"{len(sm.agents)} AutoAgent(s)" if len(sm.agents) > 1 else "AutoAgent (zero-config)"
        _status("Starting", agent_info)

        # Profile execution if enabled
        if sm.swarm_profiler:
            profile_context = sm.swarm_profiler.profile("Orchestrator.run", metadata={"goal": goal, "mode": sm.mode})
            profile_context.__enter__()
        else:
            profile_context = None

        try:
            # Store goal in shared context for state management
            if sm.shared_context:
                sm.shared_context.set('goal', goal)
                sm.shared_context.set('query', goal)

            # Autonomous planning: Research, install, configure if needed
            # For multi-agent mode with zero-config: skip full setup (agents have specific sub-goals)
            # This reduces latency significantly for parallel agent execution
            if not skip_autonomous_setup and sm.mode == "single":
                _status("Autonomous setup", "analyzing requirements")
                await self.autonomous_setup(goal, status_callback=status_callback)
            elif not skip_autonomous_setup and sm.mode == "multi":
                _status("Fast mode", "multi-agent (agents configured with sub-goals)")
            else:
                _status("Fast mode", "skipping autonomous setup")

            # Set root task in SwarmTaskBoard
            sm.swarm_task_board.root_task = goal

            # Record swarm-level step: goal received
            if sm.swarm_state_manager:
                sm.swarm_state_manager.record_swarm_step({
                    'step': 'goal_received',
                    'goal': goal,
                    'mode': sm.mode,
                    'agent_count': len(sm.agents)
                })

            # Ensemble mode: Multi-perspective analysis
            # For multi-agent: ensemble happens per-agent (each agent has different sub-goal)
            # For single-agent: ensemble happens at swarm level
            ensemble_result = None
            if ensemble and sm.mode == "single":
                _status("Ensembling", f"strategy={ensemble_strategy}, perspectives={max_perspectives}")
                _ens_start = _time.time()
                ensemble_result = await sm._execute_ensemble(
                    goal,
                    strategy=ensemble_strategy,
                    status_callback=status_callback,
                    max_perspectives=max_perspectives,
                )
                sm._efficiency_stats['ensemble_time'] = _time.time() - _ens_start
                if ensemble_result.get('success'):
                    # Pass ensemble context via kwargs (not embedded in goal string)
                    # to avoid polluting search queries and downstream skill params.
                    kwargs['ensemble_context'] = ensemble_result

                    # Show quality scores if available
                    quality_scores = ensemble_result.get('quality_scores', {})
                    if quality_scores:
                        avg_quality = sum(quality_scores.values()) / len(quality_scores)
                        scores_str = ", ".join(f"{k}:{v:.0%}" for k, v in quality_scores.items())
                        _status("Ensemble quality", f"avg={avg_quality:.0%} ({scores_str})")
                    else:
                        _status("Ensemble complete", f"{len(ensemble_result.get('perspectives_used', []))} perspectives")
            elif ensemble and sm.mode == "multi":
                # Multi-agent mode: SKIP per-agent ensemble (too expensive)
                # With N agents × 4 perspectives = 4N LLM calls - massive overkill
                # Each agent already has a specific sub-goal, no need for multi-perspective
                _status("Ensemble mode", "DISABLED for multi-agent (agents have specific sub-goals)")
                ensemble = False  # Disable for agents

            # Clean up internal flag before passing to agents
            kwargs.pop('_swarm_gate_decision', None)

            # Single-agent mode: Simple execution
            if sm.mode == "single":
                agent_name = sm.agents[0].name if sm.agents else "auto"
                _status("Executing", f"agent '{agent_name}' with skill orchestration")
                # Architect → Actor → Auditor pipeline (now fast with max_eval_iters=2)
                # skip_validation: explicit kwarg wins, else derive from skip_autonomous_setup
                skip_val = skip_validation if skip_validation is not None else skip_autonomous_setup
                # Forward ensemble flag so AutoAgent doesn't auto-detect independently
                if ensemble is not None:
                    kwargs['ensemble'] = ensemble
                # Avoid duplicate kwarg: ensemble_context may already be in kwargs (line 748)
                kwargs.pop('ensemble_context', None)
                result = await self._execute_single_agent(
                    goal,
                    skip_validation=skip_val,
                    status_callback=status_callback,
                    ensemble_context=ensemble_result if ensemble else None,
                    **kwargs
                )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = sm._efficiency_stats.get('ensemble_time', 0)
                exec_t = getattr(result, 'execution_time', total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                sm._efficiency_stats.update({'total_time': total_elapsed, 'overhead_pct': overhead_pct})
                _status("Complete", f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)")

                # Surface user-friendly summary with artifacts
                sm._log_execution_summary(result)

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                # Auto-drain one training task if result succeeded and tasks pending
                sm._maybe_drain_training_task(result)

                return result

            # Multi-agent mode: Use SwarmTaskBoard for coordination
            else:
                # MAS-ZERO: Dynamic reduction — check if multi-agent is overkill
                if sm._mas_zero_should_reduce(goal):
                    _status("MAS-ZERO reduction", "reverting to single agent (simpler is better)")
                    sm.mode = "single"
                    result = await self._execute_single_agent(
                        goal,
                        skip_validation=skip_validation if skip_validation is not None else skip_autonomous_setup,
                        status_callback=status_callback,
                        **kwargs
                    )
                else:
                    paradigm = kwargs.pop('discussion_paradigm', 'auto')
                    _status("Executing", f"{len(sm.agents)} agents — paradigm: {paradigm}")
                    _sv = skip_validation if skip_validation is not None else skip_autonomous_setup
                    result = await self._execute_multi_agent(
                        goal,
                        ensemble_context=ensemble_result if ensemble else None,
                        status_callback=status_callback,
                        ensemble=ensemble,
                        ensemble_strategy=ensemble_strategy,
                        discussion_paradigm=paradigm,
                        skip_validation=_sv,
                        **kwargs
                    )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = sm._efficiency_stats.get('ensemble_time', 0)
                exec_t = getattr(result, 'execution_time', total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                sm._efficiency_stats.update({'total_time': total_elapsed, 'overhead_pct': overhead_pct})
                _status("Complete", f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)")

                # Surface user-friendly summary with artifacts
                sm._log_execution_summary(result)

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                return result
        except (AgentExecutionError, LLMError) as e:
            _status("Error", f"{type(e).__name__}: {str(e)[:50]}")
            # Exit profiling context on error
            if profile_context:
                profile_context.__exit__(type(e), e, None)
            raise
        except Exception as e:
            _status("Error", f"unexpected: {str(e)[:50]}")
            logger.error(f"Unexpected error in run(): {e}", exc_info=True)
            # Exit profiling context on error
            if profile_context:
                profile_context.__exit__(type(e), e, None)
            raise

    async def _execute_single_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute single-agent mode.

        MAS-ZERO: For comparison/analysis tasks, runs building blocks
        (direct + ensemble) in parallel and verifies the best answer.
        For simple tasks, runs direct execution as before.

        Args:
            goal: Task goal
            **kwargs: Additional arguments

        Returns:
            EpisodeResult
        """
        sm = self._manager
        # Remove ensemble_context from kwargs before passing to runner
        kwargs.pop('ensemble_context', None)
        status_callback = kwargs.pop('status_callback', None)

        _status = StatusReporter(status_callback)

        # ── Router-guided agent selection (closes RL loop for single-agent) ──
        # When multiple agents are available, ask the router instead of
        # blindly taking agents[0].  Router uses SwarmIntelligence (RL +
        # stigmergy + TRAS) when learning data exists.
        agent_config = sm.agents[0]
        if len(sm.agents) > 1:
            try:
                route = sm._router.select_agent(goal)
                picked = route.get('agent')
                if picked and route.get('method') != 'fallback':
                    for ac in sm.agents:
                        if getattr(ac, 'name', None) == picked:
                            agent_config = ac
                            logger.info(
                                f"Router selected '{picked}' for single-agent "
                                f"(method={route['method']}, conf={route['confidence']:.2f})"
                            )
                            break
            except (ConfigurationError, LearningError) as e:
                logger.debug(f"Router select_agent skipped (recoverable): {e}")
            except Exception as e:
                logger.debug(f"Router select_agent skipped (unexpected): {e}", exc_info=True)

        runner = sm.runners[agent_config.name]

        # ── READ LEARNED INTELLIGENCE (closes the learning loop) ──────────
        # Post-episode writes: stigmergy outcomes, byzantine trust, morph scores.
        # Pre-execution reads them back to influence this run.
        # Without this read, the write side is wasted — learning never loops.
        learned_hints = []
        try:
            lp = sm.learning
            task_type = lp.transfer_learning.extractor.extract_task_type(goal)

            # 1. Stigmergy APPROACH guidance (single-agent-aware).
            #    Instead of "use agent X" (useless with one agent),
            #    inject "use these tools/approaches" and "avoid these".
            approach_rec = lp.stigmergy.recommend_approach(task_type, top_k=2)
            for good in approach_rec.get('use', []):
                tools_str = ', '.join(good['tools'][:5]) if good.get('tools') else 'N/A'
                learned_hints.append(
                    f"[Learned] Successful approach for '{task_type}': "
                    f"{good['approach'][:120]} (tools: {tools_str}, "
                    f"quality: {good.get('quality', 0):.0%})"
                )
            for bad in approach_rec.get('avoid', []):
                tools_str = ', '.join(bad['tools'][:5]) if bad.get('tools') else 'N/A'
                learned_hints.append(
                    f"[Learned] Failed approach for '{task_type}': "
                    f"{bad['approach'][:120]} — avoid this"
                )
            # Fallback: if no approach data yet, use old agent-routing signals
            if not approach_rec.get('use') and not approach_rec.get('avoid'):
                warnings = lp.get_stigmergy_warnings(task_type)
                if warnings:
                    warn_msgs = [
                        getattr(w, 'content', {}).get('goal', 'unknown')
                        if isinstance(getattr(w, 'content', None), dict) else str(w)
                        for w in warnings[:3]
                    ]
                    learned_hints.append(
                        f"[Learned] Previous failures on '{task_type}': "
                        f"{'; '.join(warn_msgs)}"
                    )

            # 2. Effectiveness: are we improving or degrading on this type?
            eff_report = lp.effectiveness.improvement_report()
            task_eff = eff_report.get(task_type)
            if task_eff and task_eff.get('trend') is not None:
                trend = task_eff['trend']
                if trend < -0.1:
                    learned_hints.append(
                        f"[Learned] Performance DECLINING on '{task_type}' "
                        f"(recent={task_eff['recent_rate']:.0%} vs "
                        f"historical={task_eff['historical_rate']:.0%}). "
                        f"Consider a different approach."
                    )
                elif trend > 0.1:
                    learned_hints.append(
                        f"[Learned] Performance IMPROVING on '{task_type}' — "
                        f"current approach is working."
                    )

            # 3. Transfer learning: relevant past learnings
            learnings = lp.transfer_learning.get_relevant_learnings(goal, top_k=2)
            for exp in learnings.get('similar_experiences', []):
                query_text = exp.get('query', '') if isinstance(exp, dict) else str(exp)
                success = exp.get('success', True) if isinstance(exp, dict) else True
                if success:
                    learned_hints.append(
                        f"[Learned] Succeeded on similar task: {query_text[:100]}"
                    )
                else:
                    err = (exp.get('error', '')[:80]) if isinstance(exp, dict) else ''
                    learned_hints.append(
                        f"[Learned] Failed on similar task: {query_text[:80]}"
                        + (f" (error: {err})" if err else "")
                    )
            if learnings.get('task_pattern'):
                learned_hints.append(
                    f"[Learned] Pattern: {str(learnings['task_pattern'])[:150]}"
                )
            for err_pat in learnings.get('error_patterns', [])[:2]:
                learned_hints.append(
                    f"[Learned] Common error: {str(err_pat)[:120]}"
                )

            # 4. Workflow patterns: known successful tool sequences for this task type
            if sm.swarm_workflow_learner:
                try:
                    from Jotty.core.learning.transfer_learning import PatternExtractor
                    norm_type = PatternExtractor.normalize_task_type(task_type)
                    wf_patterns = sm.swarm_workflow_learner.get_patterns_by_task_type(norm_type)
                    for pat in wf_patterns[:3]:
                        if pat.success_rate >= 0.5:
                            ops = ' → '.join(pat.operations[:6]) if pat.operations else 'N/A'
                            tools = ', '.join(pat.tools_used[:5]) if pat.tools_used else 'N/A'
                            learned_hints.append(
                                f"[Learned] Proven workflow for '{norm_type}': "
                                f"{ops} (tools: {tools}, "
                                f"success: {pat.success_rate:.0%}, "
                                f"time: {pat.execution_time:.0f}s)"
                            )
                except Exception as wf_err:
                    logger.debug(f"Workflow pattern lookup skipped: {wf_err}")

            # Inject learned context into kwargs for the agent
            if learned_hints:
                existing_ctx = kwargs.get('learning_context', '')
                kwargs['learning_context'] = (
                    existing_ctx + "\n\n" + "\n".join(learned_hints)
                ).strip()
                _status("Intelligence", f"{len(learned_hints)} learned hints applied")
                logger.info(
                    f" Single-agent intelligence: {len(learned_hints)} hints "
                    f"for task_type='{task_type}'"
                )
                for i, hint in enumerate(learned_hints, 1):
                    logger.info(f" Hint {i}: {hint}")

        except LearningError as e:
            logger.warning(f"Pre-execution intelligence read failed (learning): {e}")
        except Exception as e:
            logger.warning(f"Pre-execution intelligence read failed (unexpected): {e}", exc_info=True)

        # Pass status_callback back for downstream
        if status_callback:
            kwargs['status_callback'] = status_callback

        # Standard single-agent execution
        import time as _t
        _exec_start = _t.time()

        # Observability: trace agent execution
        _tracer = None
        try:
            from Jotty.core.observability import get_tracer, get_metrics
            _tracer = get_tracer()
        except ImportError:
            pass

        if _tracer:
            with _tracer.span("agent_execute", agent=agent_config.name, mode="single") as _span:
                result = await runner.run(goal=goal, **kwargs)
                _exec_elapsed = _t.time() - _exec_start
                _span.set_attribute("success", result.success)
                _span.set_attribute("execution_time_s", round(_exec_elapsed, 2))
                if hasattr(result, 'execution_time'):
                    _span.set_attribute("inner_time_s", round(result.execution_time, 2))
        else:
            result = await runner.run(goal=goal, **kwargs)
            _exec_elapsed = _t.time() - _exec_start

        # Observability: record metrics
        try:
            from Jotty.core.observability import get_metrics
            get_metrics().record_execution(
                agent_name=agent_config.name,
                task_type="single_agent",
                duration_s=_exec_elapsed,
                success=result.success,
            )
        except (ImportError, Exception):
            pass

        # Learn from execution (DRY: reuse workflow learner)
        if result.success:
            sm._learn_from_result(result, agent_config, goal=goal)

        # Post-episode learning + auto-save (fire-and-forget background)
        sm._schedule_background_learning(result, goal)

        return result

    async def _execute_multi_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute multi-agent mode with configurable discussion paradigm.

        MALLM-inspired paradigms (Becker et al., EMNLP 2025):
            fanout      — All agents run in parallel on decomposed tasks (default)
            relay       — Sequential chain; each agent builds on previous output
            debate      — Agents critique each other's outputs in rounds
            refinement  — Iterative improve loop until quality stabilizes

        DRY: All paradigms reuse the same AgentRunner.run() and semaphore.
        """
        sm = self._manager
        from Jotty.core.orchestration.swarm_roadmap import TaskStatus
        from Jotty.core.learning.predictive_marl import ActualTrajectory
        from Jotty.core.agents.feedback_channel import FeedbackMessage, FeedbackType

        # Extract callbacks and ensemble params before passing to runners
        kwargs.pop('ensemble_context', None)
        status_callback = kwargs.pop('status_callback', None)
        ensemble = kwargs.pop('ensemble', False)
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')
        discussion_paradigm = kwargs.pop('discussion_paradigm', 'auto')

        # ── Intelligence-guided agent selection (single entry point: router) ──
        # Router delegates to LearningPipeline.order_agents_for_goal (trust + stigmergy + TRAS).
        # This closes the learning loop: post_episode writes → run() reads.
        _intelligence_applied = False
        try:
            ordered = sm._router.order_agents_for_goal(goal)
            if ordered:
                sm.agents = ordered
                sm._runners_built = False
                sm._ensure_runners()
                _intelligence_applied = bool(getattr(sm, 'learning', None))
                if _intelligence_applied and sm.agents:
                    top = sm.agents[0].name if hasattr(sm.agents[0], 'name') else '?'
                    logger.info(f" Router: agents ordered for goal (lead={top})")
        except LearningError as e:
            logger.warning(f"Intelligence-guided selection failed (learning): {e}")
        except Exception as e:
            logger.warning(f"Intelligence-guided selection failed (unexpected): {e}", exc_info=True)

        # Track guidance for A/B effectiveness metrics (per task_type)
        sm._last_run_guided = _intelligence_applied
        try:
            _im_task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
        except Exception as e:
            logger.debug(f"Task type extraction for A/B metrics failed: {e}")
            _im_task_type = '_global'
        sm._last_task_type = _im_task_type

        for _tt in (_im_task_type, '_global') if _im_task_type != '_global' else ('_global',):
            if _tt not in sm._intelligence_metrics:
                sm._intelligence_metrics[_tt] = {
                    'guided_runs': 0, 'guided_successes': 0,
                    'unguided_runs': 0, 'unguided_successes': 0,
                }
            if _intelligence_applied:
                sm._intelligence_metrics[_tt]['guided_runs'] += 1
            else:
                sm._intelligence_metrics[_tt]['unguided_runs'] += 1

        # Auto paradigm selection: use learning data to pick the best paradigm
        if discussion_paradigm == 'auto':
            try:
                _task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
                discussion_paradigm = sm.learning.recommend_paradigm(_task_type)
                logger.info(
                    f" Auto paradigm: selected '{discussion_paradigm}' "
                    f"for task_type='{_task_type}'"
                )
            except LearningError as e:
                logger.debug(f"Auto paradigm selection failed (learning): {e}")
                discussion_paradigm = 'fanout'
            except Exception as e:
                logger.debug(f"Auto paradigm selection failed (unexpected): {e}", exc_info=True)
                discussion_paradigm = 'fanout'

        # Track which paradigm is used (for _post_episode_learning)
        sm._last_paradigm = discussion_paradigm

        # MALLM-inspired: Dispatch to alternative paradigms before fan-out
        if discussion_paradigm == 'relay':
            # Wire coordination: initiate handoff between relay agents
            if sm.swarm_intelligence:
                try:
                    available = [a.name for a in sm.agents]
                    if len(available) >= 2:
                        sm.swarm_intelligence.initiate_handoff(
                            from_agent=available[0],
                            to_agent=available[1],
                            task=goal,
                            context={'paradigm': 'relay', 'agents': available}
                        )
                except Exception as e:
                    logger.debug(f"Relay handoff coordination skipped: {e}")
            return await self._paradigm_relay(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == 'debate':
            return await self._paradigm_debate(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == 'refinement':
            return await self._paradigm_refinement(goal, status_callback=status_callback, **kwargs)
        # else: 'fanout' — fall through to existing parallel execution below

        # Wire coordination: form coalition for fanout tasks (agents collaborate)
        if sm.swarm_intelligence and len(sm.agents) >= 2:
            try:
                available = [a.name for a in sm.agents]
                _task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
                sm.swarm_intelligence.form_coalition(
                    task_type=_task_type,
                    min_agents=min(2, len(available)),
                    available_agents=available
                )
                logger.info(f" Coalition formed for '{_task_type}' with {len(available)} agents")
            except LearningError as e:
                logger.debug(f"Coalition formation skipped (learning): {e}")
            except Exception as e:
                logger.debug(f"Coalition formation skipped (unexpected): {e}", exc_info=True)

        # Status update at method start
        safe_status(status_callback, "Multi-agent exec", f"starting {len(sm.agents)} parallel agents")

        max_attempts = getattr(sm.config, 'max_task_attempts', 2)

        # Clear task board for fresh run (avoid stale tasks from previous runs)
        sm.swarm_task_board.subtasks.clear()
        sm.swarm_task_board.completed_tasks.clear()
        sm.swarm_task_board.execution_order.clear()

        # Add tasks to SwarmTaskBoard
        # Zero-config agents from LLM are PARALLEL (independent sub-goals)
        # Only add dependencies if explicitly specified in agent config
        for i, agent_config in enumerate(sm.agents):
            task_id = f"task_{i+1}"
            # Check if agent has explicit dependencies
            deps = getattr(agent_config, 'depends_on', []) or []

            # Use agent's sub-goal (from capabilities) as task description
            sub_goal = agent_config.capabilities[0] if agent_config.capabilities else f"{goal} (agent: {agent_config.name})"

            sm.swarm_task_board.add_task(
                task_id=task_id,
                description=sub_goal,
                actor=agent_config.name,
                depends_on=deps  # Empty for parallel execution
            )
            logger.info(f" Added task {task_id} for {agent_config.name}: {sub_goal[:50]}... (parallel: {len(deps)==0})")

        all_results = {}  # agent_name -> EpisodeResult
        attempt_counts = {}  # task_id -> attempts
        _max_iters = getattr(sm.config, 'max_episode_iterations', 12)
        _iter_count = 0

        while _iter_count < _max_iters:
            # Collect all ready tasks (no unresolved dependencies)
            batch = []

            while True:
                next_task = sm.swarm_task_board.get_next_task()
                if next_task is None:
                    break
                # Mark as IN_PROGRESS so it's not returned again
                next_task.status = TaskStatus.IN_PROGRESS
                batch.append(next_task)

            if not batch:
                break
            _iter_count += 1

            # Show batch info immediately with agent names for better UX
            if status_callback and len(batch) > 0:
                try:
                    agent_names = [t.actor for t in batch]
                    status_callback("Running batch", f"{len(batch)} agents: {', '.join(agent_names[:5])}")
                    # Show each agent's task for clarity
                    for task in batch:
                        agent_cfg = next((a for a in sm.agents if a.name == task.actor), None)
                        sub_goal = agent_cfg.capabilities[0] if agent_cfg and agent_cfg.capabilities else task.description[:50]
                        status_callback(f"  {task.actor}", sub_goal[:60])
                except Exception as e:
                    logger.debug(f"Batch status callback failed: {e}")

            # Pre-execution: trajectory prediction (non-blocking, run in background)
            predictions = {}
            # Skip trajectory prediction to reduce latency - agents start immediately
            # Prediction can happen asynchronously after execution starts
            if sm.trajectory_predictor and len(batch) <= 2:  # Only for small batches
                for task in batch:
                    try:
                        prediction = sm.trajectory_predictor.predict(
                            current_state=sm.get_current_state(),
                            acting_agent=task.actor,
                            proposed_action={'task': task.description},
                            other_agents=[a.name for a in sm.agents if a.name != task.actor],
                            goal=goal
                        )
                        predictions[task.actor] = prediction
                    except Exception as e:
                        logger.debug(f"Trajectory prediction skipped for {task.actor}: {e}")

            # Execute batch concurrently (status_callback already extracted at method start)
            # AIOS-inspired: Semaphore limits how many agents call LLM simultaneously.
            # Without this, N agents × (architect + agent + auditor) = 3N concurrent API calls.
            async def _run_task(task):
                # Check if we'll need to wait for a slot
                if sm.agent_semaphore._value == 0:
                    sm._scheduling_stats['total_waited'] += 1
                    safe_status(status_callback, f"Agent {task.actor}", "waiting for LLM slot...")

                async with sm.agent_semaphore:
                    # Track concurrency stats
                    sm._scheduling_stats['total_scheduled'] += 1
                    sm._scheduling_stats['current_concurrent'] += 1
                    if sm._scheduling_stats['current_concurrent'] > sm._scheduling_stats['peak_concurrent']:
                        sm._scheduling_stats['peak_concurrent'] = sm._scheduling_stats['current_concurrent']

                    try:
                        # Show which agent is executing
                        agent_cfg = next((a for a in sm.agents if a.name == task.actor), None)
                        sub_goal = agent_cfg.capabilities[0] if agent_cfg and agent_cfg.capabilities else task.description[:60]

                        safe_status(status_callback, f"Agent {task.actor}", f"starting: {sub_goal}")

                        # Create agent-specific status callback that prefixes with agent name
                        _agent_reporter = StatusReporter(status_callback).with_prefix(f"  [{task.actor}]")
                        agent_status_callback = _agent_reporter

                        runner = sm.runners[task.actor]
                        # Pass the agent-specific callback and ensemble params
                        task_kwargs = dict(kwargs)
                        task_kwargs['status_callback'] = agent_status_callback
                        # Forward ensemble flag explicitly so AutoAgent doesn't auto-detect
                        task_kwargs['ensemble'] = ensemble
                        if ensemble:
                            task_kwargs['ensemble_strategy'] = ensemble_strategy
                        # Forward the swarm-level gate decision to skip redundant
                        # per-agent architect/auditor if the swarm already decided
                        if '_swarm_gate_decision' in task_kwargs:
                            task_kwargs.pop('_swarm_gate_decision')  # clean up internal flag

                        # MULTI-AGENT OPTIMIZATION: Sub-agents with system_prompt
                        # are specialized for analysis/synthesis — they don't need
                        # the full skill pipeline (saves ~100s per agent).
                        # Use direct_llm=True to bypass skill discovery/selection/planning.
                        if agent_cfg and hasattr(agent_cfg, 'agent') and agent_cfg.agent:
                            _agent_obj = agent_cfg.agent
                            _has_system_prompt = (
                                hasattr(_agent_obj, 'config')
                                and hasattr(_agent_obj.config, 'system_prompt')
                                and _agent_obj.config.system_prompt
                            )
                            if _has_system_prompt:
                                task_kwargs['direct_llm'] = True

                        return task, await runner.run(goal=task.description, **task_kwargs)
                    finally:
                        sm._scheduling_stats['current_concurrent'] -= 1

            # Per-agent timeout: prevent any single agent from blocking the entire swarm.
            # Reads from SwarmConfig.actor_timeout (default 900s).
            PER_AGENT_TIMEOUT = getattr(sm.config, 'actor_timeout', 900.0)

            async def _run_task_with_timeout(task):
                try:
                    return await asyncio.wait_for(_run_task(task), timeout=PER_AGENT_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {task.actor} timed out after {PER_AGENT_TIMEOUT}s")
                    safe_status(status_callback, f"Agent {task.actor}", f"TIMEOUT ({PER_AGENT_TIMEOUT:.0f}s)")
                    return task, EpisodeResult(
                        output=f"Agent {task.actor} timed out after {PER_AGENT_TIMEOUT:.0f}s",
                        success=False,
                        trajectory=[{'step': 0, 'action': 'timeout'}],
                        tagged_outputs=[],
                        episode=sm.episode_count,
                        execution_time=PER_AGENT_TIMEOUT,
                        architect_results=[],
                        auditor_results=[],
                        agent_contributions={},
                    )

            coro_results = await asyncio.gather(
                *[_run_task_with_timeout(t) for t in batch],
                return_exceptions=True
            )

            # Process results
            for coro_result in coro_results:
                if isinstance(coro_result, Exception):
                    logger.error(f"Task execution exception: {coro_result}")
                    safe_status(status_callback, "Agent error", str(coro_result)[:60])
                    continue

                task, result = coro_result
                attempt_counts[task.task_id] = attempt_counts.get(task.task_id, 0) + 1

                # Show agent completion status
                status_icon = "" if result.success else ""
                safe_status(status_callback, f"{status_icon} Agent {task.actor}", "completed" if result.success else "failed")
                reward = 1.0 if result.success else -0.5

                # Post-execution: divergence learning
                if sm.trajectory_predictor and task.actor in predictions:
                    try:
                        prediction = predictions[task.actor]
                        actual = ActualTrajectory(
                            steps=result.trajectory or [],
                            actual_reward=reward
                        )
                        divergence = sm.trajectory_predictor.compute_divergence(prediction, actual)
                        sm.divergence_memory.store(divergence)
                        sm.trajectory_predictor.update_from_divergence(divergence)

                        # Use divergence as TD error weight for Q-update
                        divergence_penalty = 1.0 - min(1.0, divergence.total_divergence())
                        adjusted_reward = reward * divergence_penalty
                        state = {'query': goal, 'agent': task.actor}
                        action = {'actor': task.actor, 'task': task.description[:100]}
                        sm.learning_manager.record_outcome(state, action, adjusted_reward)
                    except LearningError as e:
                        logger.debug(f"Divergence learning skipped for {task.actor} (learning): {e}")
                    except Exception as e:
                        logger.debug(f"Divergence learning skipped for {task.actor} (unexpected): {e}", exc_info=True)

                if result.success:
                    sm.swarm_task_board.complete_task(task.task_id, result={'output': result.output})
                    all_results[task.actor] = result

                    agent_config = next((a for a in sm.agents if a.name == task.actor), None)
                    if agent_config:
                        sm._learn_from_result(result, agent_config, goal=task.description or goal)
                else:
                    # Retry with enriched context if attempts remain
                    if attempt_counts.get(task.task_id, 1) < max_attempts:
                        error_msg = str(getattr(result, 'error', None) or 'Execution failed')
                        fb = FeedbackMessage(
                            source_actor='swarm_manager',
                            target_actor=task.actor,
                            feedback_type=FeedbackType.ERROR,
                            content=f"Previous attempt failed: {error_msg}. Please try a different approach.",
                            context={'attempt': attempt_counts[task.task_id], 'error': error_msg},
                            requires_response=False,
                            priority=1,
                        )
                        sm.feedback_channel.send(fb)
                        # Reset task to PENDING for retry
                        try:
                            sm.swarm_task_board.add_task(
                                task_id=f"{task.task_id}_retry{attempt_counts[task.task_id]}",
                                description=task.description,
                                actor=task.actor
                            )
                        except Exception:
                            sm.swarm_task_board.fail_task(task.task_id, error=error_msg)
                    else:
                        sm.swarm_task_board.fail_task(
                            task.task_id,
                            error=str(getattr(result, 'error', None) or 'Execution failed')
                        )
                    all_results[task.actor] = result

        # MAS-ZERO: Handle TOO_HARD signals from agents
        # If any agent signaled TOO_HARD, try re-routing its task
        for agent_name, result in list(all_results.items()):
            output = result.output if hasattr(result, 'output') else result
            is_too_hard = (
                isinstance(output, dict) and output.get('too_hard')
            ) or (
                hasattr(result, 'output') and isinstance(result.output, dict)
                and result.output.get('too_hard')
            )
            if is_too_hard and not result.success:
                logger.info(f"MAS-ZERO: Agent '{agent_name}' signaled TOO_HARD, "
                           f"task may need re-decomposition")

        # MAS-ZERO: Meta-feedback evaluation (solvability + completeness)
        meta_feedback = sm._mas_zero_evaluate(goal, all_results)

        # MAS-ZERO: Iterative refinement if meta-feedback says refine
        if meta_feedback.get('should_refine') and len(all_results) > 0:
            safe_status(status_callback, "MAS-Evolve", "refining based on meta-feedback")
            all_results = await sm._mas_zero_evolve(
                goal, all_results,
                max_iterations=2,
                status_callback=status_callback,
                **kwargs,
            )

        # Cooperative credit assignment
        self._assign_cooperative_credit(all_results, goal)

        # Post-episode learning + auto-save (fire-and-forget background)
        combined_result = self._aggregate_results(all_results, goal)
        sm._schedule_background_learning(combined_result, goal)

        # Observability: record per-agent metrics
        try:
            from Jotty.core.observability import get_metrics
            _metrics = get_metrics()
            for agent_name, result in all_results.items():
                _metrics.record_execution(
                    agent_name=agent_name,
                    task_type="multi_agent",
                    duration_s=getattr(result, 'execution_time', 0.0),
                    success=result.success if hasattr(result, 'success') else False,
                )
        except (ImportError, Exception):
            pass

        return combined_result

    # =========================================================================
    # DISCUSSION PARADIGMS — delegated to ParadigmExecutor
    # =========================================================================

    async def _paradigm_run_agent(self, runner, sub_goal, agent_name, **kwargs):
        return await self._paradigms.run_agent(runner, sub_goal, agent_name, **kwargs)

    async def _paradigm_relay(self, goal, **kwargs):
        return await self._paradigms.relay(goal, **kwargs)

    async def _paradigm_debate(self, goal, **kwargs):
        return await self._paradigms.debate(goal, **kwargs)

    async def _paradigm_refinement(self, goal, **kwargs):
        return await self._paradigms.refinement(goal, **kwargs)

    def _aggregate_results(self, results, goal):
        return self._paradigms.aggregate_results(results, goal)

    def _assign_cooperative_credit(self, results, goal):
        return self._paradigms.assign_cooperative_credit(results, goal)

    async def autonomous_setup(self, goal: str, status_callback=None):
        """
        Autonomous setup: Research, install, configure.

        Public method for manual autonomous setup.
        DRY: Reuses all autonomous components.

        Args:
            goal: Task goal
            status_callback: Optional callback(stage, detail) for progress

        Example:
            await swarm.autonomous_setup("Set up Reddit scraping")
        """
        sm = self._manager
        _status = StatusReporter(status_callback, logger, emoji="")

        # Cache check: skip if already set up for this goal
        cache_key = hash(goal)
        if cache_key in sm._setup_cache:
            _status("Setup", "using cached")
            return

        # Parse intent to understand requirements
        _status("Parsing intent", goal)
        task_graph = sm.swarm_intent_parser.parse(goal)

        # Research solutions if needed
        if task_graph.requirements or task_graph.integrations:
            # Filter out stop words and meaningless single-word requirements
            stop_words = {'existing', 'use', 'check', 'find', 'get', 'the', 'a', 'an', 'and', 'or', 'for', 'with'}
            meaningful_requirements = [
                req for req in task_graph.requirements
                if req.lower() not in stop_words and len(req.split()) > 1
            ]

            if meaningful_requirements:
                _status("Researching", f"{len(meaningful_requirements)} requirements")

            for i, requirement in enumerate(meaningful_requirements):
                if not requirement.strip():
                    continue
                _status("Research", f"[{i+1}/{len(meaningful_requirements)}] {requirement[:30]}")
                research_result = await sm.swarm_researcher.research(requirement)
                if research_result.tools_found:
                    _status("Found tools", ", ".join(research_result.tools_found[:3]))
                    for tool in research_result.tools_found:
                        _status("Installing", tool)
                        await sm.swarm_installer.install(tool)

        # Configure integrations
        if task_graph.integrations:
            _status("Configuring", f"{len(task_graph.integrations)} integrations")
            for integration in task_graph.integrations:
                _status("Configure", integration)
                await sm.swarm_configurator.configure(integration)

        # Mark as cached
        sm._setup_cache[cache_key] = True
        _status("Setup complete", "")


class Orchestrator:
    """
    Composable swarm orchestrator with lazy initialization.

    All heavyweight components are lazy-loaded via descriptors.
    Init is fast (~10ms). Components are created on first access.

    Modes:
        - single: 1 AutoAgent (default)
        - multi: N agents with SwarmTaskBoard coordination

    Key features:
        - MAS-ZERO: parallel strategies, meta-feedback, candidate verification
        - Concurrency semaphore: limits parallel LLM calls (default 3)
        - Learning pipeline: TD(λ), credit assignment, memory consolidation
        - Effectiveness tracking: measures actual improvement over time
    """

    # =========================================================================
    # LAZY COMPONENTS - Only created when first accessed
    # =========================================================================

    # Planning
    swarm_task_board = LazyComponent(lambda self: _create_task_board())
    swarm_planner = LazyComponent(lambda self: _create_planner())
    swarm_intent_parser = LazyComponent(lambda self: _create_intent_parser(self.swarm_planner))

    # Memory
    swarm_memory = LazyComponent(lambda self: _create_memory(self.config))

    # Provider Gateway
    swarm_provider_gateway = LazyComponent(lambda self: _create_provider_gateway(self.config))

    # Autonomous (only used in autonomous_setup)
    swarm_researcher = LazyComponent(lambda self: _create_researcher(self.config))
    swarm_installer = LazyComponent(lambda self: _create_installer(self.config))
    swarm_configurator = LazyComponent(lambda self: _create_configurator(self.config))
    swarm_code_generator = LazyComponent(lambda self: _create_code_generator(self.config))
    swarm_workflow_learner = LazyComponent(lambda self: _create_workflow_learner(self.swarm_memory))
    swarm_integrator = LazyComponent(lambda self: _create_integrator(self.config))
    swarm_terminal = LazyComponent(lambda self: _create_terminal(self.config))

    # Feature components
    swarm_ui_registry = LazyComponent(lambda self: _create_ui_registry())
    swarm_tool_validator = LazyComponent(lambda self: _create_tool_validator())
    swarm_tool_registry = LazyComponent(lambda self: _create_tool_registry())
    swarm_profiler = LazyComponent(lambda self: _create_profiler(self.config))

    # State management
    swarm_state_manager = LazyComponent(lambda self: _create_state_manager(self))
    shared_context = LazyComponent(lambda self: _create_shared_context())
    io_manager = LazyComponent(lambda self: _create_io_manager())
    data_registry = LazyComponent(lambda self: _create_data_registry())
    context_guard = LazyComponent(lambda self: _create_context_guard())

    # Learning (single pipeline, components accessed via .learning.xxx)
    learning = LazyComponent(lambda self: _create_learning_pipeline(self.config))
    mas_learning = LazyComponent(lambda self: _create_mas_learning(self))

    # =========================================================================
    # COMPOSED MANAGERS (has-a, not is-a) — replaces mixin inheritance
    # Each manager takes explicit dependencies instead of implicit self.xxx
    # =========================================================================

    _providers = LazyComponent(lambda self: ProviderManager(
        config=self.config,
        get_swarm_intelligence=lambda: self.swarm_intelligence,
    ))
    _ensemble = LazyComponent(lambda self: EnsembleManager())
    _learning_ops = LazyComponent(lambda self: LearningDelegate(
        get_learning=lambda: self.learning,
        get_mas_learning=lambda: self.mas_learning,
        get_agents=lambda: self.agents,
    ))
    _mas_zero = LazyComponent(lambda self: MASZeroController(
        get_agents=lambda: self.agents,
        get_runners=lambda: self.runners,
    ))
    _router = LazyComponent(lambda self: SwarmRouter(
        get_swarm_intelligence=lambda: getattr(self, 'swarm_intelligence', None),
        get_agents=lambda: self.agents,
        get_model_tier_router=lambda: self._model_tier_router,
        get_learning=lambda: getattr(self, 'learning', None),
    ))

    # =========================================================================
    # INIT - Fast, minimal, no I/O
    # =========================================================================

    def __init__(
        self,
        agents: Optional[Union[AgentConfig, List[AgentConfig], str]] = None,
        config: Optional[SwarmConfig] = None,
        architect_prompts: Optional[List[str]] = None,
        auditor_prompts: Optional[List[str]] = None,
        enable_zero_config: bool = True,
        enable_lotus: bool = True,
        max_concurrent_agents: int = 3,
    ):
        """
        Initialize Orchestrator.

        Fast init (~10ms). All heavyweight components are lazy-loaded.

        Args:
            agents: AgentConfig, list of AgentConfigs, or natural language (zero-config)
            config: SwarmConfig (defaults if None)
            architect_prompts: Architect prompt paths
            auditor_prompts: Auditor prompt paths
            enable_zero_config: Enable natural language -> agent conversion
            enable_lotus: Enable LOTUS optimization layer
            max_concurrent_agents: Max agents calling LLM concurrently (AIOS-inspired, default 3)
        """
        import warnings
        warnings.warn(
            "Orchestrator is deprecated. Use Jotty() instead:\n"
            "  jotty = Jotty()\n"
            "  result = await jotty.run('task')  # auto-detects tier\n"
            "  result = await jotty.swarm('task', swarm_name='coding')  # specific swarm\n"
            "  result = await jotty.autonomous('task')  # full features\n",
            DeprecationWarning, stacklevel=2
        )
        self.config = config or SwarmConfig()
        self.enable_zero_config = enable_zero_config
        self.enable_lotus = enable_lotus
        self.episode_count = 0

        # AIOS-inspired: Concurrency control for multi-agent LLM fan-out.
        # Prevents API rate-limit errors when N agents fire in parallel.
        # DRY: Single semaphore, no wrapper classes needed.
        self.max_concurrent_agents = max_concurrent_agents
        self._agent_semaphore = None  # Lazy-created in current event loop
        self._scheduling_stats: Dict[str, int] = {
            'total_scheduled': 0,
            'total_waited': 0,    # times an agent had to wait for a slot
            'peak_concurrent': 0,
            'current_concurrent': 0,
        }

        # Prompts
        self.architect_prompts = architect_prompts or ["configs/prompts/architect/base_architect.md"]
        self.auditor_prompts = auditor_prompts or ["configs/prompts/auditor/base_auditor.md"]

        # Zero-config: natural language -> agents
        if isinstance(agents, str) and enable_zero_config:
            logger.info("Zero-config mode: analyzing task for agent configuration")
            agents = self._create_zero_config_agents(agents)

        # Normalize agents
        if agents is None:
            from Jotty.core.agents.auto_agent import AutoAgent
            agents = [AgentConfig(name="auto", agent=AutoAgent())]
        elif isinstance(agents, AgentConfig):
            agents = [agents]

        self.agents = agents
        self.mode = "multi" if len(agents) > 1 else "single"

        # Runner and LOTUS state (created lazily in _ensure_runners)
        self.runners: Dict[str, 'AgentRunner'] = {}
        self._runners_built = False
        self.lotus = None
        self.lotus_optimizer = None

        # Provider registry (delegated to _providers manager)
        # Backward compat: self.provider_registry -> self._providers.provider_registry

        # Setup cache
        self._setup_cache = {}

        # Optima-inspired efficiency tracking (Chen et al., 2024):
        # Track orchestration overhead vs. actual execution per run.
        # KISS: Just a dict, no new classes. Reset each run().
        self._efficiency_stats: Dict[str, float] = {}

        # Background training daemon (composed)
        self._training = TrainingDaemon(self)

        # Learning readiness: set by _ensure_runners background init,
        # awaited by run() to prevent operating with partially-loaded state.
        self._learning_ready = asyncio.Event()

        # Composed ExecutionOrchestrator — separates "how to run" from "how to manage"
        from Jotty.core.orchestration.execution_orchestrator import ExecutionOrchestrator
        self._execution = ExecutionOrchestrator(self)

        # Intelligence effectiveness A/B metrics:
        # Tracks whether stigmergy/byzantine guidance improves success rate.
        # Keyed by task_type for fine-grained analysis.
        # KISS: Nested dict, no classes.
        self._intelligence_metrics: Dict[str, Dict[str, int]] = {}

        # Model tier router: maps task complexity -> cheap/balanced/quality LM
        # Lazy-init on first use. Integrates with ValidationGate decisions.
        self._model_tier_router: Optional[ModelTierRouter] = None

        # Composed AgentFactory — separates agent/runner creation from orchestration
        self._agent_factory = AgentFactory(self)

        # Composed ExecutionEngine — separates task execution from management
        self._engine = ExecutionEngine(self)

        logger.info(f"Orchestrator: {self.mode} mode, {len(self.agents)} agents (lazy init)")

    # =========================================================================
    # LAZY RUNNER CREATION — delegated to AgentFactory
    # =========================================================================

    def _ensure_runners(self): self._agent_factory.ensure_runners()

    # =========================================================================
    # DELEGATION: Single __getattr__ replaces 15+ @property boilerplate
    # =========================================================================
    #
    # Learning sub-components (sm.learning_manager, sm.swarm_intelligence, etc.)
    # are forwarded to self.learning.xxx automatically.
    # Composed manager methods (_execute_ensemble, etc.) are forwarded to the
    # appropriate composed manager.
    #
    # This eliminates ~120 lines of repetitive @property definitions while
    # maintaining full backward compatibility.

    # Attributes forwarded to self.learning
    _LEARNING_ATTRS = frozenset({
        'learning_manager', 'transfer_learning', 'swarm_intelligence',
        'credit_weights', 'trajectory_predictor', 'divergence_memory',
        'brain_state', 'agent_abstractor',
        'swarm_learner', 'agent_slack', 'feedback_channel',
    })

    @property
    def agent_semaphore(self) -> asyncio.Semaphore:
        """Lazy-create asyncio.Semaphore in the current event loop."""
        if self._agent_semaphore is None:
            self._agent_semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        return self._agent_semaphore

    def __getattr__(self, name: str):
        """
        Delegate attribute access to composed managers.

        Order: learning pipeline attrs → _providers → raise AttributeError.
        Only called when normal attribute lookup fails (i.e., LazyComponent
        descriptors and instance __dict__ are checked first).
        """
        # Learning pipeline sub-components
        if name in self._LEARNING_ATTRS:
            try:
                return getattr(self.learning, name)
            except AttributeError:
                raise
            except Exception as e:
                raise AttributeError(
                    f"Failed to delegate '{name}' to learning pipeline: {e}"
                ) from e

        # Provider registry
        if name == 'provider_registry':
            return self._providers.provider_registry

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        """Handle setting delegated attributes (credit_weights, provider_registry)."""
        if name == 'credit_weights' and '_lazy_learning' in self.__dict__:
            self.learning.credit_weights = value
        elif name == 'provider_registry' and '_lazy__providers' in self.__dict__:
            self._providers.provider_registry = value
        else:
            super().__setattr__(name, value)

    def _maybe_drain_training_task(self, result) -> None:
        """Drain one queued training task after a successful run (placeholder)."""
        pass

    # --- Composed manager delegation (thin methods, not properties) ---

    async def execute_with_provider(self, category, task, context=None, provider_name=None):
        return await self._providers.execute_with_provider(category, task, context, provider_name)

    def get_provider_summary(self):
        return self._providers.get_provider_summary()

    async def _execute_ensemble(self, goal, strategy='multi_perspective',
                                status_callback=None, max_perspectives=4):
        return await self._ensemble.execute_ensemble(goal, strategy, status_callback, max_perspectives)

    def _should_auto_ensemble(self, goal):
        return self._ensemble.should_auto_ensemble(goal)

    def _auto_load_learnings(self):
        self._learning_ops.auto_load_learnings()

    def _auto_save_learnings(self):
        self._learning_ops.auto_save_learnings(
            mas_learning=getattr(self, 'mas_learning', None),
            swarm_terminal=getattr(self, 'swarm_terminal', None),
            provider_registry=self._providers.provider_registry,
            memory_persistence=getattr(self, 'memory_persistence', None),
        )

    def _save_learnings(self):
        self._auto_save_learnings()

    def load_relevant_learnings(self, task_description, agent_types=None):
        return self._learning_ops.load_relevant_learnings(task_description, agent_types)

    async def train(
        self,
        num_tasks: int = 5,
        status_callback=None,
    ) -> Dict[str, Any]:
        """
        Run self-curriculum training: generate tasks from weaknesses and execute them.

        This is the CONSUMER for CurriculumGenerator. Without this method,
        generated curriculum tasks have no execution path.

        DrZero loop: generate task → execute → record outcome → adjust difficulty.

        Args:
            num_tasks: Number of training tasks to run (default 5)
            status_callback: Optional progress callback

        Returns:
            Dict with training results and improvement metrics
        """
        _status = StatusReporter(status_callback, logger, emoji="")

        # Checkpoint before training — enables rollback if training degrades
        checkpoint_path = None
        try:
            checkpoint_path = self.learning.save_checkpoint(label="pre_train")
            _status("Checkpoint", f"saved to {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Pre-training checkpoint failed: {e}")

        _status("Training", f"starting {num_tasks} curriculum tasks")

        lp = self.learning
        curriculum = lp.curriculum_generator
        si = lp.swarm_intelligence
        profiles = si.agent_profiles

        # Pre-training effectiveness snapshot
        pre_report = lp.effectiveness.improvement_report()

        results = []
        for i in range(num_tasks):
            # Generate a task targeting weaknesses
            task = curriculum.generate_smart_task(
                profiles=profiles,
                collective_memory=list(si.collective_memory),
            )
            _status(f"Task {i+1}/{num_tasks}", f"[{task.task_type}] {task.description[:60]}")

            try:
                result = await self.run(
                    goal=task.description,
                    skip_autonomous_setup=True,
                    status_callback=status_callback,
                )
                results.append({
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'difficulty': task.difficulty,
                    'success': result.success,
                    'execution_time': getattr(result, 'execution_time', 0),
                })

                # Feed result back to curriculum for difficulty adjustment
                curriculum.update_from_result(
                    task=task,
                    success=result.success,
                    execution_time=getattr(result, 'execution_time', 0.0),
                )

                _status(
                    f"Task {i+1} {'passed' if result.success else 'failed'}",
                    f"type={task.task_type}, difficulty={task.difficulty:.1f}"
                )
            except (AgentExecutionError, LLMError) as e:
                logger.warning(f"Training task {i+1} failed (recoverable: {type(e).__name__}): {e}")
                results.append({
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'difficulty': task.difficulty,
                    'success': False,
                    'error': str(e),
                })
            except Exception as e:
                logger.warning(f"Training task {i+1} failed (unexpected): {e}", exc_info=True)
                results.append({
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'difficulty': task.difficulty,
                    'success': False,
                    'error': str(e),
                })

        # Post-training effectiveness snapshot
        post_report = lp.effectiveness.improvement_report()

        # Save all learnings
        self._auto_save_learnings()

        successes = sum(1 for r in results if r.get('success'))
        _status("Training complete", f"{successes}/{num_tasks} passed")

        return {
            'total_tasks': num_tasks,
            'successes': successes,
            'success_rate': successes / max(1, num_tasks),
            'results': results,
            'pre_effectiveness': pre_report.get('_global', {}),
            'post_effectiveness': post_report.get('_global', {}),
            'checkpoint': checkpoint_path,  # rollback with self.learning.restore_checkpoint(path)
        }

    def record_agent_result(self, agent_name, task_type, success, time_taken, output_quality=0.0) -> None:
        self._learning_ops.record_agent_result(agent_name, task_type, success, time_taken, output_quality)

    def record_session_result(self, task_description, agent_performances,
                              total_time, success, fixes_applied=None, stigmergy_signals=0):
        self._learning_ops.record_session_result(
            task_description, agent_performances, total_time, success,
            fixes_applied, stigmergy_signals,
        )

    def get_transferable_context(self, query, agent=None):
        return self._learning_ops.get_transferable_context(query, agent)

    def get_swarm_wisdom(self, query):
        return self._learning_ops.get_swarm_wisdom(query)

    def get_agent_specializations(self):
        return self._learning_ops.get_agent_specializations()

    def get_best_agent_for_task(self, query):
        return self._learning_ops.get_best_agent_for_task(query)

    def _mas_zero_verify(self, goal, results):
        return self._mas_zero.verify(goal, results)

    def _mas_zero_evaluate(self, goal, results):
        return self._mas_zero.evaluate(goal, results)

    def _mas_zero_should_reduce(self, goal):
        return self._mas_zero.should_reduce(goal)

    async def _mas_zero_evolve(self, goal, initial_results, max_iterations=2,
                               status_callback=None, **kwargs):
        return await self._mas_zero.evolve(
            goal, initial_results, max_iterations, status_callback, **kwargs
        )

    def _reset_experience(self):
        self._mas_zero.reset_experience()

    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    async def startup(self):
        """
        Async startup - prepare Orchestrator for execution.

        Call this before run() for controlled initialization.
        If not called, run() will auto-initialize (lazy).

        Returns:
            Self for chaining: `sm = await Orchestrator().startup()`
        """
        self._ensure_runners()
        logger.info("Orchestrator startup complete")
        return self

    async def shutdown(self):
        """
        Graceful shutdown - persist learnings and release resources.

        Should be called when Orchestrator is no longer needed.
        Safe to call multiple times.
        """
        try:
            # Await any in-flight background learning tasks
            await self._drain_background_tasks()

            # Persist all learnings
            if '_lazy_learning' in self.__dict__:
                self._auto_save_learnings()
                logger.info("Learnings saved on shutdown")

            # Clear runners
            self.runners.clear()
            self._runners_built = False

            logger.info("Orchestrator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def __aenter__(self):
        """Async context manager: `async with Orchestrator() as sm:`"""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        await self.shutdown()
        return False

    # =========================================================================
    # INTROSPECTION & METRICS
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """
        Get full introspection of Orchestrator state.

        Shows which lazy components have been created, runner status,
        learning stats, and execution metrics.

        Returns:
            Dict with component status, metrics, and health info.
        """
        lazy_names = [
            'swarm_planner', 'swarm_task_board', 'swarm_memory',
            'swarm_intent_parser', 'swarm_provider_gateway',
            'swarm_researcher', 'swarm_installer', 'swarm_configurator',
            'swarm_code_generator', 'swarm_workflow_learner',
            'swarm_integrator', 'swarm_terminal',
            'swarm_ui_registry', 'swarm_tool_validator', 'swarm_tool_registry',
            'swarm_profiler', 'swarm_state_manager', 'shared_context',
            'io_manager', 'data_registry', 'context_guard',
            'learning', 'mas_learning',
        ]
        created = {n: f'_lazy_{n}' in self.__dict__ for n in lazy_names}
        created_count = sum(created.values())

        result = {
            'mode': self.mode,
            'agents': [a.name for a in self.agents],
            'runners_built': self._runners_built,
            'runners': list(self.runners.keys()),
            'episode_count': self.episode_count,
            'lotus_enabled': self.enable_lotus,
            'lotus_active': self.lotus is not None,
            'zero_config': self.enable_zero_config,
            'components': {
                'total': len(lazy_names),
                'created': created_count,
                'pending': len(lazy_names) - created_count,
                'detail': created,
            },
        }

        # Scheduling stats (AIOS-inspired)
        result['scheduling'] = {
            'max_concurrent_agents': self.max_concurrent_agents,
            'semaphore_available': self._agent_semaphore._value if self._agent_semaphore else self.max_concurrent_agents,
            **self._scheduling_stats,
        }

        # Add learning stats if pipeline is active
        if '_lazy_learning' in self.__dict__:
            try:
                lp = self.learning
                result['learning'] = {
                    'episode_count': lp.episode_count,
                    'has_intelligence': self.swarm_intelligence is not None,
                    'stigmergy_signals': len(lp.stigmergy.signals),
                    'byzantine_verifications': lp.byzantine_verifier.verified_count,
                    'byzantine_inconsistencies': lp.byzantine_verifier.inconsistent_count,
                    'credit_stats': lp.get_credit_stats(),
                    'adaptive_learning': lp.get_learning_state(),
                }
            except Exception as e:
                logger.debug(f"Learning stats collection failed: {e}")
                result['learning'] = {'status': 'error'}

        # Training daemon status
        result['training_daemon'] = self.training_daemon_status()

        # Intelligence effectiveness A/B metrics (per task_type)
        def _format_im(bucket) -> Dict:
            gr = bucket.get('guided_runs', 0)
            gs = bucket.get('guided_successes', 0)
            ur = bucket.get('unguided_runs', 0)
            us = bucket.get('unguided_successes', 0)
            guided_rate = gs / gr if gr > 0 else None
            unguided_rate = us / ur if ur > 0 else None
            return {
                **bucket,
                'guided_success_rate': guided_rate,
                'unguided_success_rate': unguided_rate,
                'guidance_lift': (
                    guided_rate - unguided_rate
                    if guided_rate is not None and unguided_rate is not None
                    else None
                ),
            }

        result['intelligence_effectiveness'] = {
            tt: _format_im(bucket)
            for tt, bucket in self._intelligence_metrics.items()
        }

        # Paradigm effectiveness stats
        if '_lazy_learning' in self.__dict__:
            try:
                result['paradigm_stats'] = self.learning.get_paradigm_stats()
            except Exception as e:
                logger.debug(f"Paradigm stats unavailable: {e}")

        # Add LOTUS stats if active
        if self.lotus:
            result['lotus_stats'] = self.get_lotus_stats()

        # Add observability metrics
        try:
            from Jotty.core.observability import get_metrics, get_tracer
            _metrics = get_metrics()
            result['observability'] = {
                'metrics': _metrics.get_summary(),
                'cost_breakdown': _metrics.get_cost_breakdown(),
            }
            _tracer = get_tracer()
            _trace = _tracer.get_current_trace()
            if _trace:
                result['observability']['last_trace'] = {
                    'trace_id': _trace.trace_id[:8],
                    'spans': _trace.span_count,
                    'duration_ms': round(_trace.duration_ms, 0),
                    'total_cost_usd': round(_trace.total_cost, 6),
                    'total_tokens': _trace.total_tokens,
                }
        except (ImportError, Exception) as e:
            logger.debug(f"Observability metrics unavailable: {e}")

        # Add model tier routing stats
        if self._model_tier_router:
            result['model_tier_routing'] = self._model_tier_router.get_savings_estimate()

        return result

    @property
    def metrics(self) -> Dict[str, Any]:
        """Quick access to execution metrics."""
        return {
            'episodes': self.episode_count,
            'agents': len(self.agents),
            'mode': self.mode,
            'runners_built': self._runners_built,
            'components_loaded': sum(
                1 for k in self.__dict__ if k.startswith('_lazy_')
            ),
        }

    # =========================================================================
    # LOTUS OPTIMIZATION — delegated to AgentFactory
    # =========================================================================

    def _init_lotus_optimization(self): self._agent_factory.init_lotus_optimization()
    def get_lotus_stats(self): return self._agent_factory.get_lotus_stats()
    def get_lotus_savings(self): return self._agent_factory.get_lotus_savings()

    # =========================================================================
    # ML Learning Bridge (for SkillOrchestrator / Swarm pipeline integration)
    # =========================================================================

    def get_ml_learning(self):
        """Get MASLearning instance for ML pipeline integration."""
        return self.mas_learning

    def record_report_section_outcome(self, section_name: str, success: bool, error: str = None) -> None:
        """Record report section outcome for cross-run learning."""
        try:
            if self.learning_manager and hasattr(self.learning_manager, 'record_experience'):
                self.learning_manager.record_experience(
                    agent_name='report_generator',
                    state={'section': section_name, 'error': (error or '')[:200]},
                    action={'type': 'generate_section', 'section': section_name},
                    reward=1.0 if success else -1.0,
                    domain='report_sections',
                )
        except Exception as e:
            logger.debug(f"Record section outcome failed: {e}")

    def should_skip_report_section(self, section_name: str) -> bool:
        """Check if a report section should be skipped based on learned failures."""
        try:
            if self.learning_manager and hasattr(self.learning_manager, 'get_learned_context'):
                context = self.learning_manager.get_learned_context(
                    state={'section': section_name, 'task': 'report_generation'},
                    action={'type': 'generate_section', 'section': section_name}
                )
                if context and 'negative reward' in context.lower():
                    return True
        except Exception as e:
            logger.debug(f"Section skip check failed for '{section_name}': {e}")
        return False

    # =========================================================================
    # Provider, Ensemble, Learning methods — see _provider_mixin.py,
    # _ensemble_mixin.py, _learning_delegation_mixin.py
    # =========================================================================

    def _register_agents_with_axon(self): self._agent_factory.register_agents_with_axon()

    # list_capabilities() and get_help() removed — see CLAUDE.md or docs/
    
    def parse_intent_to_agent_config(self, natural_language: str) -> AgentConfig:
        """
        Convert natural language to AgentConfig (zero-config).
        
        Public utility method for intent parsing.
        DRY: Reuses IntentParser and AutoAgent.
        
        Args:
            natural_language: Natural language request
            
        Returns:
            AgentConfig with AutoAgent
            
        Example:
            agent_config = swarm.parse_intent_to_agent_config("Research topic")
        """
        # Parse intent (DRY: reuse IntentParser)
        task_graph = self.swarm_intent_parser.parse(natural_language)
        
        # Create AutoAgent (DRY: reuse existing AutoAgent)
        agent = AutoAgent()
        
        # Create AgentConfig with parsed metadata
        agent_config = AgentConfig(
            name="auto",
            agent=agent,
            metadata={
                "original_request": natural_language,
                "task_type": task_graph.task_type.value,
                "workflow": task_graph.workflow,
                "operations": task_graph.operations,
                "integrations": task_graph.integrations,
                "requirements": task_graph.requirements,
            }
        )
        
        logger.info(f" Converted natural language to AgentConfig: {task_graph.task_type.value}")
        return agent_config
    
    def compose_prompt(
        self,
        agent_name: str = "",
        task: str = "",
        learning_context: Optional[list] = None,
        constraints: Optional[list] = None,
        extra_sections: Optional[dict] = None,
    ) -> str:
        """
        Compose a model-family-aware agent prompt using PromptComposer.

        Adapts prompt structure to the LLM provider (Claude: XML, GPT: Markdown,
        Groq: minimal). Integrates tool trust levels and context gates.

        This is a utility — the existing raw string path still works.
        Use this when you want model-optimized prompts.
        """
        from Jotty.core.prompts import PromptComposer

        # Detect model from agent config
        model = ""
        for ac in self.agents:
            if ac.name == agent_name or not agent_name:
                model = getattr(ac, 'model', '') or getattr(ac.agent, 'model', '') or ''
                break
        if not model:
            model = getattr(self.config, 'model', '')

        composer = PromptComposer(model=model)

        # Gather tool info with trust levels
        tool_names = []
        tool_descs = {}
        trust_levels = {}
        try:
            from Jotty.core.registry import get_unified_registry
            registry = get_unified_registry()
            skills = registry.list_skills()
            for s in skills[:30]:  # Top 30 to keep prompt manageable
                tool_names.append(s['name'])
                tool_descs[s['name']] = s.get('description', '')
                trust_levels[s['name']] = s.get('trust_level', 'safe')
        except Exception as e:
            logger.debug(f"Tool registry lookup failed: {e}")

        # Agent identity
        identity = ""
        for ac in self.agents:
            if ac.name == agent_name or not agent_name:
                identity = getattr(ac.agent, 'system_prompt', '') or getattr(ac, 'system_prompt', '')
                break

        import os as _os
        return composer.compose(
            identity=identity,
            tools=tool_names if tool_names else None,
            tool_descriptions=tool_descs,
            trust_levels=trust_levels,
            learning_context=learning_context,
            constraints=constraints,
            task=task,
            extra_sections=extra_sections,
            workspace_dir=_os.getcwd(),
        )

    async def run(self, goal, **kwargs): return await self._engine.run(goal, **kwargs)

    # _execute_ensemble and _should_auto_ensemble — delegated to EnsembleManager

    async def _execute_single_agent(self, goal, **kwargs): return await self._engine._execute_single_agent(goal, **kwargs)
    async def _execute_multi_agent(self, goal, **kwargs): return await self._engine._execute_multi_agent(goal, **kwargs)
    async def _paradigm_run_agent(self, *args, **kwargs): return await self._engine._paradigm_run_agent(*args, **kwargs)
    async def _paradigm_relay(self, goal, **kwargs): return await self._engine._paradigm_relay(goal, **kwargs)
    async def _paradigm_debate(self, goal, **kwargs): return await self._engine._paradigm_debate(goal, **kwargs)
    async def _paradigm_refinement(self, goal, **kwargs): return await self._engine._paradigm_refinement(goal, **kwargs)
    def _aggregate_results(self, results, goal): return self._engine._aggregate_results(results, goal)
    def _assign_cooperative_credit(self, results, goal): return self._engine._assign_cooperative_credit(results, goal)

    def _create_zero_config_agents(self, task, status_callback=None): return self._agent_factory.create_zero_config_agents(task, status_callback)

    # _should_auto_ensemble — see _ensemble_mixin.py

    def _post_episode_learning(self, result: EpisodeResult, goal: str):
        """Delegate to SwarmLearningPipeline + update intelligence metrics."""
        self.learning.post_episode(
            result=result,
            goal=goal,
            agents=self.agents,
            architect_prompts=self.architect_prompts,
            mas_learning=getattr(self, 'mas_learning', None),
            swarm_terminal=getattr(self, 'swarm_terminal', None),
        )
        self.episode_count = self.learning.episode_count

        # Track intelligence A/B effectiveness (per task_type)
        if result.success:
            _guided = getattr(self, '_last_run_guided', False)
            _tt = getattr(self, '_last_task_type', '_global')
            for _bucket_key in (_tt, '_global') if _tt != '_global' else ('_global',):
                bucket = self._intelligence_metrics.get(_bucket_key)
                if bucket:
                    if _guided:
                        bucket['guided_successes'] += 1
                    else:
                        bucket['unguided_successes'] += 1

        # Track paradigm effectiveness (for auto paradigm selection)
        paradigm = getattr(self, '_last_paradigm', None)
        if paradigm:
            try:
                task_type = self.learning.transfer_learning.extractor.extract_task_type(goal)
                self.learning.record_paradigm_result(paradigm, result.success, task_type)
            except Exception as e:
                logger.debug(f"Paradigm result recording with task_type failed: {e}")
                try:
                    self.learning.record_paradigm_result(paradigm, result.success)
                except Exception as e2:
                    logger.debug(f"Paradigm result recording failed: {e2}")

    def _schedule_background_learning(self, result: EpisodeResult, goal: str):
        """
        Fire-and-forget: run post-episode learning + auto-save in a background task.
        
        Users get their result immediately. Learning/saving happens concurrently.
        If the event loop shuts down before completion, learnings are best-effort
        (next successful run will re-save).
        """
        import asyncio

        async def _background():
            try:
                self._post_episode_learning(result, goal)
            except Exception as e:
                logger.warning(f"Background post-episode learning failed: {e}")
            try:
                self._auto_save_learnings()
            except Exception as e:
                logger.warning(f"Background auto-save failed: {e}")

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_background())
            # Track background tasks so shutdown() can await them
            if not hasattr(self, '_background_tasks'):
                self._background_tasks = set()
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No running event loop — fall back to synchronous
            self._post_episode_learning(result, goal)
            self._auto_save_learnings()

    async def _drain_background_tasks(self, timeout: float = 10.0):
        """Await all pending background learning tasks (with timeout)."""
        import asyncio
        tasks = getattr(self, '_background_tasks', set())
        if tasks:
            pending = [t for t in tasks if not t.done()]
            if pending:
                logger.info(f"⏳ Waiting for {len(pending)} background learning task(s)...")
                done, still_pending = await asyncio.wait(pending, timeout=timeout)
                if still_pending:
                    logger.warning(
                        f" {len(still_pending)} background task(s) didn't finish in {timeout}s, "
                        f"cancelling (learnings will be saved next run)"
                    )
                    for t in still_pending:
                        t.cancel()

    def _log_execution_summary(self, result: EpisodeResult):
        """Log a user-friendly summary with artifacts after execution.
        
        Uses the OUTER result.success (which includes auditor verdict) 
        to avoid contradictions like "completed successfully" + SUCCESS: False.
        """
        try:
            _output = getattr(result, 'output', None)
            outer_success = result.success

            if hasattr(_output, 'artifacts') or hasattr(_output, 'skills_used'):
                # Build our own summary using the outer success status
                parts = []
                status = "completed successfully" if outer_success else "failed (auditor rejected)"
                exec_time = getattr(_output, 'execution_time', 0) or 0
                steps = getattr(_output, 'steps_executed', 0) or 0
                parts.append(f"Task {status} in {exec_time:.1f}s ({steps} steps)")

                skills = getattr(_output, 'skills_used', [])
                if skills:
                    parts.append(f"Skills used: {', '.join(skills)}")

                if hasattr(_output, 'artifacts'):
                    artifacts = _output.artifacts
                    if artifacts:
                        parts.append("Files created:")
                        for a in artifacts:
                            size = f" ({a['size_bytes']} bytes)" if a.get('size_bytes') else ""
                            parts.append(f"  → {a['path']}{size}")

                errors = getattr(_output, 'errors', [])
                if errors:
                    parts.append(f"Errors: {'; '.join(str(e) for e in errors[:3])}")

                logger.info(f"\n Execution Summary:\n" + '\n'.join(parts))
            elif hasattr(_output, 'summary'):
                logger.info(f"\n Execution Summary:\n{_output.summary}")
        except Exception as e:
            logger.debug(f"Summary logging skipped: {e}")

    def _learn_from_result(self, result: EpisodeResult, agent_config: AgentConfig, goal: str = ''):
        """Delegate to SwarmLearningPipeline."""
        self.learning.learn_from_result(
            result=result,
            agent_config=agent_config,
            workflow_learner=self.swarm_workflow_learner,
            goal=goal,
        )
    
    async def autonomous_setup(self, goal, status_callback=None): return await self._engine.autonomous_setup(goal, status_callback)

    # =====================================================================
    # State Management Methods (V1 capabilities integrated)
    # =====================================================================
    
    # State delegation — use self.swarm_state_manager directly.
    # Kept get_current_state as it's used internally by _execute_multi_agent.
    def get_current_state(self) -> Dict[str, Any]:
        """Get current swarm-level state."""
        if not self.swarm_state_manager:
            return {}
        return self.swarm_state_manager.get_current_state()

    # =====================================================================
    # Warmup — delegated to SwarmWarmup
    # =====================================================================

    async def warmup(self, **kwargs) -> Dict[str, Any]:
        """DrZero-inspired zero-data bootstrapping. See SwarmWarmup."""
        if not hasattr(self, '_warmup') or self._warmup is None:
            from Jotty.core.orchestration.swarm_warmup import SwarmWarmup
            self._warmup = SwarmWarmup(self)
        return await self._warmup.warmup(**kwargs)

    def get_warmup_recommendation(self) -> Dict[str, Any]:
        """Check if warmup would be beneficial."""
        if not hasattr(self, '_warmup') or self._warmup is None:
            from Jotty.core.orchestration.swarm_warmup import SwarmWarmup
            self._warmup = SwarmWarmup(self)
        return self._warmup.get_recommendation()

    # =====================================================================
    # DAG — delegated to SwarmDAGExecutor
    # =====================================================================

    async def run_with_dag(self, implementation_plan: str, **kwargs) -> EpisodeResult:
        """Execute via DAG-based orchestration. See SwarmDAGExecutor."""
        if not hasattr(self, '_dag_executor') or self._dag_executor is None:
            from Jotty.core.orchestration.swarm_dag_executor import SwarmDAGExecutor
            self._dag_executor = SwarmDAGExecutor(self)
        return await self._dag_executor.run(implementation_plan, **kwargs)

    def get_dag_agents(self):
        """Get DAG agents for external use."""
        if not hasattr(self, '_dag_executor') or self._dag_executor is None:
            from Jotty.core.orchestration.swarm_dag_executor import SwarmDAGExecutor
            self._dag_executor = SwarmDAGExecutor(self)
        return self._dag_executor.get_agents()

    # =========================================================================
    # Self-improvement — delegated to TrainingDaemon
    # =========================================================================

    async def run_training_task(self):
        return await self._training.run_training_task()

    @property
    def pending_training_tasks(self):
        return self._training.pending_count

    async def start_training_loop(self, **kwargs):
        return await self._training.start_training_loop(**kwargs)

    def start_training_daemon(self, **kwargs):
        return self._training.start(**kwargs)

    def stop_training_daemon(self):
        return self._training.stop()

    def training_daemon_status(self):
        return self._training.status()
