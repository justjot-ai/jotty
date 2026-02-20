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
        transfer_learning, swarm_intelligence,
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from Jotty.core.capabilities.registry.tool_validation import ToolValidator
    from Jotty.core.infrastructure.context.context_guard import (
        LLMContextManager,  # type: ignore[import-not-found]
    )
    from Jotty.core.infrastructure.data.data_registry import (
        DataRegistry,  # type: ignore[import-not-found, import]
    )
    from Jotty.core.infrastructure.data.io_manager import (
        IOManager,  # type: ignore[import-not-found, import]
    )
    from Jotty.core.infrastructure.monitoring.metrics.profiler import (
        PerformanceProfiler,  # type: ignore[import]
    )
    from Jotty.core.infrastructure.persistence.shared_context import (
        SharedContext,  # type: ignore[import-not-found, import]
    )
    from Jotty.core.intelligence.memory.cortex import SwarmMemory
    from Jotty.core.intelligence.orchestration.learning.mas_learning import MASLearning
    from Jotty.core.intelligence.orchestration.learning.swarm_learning_pipeline import (
        SwarmLearningPipeline,
    )
    from Jotty.core.intelligence.orchestration.routing.swarm_provider_gateway import (
        SwarmProviderGateway,
    )
    from Jotty.core.intelligence.orchestration.state.swarm_roadmap import SwarmTaskBoard
    from Jotty.core.intelligence.orchestration.state.swarm_state_manager import SwarmStateManager
    from Jotty.core.intelligence.orchestration.state.swarm_terminal import SwarmTerminal
    from Jotty.core.intelligence.orchestration.swarm_code_generator import SwarmCodeGenerator
    from Jotty.core.intelligence.orchestration.swarm_installer import SwarmInstaller
    from Jotty.core.intelligence.orchestration.swarm_researcher import SwarmResearcher
    from Jotty.core.intelligence.reasoning.autonomous.intent_parser import (
        IntentParser,  # type: ignore[import]
    )
    from Jotty.core.intelligence.reasoning.planners.agentic_planner import (
        TaskPlanner,  # type: ignore[import]
    )

from Jotty.core.infrastructure.foundation.agent_config import (
    AgentConfig,  # type: ignore[import-not-found, import]
)
from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult, SwarmConfig
from Jotty.core.infrastructure.foundation.exceptions import (  # type: ignore[import-not-found]
    AgentExecutionError,
    ConfigurationError,
    LearningError,
    LLMError,
)
from Jotty.core.infrastructure.utils.async_utils import (  # type: ignore[import-not-found, import]
    StatusReporter,
    safe_status,
)
from Jotty.core.intelligence.reasoning.agents.auto_agent import (
    AutoAgent,  # type: ignore[import-not-found]
)

from ..coordination.ensemble_manager import EnsembleManager
from ..coordination.mas_zero_controller import MASZeroController
from ..coordination.paradigm_executor import ParadigmExecutor
from ..execution.agent_runner import AgentRunner, AgentRunnerConfig
from ..learning.learning_delegate import LearningDelegate
from ..learning.training_daemon import TrainingDaemon
from ..routing.model_tier_router import ModelTierRouter

# Composed managers (has-a, not is-a) — replaces mixin inheritance
from ..routing.provider_manager import ProviderManager
from ..routing.swarm_router import SwarmRouter
from ._lazy import LazyComponent

# Optional feedback channel imports
try:
    from Jotty.core.intelligence.reasoning.tools.feedback_channel import (  # type: ignore[import-not-found]
        FeedbackMessage,
        FeedbackType,
    )
except ImportError:
    FeedbackMessage = None  # type: ignore
    FeedbackType = None  # type: ignore

# Optional observability imports
try:
    from Jotty.core.infrastructure.monitoring.observability import (  # type: ignore[import-not-found, import]
        get_metrics,
        get_tracer,
    )
except ImportError:
    get_metrics = None  # type: ignore
    get_tracer = None  # type: ignore

# Optional dspy import
try:
    import dspy
except ImportError:
    dspy = None  # type: ignore

logger = logging.getLogger(__name__)


# Suppress noisy LiteLLM CancelledError on asyncio loop shutdown.
# This is a known issue: LiteLLM's background LoggingWorker gets cancelled
# when asyncio.run() tears down, producing harmless but alarming tracebacks.
class _LiteLLMCancelledFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "CancelledError" not in msg and "LoggingWorker cancelled" not in msg


logging.getLogger("LiteLLM").addFilter(_LiteLLMCancelledFilter())


# =============================================================================
# LANE QUEUE — Per-session serialization with TTL cleanup
# =============================================================================

import time as _time_module


class _SessionLockManager:
    """
    Manages per-session asyncio locks with TTL-based auto-cleanup.

    Requests for the same session_id are serialized (queued), while
    different sessions run in parallel.
    """

    _TTL_SECONDS = 1800  # 30 minutes

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_used: Dict[str, float] = {}
        self._mgr_lock = asyncio.Lock()

    async def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session_id, with lazy TTL cleanup."""
        async with self._mgr_lock:
            # Cleanup expired locks (lightweight, O(n) but n is small)
            now = _time_module.time()
            expired = [
                sid
                for sid, ts in self._last_used.items()
                if now - ts > self._TTL_SECONDS
                and sid in self._locks
                and not self._locks[sid].locked()
            ]
            for sid in expired:
                del self._locks[sid]
                del self._last_used[sid]

            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            self._last_used[session_id] = now
            return self._locks[session_id]


# Skill Provider System - Lazy loaded via cache dict (no globals)
_provider_cache: Dict[str, Any] = {}


def _load_providers() -> bool:
    """
    Lazy load skill providers to avoid circular imports.

    Caches result in _provider_cache so the import cost is paid once.
    Returns True if providers are available.
    """
    if _provider_cache.get("available"):
        return True

    try:
        from Jotty.skills._infrastructure import (  # type: ignore[import]
            ProviderRegistry,
            SkillCategory,
        )
        from Jotty.skills._providers.agent_s_provider import AgentSProvider  # type: ignore[import]
        from Jotty.skills._providers.browser_use_provider import (
            BrowserUseProvider,  # type: ignore[import]
        )
        from Jotty.skills._providers.composite_provider import (  # type: ignore[import]
            AutomateWorkflowProvider,
            FullStackAgentProvider,
            ResearchAndAnalyzeProvider,
        )
        from Jotty.skills._providers.open_interpreter_provider import (  # type: ignore[import]
            OpenInterpreterProvider,
        )
        from Jotty.skills._providers.openhands_provider import (
            OpenHandsProvider,  # type: ignore[import]
        )

        _provider_cache.update(
            {
                "available": True,
                "ProviderRegistry": ProviderRegistry,
                "SkillCategory": SkillCategory,
                "BrowserUseProvider": BrowserUseProvider,
                "OpenHandsProvider": OpenHandsProvider,
                "AgentSProvider": AgentSProvider,
                "OpenInterpreterProvider": OpenInterpreterProvider,
                "ResearchAndAnalyzeProvider": ResearchAndAnalyzeProvider,
                "AutomateWorkflowProvider": AutomateWorkflowProvider,
                "FullStackAgentProvider": FullStackAgentProvider,
            }
        )
        return True

    except ImportError as e:
        logger.debug(f"Skill providers not available: {e}")
        return False


# =========================================================================
# LAZY FACTORY FUNCTIONS - Called by LazyComponent descriptors
# =========================================================================


def _create_task_board() -> "SwarmTaskBoard":
    from Jotty.core.intelligence.orchestration.state.swarm_roadmap import SwarmTaskBoard

    return SwarmTaskBoard()


def _create_planner() -> "TaskPlanner":
    from Jotty.core.intelligence.reasoning.planners.agentic_planner import TaskPlanner

    return TaskPlanner()


def _create_intent_parser(planner: "TaskPlanner") -> "IntentParser":
    from Jotty.core.intelligence.reasoning.autonomous.intent_parser import IntentParser

    return IntentParser(planner=planner)


def _create_memory(config: SwarmConfig) -> "SwarmMemory":
    from Jotty.core.intelligence.memory.cortex import SwarmMemory

    return SwarmMemory(config=config, agent_name="SwarmShared")


def _create_provider_gateway(config: SwarmConfig) -> "SwarmProviderGateway":
    from Jotty.core.intelligence.orchestration.routing.swarm_provider_gateway import (
        SwarmProviderGateway,
    )

    provider_preference = getattr(config, "provider", None)
    return SwarmProviderGateway(config=config, provider=provider_preference)


def _create_researcher(config: SwarmConfig) -> "SwarmResearcher":
    from Jotty.core.intelligence.orchestration.swarm_researcher import SwarmResearcher

    return SwarmResearcher(config=config)


def _create_installer(config: SwarmConfig) -> "SwarmInstaller":
    from Jotty.core.intelligence.orchestration.swarm_installer import SwarmInstaller

    return SwarmInstaller(config=config)


def _create_code_generator(config: SwarmConfig) -> "SwarmCodeGenerator":
    from Jotty.core.intelligence.orchestration.swarm_code_generator import SwarmCodeGenerator

    return SwarmCodeGenerator(config=config)


def _create_terminal(config: SwarmConfig) -> "SwarmTerminal":
    from Jotty.core.intelligence.orchestration.state.swarm_terminal import SwarmTerminal

    return SwarmTerminal(config=config, auto_fix=True, max_fix_attempts=3)


def _create_ui_registry() -> Any:
    from Jotty.core.capabilities.registry.agui_component_registry import get_agui_registry

    return get_agui_registry()


def _create_tool_validator() -> "ToolValidator":
    from Jotty.core.capabilities.registry.tool_validation import ToolValidator

    return ToolValidator()


def _create_tool_registry() -> Any:
    from Jotty.core.capabilities.registry.tools_registry import get_tools_registry

    return get_tools_registry()


def _create_profiler(config: SwarmConfig) -> Optional["PerformanceProfiler"]:
    enable = getattr(config, "enable_profiling", False)
    if not enable:
        return None
    from Jotty.core.infrastructure.monitoring.metrics.profiler import PerformanceProfiler

    return PerformanceProfiler(enable_cprofile=True)


def _create_state_manager(sm: "Orchestrator") -> "SwarmStateManager":
    from Jotty.core.intelligence.orchestration.state.swarm_state_manager import SwarmStateManager

    agents_dict = {a.name: a for a in sm.agents}  # type: ignore[union-attr]
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
    from Jotty.core.infrastructure.persistence.shared_context import SharedContext

    return SharedContext()


def _create_io_manager() -> "IOManager":
    from Jotty.core.infrastructure.data.io_manager import IOManager

    return IOManager()


def _create_data_registry() -> "DataRegistry":
    from Jotty.core.infrastructure.data.data_registry import DataRegistry

    return DataRegistry()


def _create_context_guard() -> "LLMContextManager":
    from Jotty.core.infrastructure.context.context_manager import (
        SmartContextManager,  # type: ignore[import-not-found, import]
    )

    return SmartContextManager()


def _create_learning_pipeline(config: SwarmConfig) -> "SwarmLearningPipeline":
    from Jotty.core.intelligence.orchestration.learning.swarm_learning_pipeline import (
        SwarmLearningPipeline,
    )

    return SwarmLearningPipeline(config)


def _create_mas_learning(sm: "Orchestrator") -> "MASLearning":
    from Jotty.core.intelligence.orchestration.learning.mas_learning import MASLearning

    workspace_path = getattr(sm.config, "base_path", None)
    return MASLearning(
        config=sm.config,
        workspace_path=workspace_path,
        swarm_intelligence=sm.swarm_intelligence,
        transfer_learning=sm.transfer_learning,
    )


class AgentFactory:
    """Creates and manages AgentRunners and LOTUS optimization."""

    def __init__(self, manager: "Orchestrator") -> None:
        self._manager = manager

    def ensure_runners(self) -> None:
        """Build AgentRunners on first run() call (not in __init__)."""
        sm = self._manager
        if sm._runners_built:
            return

        for agent_config in sm.agents:
            if agent_config.name in sm.runners:  # type: ignore[union-attr]
                continue

            runner_config = AgentRunnerConfig(
                architect_prompts=sm.architect_prompts,
                auditor_prompts=sm.auditor_prompts,
                config=sm.config,
                agent_name=agent_config.name,  # type: ignore[union-attr]
                enable_learning=True,
                enable_memory=True,
            )

            # Propagate SwarmConfig to agent so lazy-loaded components
            # (memory, context, etc.) use the same config as Orchestrator.
            agent = agent_config.agent  # type: ignore[union-attr]
            if hasattr(agent, "set_jotty_config"):
                agent.set_jotty_config(sm.config)

            runner = AgentRunner(
                agent=agent,
                config=runner_config,
                task_planner=sm.swarm_planner,
                task_board=sm.swarm_task_board,
                swarm_memory=sm.swarm_memory,
                swarm_state_manager=sm.swarm_state_manager,
                learning_manager=sm.learning,
                transfer_learning=sm.transfer_learning,
                swarm_terminal=sm.swarm_terminal,
                swarm_intelligence=sm.swarm_intelligence,
            )
            sm.runners[agent_config.name] = runner  # type: ignore[union-attr]

        # Register agents with Axon for inter-agent communication
        self.register_agents_with_axon()

        # LOTUS optimization
        if sm.enable_lotus:
            self.init_lotus_optimization()

        # Auto-load previous learnings + integrate MAS terminal in background.
        # Uses asyncio task instead of raw thread so run() can await readiness
        # via sm._learning_ready event before executing with partial state.
        async def _bg_learning_init() -> Any:
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

        # Inject learning components into all runners
        try:
            if hasattr(sm, "learning") and sm.learning:
                components = sm.learning.learning_components
                for runner in sm.runners.values():
                    runner.inject_learning(components)
                logger.info("Learning components injected into all runners")
        except Exception as e:
            logger.debug(f"Learning injection skipped: {e}")

    def register_agents_with_axon(self) -> None:
        """Register all agents with SmartAgentSlack for inter-agent messaging."""
        sm = self._manager

        def _make_slack_callback(target_actor_name: str) -> Any:
            def _callback(message: Any) -> Any:
                try:
                    fb = FeedbackMessage(
                        source_actor=message.from_agent,
                        target_actor=target_actor_name,
                        feedback_type=FeedbackType.RESPONSE,
                        content=str(message.data),
                        context={
                            "format": getattr(message, "format", "unknown"),
                            "size_bytes": getattr(message, "size_bytes", None),
                            "metadata": getattr(message, "metadata", {}) or {},
                            "timestamp": getattr(message, "timestamp", None),
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
                agent_obj = agent_config.agent  # type: ignore[union-attr]
                signature_obj = getattr(agent_obj, "signature", None)
                sm.agent_slack.register_agent(
                    agent_name=agent_config.name,  # type: ignore[union-attr]
                    signature=signature_obj if hasattr(signature_obj, "input_fields") else None,
                    callback=_make_slack_callback(agent_config.name),  # type: ignore[union-attr]
                    max_context=getattr(sm.config, "max_context_tokens", 16000),
                )
            except Exception as e:
                logger.warning(f"Could not register {agent_config.name} with SmartAgentSlack: {e}")  # type: ignore[union-attr]

    def create_zero_config_agents(
        self, task: str, status_callback: Optional[Callable] = None
    ) -> List[AgentConfig]:
        """Delegate to ZeroConfigAgentFactory."""
        sm = self._manager
        if not hasattr(sm, "_zero_config_factory") or sm._zero_config_factory is None:
            from Jotty.core.intelligence.orchestration.zero_config_factory import (
                ZeroConfigAgentFactory,
            )

            sm._zero_config_factory = ZeroConfigAgentFactory()
        return sm._zero_config_factory.create_agents(task, status_callback)  # type: ignore[no-any-return]

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
            from Jotty.core.infrastructure.integration.lotus.integration import (  # type: ignore[import]
                LotusEnhancement,
                _enhance_agent_runner,
            )

            # Create LOTUS enhancement with default config
            sm.lotus = LotusEnhancement(
                enable_cascade=True,
                enable_cache=True,
                enable_adaptive_validation=True,
            )
            sm.lotus_optimizer = sm.lotus.lotus_optimizer  # type: ignore[attr-defined]

            # Enhance all agent runners with adaptive validation
            for name, runner in sm.runners.items():
                _enhance_agent_runner(runner, sm.lotus)

                # Pre-warm the adaptive validator with initial trust
                # This allows validation skipping from the start
                # (simulates 15 successful validations per agent)
                for _ in range(15):
                    sm.lotus.adaptive_validator.record_result(name, "architect", success=True)  # type: ignore[attr-defined]
                    sm.lotus.adaptive_validator.record_result(name, "auditor", success=True)  # type: ignore[attr-defined]
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
            return sm.lotus.get_stats()  # type: ignore[unreachable]
        return {}

    def get_lotus_savings(self) -> Dict[str, float]:
        """Get estimated cost savings from LOTUS optimization."""
        sm = self._manager
        if sm.lotus:
            return sm.lotus.get_savings()  # type: ignore[unreachable]
        return {}


class ExecutionEngine:
    """Executes tasks via single/multi-agent paradigms."""

    def __init__(self, manager: "Orchestrator") -> None:
        self._manager = manager
        self._paradigms = ParadigmExecutor(manager)

    async def run(self, goal: str, **kwargs: Any) -> EpisodeResult:
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
        _tracer = None
        if get_tracer:
            _tracer = get_tracer()
            _tracer.new_trace(metadata={"goal": goal[:200], "mode": sm.mode})

        # Lazy init: Build runners on first run
        sm._ensure_runners()

        # Wait for background learning init (optional/short for latency-sensitive deployments)
        # learning_wait_timeout_seconds <= 0 skips wait; default 5.0s.
        learning_wait_timeout = getattr(sm.config, "learning_wait_timeout_seconds", 5.0)
        if learning_wait_timeout > 0:
            try:
                await asyncio.wait_for(
                    sm._learning_ready.wait(), timeout=float(learning_wait_timeout)
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Learning init timed out after %.1fs, proceeding without full learning state",
                    learning_wait_timeout,
                )
        else:
            logger.debug("Learning wait skipped (learning_wait_timeout_seconds <= 0)")

        # MAS-ZERO: Reset per-problem experience library
        sm._reset_experience()
        sm._efficiency_stats = {}  # Reset per-run
        sm._execution._efficiency_stats = {}  # Reset orchestrator stats too

        # Extract ExecutionContext if provided by ModeRouter
        exec_context = kwargs.pop("context", None)

        # Extract special kwargs (pop ALL so they don't leak into **kwargs downstream)
        skip_autonomous_setup = kwargs.pop("skip_autonomous_setup", False)
        skip_validation = kwargs.pop("skip_validation", None)  # Explicit override
        status_callback = kwargs.pop("status_callback", None)
        ensemble = kwargs.pop("ensemble", None)  # None = auto-detect, True/False = explicit
        ensemble_strategy = kwargs.pop("ensemble_strategy", "multi_perspective")

        # If ExecutionContext provided, extract callbacks from it
        if exec_context is not None:
            if status_callback is None and hasattr(exec_context, "status_callback"):
                status_callback = exec_context.status_callback
            # Emit start event
            if hasattr(exec_context, "emit_event"):
                from Jotty.core.infrastructure.foundation.types.sdk_types import (
                    SDKEventType,  # type: ignore[import-not-found, import]
                )

                exec_context.emit_event(
                    SDKEventType.PLANNING,
                    {
                        "goal": goal,
                        "mode": sm.mode,
                        "agents": len(sm.agents),
                    },
                )

        # Auto-detect ensemble for certain task types (if not explicitly set)
        # Optima-inspired: adaptive sizing returns (should_ensemble, max_perspectives)
        max_perspectives = 4  # default
        if ensemble is None:
            ensemble, max_perspectives = sm._should_auto_ensemble(goal)
            if ensemble:
                logger.info(
                    f"Auto-enabled ensemble: {max_perspectives} perspectives (use ensemble=False to override)"
                )

        # Ensure DSPy LM is configured (critical for all agent operations)
        if dspy and (not hasattr(dspy.settings, "lm") or dspy.settings.lm is None):
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
        from Jotty.core.intelligence.orchestration.execution.validation_gate import (
            ValidationMode,
            get_validation_gate,
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
                agent_name=sm.agents[0].name if sm.agents else "auto",  # type: ignore[union-attr]
                force_mode=_force_mode,
            )

        if _gate_decision and _gate_decision.mode == ValidationMode.DIRECT and sm.mode == "single":
            _status(
                "Fast path", f"DIRECT mode — bypassing agent pipeline ({_gate_decision.reason})"
            )

            # Model tier routing: use cheapest LM for DIRECT tasks
            try:
                if sm._model_tier_router is None:
                    sm._model_tier_router = ModelTierRouter()
                tier_decision = sm._model_tier_router.get_model_for_mode(ValidationMode.DIRECT)
                tier_lm = sm._model_tier_router.get_lm_for_mode(ValidationMode.DIRECT)  # type: ignore[func-returns-value]
                if tier_lm:
                    lm = tier_lm
                    _status(
                        "Model tier",
                        f"{tier_decision.tier.value} ({tier_decision.model}) — cost ratio {tier_decision.estimated_cost_ratio:.1f}x",
                    )
                else:
                    lm = dspy.settings.lm if dspy else None
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
                        is_rate_limit = (
                            "429" in err_str
                            or "RateLimit" in err_str
                            or "rate limit" in err_str.lower()
                        )
                        if is_rate_limit:
                            logger.info(
                                "Fast path rate limited — falling through to full pipeline (no retry)"
                            )
                            _rate_limited = True
                            break
                        continue
                    except Exception as e:
                        err_str = str(e)
                        is_rate_limit = (
                            "429" in err_str
                            or "RateLimit" in err_str
                            or "rate limit" in err_str.lower()
                        )
                        if is_rate_limit:
                            logger.info(
                                "Fast path rate limited — falling through to full pipeline (no retry)"
                            )
                            _rate_limited = True
                            break
                        continue

                if response is None:
                    raise AgentExecutionError(
                        "All LM calling conventions failed"
                        + (" (rate limited)" if _rate_limited else "")
                    )

                if isinstance(response, list):
                    response = response[0] if response else ""
                elif hasattr(response, "text"):
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
                    trajectory=[{"step": 1, "action": "direct_llm", "output": response[:200]}],
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
                logger.info(
                    f"Fast path failed (recoverable: {type(e).__name__}): {e}, falling back to full pipeline"
                )
                # Fall through to normal pipeline
            except Exception as e:
                logger.info(
                    f"Fast path failed (unexpected: {type(e).__name__}): {e}, falling back to full pipeline"
                )
                # Fall through to normal pipeline

        # Store gate decision for downstream use — pass to AgentRunner to avoid redundant decide()
        kwargs["gate_decision"] = _gate_decision

        # ── MODEL TIER ROUTING: Select LM quality based on task complexity ──
        # DIRECT → cheap (Haiku), AUDIT_ONLY → balanced (Sonnet), FULL → quality (Opus/Sonnet)
        if _gate_decision and _gate_decision.mode != ValidationMode.DIRECT:
            try:
                if sm._model_tier_router is None:
                    sm._model_tier_router = ModelTierRouter()
                tier_decision = sm._model_tier_router.get_model_for_mode(_gate_decision.mode)
                tier_lm = sm._model_tier_router.get_lm_for_mode(_gate_decision.mode)  # type: ignore[func-returns-value]
                if tier_lm and dspy:
                    dspy.configure(lm=tier_lm)
                    _status("Model tier", f"{tier_decision.tier.value} ({tier_decision.model})")
            except (ConfigurationError, LLMError) as e:
                logger.debug(f"Model tier routing skipped (recoverable): {e}")
            except Exception as e:
                logger.debug(f"Model tier routing skipped (unexpected): {e}", exc_info=True)

        # Zero-config: LLM decides single vs multi-agent at RUN TIME (when goal is available)
        # SKIP when gate already classified as DIRECT — it's a simple task, no need
        # to burn an LLM call deciding single vs multi-agent.
        _gate_is_direct = _gate_decision and _gate_decision.mode == ValidationMode.DIRECT
        if sm.enable_zero_config and sm.mode == "single" and not _gate_is_direct:
            _status("Analyzing task", "deciding single vs multi-agent")
            new_agents = sm._create_zero_config_agents(goal, status_callback)
            if len(new_agents) > 1:
                # LLM detected parallel sub-goals - upgrade to multi-agent
                sm.agents = new_agents
                sm.mode = "multi"
                logger.info(
                    f" Zero-config: Upgraded to {len(sm.agents)} agents for parallel execution"
                )

                # Create runners for new agents
                for agent_config in sm.agents:
                    if agent_config.name not in sm.runners:
                        runner_config = AgentRunnerConfig(
                            architect_prompts=sm.architect_prompts,
                            auditor_prompts=sm.auditor_prompts,
                            config=sm.config,
                            agent_name=agent_config.name,
                            enable_learning=True,
                            enable_memory=True,
                        )
                        runner = AgentRunner(
                            agent=agent_config.agent,
                            config=runner_config,
                            task_planner=sm.swarm_planner,
                            task_board=sm.swarm_task_board,
                            swarm_memory=sm.swarm_memory,
                            swarm_state_manager=sm.swarm_state_manager,
                            learning_manager=sm.learning,
                            transfer_learning=sm.transfer_learning,
                            swarm_terminal=sm.swarm_terminal,  # Shared intelligent terminal
                        )
                        sm.runners[agent_config.name] = runner

        agent_info = (
            f"{len(sm.agents)} AutoAgent(s)" if len(sm.agents) > 1 else "AutoAgent (zero-config)"
        )
        _status("Starting", agent_info)

        # Profile execution if enabled
        if sm.swarm_profiler:
            profile_context = sm.swarm_profiler.profile(
                "Orchestrator.run", metadata={"goal": goal, "mode": sm.mode}
            )
            profile_context.__enter__()
        else:
            profile_context = None

        try:
            # Store goal in shared context for state management
            if sm.shared_context:
                sm.shared_context.set("goal", goal)
                sm.shared_context.set("query", goal)

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
                sm.swarm_state_manager.record_swarm_step(
                    {
                        "step": "goal_received",
                        "goal": goal,
                        "mode": sm.mode,
                        "agent_count": len(sm.agents),
                    }
                )

            # Ensemble mode: Multi-perspective analysis
            # For multi-agent: ensemble happens per-agent (each agent has different sub-goal)
            # For single-agent: ensemble happens at swarm level
            ensemble_result = None
            if ensemble and sm.mode == "single":
                _status(
                    "Ensembling", f"strategy={ensemble_strategy}, perspectives={max_perspectives}"
                )
                _ens_start = _time.time()
                ensemble_result = await sm._execute_ensemble(
                    goal,
                    strategy=ensemble_strategy,
                    status_callback=status_callback,
                    max_perspectives=max_perspectives,
                )
                sm._efficiency_stats["ensemble_time"] = _time.time() - _ens_start
                if ensemble_result.get("success"):
                    # Pass ensemble context via kwargs (not embedded in goal string)
                    # to avoid polluting search queries and downstream skill params.
                    kwargs["ensemble_context"] = ensemble_result

                    # Show quality scores if available
                    quality_scores = ensemble_result.get("quality_scores", {})
                    if quality_scores:
                        avg_quality = sum(quality_scores.values()) / len(quality_scores)
                        scores_str = ", ".join(f"{k}:{v:.0%}" for k, v in quality_scores.items())
                        _status("Ensemble quality", f"avg={avg_quality:.0%} ({scores_str})")
                    else:
                        _status(
                            "Ensemble complete",
                            f"{len(ensemble_result.get('perspectives_used', []))} perspectives",
                        )
            elif ensemble and sm.mode == "multi":
                # Multi-agent mode: SKIP per-agent ensemble (too expensive)
                # With N agents × 4 perspectives = 4N LLM calls - massive overkill
                # Each agent already has a specific sub-goal, no need for multi-perspective
                _status(
                    "Ensemble mode", "DISABLED for multi-agent (agents have specific sub-goals)"
                )
                ensemble = False  # Disable for agents

            # Keep gate_decision in kwargs so AgentRunner can reuse it (no second decide() call)

            # Single-agent mode: Simple execution
            if sm.mode == "single":
                agent_name = sm.agents[0].name if sm.agents else "auto"  # type: ignore[union-attr]
                _status("Executing", f"agent '{agent_name}' with skill orchestration")
                # Architect → Actor → Auditor pipeline (now fast with max_eval_iters=2)
                # skip_validation: explicit kwarg wins, else derive from skip_autonomous_setup
                skip_val = skip_validation if skip_validation is not None else skip_autonomous_setup
                # Forward ensemble flag so AutoAgent doesn't auto-detect independently
                if ensemble is not None:
                    kwargs["ensemble"] = ensemble
                # Avoid duplicate kwarg: ensemble_context may already be in kwargs (line 748)
                kwargs.pop("ensemble_context", None)
                result = await self._execute_single_agent(
                    goal,
                    skip_validation=skip_val,
                    status_callback=status_callback,
                    ensemble_context=ensemble_result if ensemble else None,
                    **kwargs,
                )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = sm._efficiency_stats.get("ensemble_time", 0)
                exec_t = getattr(result, "execution_time", total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                sm._efficiency_stats.update(
                    {"total_time": total_elapsed, "overhead_pct": overhead_pct}
                )
                _status(
                    "Complete",
                    f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)",
                )

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
                        skip_validation=(
                            skip_validation
                            if skip_validation is not None
                            else skip_autonomous_setup
                        ),
                        status_callback=status_callback,
                        **kwargs,
                    )
                else:
                    paradigm = kwargs.pop("discussion_paradigm", "auto")
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
                        **kwargs,
                    )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = sm._efficiency_stats.get("ensemble_time", 0)
                exec_t = getattr(result, "execution_time", total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                sm._efficiency_stats.update(
                    {"total_time": total_elapsed, "overhead_pct": overhead_pct}
                )
                _status(
                    "Complete",
                    f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)",
                )

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

    async def _execute_single_agent(self, goal: str, **kwargs: Any) -> EpisodeResult:
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
        kwargs.pop("ensemble_context", None)
        status_callback = kwargs.pop("status_callback", None)

        _status = StatusReporter(status_callback)

        # ── Pre-execution strategy: leverage learned intelligence ──
        # Use stigmergy + MAS learning to inform agent selection
        try:
            if hasattr(sm, "learning") and sm.learning:
                components = sm.learning.learning_components
                stig = components.get("stigmergy")
                mas = components.get("mas_learning")

                # Stigmergy: recommend approach from historical signals
                if stig:
                    approach = stig.recommend_approach(goal[:100])
                    if approach:
                        kwargs.setdefault("learning_context", "")
                        kwargs[
                            "learning_context"
                        ] += f"\n[Stigmergy] Recommended approach: {approach}"

                # MAS Learning: get execution strategy from history
                if mas and hasattr(mas, "get_execution_strategy"):
                    strategy = mas.get_execution_strategy(
                        goal, [a.name if hasattr(a, "name") else str(a) for a in sm.agents]
                    )
                    if (
                        strategy
                        and hasattr(strategy, "recommended_agents")
                        and strategy.recommended_agents
                    ):
                        logger.info(f"MAS strategy recommends: {strategy.recommended_agents[:3]}")
        except Exception as e:
            logger.debug(f"Pre-execution strategy skipped: {e}")

        # ── Router-guided agent selection (closes RL loop for single-agent) ──
        # When multiple agents are available, ask the router instead of
        # blindly taking agents[0].  Router uses SwarmIntelligence (RL +
        # stigmergy + TRAS) when learning data exists.
        agent_config = sm.agents[0]
        if len(sm.agents) > 1:
            try:
                route = sm._router.select_agent(goal)
                picked = route.get("agent")
                if picked and route.get("method") != "fallback":
                    for ac in sm.agents:
                        if getattr(ac, "name", None) == picked:
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

        runner = sm.runners[agent_config.name]  # type: ignore[union-attr]

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
            for good in approach_rec.get("use", []):
                tools_str = ", ".join(good["tools"][:5]) if good.get("tools") else "N/A"
                learned_hints.append(
                    f"[Learned] Successful approach for '{task_type}': "
                    f"{good['approach'][:120]} (tools: {tools_str}, "
                    f"quality: {good.get('quality', 0):.0%})"
                )
            for bad in approach_rec.get("avoid", []):
                tools_str = ", ".join(bad["tools"][:5]) if bad.get("tools") else "N/A"
                learned_hints.append(
                    f"[Learned] Failed approach for '{task_type}': "
                    f"{bad['approach'][:120]} — avoid this"
                )
            # Fallback: if no approach data yet, use old agent-routing signals
            if not approach_rec.get("use") and not approach_rec.get("avoid"):
                warnings = lp.get_stigmergy_warnings(task_type)
                if warnings:
                    warn_msgs = [
                        (
                            getattr(w, "content", {}).get("goal", "unknown")
                            if isinstance(getattr(w, "content", None), dict)
                            else str(w)
                        )
                        for w in warnings[:3]
                    ]
                    learned_hints.append(
                        f"[Learned] Previous failures on '{task_type}': " f"{'; '.join(warn_msgs)}"
                    )

            # 2. Effectiveness: are we improving or degrading on this type?
            #    Reads from BOTH in-memory EffectivenessTracker (fast, current session)
            #    AND LearningService (SQLite, cross-session history).
            eff_report = lp.effectiveness.improvement_report()
            task_eff = eff_report.get(task_type)
            if task_eff and task_eff.get("trend") is not None:
                trend = task_eff["trend"]
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

            # Cross-session effectiveness from LearningService (SQLite)
            try:
                from Jotty.core.intelligence.learning.learning_service import LearningService

                ls = LearningService.get_instance()
                ls_report = ls.improvement_report(domain=task_type)
                if ls_report and ls_report.get("recent_success_rate") is not None:
                    ls_trend = ls_report.get("trend", 0)
                    if ls_trend < -0.15 and not task_eff:
                        learned_hints.append(
                            f"[Learned/History] Cross-session performance declining "
                            f"on '{task_type}'. Try alternative approaches."
                        )
            except Exception:
                pass

            # 3. Transfer learning: relevant past learnings
            learnings = lp.transfer_learning.get_relevant_learnings(goal, top_k=2)
            for exp in learnings.get("similar_experiences", []):
                query_text = exp.get("query", "") if isinstance(exp, dict) else str(exp)
                success = exp.get("success", True) if isinstance(exp, dict) else True
                if success:
                    learned_hints.append(f"[Learned] Succeeded on similar task: {query_text[:100]}")
                else:
                    err = (exp.get("error", "")[:80]) if isinstance(exp, dict) else ""
                    learned_hints.append(
                        f"[Learned] Failed on similar task: {query_text[:80]}"
                        + (f" (error: {err})" if err else "")
                    )
            if learnings.get("task_pattern"):
                learned_hints.append(f"[Learned] Pattern: {str(learnings['task_pattern'])[:150]}")
            for err_pat in learnings.get("error_patterns", [])[:2]:
                learned_hints.append(f"[Learned] Common error: {str(err_pat)[:120]}")

            # Inject learned context into kwargs for the agent
            if learned_hints:
                existing_ctx = kwargs.get("learning_context", "")
                kwargs["learning_context"] = (
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
            logger.warning(
                f"Pre-execution intelligence read failed (unexpected): {e}", exc_info=True
            )

        # Pass status_callback back for downstream
        if status_callback:
            kwargs["status_callback"] = status_callback

        # Standard single-agent execution
        import time as _t

        _exec_start = _t.time()

        # Observability: trace agent execution
        _tracer = get_tracer() if get_tracer else None
        _metrics = get_metrics() if get_metrics else None

        if _tracer:
            with _tracer.span("agent_execute", agent=agent_config.name, mode="single") as _span:  # type: ignore[union-attr]
                result = await runner.run(goal=goal, **kwargs)
                _exec_elapsed = _t.time() - _exec_start
                _span.set_attribute("success", result.success)
                _span.set_attribute("execution_time_s", round(_exec_elapsed, 2))
                if hasattr(result, "execution_time"):
                    _span.set_attribute("inner_time_s", round(result.execution_time, 2))
        else:
            result = await runner.run(goal=goal, **kwargs)
            _exec_elapsed = _t.time() - _exec_start

        # Observability: record metrics
        if _metrics:
            _metrics.record_execution(
                agent_name=agent_config.name,  # type: ignore[union-attr]
                task_type="single_agent",
                duration_s=_exec_elapsed,
                success=result.success,
            )

        # Learn from execution (DRY: reuse workflow learner)
        if result.success:
            sm._learn_from_result(result, agent_config, goal=goal)

        # Post-episode learning + auto-save (fire-and-forget background)
        sm._schedule_background_learning(result, goal)

        return result

    async def _execute_multi_agent(self, goal: str, **kwargs: Any) -> EpisodeResult:
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
        from Jotty.core.intelligence.learning.predictive_marl import (
            ActualTrajectory,  # type: ignore[import]
        )
        from Jotty.core.intelligence.orchestration.state.swarm_roadmap import TaskStatus

        # Extract callbacks and ensemble params before passing to runners
        kwargs.pop("ensemble_context", None)
        status_callback = kwargs.pop("status_callback", None)
        ensemble = kwargs.pop("ensemble", False)
        ensemble_strategy = kwargs.pop("ensemble_strategy", "multi_perspective")
        discussion_paradigm = kwargs.pop("discussion_paradigm", "auto")

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
                _intelligence_applied = bool(getattr(sm, "learning", None))
                if _intelligence_applied and sm.agents:
                    top = sm.agents[0].name if hasattr(sm.agents[0], "name") else "?"
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
            _im_task_type = "_global"
        sm._last_task_type = _im_task_type

        for _tt in (_im_task_type, "_global") if _im_task_type != "_global" else ("_global",):
            if _tt not in sm._intelligence_metrics:
                sm._intelligence_metrics[_tt] = {
                    "guided_runs": 0,
                    "guided_successes": 0,
                    "unguided_runs": 0,
                    "unguided_successes": 0,
                }
            if _intelligence_applied:
                sm._intelligence_metrics[_tt]["guided_runs"] += 1
            else:
                sm._intelligence_metrics[_tt]["unguided_runs"] += 1

        # Auto paradigm selection: use learning data to pick the best paradigm
        if discussion_paradigm == "auto":
            try:
                _task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
                discussion_paradigm = sm.learning.recommend_paradigm(_task_type)
                logger.info(
                    f" Auto paradigm: selected '{discussion_paradigm}' "
                    f"for task_type='{_task_type}'"
                )
            except LearningError as e:
                logger.debug(f"Auto paradigm selection failed (learning): {e}")
                discussion_paradigm = "fanout"
            except Exception as e:
                logger.debug(f"Auto paradigm selection failed (unexpected): {e}", exc_info=True)
                discussion_paradigm = "fanout"

        # Track which paradigm is used (for _post_episode_learning)
        sm._last_paradigm = discussion_paradigm

        # MALLM-inspired: Dispatch to alternative paradigms before fan-out
        if discussion_paradigm == "relay":
            # Wire coordination: initiate handoff between relay agents
            if sm.swarm_intelligence:
                try:
                    available = [a.name for a in sm.agents]  # type: ignore[union-attr]
                    if len(available) >= 2:
                        sm.swarm_intelligence.initiate_handoff(
                            from_agent=available[0],
                            to_agent=available[1],
                            task=goal,
                            context={"paradigm": "relay", "agents": available},
                        )
                except Exception as e:
                    logger.debug(f"Relay handoff coordination skipped: {e}")
            return await self._paradigm_relay(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == "debate":
            return await self._paradigm_debate(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == "refinement":
            return await self._paradigm_refinement(goal, status_callback=status_callback, **kwargs)
        # else: 'fanout' — fall through to existing parallel execution below

        # Wire coordination: form coalition for fanout tasks (agents collaborate)
        if sm.swarm_intelligence and len(sm.agents) >= 2:
            try:
                available = [a.name for a in sm.agents]  # type: ignore[union-attr]
                _task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
                sm.swarm_intelligence.form_coalition(
                    task_type=_task_type,
                    min_agents=min(2, len(available)),
                    available_agents=available,
                )
                logger.info(f" Coalition formed for '{_task_type}' with {len(available)} agents")
            except LearningError as e:
                logger.debug(f"Coalition formation skipped (learning): {e}")
            except Exception as e:
                logger.debug(f"Coalition formation skipped (unexpected): {e}", exc_info=True)

        # Status update at method start
        safe_status(
            status_callback, "Multi-agent exec", f"starting {len(sm.agents)} parallel agents"
        )

        max_attempts = getattr(sm.config, "max_task_attempts", 2)

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
            deps = getattr(agent_config, "depends_on", []) or []

            # Use agent's sub-goal (from capabilities) as task description
            sub_goal = (
                agent_config.capabilities[0]
                if agent_config.capabilities
                else f"{goal} (agent: {agent_config.name})"
            )

            sm.swarm_task_board.add_task(
                task_id=task_id,
                description=sub_goal,
                actor=agent_config.name,
                depends_on=deps,  # Empty for parallel execution
            )
            logger.info(
                f" Added task {task_id} for {agent_config.name}: {sub_goal[:50]}... (parallel: {len(deps)==0})"
            )

        all_results = {}  # agent_name -> EpisodeResult
        attempt_counts: Dict[str, Any] = {}  # task_id -> attempts
        _max_iters = getattr(sm.config, "max_episode_iterations", 12)
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
                    status_callback(
                        "Running batch", f"{len(batch)} agents: {', '.join(agent_names[:5])}"
                    )
                    # Show each agent's task for clarity
                    for task in batch:
                        agent_cfg = next((a for a in sm.agents if a.name == task.actor), None)  # type: ignore[union-attr]
                        sub_goal = (
                            agent_cfg.capabilities[0]  # type: ignore[union-attr]
                            if agent_cfg and agent_cfg.capabilities  # type: ignore[union-attr]
                            else task.description[:50]
                        )
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
                            proposed_action={"task": task.description},
                            other_agents=[a.name for a in sm.agents if a.name != task.actor],  # type: ignore[union-attr]
                            goal=goal,
                        )
                        predictions[task.actor] = prediction
                    except Exception as e:
                        logger.debug(f"Trajectory prediction skipped for {task.actor}: {e}")

            # Execute batch concurrently (status_callback already extracted at method start)
            # AIOS-inspired: Semaphore limits how many agents call LLM simultaneously.
            # Without this, N agents × (architect + agent + auditor) = 3N concurrent API calls.
            async def _run_task(task: Any) -> Any:
                # Check if we'll need to wait for a slot
                if sm.agent_semaphore._value == 0:
                    sm._scheduling_stats["total_waited"] += 1
                    safe_status(status_callback, f"Agent {task.actor}", "waiting for LLM slot...")

                async with sm.agent_semaphore:
                    # Track concurrency stats
                    sm._scheduling_stats["total_scheduled"] += 1
                    sm._scheduling_stats["current_concurrent"] += 1
                    if (
                        sm._scheduling_stats["current_concurrent"]
                        > sm._scheduling_stats["peak_concurrent"]
                    ):
                        sm._scheduling_stats["peak_concurrent"] = sm._scheduling_stats[
                            "current_concurrent"
                        ]

                    try:
                        # Show which agent is executing
                        agent_cfg = next((a for a in sm.agents if a.name == task.actor), None)  # type: ignore[union-attr]
                        sub_goal = (
                            agent_cfg.capabilities[0]  # type: ignore[union-attr]
                            if agent_cfg and agent_cfg.capabilities  # type: ignore[union-attr]
                            else task.description[:60]
                        )

                        safe_status(status_callback, f"Agent {task.actor}", f"starting: {sub_goal}")

                        # Create agent-specific status callback that prefixes with agent name
                        _agent_reporter = StatusReporter(status_callback).with_prefix(
                            f"  [{task.actor}]"
                        )
                        agent_status_callback = _agent_reporter

                        runner = sm.runners[task.actor]
                        # Pass the agent-specific callback and ensemble params
                        task_kwargs = dict(kwargs)
                        task_kwargs["status_callback"] = agent_status_callback
                        # Forward ensemble flag explicitly so AutoAgent doesn't auto-detect
                        task_kwargs["ensemble"] = ensemble
                        if ensemble:
                            task_kwargs["ensemble_strategy"] = ensemble_strategy
                        # Forward the swarm-level gate decision to skip redundant
                        # per-agent architect/auditor if the swarm already decided
                        task_kwargs.pop(
                            "gate_decision", None
                        )  # single-agent only; don't pass to multi-agent tasks

                        # MULTI-AGENT OPTIMIZATION: Sub-agents with system_prompt
                        # are specialized for analysis/synthesis — they don't need
                        # the full skill pipeline (saves ~100s per agent).
                        # Use direct_llm=True to bypass skill discovery/selection/planning.
                        if agent_cfg and hasattr(agent_cfg, "agent") and agent_cfg.agent:
                            _agent_obj = agent_cfg.agent
                            _has_system_prompt = (
                                hasattr(_agent_obj, "config")
                                and hasattr(_agent_obj.config, "system_prompt")
                                and _agent_obj.config.system_prompt
                            )
                            if _has_system_prompt:
                                task_kwargs["direct_llm"] = True

                        return task, await runner.run(goal=task.description, **task_kwargs)
                    finally:
                        sm._scheduling_stats["current_concurrent"] -= 1

            # Per-agent timeout: prevent any single agent from blocking the entire swarm.
            # Reads from SwarmConfig.actor_timeout (default 900s).
            PER_AGENT_TIMEOUT = getattr(sm.config, "actor_timeout", 900.0)

            async def _run_task_with_timeout(task: Any) -> Any:
                try:
                    return await asyncio.wait_for(_run_task(task), timeout=PER_AGENT_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {task.actor} timed out after {PER_AGENT_TIMEOUT}s")
                    safe_status(
                        status_callback,
                        f"Agent {task.actor}",
                        f"TIMEOUT ({PER_AGENT_TIMEOUT:.0f}s)",
                    )
                    return task, EpisodeResult(
                        output=f"Agent {task.actor} timed out after {PER_AGENT_TIMEOUT:.0f}s",
                        success=False,
                        trajectory=[{"step": 0, "action": "timeout"}],
                        tagged_outputs=[],
                        episode=sm.episode_count,
                        execution_time=PER_AGENT_TIMEOUT,
                        architect_results=[],
                        auditor_results=[],
                        agent_contributions={},
                    )

            coro_results = await asyncio.gather(
                *[_run_task_with_timeout(t) for t in batch], return_exceptions=True
            )

            # Process results
            for coro_result in coro_results:
                if isinstance(coro_result, Exception):
                    logger.error(f"Task execution exception: {coro_result}")
                    safe_status(status_callback, "Agent error", str(coro_result)[:60])
                    continue

                task, result = coro_result  # type: ignore[misc]
                attempt_counts[task.task_id] = attempt_counts.get(task.task_id, 0) + 1

                # Show agent completion status
                status_icon = "" if result.success else ""
                safe_status(
                    status_callback,
                    f"{status_icon} Agent {task.actor}",
                    "completed" if result.success else "failed",
                )
                reward = 1.0 if result.success else -0.5

                # Post-execution: divergence learning
                if sm.trajectory_predictor and task.actor in predictions:
                    try:
                        prediction = predictions[task.actor]
                        actual = ActualTrajectory(
                            steps=result.trajectory or [], actual_reward=reward
                        )
                        divergence = sm.trajectory_predictor.compute_divergence(prediction, actual)
                        sm.divergence_memory.store(divergence)
                        sm.trajectory_predictor.update_from_divergence(divergence)

                        # Use divergence as TD error weight for Q-update
                        divergence_penalty = 1.0 - min(1.0, divergence.total_divergence())
                        adjusted_reward = reward * divergence_penalty
                        state = {"query": goal, "agent": task.actor}
                        action = {"actor": task.actor, "task": task.description[:100]}
                        from Jotty.core.intelligence.learning.learning_service import (
                            LearningService,
                        )

                        svc = LearningService.get_instance()
                        svc.record_outcome(
                            unit_name=task.actor,
                            state=f"divergence:{goal[:80]}",
                            action=f"actor:{task.actor}|task:{task.description[:60]}",
                            reward=adjusted_reward,
                            domain=getattr(sm.config, "domain", "general"),
                        )
                    except LearningError as e:
                        logger.debug(
                            f"Divergence learning skipped for {task.actor} (learning): {e}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Divergence learning skipped for {task.actor} (unexpected): {e}",
                            exc_info=True,
                        )

                if result.success:
                    sm.swarm_task_board.complete_task(
                        task.task_id, result={"output": result.output}
                    )
                    all_results[task.actor] = result

                    agent_config = next((a for a in sm.agents if a.name == task.actor), None)  # type: ignore[union-attr]
                    if agent_config:
                        sm._learn_from_result(result, agent_config, goal=task.description or goal)
                else:
                    # Retry with enriched context if attempts remain
                    if attempt_counts.get(task.task_id, 1) < max_attempts:
                        error_msg = str(getattr(result, "error", None) or "Execution failed")
                        fb = FeedbackMessage(
                            source_actor="swarm_manager",
                            target_actor=task.actor,
                            feedback_type=FeedbackType.ERROR,
                            content=f"Previous attempt failed: {error_msg}. Please try a different approach.",
                            context={"attempt": attempt_counts[task.task_id], "error": error_msg},
                            requires_response=False,
                            priority=1,
                        )
                        sm.feedback_channel.send(fb)
                        # Reset task to PENDING for retry
                        try:
                            sm.swarm_task_board.add_task(
                                task_id=f"{task.task_id}_retry{attempt_counts[task.task_id]}",
                                description=task.description,
                                actor=task.actor,
                            )
                        except Exception:
                            sm.swarm_task_board.fail_task(task.task_id, error=error_msg)
                    else:
                        sm.swarm_task_board.fail_task(
                            task.task_id,
                            error=str(getattr(result, "error", None) or "Execution failed"),
                        )
                    all_results[task.actor] = result

        # MAS-ZERO: Handle TOO_HARD signals from agents
        # If any agent signaled TOO_HARD, try re-routing its task
        for agent_name, result in list(all_results.items()):
            output = result.output if hasattr(result, "output") else result
            is_too_hard = (isinstance(output, dict) and output.get("too_hard")) or (
                hasattr(result, "output")
                and isinstance(result.output, dict)
                and result.output.get("too_hard")
            )
            if is_too_hard and not result.success:
                logger.info(
                    f"MAS-ZERO: Agent '{agent_name}' signaled TOO_HARD, "
                    f"task may need re-decomposition"
                )

        # MAS-ZERO: Meta-feedback evaluation (solvability + completeness)
        meta_feedback = sm._mas_zero_evaluate(goal, all_results)

        # MAS-ZERO: Iterative refinement if meta-feedback says refine
        if meta_feedback.get("should_refine") and len(all_results) > 0:
            safe_status(status_callback, "MAS-Evolve", "refining based on meta-feedback")
            all_results = await sm._mas_zero_evolve(
                goal,
                all_results,
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
        if get_metrics:
            try:
                _metrics = get_metrics()
                for agent_name, result in all_results.items():
                    _metrics.record_execution(
                        agent_name=agent_name,
                        task_type="multi_agent",
                        duration_s=getattr(result, "execution_time", 0.0),
                        success=result.success if hasattr(result, "success") else False,
                    )
            except Exception:
                pass

        return combined_result

    # =========================================================================
    # DISCUSSION PARADIGMS — delegated to ParadigmExecutor
    # =========================================================================

    async def _paradigm_run_agent(
        self, runner: Any, sub_goal: Any, agent_name: Any, **kwargs: Any
    ) -> Any:
        return await self._paradigms.run_agent(runner, sub_goal, agent_name, **kwargs)

    async def _paradigm_relay(self, goal: Any, **kwargs: Any) -> Any:
        return await self._paradigms.relay(goal, **kwargs)

    async def _paradigm_debate(self, goal: Any, **kwargs: Any) -> Any:
        return await self._paradigms.debate(goal, **kwargs)

    async def _paradigm_refinement(self, goal: Any, **kwargs: Any) -> Any:
        return await self._paradigms.refinement(goal, **kwargs)

    def _aggregate_results(self, results: Any, goal: Any) -> Any:
        return self._paradigms.aggregate_results(results, goal)

    def _assign_cooperative_credit(self, results: Any, goal: Any) -> Any:
        return self._paradigms.assign_cooperative_credit(results, goal)

    async def autonomous_setup(self, goal: str, status_callback: Any = None) -> Any:
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
        if cache_key in sm._setup_cache:  # type: ignore[comparison-overlap]
            _status("Setup", "using cached")
            return

        # Parse intent to understand requirements
        _status("Parsing intent", goal)
        task_graph = sm.swarm_intent_parser.parse(goal)

        # Research solutions if needed
        if task_graph.requirements or task_graph.integrations:
            # Filter out stop words and meaningless single-word requirements
            stop_words = {
                "existing",
                "use",
                "check",
                "find",
                "get",
                "the",
                "a",
                "an",
                "and",
                "or",
                "for",
                "with",
            }
            meaningful_requirements = [
                req
                for req in task_graph.requirements
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

        # Configure integrations (handled externally if needed)
        if task_graph.integrations:
            _status("Configuring", f"{len(task_graph.integrations)} integrations noted")
            for integration in task_graph.integrations:
                logger.info(f"Integration required: {integration}")

        # Mark as cached
        sm._setup_cache[cache_key] = True  # type: ignore[index]
        _status("Setup complete", "")


def _build_response_digest(content: str, max_len: int = 1500) -> str:
    """
    Build a structural digest of a response for storage in episode outcomes.
    Shows outline + representative samples instead of an arbitrary character cutoff.
    """
    import re as _re

    lines = content.split("\n")
    words = content.split()

    # Extract headings (skip code blocks)
    headings = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and stripped.startswith("#"):
            title = stripped.lstrip("# ").strip()
            if title:
                headings.append(title)

    code_blocks = len(_re.findall(r"```\w*\n.*?```", content, _re.DOTALL))

    parts = [f"[{len(words)} words, {len(headings)} sections, {code_blocks} code blocks]"]

    if headings:
        parts.append("Outline: " + " → ".join(headings[:10]))

    # Opening paragraph
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("```"):
            parts.append("Opens: " + stripped[:200])
            break

    # One sample from the middle
    mid_sections: list = []
    current_heading = ""
    current_text: list = []
    in_code_blk = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_blk = not in_code_blk
        if not in_code_blk and stripped.startswith("#") and stripped.lstrip("# ").strip():
            if current_text and current_heading:
                body = "\n".join(current_text).strip()
                if len(body) > 80:
                    mid_sections.append((current_heading, body))
            current_heading = stripped.lstrip("# ").strip()
            current_text = []
        else:
            current_text.append(line)
    if current_text and current_heading:
        body = "\n".join(current_text).strip()
        if len(body) > 80:
            mid_sections.append((current_heading, body))

    if mid_sections:
        mid = mid_sections[len(mid_sections) // 2]
        paras = [p.strip() for p in mid[1].split("\n\n") if p.strip()]
        sample = paras[0] if paras else mid[1]
        parts.append(f"[{mid[0]}]: {sample[:300]}")

    # First code block sample (if present)
    code_match = _re.search(r"```\w*\n(.*?)```", content, _re.DOTALL)
    if code_match:
        code_lines = code_match.group(1).strip().split("\n")
        if len(code_lines) > 8:
            code_sample = "\n".join(code_lines[:6]) + f"\n... ({len(code_lines)} lines)"
        else:
            code_sample = "\n".join(code_lines)
        parts.append(f"Code: ```\n{code_sample}\n```")

    digest = "\n".join(parts)
    return digest[:max_len]


class Orchestrator:
    """
    Composable swarm orchestrator with lazy initialization.

    Public API — two methods:
        result = await orchestrator.run(goal)                   # Do this thing
        result = await orchestrator.chat(message, history=...)  # Let's talk

    run() auto-routes based on complexity, or accepts explicit hints:
        run(goal)                          # Auto-detect (agent or swarm)
        run(goal, swarm=CodingSwarm)       # Explicit swarm
        run(goal, agent=my_agent)          # Explicit single agent
        run(goal, stages=[...])            # Multi-stage pipeline
        run(goal, stream=True)             # Streaming output

    chat() is conversational mode with human-in-the-loop:
        chat(message, history=[...])       # Tool calling, session context
        chat(message, stream=True)         # Streaming tokens

    All modes share LearningService, Memory, Tools, Providers.

    All heavyweight components are lazy-loaded via descriptors.
    Init is fast (~10ms). Components are created on first access.

    Key features:
        - Unified execution: run() and chat() cover 100% of use cases
        - Auto-routing: complexity detection routes to agent or swarm
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
    swarm_code_generator = LazyComponent(lambda self: _create_code_generator(self.config))
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

    _providers = LazyComponent(
        lambda self: ProviderManager(
            config=self.config,
            get_swarm_intelligence=lambda: self.swarm_intelligence,
        )
    )
    _ensemble = LazyComponent(lambda self: EnsembleManager())
    _learning_ops = LazyComponent(
        lambda self: LearningDelegate(
            get_learning=lambda: self.learning,
            get_mas_learning=lambda: self.mas_learning,
            get_agents=lambda: self.agents,
        )
    )
    _mas_zero = LazyComponent(
        lambda self: MASZeroController(
            get_agents=lambda: self.agents,
            get_runners=lambda: self.runners,
        )
    )
    _router = LazyComponent(
        lambda self: SwarmRouter(
            get_swarm_intelligence=lambda: getattr(self, "swarm_intelligence", None),
            get_agents=lambda: self.agents,
            get_model_tier_router=lambda: self._model_tier_router,
            get_learning=lambda: getattr(self, "learning", None),
        )
    )

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
    ) -> None:
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
            "total_scheduled": 0,
            "total_waited": 0,  # times an agent had to wait for a slot
            "peak_concurrent": 0,
            "current_concurrent": 0,
        }

        # Prompts
        self.architect_prompts = architect_prompts or [
            "configs/prompts/architect/base_architect.md"
        ]
        self.auditor_prompts = auditor_prompts or ["configs/prompts/auditor/base_auditor.md"]

        # Composed AgentFactory — create BEFORE zero-config (needed for agent creation)
        self._agent_factory = AgentFactory(self)

        # Zero-config: natural language -> agents
        if isinstance(agents, str) and enable_zero_config:
            logger.info("Zero-config mode: analyzing task for agent configuration")
            agents = self._create_zero_config_agents(agents)

        # Normalize agents
        if agents is None:
            agents = [AgentConfig(name="auto", agent=AutoAgent())]
        elif isinstance(agents, AgentConfig):
            agents = [agents]

        self.agents = agents
        self.mode = "multi" if len(agents) > 1 else "single"

        # Lane queue: per-session serialization
        self._session_locks = _SessionLockManager()

        # Runner and LOTUS state (created lazily in _ensure_runners)
        self.runners: Dict[str, "AgentRunner"] = {}
        self._runners_built = False
        self.lotus = None
        self.lotus_optimizer = None

        # Provider registry (delegated to _providers manager)
        # Backward compat: self.provider_registry -> self._providers.provider_registry

        # Setup cache
        self._setup_cache: Dict[str, Any] = {}

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
        from Jotty.core.intelligence.orchestration.execution.execution_orchestrator import (
            ExecutionOrchestrator,
        )

        self._execution = ExecutionOrchestrator(self)

        # Intelligence effectiveness A/B metrics:
        # Tracks whether stigmergy/byzantine guidance improves success rate.
        # Keyed by task_type for fine-grained analysis.
        # KISS: Nested dict, no classes.
        self._intelligence_metrics: Dict[str, Dict[str, int]] = {}

        # Model tier router: maps task complexity -> cheap/balanced/quality LM
        # Lazy-init on first use. Integrates with ValidationGate decisions.
        self._model_tier_router: Optional[ModelTierRouter] = None

        # Composed ExecutionEngine — separates task execution from management
        self._engine = ExecutionEngine(self)

        logger.info(f"Orchestrator: {self.mode} mode, {len(self.agents)} agents (lazy init)")

    # =========================================================================
    # LAZY RUNNER CREATION — delegated to AgentFactory
    # =========================================================================

    def _ensure_runners(self) -> None:
        self._agent_factory.ensure_runners()

    # =========================================================================
    # DELEGATION: Single __getattr__ replaces 15+ @property boilerplate
    # =========================================================================
    #
    # Learning sub-components (sm.swarm_intelligence, etc.)
    # are forwarded to self.learning.xxx automatically.
    # Composed manager methods (_execute_ensemble, etc.) are forwarded to the
    # appropriate composed manager.
    #
    # This eliminates ~120 lines of repetitive @property definitions while
    # maintaining full backward compatibility.

    # Attributes forwarded to self.learning
    _LEARNING_ATTRS = frozenset(
        {
            "transfer_learning",
            "swarm_intelligence",
            "credit_weights",
            "trajectory_predictor",
            "divergence_memory",
            "brain_state",
            "agent_abstractor",
            "swarm_learner",
            "agent_slack",
            "feedback_channel",
        }
    )

    @property
    def agent_semaphore(self) -> asyncio.Semaphore:
        """Lazy-create asyncio.Semaphore in the current event loop."""
        if self._agent_semaphore is None:
            self._agent_semaphore = asyncio.Semaphore(self.max_concurrent_agents)  # type: ignore[assignment]
        return self._agent_semaphore  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
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
        if name == "provider_registry":
            return self._providers.provider_registry

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle setting delegated attributes (credit_weights, provider_registry)."""
        if name == "credit_weights" and "_lazy_learning" in self.__dict__:
            self.learning.credit_weights = value
        elif name == "provider_registry" and "_lazy__providers" in self.__dict__:
            self._providers.provider_registry = value
        else:
            super().__setattr__(name, value)

    def _maybe_drain_training_task(self, result: Any) -> None:
        """Drain one queued training task after a successful run (placeholder)."""
        pass

    # --- Composed manager delegation (thin methods, not properties) ---

    async def execute_with_provider(
        self, category: Any, task: Any, context: Any = None, provider_name: Any = None
    ) -> Any:
        return await self._providers.execute_with_provider(category, task, context, provider_name)

    def get_provider_summary(self) -> Any:
        return self._providers.get_provider_summary()

    async def _execute_ensemble(
        self,
        goal: Any,
        strategy: Any = "multi_perspective",
        status_callback: Any = None,
        max_perspectives: Any = 4,
    ) -> Any:
        return await self._ensemble.execute_ensemble(
            goal, strategy, status_callback, max_perspectives
        )

    def _should_auto_ensemble(self, goal: Any) -> Any:
        return self._ensemble.should_auto_ensemble(goal)

    def _auto_load_learnings(self) -> Any:
        self._learning_ops.auto_load_learnings()

    def _auto_save_learnings(self) -> Any:
        self._learning_ops.auto_save_learnings(
            mas_learning=getattr(self, "mas_learning", None),
            swarm_terminal=getattr(self, "swarm_terminal", None),
            provider_registry=self._providers.provider_registry,
            memory_persistence=getattr(self, "memory_persistence", None),
        )

    def _save_learnings(self) -> Any:
        self._auto_save_learnings()

    def load_relevant_learnings(self, task_description: Any, agent_types: Any = None) -> Any:
        return self._learning_ops.load_relevant_learnings(task_description, agent_types)

    async def train(self, num_tasks: int = 5, status_callback: Any = None) -> Dict[str, Any]:
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
                results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "difficulty": task.difficulty,
                        "success": result.success,
                        "execution_time": getattr(result, "execution_time", 0),
                    }
                )

                # Feed result back to curriculum for difficulty adjustment
                curriculum.update_from_result(
                    task=task,
                    success=result.success,
                    execution_time=getattr(result, "execution_time", 0.0),
                )

                _status(
                    f"Task {i+1} {'passed' if result.success else 'failed'}",
                    f"type={task.task_type}, difficulty={task.difficulty:.1f}",
                )
            except (AgentExecutionError, LLMError) as e:
                logger.warning(f"Training task {i+1} failed (recoverable: {type(e).__name__}): {e}")
                results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "difficulty": task.difficulty,
                        "success": False,
                        "error": str(e),
                    }
                )
            except Exception as e:
                logger.warning(f"Training task {i+1} failed (unexpected): {e}", exc_info=True)
                results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "difficulty": task.difficulty,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Post-training effectiveness snapshot
        post_report = lp.effectiveness.improvement_report()

        # Save all learnings
        self._auto_save_learnings()

        successes = sum(1 for r in results if r.get("success"))
        _status("Training complete", f"{successes}/{num_tasks} passed")

        return {
            "total_tasks": num_tasks,
            "successes": successes,
            "success_rate": successes / max(1, num_tasks),
            "results": results,
            "pre_effectiveness": pre_report.get("_global", {}),
            "post_effectiveness": post_report.get("_global", {}),
            "checkpoint": checkpoint_path,  # rollback with self.learning.restore_checkpoint(path)
        }

    def record_agent_result(
        self,
        agent_name: Any,
        task_type: Any,
        success: Any,
        time_taken: Any,
        output_quality: Any = 0.0,
    ) -> None:
        self._learning_ops.record_agent_result(
            agent_name, task_type, success, time_taken, output_quality
        )

    def record_session_result(
        self,
        task_description: Any,
        agent_performances: Any,
        total_time: Any,
        success: Any,
        fixes_applied: Any = None,
        stigmergy_signals: Any = 0,
    ) -> Any:
        self._learning_ops.record_session_result(
            task_description,
            agent_performances,
            total_time,
            success,
            fixes_applied,
            stigmergy_signals,
        )

    def get_transferable_context(self, query: Any, agent: Any = None) -> Any:
        return self._learning_ops.get_transferable_context(query, agent)

    def get_swarm_wisdom(self, query: Any) -> Any:
        return self._learning_ops.get_swarm_wisdom(query)

    def get_agent_specializations(self) -> Any:
        return self._learning_ops.get_agent_specializations()

    def get_best_agent_for_task(self, query: Any) -> Any:
        return self._learning_ops.get_best_agent_for_task(query)

    def _mas_zero_verify(self, goal: Any, results: Any) -> Any:
        return self._mas_zero.verify(goal, results)

    def _mas_zero_evaluate(self, goal: Any, results: Any) -> Any:
        return self._mas_zero.evaluate(goal, results)

    def _mas_zero_should_reduce(self, goal: Any) -> Any:
        return self._mas_zero.should_reduce(goal)

    async def _mas_zero_evolve(
        self,
        goal: Any,
        initial_results: Any,
        max_iterations: Any = 2,
        status_callback: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._mas_zero.evolve(
            goal, initial_results, max_iterations, status_callback, **kwargs
        )

    def _reset_experience(self) -> Any:
        self._mas_zero.reset_experience()

    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    async def startup(self) -> Any:
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

    async def shutdown(self) -> Any:
        """
        Graceful shutdown - persist learnings and release resources.

        Should be called when Orchestrator is no longer needed.
        Safe to call multiple times.
        """
        try:
            # Await any in-flight background learning tasks
            await self._drain_background_tasks()

            # Periodic health report: log metrics summary at shutdown
            if "_lazy_learning" in self.__dict__:
                try:
                    lp = self.learning
                    if lp.metrics:
                        report = lp.metrics.get_report()
                        success_rate = report.get("success_rate", 0.0)
                        total_tasks = report.get("total_tasks", 0)
                        trend = report.get("trend", 0.0)
                        trend_label = (
                            "improving" if trend > 0 else ("declining" if trend < 0 else "stable")
                        )
                        logger.info(
                            f"Swarm Health Report: success={success_rate:.1%}, "
                            f"tasks={total_tasks}, trend={trend_label}"
                        )
                    # Also log effectiveness report
                    eff_report = lp.effectiveness.improvement_report()
                    global_eff = eff_report.get("_global", {})
                    if global_eff.get("total_episodes", 0) > 0:
                        logger.info(
                            f"Effectiveness: recent={global_eff.get('recent_success_rate', 0):.1%}, "
                            f"historical={global_eff.get('historical_success_rate', 0):.1%}, "
                            f"trend={global_eff.get('trend', 0):+.3f}, "
                            f"improving={global_eff.get('improving', False)}"
                        )
                except Exception as health_err:
                    logger.debug(f"Health report skipped: {health_err}")

            # Persist all learnings
            if "_lazy_learning" in self.__dict__:
                self._auto_save_learnings()
                logger.info("Learnings saved on shutdown")

            # Clear runners
            self.runners.clear()
            self._runners_built = False

            logger.info("Orchestrator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def __aenter__(self) -> Any:
        """Async context manager: `async with Orchestrator() as sm:`"""
        await self.startup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
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
            "swarm_planner",
            "swarm_task_board",
            "swarm_memory",
            "swarm_intent_parser",
            "swarm_provider_gateway",
            "swarm_researcher",
            "swarm_installer",
            "swarm_code_generator",
            "swarm_terminal",
            "swarm_ui_registry",
            "swarm_tool_validator",
            "swarm_tool_registry",
            "swarm_profiler",
            "swarm_state_manager",
            "shared_context",
            "io_manager",
            "data_registry",
            "context_guard",
            "learning",
            "mas_learning",
        ]
        created = {n: f"_lazy_{n}" in self.__dict__ for n in lazy_names}
        created_count = sum(created.values())

        result = {
            "mode": self.mode,
            "agents": [a.name for a in self.agents],  # type: ignore[union-attr]
            "runners_built": self._runners_built,
            "runners": list(self.runners.keys()),
            "episode_count": self.episode_count,
            "lotus_enabled": self.enable_lotus,
            "lotus_active": self.lotus is not None,
            "zero_config": self.enable_zero_config,
            "components": {
                "total": len(lazy_names),
                "created": created_count,
                "pending": len(lazy_names) - created_count,
                "detail": created,
            },
        }

        # Scheduling stats (AIOS-inspired)
        result["scheduling"] = {
            "max_concurrent_agents": self.max_concurrent_agents,
            "semaphore_available": (
                self._agent_semaphore._value
                if self._agent_semaphore
                else self.max_concurrent_agents
            ),
            **self._scheduling_stats,
        }

        # Add learning stats if pipeline is active
        if "_lazy_learning" in self.__dict__:
            try:
                lp = self.learning
                result["learning"] = {
                    "episode_count": lp.episode_count,
                    "has_intelligence": self.swarm_intelligence is not None,
                    "stigmergy_signals": len(lp.stigmergy.signals),
                    "byzantine_verifications": lp.byzantine_verifier.verified_count,
                    "byzantine_inconsistencies": lp.byzantine_verifier.inconsistent_count,
                    "credit_stats": lp.get_credit_stats(),
                    "adaptive_learning": lp.get_learning_state(),
                }
            except Exception as e:
                logger.debug(f"Learning stats collection failed: {e}")
                result["learning"] = {"status": "error"}

        # Training daemon status
        result["training_daemon"] = self.training_daemon_status()

        # Intelligence effectiveness A/B metrics (per task_type)
        def _format_im(bucket: Any) -> Dict:
            gr = bucket.get("guided_runs", 0)
            gs = bucket.get("guided_successes", 0)
            ur = bucket.get("unguided_runs", 0)
            us = bucket.get("unguided_successes", 0)
            guided_rate = gs / gr if gr > 0 else None
            unguided_rate = us / ur if ur > 0 else None
            return {
                **bucket,
                "guided_success_rate": guided_rate,
                "unguided_success_rate": unguided_rate,
                "guidance_lift": (
                    guided_rate - unguided_rate
                    if guided_rate is not None and unguided_rate is not None
                    else None
                ),
            }

        result["intelligence_effectiveness"] = {
            tt: _format_im(bucket) for tt, bucket in self._intelligence_metrics.items()
        }

        # Paradigm effectiveness stats
        if "_lazy_learning" in self.__dict__:
            try:
                result["paradigm_stats"] = self.learning.get_paradigm_stats()
            except Exception as e:
                logger.debug(f"Paradigm stats unavailable: {e}")

        # Add LOTUS stats if active
        if self.lotus:
            result["lotus_stats"] = self.get_lotus_stats()

        # Add observability metrics
        if get_metrics and get_tracer:
            try:
                _metrics = get_metrics()
                result["observability"] = {
                    "metrics": _metrics.get_summary(),
                    "cost_breakdown": _metrics.get_cost_breakdown(),
                }
                _tracer = get_tracer()
                _trace = _tracer.get_current_trace()
                if _trace:
                    result["observability"]["last_trace"] = {  # type: ignore[index]
                        "trace_id": _trace.trace_id[:8],
                        "spans": _trace.span_count,
                        "duration_ms": round(_trace.duration_ms, 0),
                        "total_cost_usd": round(_trace.total_cost, 6),
                        "total_tokens": _trace.total_tokens,
                    }
            except Exception as e:
                logger.debug(f"Observability metrics unavailable: {e}")

        # Add model tier routing stats
        if self._model_tier_router:
            result["model_tier_routing"] = self._model_tier_router.get_savings_estimate()

        return result

    @property
    def metrics(self) -> Dict[str, Any]:
        """Quick access to execution metrics."""
        return {
            "episodes": self.episode_count,
            "agents": len(self.agents),
            "mode": self.mode,
            "runners_built": self._runners_built,
            "components_loaded": sum(1 for k in self.__dict__ if k.startswith("_lazy_")),
        }

    # =========================================================================
    # LOTUS OPTIMIZATION — delegated to AgentFactory
    # =========================================================================

    def _init_lotus_optimization(self) -> None:
        self._agent_factory.init_lotus_optimization()

    def get_lotus_stats(self) -> Any:
        return self._agent_factory.get_lotus_stats()

    def get_lotus_savings(self) -> Any:
        return self._agent_factory.get_lotus_savings()

    # =========================================================================
    # ML Learning Bridge (for SkillOrchestrator / Swarm pipeline integration)
    # =========================================================================

    def get_ml_learning(self) -> Any:
        """Get MASLearning instance for ML pipeline integration."""
        return self.mas_learning

    def record_report_section_outcome(
        self, section_name: str, success: bool, error: str | None = None
    ) -> None:
        """Record report section outcome for cross-run learning."""
        try:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            svc = LearningService.get_instance()
            svc.record_outcome(
                unit_name="report_generator",
                state=f"section:{section_name}|error:{(error or '')[:200]}",
                action=f"generate_section:{section_name}",
                reward=1.0 if success else 0.0,
                domain="report_sections",
            )
        except Exception as e:
            logger.debug(f"Record section outcome failed: {e}")

    def should_skip_report_section(self, section_name: str) -> bool:
        """Check if a report section should be skipped based on learned failures."""
        try:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            svc = LearningService.get_instance()
            guidance = svc.query("report_sections", context={"section": section_name})
            failures = guidance.get("failure_analysis", [])
            for f in failures:
                if section_name in f.get("description", ""):
                    return True
        except Exception as e:
            logger.debug(f"Section skip check failed for '{section_name}': {e}")
        return False

    # =========================================================================
    # Provider, Ensemble, Learning methods — see _provider_mixin.py,
    # _ensemble_mixin.py, learning_delegate.py
    # =========================================================================

    def _register_agents_with_axon(self) -> None:
        self._agent_factory.register_agents_with_axon()

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
            },
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
        from Jotty.core.capabilities.prompts import (
            PromptComposer,  # type: ignore[import-not-found, import]
        )

        # Detect model from agent config
        model = ""
        for ac in self.agents:
            if ac.name == agent_name or not agent_name:  # type: ignore[union-attr]
                model = getattr(ac, "model", "") or getattr(ac.agent, "model", "") or ""  # type: ignore[union-attr]
                break
        if not model:
            model = getattr(self.config, "model", "")

        composer = PromptComposer(model=model)

        # Gather tool info with trust levels
        tool_names = []
        tool_descs = {}
        trust_levels = {}
        try:
            from Jotty.core.capabilities.registry import get_unified_registry

            registry = get_unified_registry()
            skills = registry.list_skills()
            for s in skills[:30]:  # Top 30 to keep prompt manageable
                tool_names.append(s["name"])
                tool_descs[s["name"]] = s.get("description", "")
                trust_levels[s["name"]] = s.get("trust_level", "safe")
        except Exception as e:
            logger.debug(f"Tool registry lookup failed: {e}")

        # Agent identity
        identity = ""
        for ac in self.agents:
            if ac.name == agent_name or not agent_name:  # type: ignore[union-attr]
                identity = getattr(ac.agent, "system_prompt", "") or getattr(  # type: ignore[union-attr]
                    ac, "system_prompt", ""
                )
                break

        import os as _os

        return composer.compose(  # type: ignore[no-any-return]
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

    def _ensure_engine(self) -> "ExecutionEngine":
        """Lazy-init ExecutionEngine (supports tests using __new__)."""
        if not hasattr(self, "_engine"):
            self._engine = ExecutionEngine(self)
        return self._engine

    # =========================================================================
    # PUBLIC API — Two methods: run() and chat()
    # =========================================================================

    async def run(
        self,
        goal: str,
        *,
        stream: bool = False,
        stages: Optional[List[Dict[str, Any]]] = None,
        swarm: Optional[Any] = None,
        agent: Optional[Any] = None,
        learn: bool = True,
        status_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Universal execution entry point.

        Auto-routes based on what's provided:
        - Just a goal          → auto-detect complexity (agent or swarm)
        - agent=MyAgent        → single agent execution
        - swarm=CodingSwarm    → specific swarm template
        - stages=[...]         → multi-stage pipeline with dependencies

        All modes share LearningService, Memory, Tools, Providers.
        Learning is fire-and-forget: never blocks the result.

        Args:
            goal: Task goal/description (natural language)
            stream: If True, returns AsyncIterator[StreamEvent]
            stages: Pipeline stage definitions (triggers pipeline mode)
            swarm: SwarmTemplate class or instance (triggers swarm mode)
            agent: BaseAgent instance (triggers single-agent mode)
            learn: If True (default), record outcomes + run post-episode learning.
                   Set False for tests, benchmarks, or latency-critical paths.
            status_callback: Optional progress callback(stage, detail)
            **kwargs: Additional arguments passed to the execution engine

        Returns:
            ExecutionResult (or AsyncIterator[StreamEvent] if stream=True)

        Examples:
            # Auto-detect with learning (default)
            result = await orchestrator.run("What is GDP?")

            # Skip learning (tests / benchmarks)
            result = await orchestrator.run("Quick test", learn=False)

            # Explicit swarm
            result = await orchestrator.run("Analyze data", swarm=DataAnalysisSwarm)

            # Pipeline
            result = await orchestrator.run("Build and test API", stages=[
                {"name": "design", "swarm": CodingSwarm},
                {"name": "test", "swarm": TestingSwarm, "depends_on": ["design"]},
            ])

            # Streaming (await first, then iterate)
            stream = await orchestrator.run("Research AI", stream=True)
            async for event in stream:
                print(event)
        """

        # Lane queue: serialize requests for the same session
        session_id = kwargs.get("session_id", "")
        if session_id:
            lock = await self._session_locks.get_lock(session_id)
            return await self._run_with_lock(
                lock,
                goal,
                stream=stream,
                stages=stages,
                swarm=swarm,
                agent=agent,
                learn=learn,
                status_callback=status_callback,
                **kwargs,
            )
        return await self._run_inner(
            goal,
            stream=stream,
            stages=stages,
            swarm=swarm,
            agent=agent,
            learn=learn,
            status_callback=status_callback,
            **kwargs,
        )

    async def _run_with_lock(self, lock: asyncio.Lock, goal: str, **kwargs: Any) -> Any:
        """Execute run() under a per-session lock."""
        async with lock:
            return await self._run_inner(goal, **kwargs)

    async def _run_inner(
        self,
        goal: str,
        *,
        stream: bool = False,
        stages: Optional[List[Dict[str, Any]]] = None,
        swarm: Optional[Any] = None,
        agent: Optional[Any] = None,
        learn: bool = True,
        status_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        """Core run() logic, may be called directly or under a session lock."""
        import time as _time

        from Jotty.core.intelligence.learning.learning_service import (
            LearningService,
            analyze_response,
            classify_domain,
        )

        learning = LearningService.get_instance()
        run_start = _time.time()

        # ── PRE-EXECUTION: classify domain + learning-steered params ──
        detected_domain, detected_task_type = classify_domain(goal)
        optimal_params: Dict[str, Any] = {}

        if learn:
            # Get optimal execution parameters from learning
            try:
                optimal_params = learning.get_optimal_execution_params(
                    domain=detected_domain,
                    task_type=detected_task_type,
                    goal=goal,
                )
            except Exception:
                pass

            # Inject learning context + retrieval-augmented examples
            try:
                ctx_parts: List[str] = []
                guidance_str = learning.build_context_string(
                    domain=detected_domain, task_type=detected_task_type
                )
                if guidance_str:
                    ctx_parts.append(guidance_str)
                if detected_domain != "general":
                    general_str = learning.build_context_string(domain="general", task_type="run")
                    if general_str and general_str != guidance_str:
                        ctx_parts.append(general_str)
                retrieval_ctx = learning.build_retrieval_context(
                    domain=detected_domain, task_type=detected_task_type, goal=goal
                )
                if retrieval_ctx:
                    ctx_parts.append(retrieval_ctx)
                if ctx_parts:
                    kwargs.setdefault("learning_context", "")
                    kwargs["learning_context"] += "\n" + "\n".join(ctx_parts)
            except Exception as e:
                logger.debug(f"Pre-execution learning guidance failed: {e}")

        # ── PRE-EXECUTION: inject budget awareness (ClawWork-inspired) ──
        try:
            from Jotty.core.infrastructure.utils.budget_tracker import BudgetTracker

            budget = BudgetTracker.get_instance()
            budget_ctx = budget.get_economic_context()
            if budget_ctx:
                kwargs.setdefault("learning_context", "")
                kwargs["learning_context"] += "\n" + budget_ctx
        except Exception as e:
            logger.debug(f"Budget context injection skipped: {e}")

        execution_mode = (
            "pipeline" if stages else "swarm" if swarm else "agent" if agent else "auto"
        )

        # ── EXECUTION: route to the appropriate path ──
        result = None
        if stages is not None:
            result = await self._run_pipeline(
                goal, stages=stages, status_callback=status_callback, **kwargs
            )
        elif swarm is not None:
            if stream:
                return self._run_swarm_stream(
                    goal, swarm=swarm, status_callback=status_callback, **kwargs
                )
            result = await self._run_swarm(
                goal, swarm=swarm, status_callback=status_callback, stream=False, **kwargs
            )
        elif agent is not None:
            result = await self._run_agent(
                goal, agent=agent, status_callback=status_callback, stream=stream, **kwargs
            )
        elif stream:
            kwargs["status_callback"] = status_callback
            return self._run_stream(goal, **kwargs)
        else:
            kwargs["status_callback"] = status_callback
            result = await self._ensure_engine().run(goal, **kwargs)

        # ── RECOVERY: salvage partial results on failure (ClawWork wrap-up inspired) ──
        if result and isinstance(result, EpisodeResult) and not result.success:
            result = self._attempt_recovery(result, goal)

        # ── POST-EXECUTION: learning (fire-and-forget, never blocks result) ──
        if learn:
            run_time = _time.time() - run_start
            success = getattr(result, "success", True) if result else False

            result_text = ""
            if isinstance(result, EpisodeResult):
                result_text = getattr(result, "output", "") or str(result)
            elif result:
                result_text = str(result)

            from Jotty.core.intelligence.learning.learning_service import analyze_response

            response_analysis = analyze_response(result_text, goal) if result_text else {}
            heuristic_quality = response_analysis.get(
                "quality_score",
                getattr(result, "quality_score", 0.8 if success else 0.0),
            )

            # Build rich outcome with excerpt for retrieval
            outcome: Dict[str, Any] = {
                "output_length": len(result_text),
                **{k: v for k, v in response_analysis.items() if k != "empty"},
            }
            if result_text:
                # Build a structural digest: first ~600 chars preserving headings/structure
                excerpt = (
                    result_text[:600].rsplit("\n", 1)[0] if len(result_text) > 600 else result_text
                )
                outcome["response_excerpt"] = excerpt

            # 1. Record with rich metadata
            import uuid as _uuid

            ep_id = f"run_{int(run_start)}_{_uuid.uuid4().hex[:6]}"
            try:
                learning.record(
                    unit_name="Orchestrator",
                    unit_type="orchestrator",
                    domain=detected_domain,
                    task_type=detected_task_type,
                    context={"goal": str(goal)[:500]},
                    action={
                        "mode": execution_mode,
                        "domain": detected_domain,
                        "task_type": detected_task_type,
                        "strategy": optimal_params.get("strategy", "default"),
                        "exploration": optimal_params.get("exploration", False),
                    },
                    outcome=outcome,
                    success=success,
                    quality=heuristic_quality,
                    execution_time=run_time,
                )
            except Exception as e:
                logger.debug(f"LearningService record failed: {e}")

            # 2. Post-execution reflection
            if success and result_text:
                try:
                    learning.post_execution_reflect(
                        episode_id=ep_id,
                        goal=goal,
                        content=result_text,
                        domain=detected_domain,
                        quality_score=heuristic_quality,
                        execution_time=run_time,
                    )
                except Exception as e:
                    logger.debug(f"Post-execution reflection failed: {e}")

            # 3. Full learning pipeline (heavy — background, fire-and-forget)
            #    For chat() path, wrap LLMExecutionResult into EpisodeResult
            #    so TD-Lambda, credit assignment, and validation all fire.
            learnable_result = result
            if result and not isinstance(result, EpisodeResult):
                learnable_result = EpisodeResult(
                    success=success,
                    output=result_text[:50000] if result_text else "",
                    trajectory=[],
                    tagged_outputs=[],
                    episode=0,
                    execution_time=run_time,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                    override_metadata={
                        "source": "chat",
                        "domain": detected_domain,
                        "task_type": detected_task_type,
                        "quality": round(heuristic_quality, 3),
                    },
                )
            if isinstance(learnable_result, EpisodeResult):
                self._schedule_background_learning(learnable_result, goal)

        # ── LLM judge (awaited inline to ensure completion) ──
        if learn and result_text and len(result_text) > 1000 and success:
            try:
                llm_score = await learning.llm_judge_quality(
                    goal=goal,
                    content=result_text,
                    domain=detected_domain,
                    heuristic_score=heuristic_quality,
                )
                blended = llm_score * 0.6 + heuristic_quality * 0.4
                learning._store.update_episode_quality(
                    episode_id=ep_id,
                    quality=blended,
                    outcome_patch={
                        "llm_judged": True,
                        "llm_score": round(llm_score, 3),
                        "heuristic_quality": round(heuristic_quality, 3),
                    },
                )
                learning._update_values(
                    detected_domain,
                    detected_task_type,
                    {"domain": detected_domain},
                    True,
                    blended,
                )
                learning.analyze_exploration_results(detected_domain)
            except Exception as e:
                logger.debug(f"LLM judge failed in _run_inner: {e}")

        return result

    async def chat(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        learn: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        enabled_tools: Optional[List[str]] = None,
        output_format: str = "auto",
        max_steps: int = 10,
        **kwargs: Any,
    ) -> Any:
        """
        Conversational mode with human-in-the-loop.

        Tool calling, streaming, session context via history.
        Uses ChatExecutor internally, wrapped with LearningService.
        Learning is fire-and-forget: never blocks the result.

        Args:
            message: User message
            history: Conversation history (list of {role, content} dicts)
            stream: If True, returns AsyncIterator[StreamEvent]
            learn: If True (default), record outcomes to LearningService.
                   Set False for tests, benchmarks, or latency-critical paths.
            provider: LLM provider ('anthropic', 'openai', etc.). Auto-detects if None.
            model: Model name (uses provider default if not specified)
            status_callback: Progress callback(stage, detail)
            stream_callback: Token-level streaming callback(chunk)
            enabled_tools: Only enable these tools (None = all)
            output_format: Force output format ('auto', 'pdf', 'docx', etc.)
            max_steps: Maximum tool-calling iterations
            **kwargs: Additional arguments

        Returns:
            LLMExecutionResult (or AsyncIterator[StreamEvent] if stream=True)

        Examples:
            # Simple chat (learning on by default)
            result = await orchestrator.chat("Hello!")

            # Skip learning (tests / benchmarks)
            result = await orchestrator.chat("Quick test", learn=False)

            # Streaming (await first, then iterate)
            stream = await orchestrator.chat("Explain quantum physics", stream=True)
            async for event in stream:
                print(event)
        """
        import time as _time

        from Jotty.core.intelligence.learning.learning_service import (
            LearningService,
            analyze_response,
            classify_domain,
        )
        from Jotty.core.intelligence.orchestration.execution.unified_executor import (
            ChatExecutor as _ChatExecutor,
        )

        learning = LearningService.get_instance()
        chat_start = _time.time()

        # ── PRE-EXECUTION: classify domain ──
        detected_domain, detected_task_type = classify_domain(message)

        # ── PRE-EXECUTION: learning steers execution parameters ──
        optimal_params: Dict[str, Any] = {}
        effective_model = model
        effective_provider = provider
        if learn:
            try:
                optimal_params = learning.get_optimal_execution_params(
                    domain=detected_domain,
                    task_type=detected_task_type,
                    goal=message,
                )
                # Learning overrides model/provider ONLY if caller didn't specify
                if not model and optimal_params.get("model"):
                    effective_model = optimal_params["model"]
                if optimal_params.get("exploration"):
                    logger.info(
                        f"Exploration: {optimal_params.get('exploration_reason', 'unknown')}"
                    )
            except Exception as e:
                logger.debug(f"Execution param optimization failed: {e}")

        # ── PRE-EXECUTION: start episode ──
        episode_id = None
        if learn:
            try:
                episode_id = learning.start_episode(
                    unit_name="Orchestrator",
                    unit_type="chat",
                    domain=detected_domain,
                    task_type=detected_task_type,
                    context={
                        "message": message[:500],
                        "history_len": len(history or []),
                        "detected_domain": detected_domain,
                        "detected_task_type": detected_task_type,
                        "provider": effective_provider or "auto",
                        "model": effective_model or "default",
                        "exploration": optimal_params.get("exploration", False),
                        "exploration_reason": optimal_params.get("exploration_reason", ""),
                        "strategy": optimal_params.get("strategy", "default"),
                    },
                )
            except Exception as e:
                logger.debug(f"LearningService episode start failed: {e}")

            # Inject learning context: domain guidance + retrieval-augmented examples
            # All context injection respects the adaptive gate — only inject when
            # the model is struggling (success_rate < 90%) or cold-starting (< 5 episodes).
            try:
                ctx_parts: List[str] = []

                # 1. Domain-specific guidance (has its own adaptive gate)
                domain_ctx = learning.build_context_string(
                    domain=detected_domain, task_type=detected_task_type
                )
                if domain_ctx:
                    ctx_parts.append(domain_ctx)

                # 2. General guidance (if different domain)
                if detected_domain != "general":
                    general_ctx = learning.build_context_string(domain="general", task_type="run")
                    if general_ctx and general_ctx != domain_ctx:
                        ctx_parts.append(general_ctx)

                # 3. Retrieval-augmented examples (has its own adaptive gate)
                retrieval_ctx = learning.build_retrieval_context(
                    domain=detected_domain, task_type=detected_task_type, goal=message
                )
                if retrieval_ctx:
                    ctx_parts.append(retrieval_ctx)

                # 4-5. Paradigm + tool guidance only when adaptive gate is open
                # (i.e., domain_ctx or retrieval_ctx was non-empty)
                if ctx_parts:
                    paradigm = optimal_params.get("paradigm")
                    if paradigm:
                        paradigm_guidance = {
                            "direct": "Respond directly and comprehensively in a single pass.",
                            "relay": "Break the task into sequential sub-tasks. Address each in order.",
                            "debate": "Consider multiple perspectives. Present arguments for and against before concluding.",
                            "refinement": "Draft an initial response, then critically review and improve it.",
                        }
                        guidance = paradigm_guidance.get(paradigm, "")
                        if guidance:
                            ctx_parts.append(f"[APPROACH] Use {paradigm} paradigm: {guidance}")

                    tools_hint = optimal_params.get("tools_hint")
                    if tools_hint:
                        ctx_parts.append(
                            f"[TOOL GUIDANCE] Recommended tools for {detected_domain}: "
                            + ", ".join(tools_hint)
                        )

                if ctx_parts:
                    kwargs.setdefault("learning_context", "")
                    kwargs["learning_context"] += "\n" + "\n".join(ctx_parts)
            except Exception:
                pass

        # ── EXECUTION (with learning-optimized parameters) ──
        executor = _ChatExecutor(
            provider=effective_provider,
            model=effective_model,
            status_callback=status_callback,
            stream_callback=stream_callback if not stream else None,
            enabled_tools=enabled_tools,
            output_format=output_format,
            max_steps=max_steps,
            learning_context=kwargs.get("learning_context"),
            temperature=kwargs.get("temperature"),
        )

        if stream:
            return self._chat_stream_with_learning(
                executor,
                message,
                history,
                episode_id,
                learning,
                chat_start,
                _domain=detected_domain,
                _task_type=detected_task_type,
                _optimal_params=optimal_params,
                _effective_provider=effective_provider,
                _effective_model=effective_model,
            )

        # Non-streaming execution
        result = None
        error_msg = None
        try:
            result = await executor.execute(message, history=history)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chat execution failed: {e}", exc_info=True)
            raise
        finally:
            # ── POST-EXECUTION: record with heuristic quality immediately ──
            if learn and episode_id:
                chat_time = _time.time() - chat_start
                content = getattr(result, "content", "") if result else ""

                response_analysis = analyze_response(content, message) if content else {}
                heuristic_quality = response_analysis.get("quality_score", 0.0)

                # Success = LLM didn't error AND quality is above threshold.
                # Without quality-based failure detection, the learning system
                # records 100% success rate even for poor responses, keeping
                # the adaptive gate permanently closed.
                llm_ok = getattr(result, "success", False) if result else False
                success = llm_ok and heuristic_quality >= 0.5

                outcome: Dict[str, Any] = {
                    "content_length": len(content),
                    **{k: v for k, v in response_analysis.items() if k != "empty"},
                }
                if content:
                    outcome["response_excerpt"] = _build_response_digest(content)

                action_meta: Dict[str, Any] = {
                    "provider": effective_provider or "auto",
                    "model": effective_model or "default",
                    "domain": detected_domain,
                    "task_type": detected_task_type,
                    "had_history": bool(history),
                    "max_steps": max_steps,
                    "tools_enabled": bool(enabled_tools),
                    "temperature": optimal_params.get("temperature") or 0.5,
                    "paradigm": optimal_params.get("paradigm") or "direct",
                }
                if optimal_params.get("tools_hint"):
                    action_meta["tools_used"] = ",".join(optimal_params["tools_hint"])
                if optimal_params.get("exploration"):
                    action_meta["exploration"] = True
                    action_meta["exploration_reason"] = optimal_params.get("exploration_reason", "")
                if optimal_params.get("strategy"):
                    action_meta["strategy"] = optimal_params["strategy"]

                try:
                    learning.end_episode(
                        episode_id=episode_id,
                        success=success,
                        quality=heuristic_quality,
                        cost=0.0,
                        outcome=outcome,
                        error_message=error_msg,
                        action_metadata=action_meta,
                    )
                except Exception as e:
                    logger.debug(f"LearningService episode end failed: {e}")

                if success and content:
                    try:
                        learning.post_execution_reflect(
                            episode_id=episode_id,
                            goal=message,
                            content=content,
                            domain=detected_domain,
                            quality_score=heuristic_quality,
                            execution_time=chat_time,
                        )
                    except Exception as e:
                        logger.debug(f"Post-execution reflection failed: {e}")

        # ── POST-EXECUTION: Adaptive LLM judge ──
        # Cold start (<10 judged): inline. Stable: background (or skip).
        if learn and episode_id and result:
            _jcontent = getattr(result, "content", "")
            _jsuccess = getattr(result, "success", False)
            if _jcontent and len(_jcontent) > 1000 and _jsuccess:
                try:
                    _jra = analyze_response(_jcontent, message)
                    _jhq = _jra.get("quality_score", 0.8)

                    if learning.should_judge_inline(detected_domain):
                        llm_score = await learning.llm_judge_quality(
                            goal=message,
                            content=_jcontent,
                            domain=detected_domain,
                            heuristic_score=_jhq,
                        )
                        blended = llm_score * 0.6 + _jhq * 0.4
                        learning._store.update_episode_quality(
                            episode_id=episode_id,
                            quality=blended,
                            outcome_patch={
                                "llm_judged": True,
                                "llm_score": round(llm_score, 3),
                                "heuristic_quality": round(_jhq, 3),
                            },
                        )
                        learning._update_values(
                            detected_domain,
                            detected_task_type,
                            {"domain": detected_domain, "provider": effective_provider or "auto"},
                            True,
                            blended,
                        )
                    else:
                        learning.schedule_background_judge(
                            episode_id=episode_id,
                            goal=message,
                            content=_jcontent,
                            domain=detected_domain,
                            heuristic_quality=_jhq,
                        )

                    learning.analyze_exploration_results(detected_domain)
                except Exception as e:
                    logger.debug(f"LLM judge failed: {e}")

        return result

    async def _chat_stream_with_learning(
        self,
        executor: Any,
        message: str,
        history: Optional[List[Dict[str, Any]]],
        episode_id: Optional[str],
        learning: Any,
        start_time: float,
        _domain: str = "general",
        _task_type: str = "chat",
        _optimal_params: Optional[Dict[str, Any]] = None,
        _effective_provider: Optional[str] = None,
        _effective_model: Optional[str] = None,
    ) -> Any:
        """Wrap streaming chat with LearningService episode tracking."""
        import time as _time

        collected_content = []
        try:
            async for event in executor.execute_stream(message, history=history):
                if hasattr(event, "type") and hasattr(event, "data"):
                    if event.type == "text" and event.data:
                        collected_content.append(str(event.data))
                yield event
        except Exception as e:
            logger.error(f"Chat stream failed: {e}")
            raise
        finally:
            if episode_id:
                chat_time = _time.time() - start_time
                full_content = "".join(collected_content)

                from Jotty.core.intelligence.learning.learning_service import analyze_response

                analysis = analyze_response(full_content, message) if full_content else {}
                quality = analysis.get("quality_score", 0.7)

                _op = _optimal_params or {}
                stream_action_meta: Dict[str, Any] = {
                    "provider": _effective_provider or "auto",
                    "model": _effective_model or "default",
                    "domain": _domain,
                    "task_type": _task_type,
                    "temperature": _op.get("temperature") or 0.5,
                    "paradigm": _op.get("paradigm") or "direct",
                    "streamed": True,
                }
                if _op.get("tools_hint"):
                    stream_action_meta["tools_used"] = ",".join(_op["tools_hint"])
                if _op.get("exploration"):
                    stream_action_meta["exploration"] = True
                    stream_action_meta["exploration_reason"] = _op.get("exploration_reason", "")
                if _op.get("strategy"):
                    stream_action_meta["strategy"] = _op["strategy"]

                try:
                    learning.end_episode(
                        episode_id=episode_id,
                        success=True,
                        quality=quality,
                        cost=0.0,
                        outcome={
                            "streamed": True,
                            "duration": chat_time,
                            "content_length": len(full_content),
                            **{k: v for k, v in analysis.items() if k != "empty"},
                        },
                        action_metadata=stream_action_meta,
                    )
                except Exception:
                    pass

                # Domain-specific record for streaming too
                try:
                    stream_outcome: Dict[str, Any] = {
                        "content_length": len(full_content),
                        **analysis,
                    }
                    if full_content:
                        excerpt = (
                            full_content[:600].rsplit("\n", 1)[0]
                            if len(full_content) > 600
                            else full_content
                        )
                        stream_outcome["response_excerpt"] = excerpt
                    learning.record(
                        unit_name="Orchestrator",
                        unit_type="chat",
                        domain=_domain,
                        task_type=_task_type,
                        context={"goal": message[:500]},
                        action={"domain": _domain, "task_type": _task_type, "streamed": True},
                        outcome=stream_outcome,
                        success=True,
                        quality=quality,
                        execution_time=chat_time,
                    )
                except Exception:
                    pass

    # =========================================================================
    # INTERNAL EXECUTION MODES — Called by run()
    # =========================================================================

    async def _run_swarm(
        self,
        goal: str,
        *,
        swarm: Any,
        status_callback: Optional[Callable] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute with a specific swarm template."""

        from Jotty.core.intelligence.learning.learning_service import LearningService

        learning = LearningService.get_instance()

        # Instantiate swarm if class was passed
        if isinstance(swarm, type):
            swarm_config = kwargs.pop("swarm_config", {})
            if isinstance(swarm_config, dict):
                from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import (
                    SwarmBaseConfig,
                )

                swarm_config = SwarmBaseConfig(
                    name=swarm.__name__,
                    domain=swarm_config.get("domain", "general"),
                    **{k: v for k, v in swarm_config.items() if k not in ("name", "domain")},
                )
            swarm = swarm(swarm_config)

        if status_callback:
            status_callback("swarm", f"Executing {swarm.__class__.__name__}")

        return await swarm.execute(task=goal, **kwargs)

    async def _run_swarm_stream(
        self,
        goal: str,
        *,
        swarm: Any,
        status_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a swarm with streaming events wrapping execution."""
        from Jotty.core.intelligence.orchestration.llm_providers.types import (
            StreamEvent,
        )

        swarm_name = swarm.__name__ if isinstance(swarm, type) else swarm.__class__.__name__
        yield StreamEvent(type="status", data=f"Starting {swarm_name}")

        import time as _t

        start = _t.time()
        try:
            result = await self._run_swarm(
                goal, swarm=swarm, status_callback=status_callback, stream=False, **kwargs
            )
            elapsed = _t.time() - start

            # Emit the result as a text event
            output_str = ""
            if hasattr(result, "output"):
                output_str = str(result.output)
            elif hasattr(result, "content"):
                output_str = str(result.content)
            else:
                output_str = str(result)

            yield StreamEvent(type="text", data=output_str)
            yield StreamEvent(
                type="complete",
                data={
                    "success": getattr(result, "success", True),
                    "elapsed": elapsed,
                    "swarm": swarm_name,
                },
            )
        except Exception as e:
            yield StreamEvent(type="error", data=str(e))

    async def _run_agent(
        self,
        goal: str,
        *,
        agent: Any,
        status_callback: Optional[Callable] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute with a specific single agent."""
        if status_callback:
            agent_name = getattr(agent, "name", agent.__class__.__name__)
            status_callback("agent", f"Executing {agent_name}")

        return await agent.execute(task=goal, **kwargs)

    async def _run_stream(self, goal: str, **kwargs: Any) -> Any:
        """Auto-detect with streaming. Delegates to ChatExecutor."""
        from Jotty.core.intelligence.orchestration.execution.unified_executor import (
            ChatExecutor as _ChatExecutor,
        )

        executor = _ChatExecutor(
            status_callback=kwargs.pop("status_callback", None),
        )
        async for event in executor.execute_stream(goal):
            yield event

    # =========================================================================
    # PIPELINE MODE — Multi-stage orchestration with learning
    # =========================================================================

    async def _run_pipeline(
        self,
        goal: str,
        stages: List[Dict[str, Any]],
        status_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> EpisodeResult:
        """
        Execute a multi-stage pipeline with integrated learning.

        Each stage can contain a swarm, an agent, or a callable.
        Learning wraps every stage: pre-stage guidance, post-stage recording,
        and inter-stage reflection when failures occur.

        Args:
            goal: Overall pipeline goal
            stages: List of stage definitions, each a dict with:
                - name (str): Stage identifier
                - swarm (optional): SwarmTemplate instance or class
                - agent (optional): BaseAgent instance
                - callable (optional): Async callable(context) -> result
                - depends_on (optional): List[str] of stage names to wait for
                - config (optional): Dict of stage-specific config
            status_callback: Optional progress callback(stage_name, detail)

        Returns:
            EpisodeResult with aggregated pipeline results

        Example:
            result = await orchestrator.run(
                "Build and test a REST API",
                stages=[
                    {"name": "design", "swarm": CodingSwarm(config), "config": {"task": "Design API"}},
                    {"name": "test", "swarm": TestingSwarm(config), "depends_on": ["design"]},
                    {"name": "review", "agent": SecurityReviewer(), "depends_on": ["design"]},
                ],
            )
        """
        import time as _time

        from Jotty.core.intelligence.learning.learning_service import LearningService

        learning = LearningService.get_instance()
        pipeline_start = _time.time()

        # Classify the pipeline goal to get actual domain (not config default)
        from Jotty.core.intelligence.learning.learning_service import classify_domain

        pipeline_domain, pipeline_task_type = classify_domain(goal)

        # Start a pipeline-level episode
        pipeline_episode = learning.start_episode(
            unit_name="Orchestrator",
            unit_type="pipeline",
            domain=pipeline_domain,
            task_type=pipeline_task_type,
            context={"goal": goal[:500], "stages": [s["name"] for s in stages]},
        )

        stage_results: Dict[str, Any] = {}
        stage_outputs: Dict[str, Any] = {}
        all_success = True
        total_cost = 0.0

        for i, stage_def in enumerate(stages):
            stage_name = stage_def.get("name", f"stage_{i}")
            depends_on = stage_def.get("depends_on", [])

            # Check dependencies
            for dep in depends_on:
                if dep not in stage_results:
                    logger.warning(f"Pipeline stage '{stage_name}': dependency '{dep}' not found")
                elif not stage_results[dep].get("success", False):
                    # Reflect: dependency failed
                    reflection = learning.reflect(
                        episode_id=pipeline_episode,
                        step=i,
                        observation=f"Dependency '{dep}' failed for stage '{stage_name}'",
                        unit_name="Orchestrator",
                    )
                    logger.info(f"Pipeline reflection: {reflection.get('adjustment', '')}")

            if status_callback:
                status_callback(stage_name, f"Starting stage {i+1}/{len(stages)}")

            # Build context from previous stages (generous limit for weaker models)
            stage_context = {"goal": goal, "stage": stage_name, "previous_outputs": {}}
            for dep in depends_on:
                if dep in stage_outputs:
                    stage_context["previous_outputs"][dep] = str(stage_outputs[dep])[:8000]

            # Query learning for THIS DOMAIN (not stage name) so we get real data
            stage_task_type = stage_def.get("task_type", pipeline_task_type)
            guidance = learning.query(
                domain=pipeline_domain,
                task_type=stage_task_type,
                context=stage_context,
            )
            if guidance.get("recommendations"):
                stage_context["learning_recommendations"] = guidance["recommendations"]

            # Record stage under pipeline domain so learning accumulates properly
            stage_episode = learning.start_episode(
                unit_name=stage_name,
                unit_type="pipeline_stage",
                domain=pipeline_domain,
                task_type=stage_task_type,
                context=stage_context,
                parent_episode_id=pipeline_episode,
            )

            stage_start = _time.time()
            stage_success = False
            stage_output = None
            stage_error = None

            try:
                # Execute the stage
                if "swarm" in stage_def:
                    swarm = stage_def["swarm"]
                    if isinstance(swarm, type):
                        swarm_config = stage_def.get("config", {})
                        swarm = swarm(swarm_config)
                    task = stage_def.get("task", goal)
                    if depends_on and stage_outputs:
                        context_str = "\n".join(
                            f"[{dep}]: {str(stage_outputs.get(dep, ''))[:4000]}"
                            for dep in depends_on
                            if dep in stage_outputs
                        )
                        task = f"{context_str}\n\n{task}"
                    stage_output = await swarm.execute(task)
                    stage_success = getattr(stage_output, "success", True)

                elif "agent" in stage_def:
                    agent = stage_def["agent"]
                    task = stage_def.get("task", goal)
                    stage_output = await agent.execute(task=task, **stage_context)
                    stage_success = getattr(stage_output, "success", True)

                elif "callable" in stage_def:
                    fn = stage_def["callable"]
                    stage_output = await fn(stage_context)
                    stage_success = True

                else:
                    logger.warning(f"Stage '{stage_name}': no swarm, agent, or callable defined")
                    stage_success = False
                    stage_error = "No execution unit defined"

            except Exception as e:
                logger.error(f"Pipeline stage '{stage_name}' failed: {e}")
                stage_success = False
                stage_error = str(e)

                # Mid-pipeline reflection
                reflection = learning.reflect(
                    episode_id=pipeline_episode,
                    step=i,
                    observation=f"Stage '{stage_name}' failed: {str(e)[:200]}",
                    unit_name="Orchestrator",
                    analysis=f"Error type: {type(e).__name__}",
                )
                logger.info(f"Pipeline reflection: {reflection.get('adjustment', '')}")

            stage_time = _time.time() - stage_start

            # Measure actual stage output quality (not hardcoded)
            stage_quality = 0.1
            stage_output_str = str(stage_output) if stage_output else ""
            if stage_success and stage_output_str:
                from Jotty.core.intelligence.learning.learning_service import (
                    analyze_response,
                )

                stage_task = stage_def.get("task", goal)
                stage_analysis = analyze_response(stage_output_str, stage_task)
                stage_quality = stage_analysis.get("quality_score", 0.5)

            # Record stage result
            stage_results[stage_name] = {
                "success": stage_success,
                "time": stage_time,
                "output": stage_output,
                "error": stage_error,
                "quality": stage_quality,
            }
            if stage_output is not None:
                stage_outputs[stage_name] = stage_output

            if not stage_success:
                all_success = False

            # End stage episode with measured quality
            learning.end_episode(
                episode_id=stage_episode,
                success=stage_success,
                quality=stage_quality,
                cost=getattr(stage_output, "cost", 0.0) if stage_output else 0.0,
                outcome={
                    "output_length": len(stage_output_str),
                    "quality": round(stage_quality, 3),
                    "output": stage_output_str[:500],
                },
                error_message=stage_error,
            )

            if status_callback:
                status = "completed" if stage_success else "failed"
                status_callback(stage_name, f"Stage {status} ({stage_time:.1f}s)")

        # End pipeline episode with aggregated quality from actual stage measurements
        pipeline_time = _time.time() - pipeline_start
        stage_qualities = [r["quality"] for r in stage_results.values()]
        pipeline_quality = sum(stage_qualities) / max(len(stage_qualities), 1)
        learning.end_episode(
            episode_id=pipeline_episode,
            success=all_success,
            quality=pipeline_quality,
            cost=total_cost,
            outcome={
                "stages_completed": sum(1 for r in stage_results.values() if r["success"]),
                "stages_total": len(stages),
                "stage_qualities": {
                    name: round(r["quality"], 3) for name, r in stage_results.items()
                },
                "pipeline_quality": round(pipeline_quality, 3),
            },
        )

        # Build final EpisodeResult
        last_stage = stages[-1]["name"] if stages else ""
        final_output = stage_outputs.get(last_stage, "")

        # Concatenate all stage outputs for a complete result
        all_outputs = []
        for sname in [s["name"] for s in stages]:
            if sname in stage_outputs:
                all_outputs.append(f"=== {sname.upper()} ===\n{str(stage_outputs[sname])}")
        combined_output = "\n\n".join(all_outputs) if all_outputs else str(final_output)

        return EpisodeResult(
            success=all_success,
            output=combined_output[:50000] if combined_output else "",
            trajectory=[],
            tagged_outputs=[],
            episode=0,
            execution_time=pipeline_time,
            architect_results=[],
            auditor_results=[],
            agent_contributions={},
            override_metadata={
                "pipeline": True,
                "pipeline_quality": round(pipeline_quality, 3),
                "stages": {
                    name: {
                        "success": r["success"],
                        "time": r["time"],
                        "quality": round(r["quality"], 3),
                        "output_length": len(str(r.get("output", ""))),
                    }
                    for name, r in stage_results.items()
                },
                "total_cost": total_cost,
            },
        )

    # Backward-compat alias: run_pipeline() → run(goal, stages=...)
    async def run_pipeline(
        self,
        goal: str,
        stages: List[Dict[str, Any]],
        status_callback: Optional[Callable] = None,
    ) -> EpisodeResult:
        """Backward-compat alias. Use run(goal, stages=...) instead."""
        return await self.run(goal, stages=stages, status_callback=status_callback)

    # _execute_ensemble and _should_auto_ensemble — delegated to EnsembleManager

    async def _execute_single_agent(self, goal: str, **kwargs: Any) -> Any:
        return await self._ensure_engine()._execute_single_agent(goal, **kwargs)

    async def _execute_multi_agent(self, goal: str, **kwargs: Any) -> Any:
        return await self._ensure_engine()._execute_multi_agent(goal, **kwargs)

    async def _paradigm_run_agent(self, *args: Any, **kwargs: Any) -> Any:
        return await self._ensure_engine()._paradigm_run_agent(*args, **kwargs)

    async def _paradigm_relay(self, goal: str, **kwargs: Any) -> Any:
        return await self._ensure_engine()._paradigm_relay(goal, **kwargs)

    async def _paradigm_debate(self, goal: str, **kwargs: Any) -> Any:
        return await self._ensure_engine()._paradigm_debate(goal, **kwargs)

    async def _paradigm_refinement(self, goal: str, **kwargs: Any) -> Any:
        return await self._ensure_engine()._paradigm_refinement(goal, **kwargs)

    def _aggregate_results(self, results: Any, goal: str) -> Any:
        return self._ensure_engine()._aggregate_results(results, goal)

    def _assign_cooperative_credit(self, results: Any, goal: str) -> Any:
        return self._ensure_engine()._assign_cooperative_credit(results, goal)

    def _create_zero_config_agents(self, task: Any, status_callback: Any = None) -> Any:
        return self._agent_factory.create_zero_config_agents(task, status_callback)

    # _should_auto_ensemble — see _ensemble_mixin.py

    # =========================================================================
    # RECOVERY — ClawWork-inspired wrap-up on failure
    # =========================================================================

    def _attempt_recovery(self, result: EpisodeResult, goal: str) -> EpisodeResult:
        """
        Attempt to salvage partial results from a failed execution.

        Inspired by ClawWork's wrap-up workflow: when an agent fails to
        complete within its iteration budget, a secondary pass extracts
        whatever partial work was produced (trajectory outputs, tagged
        outputs) and surfaces it as a degraded-but-usable result instead
        of a hard failure.

        This method is synchronous and cheap — no LLM calls. It inspects
        the trajectory and tagged_outputs for any salvageable content.

        Args:
            result: The failed EpisodeResult
            goal: Original goal string

        Returns:
            The same EpisodeResult, possibly with output updated to include
            salvaged partial content. success remains False.
        """
        try:
            existing_output = str(result.output or "")

            # Already has substantial output — nothing to recover
            if len(existing_output) > 100:
                return result

            # Collect partial outputs from trajectory steps
            partial_fragments = []
            for step in result.trajectory or []:
                if not isinstance(step, dict):
                    continue
                for key in ("output", "result", "content"):
                    fragment = step.get(key)
                    if fragment and len(str(fragment)) > 20:
                        partial_fragments.append(str(fragment))

            # Collect from tagged outputs
            for tagged in result.tagged_outputs or []:
                content = getattr(tagged, "content", None)
                if content and len(str(content)) > 20:
                    partial_fragments.append(str(content))

            if not partial_fragments:
                return result

            # Build salvaged output
            salvaged = "[Partial result — execution did not fully succeed]\n\n"
            salvaged += "\n---\n".join(partial_fragments[:5])  # Cap at 5 fragments

            # Update output with salvaged content
            if existing_output:
                result.output = existing_output + "\n\n" + salvaged
            else:
                result.output = salvaged

            result.alerts = list(result.alerts or [])
            result.alerts.append(f"Recovery: salvaged {len(partial_fragments)} partial fragments")

            logger.info(
                f"Recovery salvaged {len(partial_fragments)} fragments "
                f"from failed execution of: {goal[:80]}"
            )
        except Exception as e:
            logger.debug(f"Recovery attempt failed (non-fatal): {e}")

        return result

    def _post_episode_learning(self, result: EpisodeResult, goal: str) -> Any:
        """Delegate to SwarmLearningPipeline + update intelligence metrics."""
        self.learning.post_episode(
            result=result,
            goal=goal,
            agents=self.agents,
            architect_prompts=self.architect_prompts,
            mas_learning=getattr(self, "mas_learning", None),
            swarm_terminal=getattr(self, "swarm_terminal", None),
        )
        self.episode_count = self.learning.episode_count

        # Track intelligence A/B effectiveness (per task_type)
        if result.success:
            _guided = getattr(self, "_last_run_guided", False)
            _tt = getattr(self, "_last_task_type", "_global")
            for _bucket_key in (_tt, "_global") if _tt != "_global" else ("_global",):
                bucket = self._intelligence_metrics.get(_bucket_key)
                if bucket:
                    if _guided:
                        bucket["guided_successes"] += 1
                    else:
                        bucket["unguided_successes"] += 1

        # Track paradigm effectiveness (for auto paradigm selection)
        paradigm = getattr(self, "_last_paradigm", None)
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

    def _schedule_background_learning(self, result: EpisodeResult, goal: str) -> Any:
        """
        Fire-and-forget: run post-episode learning + auto-save in a background task.

        Users get their result immediately. Learning/saving happens concurrently.
        If the event loop shuts down before completion, learnings are best-effort
        (next successful run will re-save).
        """

        async def _background() -> Any:
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
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = set()
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No running event loop — fall back to synchronous
            self._post_episode_learning(result, goal)
            self._auto_save_learnings()

    async def _drain_background_tasks(self, timeout: float = 10.0) -> Any:
        """Await all pending background learning tasks (with timeout)."""
        tasks = getattr(self, "_background_tasks", set())  # type: ignore[var-annotated]
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

    def _log_execution_summary(self, result: EpisodeResult) -> Any:
        """Log a user-friendly summary with artifacts after execution.

        Uses the OUTER result.success (which includes auditor verdict)
        to avoid contradictions like "completed successfully" + SUCCESS: False.
        """
        try:
            _output = getattr(result, "output", None)
            outer_success = result.success

            if hasattr(_output, "artifacts") or hasattr(_output, "skills_used"):
                # Build our own summary using the outer success status
                parts = []
                status = "completed successfully" if outer_success else "failed (auditor rejected)"
                exec_time = getattr(_output, "execution_time", 0) or 0
                steps = getattr(_output, "steps_executed", 0) or 0
                parts.append(f"Task {status} in {exec_time:.1f}s ({steps} steps)")

                skills = getattr(_output, "skills_used", [])
                if skills:
                    parts.append(f"Skills used: {', '.join(skills)}")

                if hasattr(_output, "artifacts"):
                    artifacts = _output.artifacts  # type: ignore[union-attr]
                    if artifacts:
                        parts.append("Files created:")
                        for a in artifacts:
                            size = f" ({a['size_bytes']} bytes)" if a.get("size_bytes") else ""
                            parts.append(f"  → {a['path']}{size}")

                errors = getattr(_output, "errors", [])
                if errors:
                    parts.append(f"Errors: {'; '.join(str(e) for e in errors[:3])}")

                logger.info("\n Execution Summary:\n" + "\n".join(parts))
            elif hasattr(_output, "summary"):
                logger.info(f"\n Execution Summary:\n{_output.summary}")  # type: ignore[union-attr]
        except Exception as e:
            logger.debug(f"Summary logging skipped: {e}")

    def _learn_from_result(
        self, result: EpisodeResult, agent_config: AgentConfig, goal: str = ""
    ) -> Any:
        """Delegate to SwarmLearningPipeline."""
        self.learning.learn_from_result(
            result=result,
            agent_config=agent_config,
            goal=goal,
        )

    async def autonomous_setup(self, goal: str, status_callback: Any = None) -> Any:
        return await self._engine.autonomous_setup(goal, status_callback)

    # =====================================================================
    # State Management Methods (V1 capabilities integrated)
    # =====================================================================

    # State delegation — use self.swarm_state_manager directly.
    # Kept get_current_state as it's used internally by _execute_multi_agent.
    def get_current_state(self) -> Dict[str, Any]:
        """Get current swarm-level state."""
        if not self.swarm_state_manager:
            return {}
        return self.swarm_state_manager.get_current_state()  # type: ignore[no-any-return]

    # =====================================================================
    # Warmup — delegated to SwarmWarmup
    # =====================================================================

    def _ensure_warmup(self) -> Any:
        """Lazy-load SwarmWarmup instance."""
        if not hasattr(self, "_warmup") or self._warmup is None:  # type: ignore[has-type]
            from Jotty.core.intelligence.orchestration.execution.swarm_warmup import SwarmWarmup

            self._warmup = SwarmWarmup(self)
        return self._warmup

    async def warmup(self, **kwargs: Any) -> Dict[str, Any]:
        """DrZero-inspired zero-data bootstrapping. See SwarmWarmup."""
        return await self._ensure_warmup().warmup(**kwargs)  # type: ignore[no-any-return]

    def get_warmup_recommendation(self) -> Dict[str, Any]:
        """Check if warmup would be beneficial."""
        return self._ensure_warmup().get_recommendation()  # type: ignore[no-any-return]

    # =====================================================================
    # DAG — delegated to SwarmDAGExecutor
    # =====================================================================

    def _ensure_dag_executor(self) -> Any:
        """Lazy-load SwarmDAGExecutor instance."""
        if not hasattr(self, "_dag_executor") or self._dag_executor is None:  # type: ignore[has-type]
            from Jotty.core.intelligence.orchestration.execution.swarm_dag_executor import (
                SwarmDAGExecutor,
            )

            self._dag_executor = SwarmDAGExecutor(self)
        return self._dag_executor

    async def run_with_dag(self, implementation_plan: str, **kwargs: Any) -> EpisodeResult:
        """Execute via DAG-based orchestration. See SwarmDAGExecutor."""
        return await self._ensure_dag_executor().run(implementation_plan, **kwargs)

    def get_dag_agents(self) -> Any:
        """Get DAG agents for external use."""
        return self._ensure_dag_executor().get_agents()

    # =========================================================================
    # Self-improvement — delegated to TrainingDaemon
    # =========================================================================

    async def run_training_task(self) -> Any:
        return await self._training.run_training_task()

    @property
    def pending_training_tasks(self) -> Any:
        return self._training.pending_count

    async def start_training_loop(self, **kwargs: Any) -> Any:
        return await self._training.start_training_loop(**kwargs)

    def start_training_daemon(self, **kwargs: Any) -> Any:
        return self._training.start(**kwargs)

    def stop_training_daemon(self) -> Any:
        return self._training.stop()

    def training_daemon_status(self) -> Any:
        return self._training.status()
