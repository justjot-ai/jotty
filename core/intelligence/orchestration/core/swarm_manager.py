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

Sub-modules (extracted for maintainability):
    lazy_components.py  — Factory functions, LiteLLM filter, SessionLockManager
    agent_factory.py    — AgentFactory (runner creation, LOTUS, Axon registration)
    execution_engine.py — ExecutionEngine (run, single/multi-agent, paradigms)

Usage:
    sm = Orchestrator()  # Fast: ~10ms
    result = await sm.run("Research AI trends")  # Components init on demand
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from Jotty.core.infrastructure.foundation.agent_config import (
    AgentConfig,  # type: ignore[import-not-found, import]
)
from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult, SwarmConfig
from Jotty.core.infrastructure.foundation.exceptions import (  # type: ignore[import-not-found]
    AgentExecutionError,
    LLMError,
)
from Jotty.core.infrastructure.utils.async_utils import (  # type: ignore[import-not-found, import]
    StatusReporter,
)
from Jotty.core.intelligence.reasoning.agents.auto_agent import (
    AutoAgent,  # type: ignore[import-not-found]
)

from ..coordination.ensemble_manager import EnsembleManager
from ..coordination.mas_zero_controller import MASZeroController
from ..execution.agent_runner import AgentRunner
from ..learning.learning_delegate import LearningDelegate
from ..learning.training_daemon import TrainingDaemon
from ..routing.model_tier_router import ModelTierRouter

# Composed managers (has-a, not is-a) — replaces mixin inheritance
from ..routing.provider_manager import ProviderManager
from ..routing.swarm_router import SwarmRouter
from ._lazy import LazyComponent

# Extracted sub-modules
from .agent_factory import AgentFactory
from .execution_engine import ExecutionEngine, _build_response_digest
from .lazy_components import (
    _create_code_generator,
    _create_context_guard,
    _create_data_registry,
    _create_installer,
    _create_intent_parser,
    _create_io_manager,
    _create_learning_pipeline,
    _create_mas_learning,
    _create_memory,
    _create_planner,
    _create_profiler,
    _create_provider_gateway,
    _create_researcher,
    _create_shared_context,
    _create_state_manager,
    _create_task_board,
    _create_terminal,
    _create_tool_registry,
    _create_tool_validator,
    _create_ui_registry,
    _SessionLockManager,
)

# Optional observability imports
try:
    from Jotty.core.infrastructure.monitoring.observability import (  # type: ignore[import-not-found, import]
        get_metrics,
        get_tracer,
    )
except ImportError:
    get_metrics = None  # type: ignore
    get_tracer = None  # type: ignore

logger = logging.getLogger(__name__)


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

    # Q-learning predictor (lazy, non-fatal)
    _Q_UNSET = object()  # sentinel: "not yet loaded"
    _q_predictor: Any

    @property
    def q_predictor(self) -> Any:
        """Lazy LLMQPredictor for ε-greedy task selection."""
        if self._q_predictor is self._Q_UNSET:
            try:
                from Jotty.core.intelligence.learning.q_learning import LLMQPredictor

                self._q_predictor = LLMQPredictor(self.config)
            except Exception:
                self._q_predictor = None
        return self._q_predictor

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
        self._q_predictor: Any = Orchestrator._Q_UNSET  # Lazy-created by q_predictor property

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
        trace: bool = False,
        report_dir: str = "reports",
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
            trace: If True, capture full execution trace and save a markdown
                   planning document to report_dir.
            report_dir: Directory for trace reports (default: "reports")
            status_callback: Optional progress callback(stage, detail)
            **kwargs: Additional arguments passed to the execution engine

        Returns:
            ExecutionResult (or AsyncIterator[StreamEvent] if stream=True).
            When trace=True, result has .trace_report_path attribute set.

        Examples:
            # Auto-detect with learning (default)
            result = await orchestrator.run("What is GDP?")

            # With tracing — generates a markdown report
            result = await orchestrator.run("Research AI", trace=True)
            print(result.trace_report_path)  # reports/20260222_..._research_ai.md

            # Streaming (await first, then iterate)
            stream = await orchestrator.run("Research AI", stream=True)
            async for event in stream:
                print(event)
        """

        # ── TRACING: wrap status_callback to capture all events ──
        _tracer_inst = None
        if trace and not stream:
            from Jotty.core.infrastructure.monitoring.execution_tracer import ExecutionTracer

            _tracer_inst = ExecutionTracer()
            _tracer_inst.goal = goal
            _tracer_inst.mode = "run"
            _tracer_inst._user_callback = status_callback
            _tracer_inst.take_pre_snapshot()
            status_callback = _tracer_inst.callback

        # Lane queue: serialize requests for the same session
        session_id = kwargs.get("session_id", "")
        if session_id:
            lock = await self._session_locks.get_lock(session_id)
            result = await self._run_with_lock(
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
        else:
            result = await self._run_inner(
                goal,
                stream=stream,
                stages=stages,
                swarm=swarm,
                agent=agent,
                learn=learn,
                status_callback=status_callback,
                **kwargs,
            )

        # ── TRACING: generate and save report ──
        if _tracer_inst is not None and result is not None:
            _tracer_inst.take_post_snapshot()
            from datetime import datetime as _dt

            report = _tracer_inst.generate_report(result, goal=goal, mode="run")
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            slug = goal[:40].lower().replace(" ", "_").replace("/", "_")
            report_path = _tracer_inst.save_report(report, f"{report_dir}/{ts}_{slug}.md")
            try:
                result.trace_report_path = report_path
            except (AttributeError, TypeError):
                pass
            logger.info(f"Execution trace saved: {report_path}")

        return result

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
                    domain=detected_domain, task_type=detected_task_type, goal=goal
                )
                if guidance_str:
                    ctx_parts.append(guidance_str)
                if detected_domain != "general":
                    general_str = learning.build_context_string(
                        domain="general", task_type="run", goal=goal
                    )
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
            _raw_output = None
            if isinstance(result, EpisodeResult):
                _raw_output = getattr(result, "output", None)
                if isinstance(_raw_output, str):
                    result_text = _raw_output
                elif _raw_output is not None:
                    # output may be AgenticExecutionResult — extract text from it
                    for _attr in ("final_output", "output", "content"):
                        _val = getattr(_raw_output, _attr, None)
                        if isinstance(_val, str) and len(_val) > len(result_text):
                            result_text = _val
                        elif isinstance(_val, dict):
                            _text = (
                                _val.get("response", "")
                                or _val.get("text", "")
                                or _val.get("content", "")
                                or _val.get("output", "")
                            )
                            if isinstance(_text, str) and len(_text) > len(result_text):
                                result_text = _text
                    _outputs = getattr(_raw_output, "outputs", None)
                    if isinstance(_outputs, dict) and not result_text:
                        for _v in _outputs.values():
                            if isinstance(_v, dict):
                                _txt = (
                                    _v.get("response", "")
                                    or _v.get("text", "")
                                    or _v.get("content", "")
                                )
                                if isinstance(_txt, str) and len(_txt) > len(result_text):
                                    result_text = _txt
                            elif isinstance(_v, str) and len(_v) > len(result_text):
                                result_text = _v
            elif result:
                for _attr in ("final_output", "output", "content"):
                    _val = getattr(result, _attr, None)
                    if _val is None:
                        continue
                    if isinstance(_val, str):
                        if len(_val) > len(result_text):
                            result_text = _val
                    elif isinstance(_val, dict):
                        _text = (
                            _val.get("response", "")
                            or _val.get("text", "")
                            or _val.get("content", "")
                            or _val.get("output", "")
                        )
                        if isinstance(_text, str) and len(_text) > len(result_text):
                            result_text = _text
                        elif not result_text:
                            result_text = json.dumps(_val, default=str)[:10000]
                    elif isinstance(_val, list):
                        # Concatenate list items
                        _parts = [str(v) for v in _val if v]
                        _text = "\n".join(_parts)
                        if len(_text) > len(result_text):
                            result_text = _text
                # Also check outputs dict (common in AgenticExecutionResult)
                _outputs = getattr(result, "outputs", None)
                if isinstance(_outputs, dict) and not result_text:
                    for _v in _outputs.values():
                        if isinstance(_v, str) and len(_v) > len(result_text):
                            result_text = _v
            if not isinstance(result_text, str):
                result_text = str(result_text) if result_text else ""

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
                excerpt = (
                    result_text[:600].rsplit("\n", 1)[0] if len(result_text) > 600 else result_text
                )
                outcome["response_excerpt"] = excerpt
                # Store actual response content for few-shot retrieval.
                # Retrieval depends on this key being present and non-trivial.
                outcome["content"] = result_text[:2000]

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
        trace: bool = False,
        report_dir: str = "reports",
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
            trace: If True, capture full execution trace and save a markdown
                   planning document to report_dir.
            report_dir: Directory for trace reports (default: "reports")
            provider: LLM provider ('anthropic', 'openai', etc.). Auto-detects if None.
            model: Model name (uses provider default if not specified)
            status_callback: Progress callback(stage, detail)
            stream_callback: Token-level streaming callback(chunk)
            enabled_tools: Only enable these tools (None = all)
            output_format: Force output format ('auto', 'pdf', 'docx', etc.)
            max_steps: Maximum tool-calling iterations
            **kwargs: Additional arguments

        Returns:
            LLMExecutionResult (or AsyncIterator[StreamEvent] if stream=True).
            When trace=True, result has .trace_report_path attribute set.

        Examples:
            # Simple chat with tracing
            result = await orchestrator.chat("Hello!", trace=True)
            print(result.trace_report_path)

            # Streaming (await first, then iterate)
            stream = await orchestrator.chat("Explain quantum physics", stream=True)
            async for event in stream:
                print(event)
        """
        # ── TRACING: wrap status_callback ──
        _tracer_inst = None
        if trace and not stream:
            from Jotty.core.infrastructure.monitoring.execution_tracer import ExecutionTracer

            _tracer_inst = ExecutionTracer()
            _tracer_inst.goal = message
            _tracer_inst.mode = "chat"
            _tracer_inst._user_callback = status_callback
            _tracer_inst.take_pre_snapshot()
            status_callback = _tracer_inst.callback

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
                    domain=detected_domain, task_type=detected_task_type, goal=message
                )
                if domain_ctx:
                    ctx_parts.append(domain_ctx)

                # 2. General guidance (if different domain)
                if detected_domain != "general":
                    general_ctx = learning.build_context_string(
                        domain="general", task_type="run", goal=message
                    )
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
                    # Store actual content for semantic retrieval (mem0-style)
                    outcome["content"] = content[:1500]

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

        # ── TRACING: generate and save report ──
        if _tracer_inst is not None:
            _tracer_inst.take_post_snapshot()
            from datetime import datetime as _dt

            report = _tracer_inst.generate_report(result, goal=message, mode="chat")
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            slug = message[:40].lower().replace(" ", "_").replace("/", "_")
            report_path = _tracer_inst.save_report(report, f"{report_dir}/{ts}_chat_{slug}.md")
            try:
                result.trace_report_path = report_path
            except (AttributeError, TypeError):
                pass
            logger.info(f"Chat trace saved: {report_path}")

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

        # Strip orchestrator-internal kwargs that swarm templates don't accept
        swarm_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("learning_context", "status_callback", "session_id")
        }
        return await swarm.execute(task=goal, **swarm_kwargs)

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
