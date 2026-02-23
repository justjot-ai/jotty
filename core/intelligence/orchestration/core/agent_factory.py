"""
Agent Factory
=============

Creates and manages AgentRunners and LOTUS optimization for the Orchestrator.

Responsibilities:
    - Building AgentRunners from AgentConfig
    - Registering agents with Axon (inter-agent communication)
    - Zero-config agent creation (natural language -> agents)
    - LOTUS optimization initialization and stats

Extracted from swarm_manager.py for maintainability.
"""

# mypy: disable-error-code="has-type"
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .swarm_manager import Orchestrator

from Jotty.core.infrastructure.foundation.agent_config import (
    AgentConfig,  # type: ignore[import-not-found, import]
)
from Jotty.core.infrastructure.foundation.exceptions import (  # type: ignore[import-not-found]
    LearningError,
)

from ..execution.agent_runner import AgentRunner, AgentRunnerConfig
from .lazy_components import _load_providers

# Optional feedback channel imports
try:
    from Jotty.core.intelligence.reasoning.tools.feedback_channel import (  # type: ignore[import-not-found]
        FeedbackMessage,
        FeedbackType,
    )
except ImportError:
    FeedbackMessage = None  # type: ignore
    FeedbackType = None  # type: ignore

logger = logging.getLogger(__name__)


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
                # Include Q-predictor so runners can feed experiences
                if sm.q_predictor is not None:
                    components["q_predictor"] = sm.q_predictor
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
