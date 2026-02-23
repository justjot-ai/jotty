"""
Lazy Component Factory Functions
================================

Module-level constants and factory functions used by LazyComponent
descriptors in the Orchestrator class.

All _create_*() functions are called lazily on first access.
They import heavyweight modules inside the function body to avoid
loading them at import time.

Extracted from swarm_manager.py for maintainability.
"""

from __future__ import annotations

import logging
import time as _time_module
from typing import TYPE_CHECKING, Any, Dict, Optional

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

from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

logger = logging.getLogger(__name__)


# =============================================================================
# LiteLLM NOISE FILTER
# =============================================================================


# Suppress noisy LiteLLM CancelledError on asyncio loop shutdown.
# This is a known issue: LiteLLM's background LoggingWorker gets cancelled
# when asyncio.run() tears down, producing harmless but alarming tracebacks.
class _LiteLLMCancelledFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "CancelledError" not in msg and "LoggingWorker cancelled" not in msg


logging.getLogger("LiteLLM").addFilter(_LiteLLMCancelledFilter())


# =============================================================================
# LANE QUEUE - Per-session serialization with TTL cleanup
# =============================================================================

import asyncio


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


# =============================================================================
# SKILL PROVIDER SYSTEM - Lazy loaded via cache dict
# =============================================================================

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


def _create_state_manager(sm: "Orchestrator") -> "SwarmStateManager":  # type: ignore[name-defined]
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


def _create_mas_learning(sm: "Orchestrator") -> "MASLearning":  # type: ignore[name-defined]
    from Jotty.core.intelligence.orchestration.learning.mas_learning import MASLearning

    workspace_path = getattr(sm.config, "base_path", None)
    return MASLearning(
        config=sm.config,
        workspace_path=workspace_path,
        swarm_intelligence=sm.swarm_intelligence,
        transfer_learning=sm.transfer_learning,
    )
