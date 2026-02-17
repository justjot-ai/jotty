"""
Orchestration & Intelligence Subsystem Facade
===============================================

Clean, discoverable API for orchestration components that are normally
hidden inside the Orchestrator's lazy descriptors.

No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.intelligence.orchestration.facade import get_swarm_intelligence, list_components

    si = get_swarm_intelligence()
    components = list_components()
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

if TYPE_CHECKING:
    from Jotty.core.infrastructure.foundation.data_structures import (
        SwarmConfig,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.coordination.ensemble_manager import (
        EnsembleManager,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
        ParadigmExecutor,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
        SwarmIntelligence,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.learning.training_daemon import (
        TrainingDaemon,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.routing.model_tier_router import (
        ModelTierRouter,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.routing.provider_manager import (
        ProviderManager,  # type: ignore[import]
    )
    from Jotty.core.intelligence.orchestration.routing.swarm_router import (
        SwarmRouter,  # type: ignore[import]
    )

_lock = threading.Lock()
_singletons: Dict[str, object] = {}


def get_swarm_intelligence(config: Any = None) -> "SwarmIntelligence":
    """
    Return a SwarmIntelligence singleton for multi-agent coordination.

    Thread-safe with double-checked locking. Parameterized configs
    bypass the cache and return fresh instances.

    Args:
        config: Optional SwarmConfig.

    Returns:
        SwarmIntelligence instance.
    """
    if config is not None:
        from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
            SwarmIntelligence,
        )

        return SwarmIntelligence(config=config)
    key = "swarm_intelligence"
    if key not in _singletons:
        with _lock:
            if key not in _singletons:
                from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
                    SwarmIntelligence,
                )

                _singletons[key] = SwarmIntelligence(config=config)
    return _singletons[key]


def get_paradigm_executor(
    manager: Any = None,
) -> Union["ParadigmExecutor", Type["ParadigmExecutor"]]:
    """
    Return a ParadigmExecutor for discussion paradigms (relay, debate, refinement).

    Args:
        manager: Orchestrator instance. If None, returns the class for manual instantiation.

    Returns:
        ParadigmExecutor instance (if manager provided) or class (if not).
    """
    from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
        ParadigmExecutor,
    )

    if manager is not None:
        return ParadigmExecutor(manager)
    return ParadigmExecutor


def get_training_daemon(manager: Any = None) -> Union["TrainingDaemon", Type["TrainingDaemon"]]:
    """
    Return a TrainingDaemon for background self-improvement.

    Args:
        manager: Orchestrator instance. If None, returns the class for manual instantiation.

    Returns:
        TrainingDaemon instance (if manager provided) or class (if not).
    """
    from Jotty.core.intelligence.orchestration.learning.training_daemon import TrainingDaemon

    if manager is not None:
        return TrainingDaemon(manager)
    return TrainingDaemon


def get_ensemble_manager() -> "EnsembleManager":
    """
    Return an EnsembleManager singleton for prompt ensembling.

    Thread-safe with double-checked locking.

    Returns:
        EnsembleManager instance (stateless).
    """
    key = "ensemble_manager"
    if key not in _singletons:
        with _lock:
            if key not in _singletons:
                from Jotty.core.intelligence.orchestration.coordination.ensemble_manager import (
                    EnsembleManager,
                )

                _singletons[key] = EnsembleManager()
    return _singletons[key]


def get_provider_manager(config: Optional["SwarmConfig"] = None) -> "ProviderManager":
    """
    Return a ProviderManager for skill provider registry management.

    Args:
        config: Optional SwarmConfig.

    Returns:
        ProviderManager instance.
    """
    from Jotty.core.intelligence.orchestration.routing.provider_manager import ProviderManager

    return ProviderManager(
        config=config,
        get_swarm_intelligence=lambda: None,
    )


def get_model_tier_router(default_provider: str | None = None) -> "ModelTierRouter":
    """
    Return a ModelTierRouter instance for complexity-based LLM model selection.

    Args:
        default_provider: Optional default LLM provider name.

    Returns:
        ModelTierRouter instance.
    """
    from Jotty.core.intelligence.orchestration.routing.model_tier_router import ModelTierRouter

    return ModelTierRouter(default_provider=default_provider)


def get_swarm_router() -> "SwarmRouter":
    """
    Return a SwarmRouter singleton for centralized task routing.

    Thread-safe with double-checked locking.

    Returns:
        SwarmRouter instance.
    """
    key = "swarm_router"
    if key not in _singletons:
        with _lock:
            if key not in _singletons:
                from Jotty.core.intelligence.orchestration.routing.swarm_router import SwarmRouter

                _singletons[key] = SwarmRouter()
    return _singletons[key]


def list_components() -> Dict[str, str]:
    """
    List all orchestration subsystem components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "Orchestrator": "Main composable swarm orchestrator with lazy initialization",
        "SwarmIntelligence": "Emergent specialization, consensus voting, RL-informed routing",
        "ParadigmExecutor": "MALLM discussion paradigms: relay, debate, refinement",
        "TrainingDaemon": "Background self-improvement loop with curriculum generation",
        "EnsembleManager": "Prompt ensembling for multi-perspective analysis",
        "ProviderManager": "Skill provider registry (browser-use, openhands, etc.)",
        "ModelTierRouter": "Maps task complexity to LLM model tiers (cheap/balanced/quality)",
        "SwarmRouter": "Centralized task routing and agent selection",
        "SwarmTaskBoard": "Task decomposition and subtask state tracking",
        "AgentRunner": "Per-agent execution wrapper with retries and metrics",
        "SandboxManager": "Sandboxed code execution with trust levels",
        "ChatExecutor": "Direct LLM tool-calling executor (fast path)",
        "SwarmLearningPipeline": "Combined TD(lambda) + credit + memory pipeline",
        "MASZeroController": "MAS-ZERO: parallel strategies, meta-feedback, verification",
    }
