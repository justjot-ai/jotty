"""
Orchestration & Intelligence Subsystem Facade
===============================================

Clean, discoverable API for orchestration components that are normally
hidden inside the Orchestrator's lazy descriptors.

No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.orchestration.facade import get_swarm_intelligence, list_components

    si = get_swarm_intelligence()
    components = list_components()
"""

from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.foundation.data_structures import SwarmConfig


def get_swarm_intelligence(config=None):
    """
    Return a SwarmIntelligence instance for multi-agent coordination.

    Args:
        config: Optional SwarmConfig.

    Returns:
        SwarmIntelligence instance.
    """
    from Jotty.core.orchestration.swarm_intelligence import SwarmIntelligence
    return SwarmIntelligence(config=config)


def get_paradigm_executor(manager=None):
    """
    Return a ParadigmExecutor for discussion paradigms (relay, debate, refinement).

    Args:
        manager: Orchestrator instance. If None, returns the class for manual instantiation.

    Returns:
        ParadigmExecutor instance (if manager provided) or class (if not).
    """
    from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
    if manager is not None:
        return ParadigmExecutor(manager)
    return ParadigmExecutor


def get_training_daemon(manager=None):
    """
    Return a TrainingDaemon for background self-improvement.

    Args:
        manager: Orchestrator instance. If None, returns the class for manual instantiation.

    Returns:
        TrainingDaemon instance (if manager provided) or class (if not).
    """
    from Jotty.core.orchestration.training_daemon import TrainingDaemon
    if manager is not None:
        return TrainingDaemon(manager)
    return TrainingDaemon


def get_ensemble_manager():
    """
    Return an EnsembleManager instance for prompt ensembling.

    Returns:
        EnsembleManager instance (stateless).
    """
    from Jotty.core.orchestration.ensemble_manager import EnsembleManager
    return EnsembleManager()


def get_provider_manager(config: Optional['SwarmConfig'] = None):
    """
    Return a ProviderManager for skill provider registry management.

    Args:
        config: Optional SwarmConfig.

    Returns:
        ProviderManager instance.
    """
    from Jotty.core.orchestration.provider_manager import ProviderManager
    return ProviderManager(
        config=config,
        get_swarm_intelligence=lambda: None,
    )


def get_model_tier_router(default_provider: str = None):
    """
    Return a ModelTierRouter instance for complexity-based LLM model selection.

    Args:
        default_provider: Optional default LLM provider name.

    Returns:
        ModelTierRouter instance.
    """
    from Jotty.core.orchestration.model_tier_router import ModelTierRouter
    return ModelTierRouter(default_provider=default_provider)


def get_swarm_router():
    """
    Return a SwarmRouter instance for centralized task routing.

    Returns:
        SwarmRouter instance.
    """
    from Jotty.core.orchestration.swarm_router import SwarmRouter
    return SwarmRouter()


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
