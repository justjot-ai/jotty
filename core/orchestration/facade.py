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


def get_swarm_intelligence():
    """
    Return the SwarmIntelligence class for multi-agent coordination.

    Returns:
        SwarmIntelligence class (instantiate with agents list).
    """
    from Jotty.core.orchestration.swarm_intelligence import SwarmIntelligence
    return SwarmIntelligence


def get_paradigm_executor():
    """
    Return the ParadigmExecutor class for discussion paradigms (relay, debate, refinement).

    Returns:
        ParadigmExecutor class (instantiate with orchestrator reference).
    """
    from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
    return ParadigmExecutor


def get_training_daemon():
    """
    Return the TrainingDaemon class for background self-improvement.

    Returns:
        TrainingDaemon class (instantiate with orchestrator reference).
    """
    from Jotty.core.orchestration.training_daemon import TrainingDaemon
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


def get_model_tier_router():
    """
    Return a ModelTierRouter for complexity-based LLM model selection.

    Returns:
        ModelTierRouter class.
    """
    from Jotty.core.orchestration.model_tier_router import ModelTierRouter
    return ModelTierRouter


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
