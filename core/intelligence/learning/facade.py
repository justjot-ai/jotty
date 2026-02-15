"""
Learning Subsystem Facade
==========================

Clean, discoverable API for all learning components.
No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.intelligence.learning.facade import get_learning_system, list_components

    # Get the unified learning coordinator
    manager = get_learning_system(config)

    # Explore what's available
    for name, desc in list_components().items():
        print(f"{name}: {desc}")
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from Jotty.core.infrastructure.foundation.configs import LearningConfig
    from Jotty.core.intelligence.learning.learning_coordinator import LearningManager
    from Jotty.core.intelligence.learning.reasoning_credit import ReasoningCreditAssigner
    from Jotty.core.intelligence.learning.shaped_rewards import ShapedRewardManager
    from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner


def _resolve_learning_config(config: Any) -> "SwarmConfig":
    """Convert LearningConfig or SwarmConfig to SwarmConfig for internal use.

    Accepts:
        - None → default SwarmConfig
        - LearningConfig → SwarmConfig.from_configs(learning=config)
        - SwarmConfig → pass through
    """
    if config is None:
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        return SwarmConfig()

    from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

    if isinstance(config, LearningConfig):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        return SwarmConfig.from_configs(learning=config)

    # Assume SwarmConfig
    return config


def get_learning_system(
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,
) -> "LearningManager":
    """
    Return a configured LearningManager (unified learning coordinator).

    Args:
        config: Optional LearningConfig or SwarmConfig. If None, uses defaults.

    Returns:
        LearningManager instance.
    """
    from Jotty.core.intelligence.learning.learning_coordinator import LearningManager

    resolved = _resolve_learning_config(config)
    return LearningManager(resolved)


def get_td_lambda(
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,
) -> "TDLambdaLearner":
    """
    Return a TDLambdaLearner for temporal-difference learning.

    Args:
        config: Optional LearningConfig or SwarmConfig. If None, uses defaults.

    Returns:
        TDLambdaLearner instance.
    """
    from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner

    resolved = _resolve_learning_config(config)
    return TDLambdaLearner(config=resolved)


def get_credit_assigner(
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,
) -> "ReasoningCreditAssigner":
    """
    Return a ReasoningCreditAssigner for multi-step reasoning credit.

    Args:
        config: Optional LearningConfig or SwarmConfig. If None, uses defaults.

    Returns:
        ReasoningCreditAssigner instance.
    """
    from Jotty.core.intelligence.learning.reasoning_credit import ReasoningCreditAssigner

    resolved = _resolve_learning_config(config)
    return ReasoningCreditAssigner(config=resolved)


def get_offline_learner(
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,
) -> Dict[str, Any]:
    """
    Return offline learning components: OfflineLearner, CounterfactualLearner, PatternDiscovery.

    Args:
        config: Optional LearningConfig or SwarmConfig. If None, uses defaults.

    Returns:
        Dict with 'offline_learner', 'counterfactual', and 'pattern_discovery' keys.
    """
    from Jotty.core.intelligence.learning.offline_learning import (
        CounterfactualLearner,
        OfflineLearner,
        PatternDiscovery,
    )

    resolved = _resolve_learning_config(config)
    return {
        "offline_learner": OfflineLearner(config=resolved),
        "counterfactual": CounterfactualLearner(config=resolved),
        "pattern_discovery": PatternDiscovery(config=resolved),
    }


def get_reward_manager() -> "ShapedRewardManager":
    """
    Return a ShapedRewardManager for reward shaping.

    Returns:
        ShapedRewardManager instance.
    """
    from Jotty.core.intelligence.learning.shaped_rewards import ShapedRewardManager

    return ShapedRewardManager()


def get_cooperative_agents() -> Dict[str, type]:
    """
    Return cooperative multi-agent components.

    Returns:
        Dict with 'predictive_agent' and 'nash_solver' keys (class objects).
    """
    from Jotty.core.intelligence.learning.predictive_cooperation import (
        NashBargainingSolver,
        PredictiveCooperativeAgent,
    )

    return {
        "predictive_agent": PredictiveCooperativeAgent,
        "nash_solver": NashBargainingSolver,
    }


def list_components() -> Dict[str, str]:
    """
    List all learning subsystem components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "LearningManager": "Unified coordinator for all learning (Q-learning, TD, per-agent)",
        "TDLambdaLearner": "Temporal-difference learning with eligibility traces",
        "ReasoningCreditAssigner": "Credit assignment for multi-step reasoning chains",
        "AdaptiveLearningRate": "Learning rate that adapts to convergence dynamics",
        "IntermediateRewardCalculator": "Calculates intermediate rewards during execution",
        "LearningHealthMonitor": "Monitors learning system health and convergence",
        "DynamicBudgetManager": "Dynamically allocates exploration budget",
        "OfflineLearner": "Batch learning from stored episode buffers",
        "CounterfactualLearner": "What-if analysis for alternative action sequences",
        "PatternDiscovery": "Discovers recurring patterns in agent behavior",
        "ShapedRewardManager": "Reward shaping for faster RL convergence",
        "PredictiveCooperativeAgent": "Multi-agent cooperation with trajectory prediction",
        "NashBargainingSolver": "Game-theoretic negotiation between agents",
        "CooperativeCreditAssigner": "Credit assignment in cooperative multi-agent settings",
        "LLMTrajectoryPredictor": "Predicts future agent trajectories using LLMs",
        "AlgorithmicCreditAssigner": "Shapley value and difference reward estimation",
        "AdaptiveExploration": "Exploration strategy that adapts over time",
        "ToolLearningFeedback": "Tool execution → learning feedback loop (NEW 2026-02-16)",
    }


# =============================================================================
# Tool Learning Feedback (NEW 2026-02-16)
# =============================================================================


def get_tool_learning_feedback() -> Any:
    """
    Get tool learning feedback loop.

    Integrates tool execution statistics with TD-Lambda learning.

    Returns:
        ToolLearningFeedback instance

    Example:
        from Jotty.core.intelligence.learning.facade import get_tool_learning_feedback

        feedback = get_tool_learning_feedback()
        feedback.feed_from_interceptor(interceptor)
        feedback.update_registry_scores(registry)
    """
    from .tool_learning import get_tool_learning_feedback as _get

    return _get()


def feed_tool_statistics(interceptor: Any) -> int:
    """
    Convenience: Feed tool execution statistics to learning system.

    Args:
        interceptor: ToolInterceptor instance with execution history

    Returns:
        Number of tool calls processed

    Example:
        from Jotty.core.intelligence.learning.facade import feed_tool_statistics
        from Jotty.core.infrastructure.integration import ToolInterceptor

        interceptor = ToolInterceptor("executor")
        # ... execute tools ...
        feed_tool_statistics(interceptor)
    """
    from .tool_learning import feed_tool_statistics_to_learning

    return feed_tool_statistics_to_learning(interceptor)


def update_registry_with_learning(registry: Any) -> int:
    """
    Convenience: Update SkillsRegistry discovery scores with learning data.

    Args:
        registry: SkillsRegistry instance

    Returns:
        Number of skills updated

    Example:
        from Jotty.core.intelligence.learning.facade import update_registry_with_learning
        from Jotty.core.capabilities.registry import get_skills_registry

        registry = get_skills_registry()
        updated = update_registry_with_learning(registry)
        print(f"Updated {updated} skills with learning data")
    """
    from .tool_learning import update_registry_with_learning as _update

    return _update(registry)
