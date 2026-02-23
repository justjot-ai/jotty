"""
Learning Subsystem Facade
==========================

Clean, discoverable API for all learning components.

Preferred entry point for new code:
    from Jotty.core.intelligence.learning.facade import get_learning_service

    service = get_learning_service()
    service.record(...)   # Record outcome from any unit
    service.query(...)    # Get guidance
    service.reflect(...)  # Mid-execution reflection

Legacy entry point (still works, returns LearningService):
    from Jotty.core.intelligence.learning.facade import get_learning_system
    service = get_learning_system()
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from Jotty.core.infrastructure.foundation.configs import LearningConfig  # type: ignore[import]
    from Jotty.core.intelligence.learning.learning_service import LearningService
    from Jotty.core.intelligence.learning.reasoning_credit import (
        ReasoningCreditAssigner,  # type: ignore[import]
    )
    from Jotty.core.intelligence.learning.shaped_rewards import (
        ShapedRewardManager,  # type: ignore[import]
    )
    from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner  # type: ignore[import]


def _resolve_learning_config(config: Any) -> "SwarmConfig":  # type: ignore[name-defined]
    """Convert LearningConfig or SwarmConfig to SwarmConfig for internal use.

    Accepts:
        - None → default SwarmConfig
        - LearningConfig → SwarmConfig.from_configs(learning=config)
        - SwarmConfig → pass through
    """
    if config is None:
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        return SwarmConfig()

    from Jotty.core.infrastructure.foundation.configs.learning import (
        LearningConfig,  # type: ignore[import-not-found, import]
    )

    if isinstance(config, LearningConfig):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        return SwarmConfig.from_configs(learning=config)

    # Assume SwarmConfig
    return config


def get_learning_system(
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,  # type: ignore[name-defined]
) -> "LearningService":
    """
    Return the unified LearningService singleton.

    Backward-compatible entry point — now returns LearningService instead
    of the deleted LearningManager. The LearningService provides
    record_outcome() and get_learned_context() methods for callers that
    used the old LearningManager API.

    Args:
        config: Ignored (LearningService is a singleton). Kept for backward compat.

    Returns:
        LearningService instance.
    """
    from Jotty.core.intelligence.learning.learning_service import LearningService

    return LearningService.get_instance()


def get_td_lambda(
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,  # type: ignore[name-defined]
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
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,  # type: ignore[name-defined]
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
    config: Optional[Union["LearningConfig", "SwarmConfig"]] = None,  # type: ignore[name-defined]
) -> Dict[str, Any]:
    """
    Return offline learning components: OfflineLearner, CounterfactualLearner, PatternDiscovery.

    Args:
        config: Optional LearningConfig or SwarmConfig. If None, uses defaults.

    Returns:
        Dict with 'offline_learner', 'counterfactual', and 'pattern_discovery' keys.
    """
    from Jotty.core.intelligence.learning.offline_learning import (  # type: ignore[import]
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
    from Jotty.core.intelligence.learning.predictive_cooperation import (  # type: ignore[import]
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
        # Core learning pipeline
        "LearningService": "Unified learning service (SQLite-backed, record/query/reflect/transfer/judge/distill)",
        "TDLambdaLearner": "TD(λ) RL with SkillQTable + StepQTable per domain",
        "CrystallizedConfig": "Hardened SOP from graduated agent (probation → crystallization)",
        "LLMJudge": "Sonnet-based quality evaluator (5 dimensions: accuracy/completeness/structure/actionability/depth)",
        "FactDistillation": "Judge-informed lesson extraction (dedup-aware, incremental)",
        # Credit & rewards
        "ReasoningCreditAssigner": "Credit assignment for multi-step reasoning chains",
        "ShapedRewardManager": "Reward shaping for faster RL convergence",
        # Offline & patterns
        "OfflineLearner": "Batch learning from stored episode buffers",
        "CounterfactualLearner": "What-if analysis for alternative action sequences",
        "PatternDiscovery": "Discovers recurring patterns in agent behavior",
        # Cooperative
        "PredictiveCooperativeAgent": "Multi-agent cooperation with trajectory prediction",
        "NashBargainingSolver": "Game-theoretic negotiation between agents",
        # Tool feedback
        "ToolLearningFeedback": "Tool execution → learning feedback loop",
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


# =============================================================================
# Unified LearningService (NEW — preferred for all new code)
# =============================================================================


def get_crystallized(task_type: str, domain: str = "") -> Any:
    """Load a crystallized config for a domain if it exists.

    Returns CrystallizedConfig or None.

    Example:
        from Jotty.core.intelligence.learning.facade import get_crystallized
        config = get_crystallized("coding", domain="finance")
        if config:
            print(config.sop_roles)  # ('generate', 'save', 'verify')
    """
    from .crystallization import load

    return load(task_type, domain)


def get_learning_service() -> Any:
    """
    Get the unified LearningService singleton.

    This is the preferred entry point for all learning interactions.
    Agents, swarms, pipelines, and the orchestrator all share this service.

    Returns:
        LearningService instance

    Example:
        from Jotty.core.intelligence.learning.facade import get_learning_service

        service = get_learning_service()

        # Record an execution outcome
        service.record("MyAgent", "agent", "coding", "code_gen",
                       context={"task": "Build API"},
                       action={"model": "claude-sonnet"},
                       outcome={"code_lines": 150},
                       success=True, quality=0.9)

        # Query for guidance
        guidance = service.query("coding", "code_gen")

        # Mid-execution reflection
        adjustment = service.reflect(episode_id, step=2,
                                      observation="Tests failing",
                                      unit_name="TestWriter")

        # Transfer learning
        patterns = service.transfer("coding", "devops")

        # Get improvement report
        report = service.improvement_report("coding")
    """
    from .learning_service import LearningService

    return LearningService.get_instance()
