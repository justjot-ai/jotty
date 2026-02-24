from typing import Any

"""
Learning Layer - Reinforcement Learning
=======================================

RL components including TD(λ), Q-learning, and multi-agent credit assignment.
All imports are lazy to avoid pulling in DSPy/OpenAI/Anthropic at module load time.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # algorithmic_credit
    "AgentContribution": ".algorithmic_credit",
    "AlgorithmicCreditAssigner": ".algorithmic_credit",
    "Coalition": ".algorithmic_credit",
    "DifferenceRewardEstimator": ".algorithmic_credit",
    "ShapleyValueEstimator": ".algorithmic_credit",
    # adaptive_components
    "AdaptiveExploration": ".adaptive_components",
    "AdaptiveLearningRate": ".adaptive_components",
    "IntermediateRewardCalculator": ".adaptive_components",
    # health_budget
    "DynamicBudgetManager": ".health_budget",
    "LearningHealthMonitor": ".health_budget",
    # td_lambda
    "TDLambdaLearner": ".td_lambda",
    # shaped_rewards
    "AgenticRewardEvaluator": ".shaped_rewards",
    "RewardCondition": ".shaped_rewards",
    "ShapedRewardManager": ".shaped_rewards",
    # learning_service (unified service)
    "LearningService": ".learning_service",
    "get_learning_service": ".learning_service",
    # learning_store (persistent SQLite store)
    "LearningStore": ".learning_store",
    "EpisodeRecord": ".learning_store",
    "PatternRecord": ".learning_store",
    "ReflectionRecord": ".learning_store",
    "ValueEstimate": ".learning_store",
}

_FACADE_IMPORTS = {
    "get_learning_system",
    "get_td_lambda",
    "get_credit_assigner",
    "get_reward_manager",
    "get_learning_service",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _FACADE_IMPORTS:
        from . import facade

        value = getattr(facade, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys()) + list(_FACADE_IMPORTS)

from .cost_aware_td import CostAwareTDLambda, get_cost_aware_td_lambda
