from typing import Any
"""
Learning Layer - Reinforcement Learning
=======================================

RL components including TD(λ), Q-learning, and multi-agent credit assignment.
All imports are lazy to avoid pulling in DSPy/OpenAI/Anthropic at module load time.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # base_learning_manager
    "BaseLearningManager": ".base_learning_manager",
    "ValueBasedLearningManager": ".base_learning_manager",
    "RewardShapingManager": ".base_learning_manager",
    "MultiAgentLearningManager": ".base_learning_manager",
    # algorithmic_credit
    "AgentContribution": ".algorithmic_credit",
    "AlgorithmicCreditAssigner": ".algorithmic_credit",
    "Coalition": ".algorithmic_credit",
    "DifferenceRewardEstimator": ".algorithmic_credit",
    "ShapleyValueEstimator": ".algorithmic_credit",
    # learning
    "AdaptiveExploration": ".learning",
    "AdaptiveLearningRate": ".learning",
    "DynamicBudgetManager": ".learning",
    "IntermediateRewardCalculator": ".learning",
    "LearningHealthMonitor": ".learning",
    "ReasoningCreditAssigner": ".learning",
    "TDLambdaLearner": ".learning",
    # offline_learning
    "CounterfactualLearner": ".offline_learning",
    "OfflineLearner": ".offline_learning",
    "PatternDiscovery": ".offline_learning",
    "PrioritizedEpisodeBuffer": ".offline_learning",
    # predictive_cooperation
    "CooperationPrinciples": ".predictive_cooperation",
    "CooperationReasoner": ".predictive_cooperation",
    "CooperationState": ".predictive_cooperation",
    "NashBargainingSolver": ".predictive_cooperation",
    "PredictiveCooperativeAgent": ".predictive_cooperation",
    # predictive_marl
    "ActualTrajectory": ".predictive_marl",
    "AgentModel": ".predictive_marl",
    "CooperativeCreditAssigner": ".predictive_marl",
    "Divergence": ".predictive_marl",
    "DivergenceMemory": ".predictive_marl",
    "LLMTrajectoryPredictor": ".predictive_marl",
    "PredictedAction": ".predictive_marl",
    "PredictedTrajectory": ".predictive_marl",
    # q_learning
    "LLMQPredictor": ".q_learning",
    # rl_components
    "RLComponents": ".rl_components",
    # shaped_rewards
    "AgenticRewardEvaluator": ".shaped_rewards",
    "RewardCondition": ".shaped_rewards",
    "ShapedRewardManager": ".shaped_rewards",
}


_FACADE_IMPORTS = {
    'get_learning_system',
    'get_td_lambda',
    'get_credit_assigner',
    'get_offline_learner',
    'get_reward_manager',
    'get_cooperative_agents',
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


__all__ = list(_LAZY_IMPORTS.keys()) + [
    'get_learning_system',
    'get_td_lambda',
    'get_credit_assigner',
    'get_offline_learner',
    'get_reward_manager',
    'get_cooperative_agents',
]

from .cost_aware_td import (
    CostAwareTDLambda,
    get_cost_aware_td_lambda,
)
