"""
Learning Layer - Reinforcement Learning
=======================================

RL components including TD(λ), Q-learning, and multi-agent credit assignment.

Modules:
--------
- learning: TD(λ) learner, adaptive learning rate
- q_learning: Q-learning with LLM Q-predictor
- rl_components: Core RL building blocks
- offline_learning: Offline training
- shaped_rewards: Reward shaping
- predictive_marl: Predictive multi-agent RL
- predictive_cooperation: Cooperation prediction
- algorithmic_credit: Credit assignment algorithms
"""

from .algorithmic_credit import (
    AgentContribution,
    AlgorithmicCreditAssigner,
    Coalition,
    CounterfactualSignature as AlgorithmicCounterfactualSignature,
    DifferenceRewardEstimator,
    ShapleyEstimatorSignature,
    ShapleyValueEstimator,
)
from .learning import (
    AdaptiveExploration,
    AdaptiveLearningRate,
    DynamicBudgetManager,
    IntermediateRewardCalculator,
    LearningHealthMonitor,
    ReasoningCreditAssigner,
    TDLambdaLearner,
)
from .offline_learning import (
    CounterfactualLearner,
    CounterfactualSignature as OfflineCounterfactualSignature,
    OfflineLearner,
    PatternDiscovery,
    PatternDiscoverySignature,
    PrioritizedEpisodeBuffer,
)
from .predictive_cooperation import (
    CooperationPrinciples,
    CooperationReasoningSignature,
    CooperationReasoner,
    CooperationState,
    NashBargainingSignature,
    NashBargainingSolver,
    PredictiveCooperativeAgent,
)
from .predictive_marl import (
    ActualTrajectory,
    AgentModel,
    CooperativeCreditAssigner,
    Divergence,
    DivergenceMemory,
    LLMTrajectoryPredictor,
    PredictedAction,
    PredictedTrajectory,
    TrajectoryPredictionSignature,
)
from .q_learning import (
    LLMQPredictor,
    LLMQPredictorSignature,
)
from .rl_components import (
    RLComponents,
)
from .shaped_rewards import (
    AgenticRewardEvaluator,
    RewardCondition,
    ShapedRewardManager,
)

__all__ = [
    # algorithmic_credit
    'AgentContribution',
    'AlgorithmicCounterfactualSignature',
    'AlgorithmicCreditAssigner',
    'Coalition',
    'DifferenceRewardEstimator',
    'ShapleyEstimatorSignature',
    'ShapleyValueEstimator',
    # learning
    'AdaptiveExploration',
    'AdaptiveLearningRate',
    'DynamicBudgetManager',
    'IntermediateRewardCalculator',
    'LearningHealthMonitor',
    'ReasoningCreditAssigner',
    'TDLambdaLearner',
    # offline_learning
    'CounterfactualLearner',
    'OfflineCounterfactualSignature',
    'OfflineLearner',
    'PatternDiscovery',
    'PatternDiscoverySignature',
    'PrioritizedEpisodeBuffer',
    # predictive_cooperation
    'CooperationPrinciples',
    'CooperationReasoningSignature',
    'CooperationReasoner',
    'CooperationState',
    'NashBargainingSignature',
    'NashBargainingSolver',
    'PredictiveCooperativeAgent',
    # predictive_marl
    'ActualTrajectory',
    'AgentModel',
    'CooperativeCreditAssigner',
    'Divergence',
    'DivergenceMemory',
    'LLMTrajectoryPredictor',
    'PredictedAction',
    'PredictedTrajectory',
    'TrajectoryPredictionSignature',
    # q_learning
    'LLMQPredictor',
    'LLMQPredictorSignature',
    # rl_components
    'RLComponents',
    # shaped_rewards
    'AgenticRewardEvaluator',
    'RewardCondition',
    'ShapedRewardManager',
]
