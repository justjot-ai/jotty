"""
Jotty v6.0 - Enhanced Learning Module (Re-export Facade)
=========================================================

All classes have been extracted into focused modules:
- td_lambda.py: TDLambdaLearner, GroupedValueBaseline
- adaptive_components.py: AdaptiveLearningRate, IntermediateRewardCalculator, AdaptiveExploration
- reasoning_credit.py: ReasoningCreditAssigner
- health_budget.py: LearningHealthMonitor, DynamicBudgetManager

This file re-exports everything for backward compatibility.
"""

from .td_lambda import TDLambdaLearner, GroupedValueBaseline
from .adaptive_components import AdaptiveLearningRate, IntermediateRewardCalculator, AdaptiveExploration
from .reasoning_credit import ReasoningCreditAssigner
from .health_budget import LearningHealthMonitor, DynamicBudgetManager

__all__ = [
    'TDLambdaLearner',
    'GroupedValueBaseline',
    'AdaptiveLearningRate',
    'IntermediateRewardCalculator',
    'AdaptiveExploration',
    'ReasoningCreditAssigner',
    'LearningHealthMonitor',
    'DynamicBudgetManager',
]
