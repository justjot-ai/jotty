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

from .adaptive_components import (
    AdaptiveExploration,
    AdaptiveLearningRate,
    IntermediateRewardCalculator,
)
from .health_budget import DynamicBudgetManager, LearningHealthMonitor
from .reasoning_credit import ReasoningCreditAssigner
from .td_lambda import GroupedValueBaseline, TDLambdaLearner

__all__ = [
    "TDLambdaLearner",
    "GroupedValueBaseline",
    "AdaptiveLearningRate",
    "IntermediateRewardCalculator",
    "AdaptiveExploration",
    "ReasoningCreditAssigner",
    "LearningHealthMonitor",
    "DynamicBudgetManager",
]
