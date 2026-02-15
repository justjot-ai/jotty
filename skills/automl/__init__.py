"""
Jotty ML Skills Library
=======================

Reusable machine learning skills that can be composed
by swarm templates to solve ML problems.

Skills are ATOMIC - they do ONE thing well.
Templates ORCHESTRATE skills into pipelines.

Available Skills:
- eda: Exploratory Data Analysis
- feature_engineering: Kaggle-style feature engineering
- feature_selection: SHAP + multi-model selection
- model_selection: Evaluate 7+ models
- hyperopt: Multi-model Optuna optimization
- ensemble: Multi-level stacking
- llm_reasoner: Chain-of-thought feature generation

Usage:
    from jotty.skills.ml import EDASkill, FeatureEngineeringSkill

    eda = EDASkill()
    insights = await eda.execute(X, y)

    fe = FeatureEngineeringSkill()
    X_new = await fe.execute(X, y, insights)
"""

from .automl_skill import AutoMLSkill
from .backtest_engine import (
    ComprehensiveBacktestResult,
    FactorExposure,
    MonteCarloResult,
    PositionSizing,
    RegimeAnalysis,
    RiskMetrics,
    StatisticalTests,
    StressTester,
    TransactionCosts,
    WalkForwardResult,
    WorldClassBacktestEngine,
)
from .backtest_report import (
    BacktestChartGenerator,
    BacktestMetrics,
    BacktestReportGenerator,
    BacktestReportSkill,
    BacktestResult,
    ModelResults,
    TradeStatistics,
)
from .base import MLSkill, SkillCategory, SkillPipeline, SkillRegistry, SkillResult
from .comprehensive_backtest_report import (
    ComprehensiveBacktestChartGenerator,
    ComprehensiveBacktestReportGenerator,
)
from .eda import EDASkill
from .ensemble import EnsembleSkill
from .feature_engineering import FeatureEngineeringSkill
from .feature_selection import FeatureSelectionSkill
from .fundamental_features import FundamentalFeaturesSkill
from .hyperopt import HyperoptSkill
from .llm_reasoner import LLMFeatureReasonerSkill
from .mlflow_tracker import MLflowTrackerSkill
from .model_selection import ModelSelectionSkill

__all__ = [
    # Base classes
    "MLSkill",
    "SkillResult",
    "SkillCategory",
    "SkillPipeline",
    "SkillRegistry",
    # Skills
    "EDASkill",
    "LLMFeatureReasonerSkill",
    "FeatureEngineeringSkill",
    "FeatureSelectionSkill",
    "ModelSelectionSkill",
    "HyperoptSkill",
    "EnsembleSkill",
    "MLflowTrackerSkill",
    "AutoMLSkill",
    "FundamentalFeaturesSkill",
    # Backtest Report
    "BacktestReportSkill",
    "BacktestReportGenerator",
    "BacktestChartGenerator",
    "BacktestResult",
    "BacktestMetrics",
    "TradeStatistics",
    "ModelResults",
    # Comprehensive Backtest Engine
    "WorldClassBacktestEngine",
    "ComprehensiveBacktestResult",
    "TransactionCosts",
    "RiskMetrics",
    "StatisticalTests",
    "RegimeAnalysis",
    "FactorExposure",
    "MonteCarloResult",
    "WalkForwardResult",
    "PositionSizing",
    "StressTester",
    "ComprehensiveBacktestReportGenerator",
    "ComprehensiveBacktestChartGenerator",
]
