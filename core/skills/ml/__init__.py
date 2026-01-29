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

from .base import MLSkill, SkillResult, SkillCategory, SkillPipeline, SkillRegistry
from .eda import EDASkill
from .llm_reasoner import LLMFeatureReasonerSkill
from .feature_engineering import FeatureEngineeringSkill
from .feature_selection import FeatureSelectionSkill
from .model_selection import ModelSelectionSkill
from .hyperopt import HyperoptSkill
from .ensemble import EnsembleSkill

__all__ = [
    # Base classes
    'MLSkill',
    'SkillResult',
    'SkillCategory',
    'SkillPipeline',
    'SkillRegistry',
    # Skills
    'EDASkill',
    'LLMFeatureReasonerSkill',
    'FeatureEngineeringSkill',
    'FeatureSelectionSkill',
    'ModelSelectionSkill',
    'HyperoptSkill',
    'EnsembleSkill',
]
