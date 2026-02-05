"""
Jotty Swarm Templates
=====================

Domain-specific orchestration patterns that combine skills, agents, and pipelines
to solve complex problems autonomously.

Templates provide:
- Pre-defined agent configurations with specialized skills
- Optimized pipeline stages with parallelism hints
- Domain-specific LLM prompts (chain-of-thought)
- Feedback loop configurations for iterative improvement
- Auto-detection of problem type

Available Templates:
- SwarmML: Machine Learning (classification, regression, clustering)
- SwarmMLComprehensive: Learning-enhanced ML with cross-session learning
- SwarmLean: Claude Code-like lean execution (research, documents, checklists)
- SwarmNLP: Natural Language Processing (coming soon)
- SwarmCV: Computer Vision (coming soon)
- SwarmTimeSeries: Time Series Analysis (coming soon)

Usage:
    from jotty import Swarm

    # Explicit template
    result = await Swarm.solve(template="ml", X=X, y=y)

    # Learning-enhanced ML (improves over sessions)
    result = await Swarm.solve(template="ml_comprehensive", X=X, y=y)

    # Lean mode for simple tasks (Claude Code-like)
    result = await Swarm.solve(template="lean", task="Create a compliance checklist")

    # Auto-detect
    result = await Swarm.auto_solve(X, y)
"""

from .base import SwarmTemplate, AgentConfig, StageConfig, FeedbackConfig, ModelTier
from .registry import TemplateRegistry
from .swarm_ml import SwarmML
from .swarm_lean import SwarmLean
from .swarm_ml_comprehensive import (
    SwarmMLComprehensive, LearningConfig, LearningState,
    MLflowConfig, ReportConfig, TelegramConfig
)

__all__ = [
    # Base classes
    'SwarmTemplate',
    'AgentConfig',
    'StageConfig',
    'FeedbackConfig',
    'ModelTier',
    'TemplateRegistry',
    # ML Templates
    'SwarmML',
    'SwarmMLComprehensive',
    'LearningConfig',
    'LearningState',
    'MLflowConfig',
    'ReportConfig',
    'TelegramConfig',
    # Lean Template
    'SwarmLean',
]
