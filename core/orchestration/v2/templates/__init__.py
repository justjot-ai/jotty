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
- SwarmLean: Claude Code-like lean execution (research, documents, checklists)
- SwarmNLP: Natural Language Processing (coming soon)
- SwarmCV: Computer Vision (coming soon)
- SwarmTimeSeries: Time Series Analysis (coming soon)

Usage:
    from jotty import Swarm

    # Explicit template
    result = await Swarm.solve(template="ml", X=X, y=y)

    # Lean mode for simple tasks (Claude Code-like)
    result = await Swarm.solve(template="lean", task="Create a compliance checklist")

    # Auto-detect
    result = await Swarm.auto_solve(X, y)
"""

from .base import SwarmTemplate, AgentConfig, StageConfig, FeedbackConfig
from .registry import TemplateRegistry
from .swarm_ml import SwarmML
from .swarm_lean import SwarmLean

__all__ = [
    'SwarmTemplate',
    'AgentConfig',
    'StageConfig',
    'FeedbackConfig',
    'TemplateRegistry',
    'SwarmML',
    'SwarmLean',
]
