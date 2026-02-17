"""
Swarm Templates - All swarms as configuration on unified SwarmLearning
===================================================================

All templates inherit from SwarmLearning and define:
- AGENT_TEAM: Agent composition
- COORDINATION: Pattern (AUTO, SEQUENTIAL, PARALLEL, etc.)
- Optional: STAGES for CUSTOM pattern

Templates get ALL 8 learning layers automatically:
1. Memory (5 levels)
2. TD-Lambda reinforcement learning
3. Swarm Intelligence meta-learning
4. Gold Standard evaluation
5. Improvement Agents (Expert, Reviewer, Planner, Actor, Auditor, Learner)
6. Pattern Learning
7. Transfer Learning
8. Adaptive Components

Author: Jotty Team
Date: February 2026
"""

import importlib as _importlib
from typing import Any

_LAZY_IMPORTS: dict[str, str] = {
    # Real swarm implementations (with DSPy agents, full _execute_domain)
    "ReviewSwarm": ".review_swarm",
    "ReviewTemplate": ".review_swarm",
    "TestingSwarm": ".testing_swarm",
    "TestingTemplate": ".testing_swarm",
    "DataAnalysisSwarm": ".data_analysis_swarm",
    "DataAnalysisTemplate": ".data_analysis_swarm",
    "DevOpsSwarm": ".devops_swarm",
    "DevopsTemplate": ".devops_swarm",
    "FundamentalSwarm": ".fundamental_swarm",
    "FundamentalTemplate": ".fundamental_swarm",
    "IdeaWriterSwarm": ".idea_writer_swarm",
    "IdeaWriterTemplate": ".idea_writer_swarm",
    "LearningSwarm": ".learning_swarm",
    "LearningTemplate": ".learning_swarm",
    "SwarmML": ".swarm_ml",
    "MLTemplate": ".swarm_ml",
    "SwarmMLComprehensive": ".swarm_ml_comprehensive",
    "MlComprehensiveTemplate": ".swarm_ml_comprehensive",
    # Templates with their own modules
    "CodingTemplate": ".coding",
    "ResearchTemplate": ".research",
    "ArxivLearningTemplate": ".arxiv_learning",
    "OlympiadLearningTemplate": ".olympiad_learning",
    "PerspectiveLearningTemplate": ".perspective_learning",
    "PilotTemplate": ".pilot",
    # Team patterns
    "CollaborativeTemplate": ".team_patterns.collaborative",
    "HybridTemplate": ".team_patterns.hybrid",
    "SequentialTemplate": ".team_patterns.sequential",
}

# Backward compatibility aliases (map to their canonical class)
_ALIASES: dict[str, str] = {
    "CodingSwarm": "CodingTemplate",
    "ResearchSwarm": "ResearchTemplate",
    "ArxivLearningSwarm": "ArxivLearningTemplate",
    "OlympiadLearningSwarm": "OlympiadLearningTemplate",
    "PerspectiveLearningSwarm": "PerspectiveLearningTemplate",
    "PilotSwarm": "PilotTemplate",
    "CollaborativeTeam": "CollaborativeTemplate",
    "HybridTeam": "HybridTemplate",
    "SequentialTeam": "SequentialTemplate",
}


def __getattr__(name: str) -> Any:
    # Direct template class
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    # Backward compat alias
    if name in _ALIASES:
        target = _ALIASES[name]
        value = __getattr__(target)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*_LAZY_IMPORTS.keys(), *_ALIASES.keys()]
