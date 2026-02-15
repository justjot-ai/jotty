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
    # Template classes
    "ArxivLearningTemplate": ".arxiv_learning",
    "CodingTemplate": ".coding",
    "DataAnalysisTemplate": ".data_analysis",
    "DevopsTemplate": ".devops",
    "FundamentalTemplate": ".fundamental",
    "IdeaWriterTemplate": ".idea_writer",
    "LearningTemplate": ".learning",
    "MLTemplate": ".ml",
    "MlComprehensiveTemplate": ".ml_comprehensive",
    "OlympiadLearningTemplate": ".olympiad_learning",
    "PerspectiveLearningTemplate": ".perspective_learning",
    "PilotTemplate": ".pilot",
    "ResearchTemplate": ".research",
    "ReviewTemplate": ".review",
    "TestingTemplate": ".testing",
    # Team patterns
    "CollaborativeTemplate": ".team_patterns.collaborative",
    "HybridTemplate": ".team_patterns.hybrid",
    "SequentialTemplate": ".team_patterns.sequential",
}

# Backward compatibility aliases (map to their template class)
_ALIASES: dict[str, str] = {
    "CodingSwarm": "CodingTemplate",
    "ResearchSwarm": "ResearchTemplate",
    "ReviewSwarm": "ReviewTemplate",
    "TestingSwarm": "TestingTemplate",
    "DataAnalysisSwarm": "DataAnalysisTemplate",
    "DevOpsSwarm": "DevopsTemplate",
    "FundamentalSwarm": "FundamentalTemplate",
    "IdeaWriterSwarm": "IdeaWriterTemplate",
    "LearningSwarm": "LearningTemplate",
    "ArxivLearningSwarm": "ArxivLearningTemplate",
    "OlympiadLearningSwarm": "OlympiadLearningTemplate",
    "PerspectiveLearningSwarm": "PerspectiveLearningTemplate",
    "PilotSwarm": "PilotTemplate",
    "SwarmML": "MLTemplate",
    "SwarmMLComprehensive": "MlComprehensiveTemplate",
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
