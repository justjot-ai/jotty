"""
Workflows - Intent-Based Automation
====================================

High-level workflows that understand intent and automatically execute.

Three Domain-Specific Workflows:
1. AutoWorkflow - General software development (APIs, apps, systems)
2. ResearchWorkflow - Research and analysis (topics, markets, trends)
3. LearningWorkflow - Educational content (K-12 to Olympiad)
"""

from .auto_workflow import (
    AutoWorkflow,
    WorkflowIntent,
    build,
    research as research_stage,  # Rename to avoid conflict
    develop,
)
from .research_workflow import (
    ResearchWorkflow,
    ResearchIntent,
    ResearchDepth,
    ResearchType,
    research,  # This is the actual research workflow
)
from .learning_workflow import (
    LearningWorkflow,
    LearningIntent,
    LearningLevel,
    LearningDepth,
    Subject,
    learn,
)
from .smart_swarm_registry import (
    SmartSwarmRegistry,
    StageType,
    SwarmConfig,
    get_smart_registry,
)

__all__ = [
    # AutoWorkflow (software development)
    "AutoWorkflow",
    "WorkflowIntent",
    "build",
    "develop",

    # ResearchWorkflow (research & analysis)
    "ResearchWorkflow",
    "ResearchIntent",
    "ResearchDepth",
    "ResearchType",
    "research",

    # LearningWorkflow (educational content)
    "LearningWorkflow",
    "LearningIntent",
    "LearningLevel",
    "LearningDepth",
    "Subject",
    "learn",

    # Registry
    "SmartSwarmRegistry",
    "StageType",
    "SwarmConfig",
    "get_smart_registry",
]
