"""Perspective Learning Swarm Package.

Multi-perspective educational swarm that explores ANY topic from 6 distinct
perspectives in 4 languages, producing professional PDF + HTML output.

Usage:
    from Jotty.core.intelligence.swarms.perspective_learning_swarm import teach_perspectives, teach_perspectives_sync

    # Async
    result = await teach_perspectives("Media and its influence on decisions", student_name="Aria")

    # Sync
    result = teach_perspectives_sync("Media and its influence on decisions", student_name="Aria")

    # Full control
    from Jotty.core.intelligence.swarms.perspective_learning_swarm import (
        PerspectiveLearningSwarm, PerspectiveLearningConfig,
        PerspectiveType, AgeGroup, ContentDepth, Language,
    )

    config = PerspectiveLearningConfig(
        student_name="Aria",
        age_group=AgeGroup.PRIMARY,
        depth=ContentDepth.STANDARD,
    )
    swarm = PerspectiveLearningSwarm(config)
    result = await swarm.teach(topic="Media and its influence on decisions")
"""

from .agents import (
    ContentAssemblerAgent,
    CurriculumDesignerAgent,
    DebateArchitectAgent,
    FrameworkBuilderAgent,
    IntuitiveExplainerAgent,
    MultilingualAgent,
    NarrativeEditorAgent,
    ProjectDesignerAgent,
    RealWorldConnectorAgent,
    StorytellerAgent,
)
from .swarm import PerspectiveLearningSwarm, teach_perspectives, teach_perspectives_sync
from .types import (
    AgeGroup,
    ContentDepth,
    DebatePoint,
    FrameworkModel,
    Language,
    LanguageContent,
    LessonContent,
    PerspectiveLearningConfig,
    PerspectiveLearningResult,
    PerspectiveSection,
    PerspectiveType,
    ProjectActivity,
)

__all__ = [
    # Swarm
    "PerspectiveLearningSwarm",
    "PerspectiveLearningConfig",
    "PerspectiveLearningResult",
    # Convenience
    "teach_perspectives",
    "teach_perspectives_sync",
    # Types
    "PerspectiveType",
    "Language",
    "AgeGroup",
    "ContentDepth",
    "PerspectiveSection",
    "LanguageContent",
    "DebatePoint",
    "ProjectActivity",
    "FrameworkModel",
    "LessonContent",
    # Agents
    "CurriculumDesignerAgent",
    "IntuitiveExplainerAgent",
    "FrameworkBuilderAgent",
    "StorytellerAgent",
    "DebateArchitectAgent",
    "ProjectDesignerAgent",
    "RealWorldConnectorAgent",
    "MultilingualAgent",
    "ContentAssemblerAgent",
    "NarrativeEditorAgent",
]
