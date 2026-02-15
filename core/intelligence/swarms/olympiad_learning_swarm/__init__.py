"""Olympiad Learning Swarm Package.

World-class educational swarm for olympiad-level mastery.
Supports any subject: Mathematics, Physics, Chemistry, CS, Biology, Astronomy.

Usage:
    from Jotty.core.intelligence.swarms.olympiad_learning_swarm import learn_topic, learn_topic_sync

    # Async
    result = await learn_topic("mathematics", "Number Theory", "Aria")

    # Sync
    result = learn_topic_sync("mathematics", "Combinatorics", "Aria")

    # Full control
    from Jotty.core.intelligence.swarms.olympiad_learning_swarm import (
        OlympiadLearningSwarm, OlympiadLearningConfig, Subject, LessonDepth, DifficultyTier
    )

    config = OlympiadLearningConfig(
        subject=Subject.MATHEMATICS,
        student_name="Aria",
        depth=LessonDepth.DEEP,
        target_tier=DifficultyTier.OLYMPIAD,
    )
    swarm = OlympiadLearningSwarm(config)
    result = await swarm.teach(topic="Number Theory")
"""

from .types import (
    Subject, DifficultyTier, LessonDepth, TeachingMode,
    OlympiadLearningConfig, OlympiadLearningResult,
    BuildingBlock, ConceptCore, PatternEntry, Problem,
    StrategyCard, MistakeEntry, LessonSection, LessonContent,
)
from .agents import (
    CurriculumArchitectAgent, ConceptDecomposerAgent,
    IntuitionBuilderAgent, PatternHunterAgent,
    ProblemCrafterAgent, SolutionStrategistAgent,
    MistakeAnalyzerAgent, ConnectionMapperAgent,
    ContentAssemblerAgent, UnifiedTopicAgent,
    NarrativeEditorAgent, RankTipsAgent,
)
from .swarm import OlympiadLearningSwarm, learn_topic, learn_topic_sync

__all__ = [
    # Swarm
    'OlympiadLearningSwarm', 'OlympiadLearningConfig', 'OlympiadLearningResult',
    # Convenience
    'learn_topic', 'learn_topic_sync',
    # Types
    'Subject', 'DifficultyTier', 'LessonDepth', 'TeachingMode',
    'BuildingBlock', 'ConceptCore', 'PatternEntry', 'Problem',
    'StrategyCard', 'MistakeEntry', 'LessonSection', 'LessonContent',
    # Agents
    'CurriculumArchitectAgent', 'ConceptDecomposerAgent',
    'IntuitionBuilderAgent', 'PatternHunterAgent',
    'ProblemCrafterAgent', 'SolutionStrategistAgent',
    'MistakeAnalyzerAgent', 'ConnectionMapperAgent',
    'ContentAssemblerAgent', 'UnifiedTopicAgent',
    'NarrativeEditorAgent', 'RankTipsAgent',
]
