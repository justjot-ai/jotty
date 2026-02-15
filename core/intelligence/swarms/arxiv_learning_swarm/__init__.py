"""ArXiv Learning Swarm Package."""

from .types import (
    LearningDepth, ContentStyle, AudienceLevel,
    ArxivLearningConfig, ArxivLearningResult,
    PaperInfo, Concept, LearningSection, LearningContent,
)
from .agents import (
    PaperFetcherAgent, ConceptExtractorAgent, IntuitionBuilderAgent,
    MathSimplifierAgent, ExampleGeneratorAgent, ProgressiveBuilderAgent,
    ContentPolisherAgent, UnifiedLearningAgent,
)
from .swarm import ArxivLearningSwarm, learn_paper, learn_paper_sync

__all__ = [
    'ArxivLearningSwarm', 'ArxivLearningConfig', 'ArxivLearningResult',
    'LearningContent', 'LearningSection', 'Concept', 'PaperInfo',
    'LearningDepth', 'ContentStyle', 'AudienceLevel',
    'learn_paper', 'learn_paper_sync',
    'PaperFetcherAgent', 'ConceptExtractorAgent', 'IntuitionBuilderAgent',
    'MathSimplifierAgent', 'ExampleGeneratorAgent', 'ProgressiveBuilderAgent',
    'ContentPolisherAgent', 'UnifiedLearningAgent',
]
