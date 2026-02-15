"""ArXiv Learning Swarm Package."""

from .agents import (
    ConceptExtractorAgent,
    ContentPolisherAgent,
    ExampleGeneratorAgent,
    IntuitionBuilderAgent,
    MathSimplifierAgent,
    PaperFetcherAgent,
    ProgressiveBuilderAgent,
    UnifiedLearningAgent,
)
from .swarm import ArxivLearningSwarm, learn_paper, learn_paper_sync
from .types import (
    ArxivLearningConfig,
    ArxivLearningResult,
    AudienceLevel,
    Concept,
    ContentStyle,
    LearningContent,
    LearningDepth,
    LearningSection,
    PaperInfo,
)

__all__ = [
    "ArxivLearningSwarm",
    "ArxivLearningConfig",
    "ArxivLearningResult",
    "LearningContent",
    "LearningSection",
    "Concept",
    "PaperInfo",
    "LearningDepth",
    "ContentStyle",
    "AudienceLevel",
    "learn_paper",
    "learn_paper_sync",
    "PaperFetcherAgent",
    "ConceptExtractorAgent",
    "IntuitionBuilderAgent",
    "MathSimplifierAgent",
    "ExampleGeneratorAgent",
    "ProgressiveBuilderAgent",
    "ContentPolisherAgent",
    "UnifiedLearningAgent",
]
