"""Perspective Learning Swarm - Types, enums, and dataclasses.

Multi-perspective educational swarm that explores ANY topic from 6 distinct
perspectives in 4 languages, producing professional PDF + HTML output.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..swarm_types import SwarmConfig, SwarmResult

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PerspectiveType(Enum):
    """The 6 learning perspectives."""

    INTUITIVE_VISUAL = "intuitive_visual"
    STRUCTURED_FRAMEWORK = "structured_framework"
    STORYTELLING = "storytelling"
    DEBATE_CRITICAL = "debate_critical"
    HANDS_ON_PROJECT = "hands_on_project"
    REAL_WORLD_APPLICATION = "real_world_application"


class Language(Enum):
    """Supported languages for multilingual content."""

    ENGLISH = "english"
    HINDI = "hindi"
    KANNADA = "kannada"
    FRENCH = "french"


class AgeGroup(Enum):
    """Target age groups."""

    EARLY_PRIMARY = "early_primary"  # K-2
    PRIMARY = "primary"  # 3-5
    MIDDLE = "middle"  # 6-8
    HIGH = "high"  # 9-12
    GENERAL = "general"


class ContentDepth(Enum):
    """How deep to go in a single session."""

    QUICK = "quick"  # 15 min focused sprint
    STANDARD = "standard"  # 45 min full lesson
    DEEP = "deep"  # 90 min deep dive
    COMPREHENSIVE = "comprehensive"  # 2+ hours


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PerspectiveLearningConfig(SwarmConfig):
    """Configuration for PerspectiveLearningSwarm."""

    # Core parameters
    student_name: str = "Student"
    age_group: AgeGroup = AgeGroup.PRIMARY
    depth: ContentDepth = ContentDepth.STANDARD
    languages: List[Language] = field(
        default_factory=lambda: [
            Language.ENGLISH,
            Language.HINDI,
            Language.KANNADA,
            Language.FRENCH,
        ]
    )
    perspectives: List[PerspectiveType] = field(default_factory=lambda: list(PerspectiveType))

    # Celebration and engagement
    celebration_word: str = "Wonderful!"

    # Output options
    generate_pdf: bool = True
    generate_html: bool = True
    send_telegram: bool = False

    # Optimization
    optimization_mode: str = "parallel_deep"
    max_concurrent_llm: int = 5
    llm_model: str = "haiku"
    use_fast_predict: bool = True
    llm_timeout: int = 0

    def __post_init__(self) -> None:
        self.name = "PerspectiveLearningSwarm"
        self.domain = "perspective_learning"
        if self.llm_timeout <= 0:
            from Jotty.core.infrastructure.foundation.config_defaults import LLM_TIMEOUT_SECONDS

            self.llm_timeout = LLM_TIMEOUT_SECONDS


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PerspectiveSection:
    """A section of content from one perspective."""

    perspective: PerspectiveType
    title: str
    content: str
    key_takeaway: str
    activity: str = ""
    visual_description: str = ""


@dataclass
class LanguageContent:
    """Content in a specific language."""

    language: Language
    summary: str
    key_vocabulary: List[str] = field(default_factory=list)
    reflection_prompts: List[str] = field(default_factory=list)
    activity: str = ""
    slogans: List[str] = field(default_factory=list)


@dataclass
class DebatePoint:
    """A point in a debate/critical thinking section."""

    position: str
    argument: str
    evidence: str
    counterargument: str
    critical_question: str


@dataclass
class ProjectActivity:
    """A hands-on project activity."""

    title: str
    description: str
    materials: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    learning_outcome: str = ""
    assessment_criteria: str = ""


@dataclass
class FrameworkModel:
    """A mental model or framework for structured thinking."""

    name: str
    description: str
    how_to_use: str
    visual_layout: str
    example_applied: str


@dataclass
class LessonContent:
    """Complete assembled lesson content with all perspectives and languages."""

    topic: str
    student_name: str
    central_idea: str
    learning_objectives: List[str]
    key_concepts: List[Dict[str, str]]
    running_example: str
    vocabulary: List[Dict[str, str]]
    perspectives: List[PerspectiveSection]
    language_sections: List[LanguageContent]
    key_insights: List[str]
    parent_guide: str
    socratic_questions: List[str]
    total_words: int
    key_takeaways: List[str] = None
    transdisciplinary_connections: str = ""


@dataclass
class PerspectiveLearningResult(SwarmResult):
    """Result from PerspectiveLearningSwarm."""

    content: Optional[LessonContent] = None
    student_name: str = ""
    topic: str = ""
    perspectives_generated: int = 0
    languages_generated: int = 0
    pdf_path: Optional[str] = None
    html_path: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================


def format_steps_on_newlines(text: str) -> str:
    """Post-process text to ensure steps are on separate lines."""
    if not text:
        return text

    text = re.sub(r"(?<=[.!?:,])\s*(Step\s*\d+)\s*:", r"\n\n\1:", text)
    text = re.sub(r"(?<=[.!?])\s*(\d+)\.\s+", r"\n\n\1. ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


PERSPECTIVE_LABELS = {
    PerspectiveType.INTUITIVE_VISUAL: "See It Clearly",
    PerspectiveType.STRUCTURED_FRAMEWORK: "Think It Through",
    PerspectiveType.STORYTELLING: "Feel the Story",
    PerspectiveType.DEBATE_CRITICAL: "Debate It",
    PerspectiveType.HANDS_ON_PROJECT: "Build It",
    PerspectiveType.REAL_WORLD_APPLICATION: "Live It",
}

LANGUAGE_LABELS = {
    Language.HINDI: "\u0939\u093f\u0928\u094d\u0926\u0940 \u092e\u0947\u0902",
    Language.KANNADA: "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1\u0ca6\u0cb2\u0ccd\u0cb2\u0cbf",
    Language.FRENCH: "En Fran\u00e7ais",
}


__all__ = [
    "PerspectiveType",
    "Language",
    "AgeGroup",
    "ContentDepth",
    "PerspectiveLearningConfig",
    "PerspectiveSection",
    "LanguageContent",
    "DebatePoint",
    "ProjectActivity",
    "FrameworkModel",
    "LessonContent",
    "PerspectiveLearningResult",
    "format_steps_on_newlines",
    "PERSPECTIVE_LABELS",
    "LANGUAGE_LABELS",
]
