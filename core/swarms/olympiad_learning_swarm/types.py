"""Olympiad Learning Swarm - Types, enums, and dataclasses.

World-class educational swarm for olympiad-level mastery.
Supports any subject (Math, Physics, Chemistry, CS, etc.).
"""

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..swarm_types import SwarmConfig, SwarmResult

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class Subject(Enum):
    """Supported subjects for olympiad preparation."""
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    BIOLOGY = "biology"
    ASTRONOMY = "astronomy"
    GENERAL = "general"


class DifficultyTier(Enum):
    """Difficulty progression tiers."""
    FOUNDATION = "foundation"          # Building blocks everyone needs
    INTERMEDIATE = "intermediate"      # School competition level
    ADVANCED = "advanced"              # National olympiad level
    OLYMPIAD = "olympiad"              # International olympiad level (IMO/IPhO/IOI)
    BEYOND = "beyond"                  # Research-level extensions


class LessonDepth(Enum):
    """How deep to go in a single session."""
    QUICK = "quick"              # 15-min focused sprint
    STANDARD = "standard"        # 45-min full lesson
    DEEP = "deep"                # 90-min deep dive with all problems
    MARATHON = "marathon"        # Multi-hour comprehensive session


class TeachingMode(Enum):
    """Teaching approach for the session."""
    CONCEPT_BUILD = "concept_build"        # Learn a new concept from scratch
    PROBLEM_DRILL = "problem_drill"        # Practice problems on known concept
    PATTERN_HUNT = "pattern_hunt"          # Discover patterns across problems
    STRATEGY_LAB = "strategy_lab"          # Learn problem-solving strategies
    COMPETITION_SIM = "competition_sim"    # Simulate competition conditions
    REVIEW = "review"                      # Review and reinforce weak areas


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OlympiadLearningConfig(SwarmConfig):
    """Configuration for OlympiadLearningSwarm."""

    # Core parameters
    subject: Subject = Subject.MATHEMATICS
    student_name: str = "Student"
    depth: LessonDepth = LessonDepth.STANDARD
    mode: TeachingMode = TeachingMode.CONCEPT_BUILD
    target_tier: DifficultyTier = DifficultyTier.OLYMPIAD

    # Content generation
    include_proofs: bool = True
    include_code: bool = True
    include_visualizations: bool = True
    include_practice_problems: bool = True
    include_competition_tips: bool = True
    include_common_mistakes: bool = True
    include_connections: bool = True
    build_from_basics: bool = True

    # Problem counts per tier
    foundation_problems: int = 3
    intermediate_problems: int = 3
    advanced_problems: int = 2
    olympiad_problems: int = 2

    # Celebration and engagement
    celebration_word: str = "Brilliant!"

    # Output options
    generate_pdf: bool = True
    generate_html: bool = True
    send_telegram: bool = False

    # Optimization
    optimization_mode: str = "parallel_deep"
    max_concurrent_llm: int = 5
    use_swarm_cache: bool = True
    llm_model: str = "haiku"
    use_fast_predict: bool = True
    llm_timeout: int = 0

    # Problem set options
    max_problems_per_tier: int = 5
    include_hints: bool = True
    include_full_solutions: bool = True

    def __post_init__(self) -> None:
        self.name = "OlympiadLearningSwarm"
        self.domain = "olympiad_learning"
        if self.llm_timeout <= 0:
            from Jotty.core.foundation.config_defaults import LLM_TIMEOUT_SECONDS
            self.llm_timeout = LLM_TIMEOUT_SECONDS


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BuildingBlock:
    """A prerequisite building block that must be understood first."""
    name: str
    description: str
    why_needed: str
    quick_review: str
    check_question: str
    difficulty: int = 1  # 1-5


@dataclass
class ConceptCore:
    """Core concept being taught."""
    name: str
    description: str
    why_it_matters: str
    real_world_hook: str
    prerequisites: List[str]
    difficulty: int  # 1-5
    key_insight: str
    common_misconceptions: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)


@dataclass
class PatternEntry:
    """A problem-solving pattern."""
    name: str
    description: str
    when_to_use: str
    example_trigger: str
    template: str


@dataclass
class Problem:
    """A practice problem with solution."""
    statement: str
    tier: DifficultyTier
    hints: List[str]
    solution: str
    strategy_used: str
    time_estimate_minutes: int
    source: str = ""
    key_insight: str = ""
    common_mistakes: List[str] = field(default_factory=list)
    relates_to_pattern: str = ""
    narrative_context: str = ""


@dataclass
class StrategyCard:
    """A problem-solving strategy."""
    name: str
    description: str
    when_to_use: str
    steps: List[str]
    example_problem: str
    example_solution: str
    pitfalls: List[str] = field(default_factory=list)


@dataclass
class MistakeEntry:
    """A common mistake and how to avoid it."""
    description: str
    why_it_happens: str
    how_to_avoid: str
    example_wrong: str
    example_correct: str


@dataclass
class LessonSection:
    """A section of the lesson."""
    title: str
    content: str
    level: int  # 1=foundation, 2=core, 3=patterns, 4=problems, 5=olympiad
    has_breakthrough_moment: bool = False
    code_example: str = ""
    problems: List[Problem] = field(default_factory=list)
    visualization_desc: str = ""
    transition_text: str = ""
    breakthrough_content: str = ""


@dataclass
class LessonContent:
    """Complete lesson content."""
    subject: Subject
    topic: str
    student_name: str
    building_blocks: List[BuildingBlock]
    core_concepts: List[ConceptCore]
    patterns: List[PatternEntry]
    strategies: List[StrategyCard]
    problems: List[Problem]
    mistakes: List[MistakeEntry]
    sections: List[LessonSection]
    key_insights: List[str]
    summary: str
    next_topics: List[str]
    total_words: int
    competition_tips: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    running_example: str = ""
    rank_tips: List[str] = field(default_factory=list)


@dataclass
class OlympiadLearningResult(SwarmResult):
    """Result from OlympiadLearningSwarm."""
    content: Optional[LessonContent] = None
    student_name: str = ""
    topic: str = ""
    subject: Optional[Subject] = None
    learning_time_estimate: str = ""
    concepts_covered: int = 0
    problems_generated: int = 0
    breakthrough_moments: int = 0
    difficulty_progression: List[int] = field(default_factory=list)
    pdf_path: Optional[str] = None
    html_path: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

def format_steps_on_newlines(text: str) -> str:
    """Post-process text to ensure steps are on separate lines."""
    if not text:
        return text

    text = re.sub(r'(?<=[.!?:,])\s*(Step\s*\d+)\s*:', r'\n\n\1:', text)
    text = re.sub(r'(?<=[.!?])\s*(\d+)\.\s+', r'\n\n\1. ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def tier_to_level(tier: DifficultyTier) -> int:
    """Convert tier enum to numeric level."""
    mapping = {
        DifficultyTier.FOUNDATION: 1,
        DifficultyTier.INTERMEDIATE: 2,
        DifficultyTier.ADVANCED: 3,
        DifficultyTier.OLYMPIAD: 4,
        DifficultyTier.BEYOND: 5,
    }
    return mapping.get(tier, 1)


__all__ = [
    'Subject', 'DifficultyTier', 'LessonDepth', 'TeachingMode',
    'OlympiadLearningConfig', 'BuildingBlock', 'ConceptCore',
    'PatternEntry', 'Problem', 'StrategyCard', 'MistakeEntry',
    'LessonSection', 'LessonContent', 'OlympiadLearningResult',
    'format_steps_on_newlines', 'tier_to_level',
]
