"""ArXiv Learning Swarm - Types, enums, and dataclasses."""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..swarm_types import SwarmConfig, SwarmResult

logger = logging.getLogger(__name__)

def format_steps_on_newlines(text: str) -> str:
    """Post-process text to ensure steps are on separate lines.

    Handles patterns like:
    - "Step 1: ... Step 2: ..." -> "Step 1: ...\n\nStep 2: ..."
    - "1. ... 2. ..." -> "1. ...\n\n2. ..."
    """
    if not text:
        return text

    # Pattern 1: "Step N:" followed by text until next "Step M:"
    # Insert newlines before each "Step N:"
    text = re.sub(r'(?<=[.!?:,])\s*(Step\s*\d+)\s*:', r'\n\n\1:', text)

    # Pattern 2: Numbered items like "1." "2." etc
    # Insert newlines before numbered items that follow sentences
    text = re.sub(r'(?<=[.!?])\s*(\d+)\.\s+', r'\n\n\1. ', text)

    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# =============================================================================
# CONFIGURATION
# =============================================================================

class LearningDepth(Enum):
    """How deep to go in explanations."""
    QUICK = "quick"          # 5-minute overview
    STANDARD = "standard"    # Full understanding
    DEEP = "deep"            # Expert-level with all math


class ContentStyle(Enum):
    """Style of content generation."""
    ENGAGING = "engaging"      # Fun, intuitive, builds excitement
    TECHNICAL = "technical"    # More formal, equation-heavy
    VISUAL = "visual"          # Emphasis on diagrams and visualizations


class AudienceLevel(Enum):
    """Target audience starting level."""
    BEGINNER = "beginner"          # High school math
    INTERMEDIATE = "intermediate"  # Undergraduate
    ADVANCED = "advanced"          # Graduate level


@dataclass
class ArxivLearningConfig(SwarmConfig):
    """Configuration for ArxivLearningSwarm."""
    depth: LearningDepth = LearningDepth.STANDARD
    style: ContentStyle = ContentStyle.ENGAGING
    audience: AudienceLevel = AudienceLevel.BEGINNER
    include_code_examples: bool = True
    include_visualizations: bool = True
    include_exercises: bool = True
    build_from_basics: bool = True  # Always start from fundamentals
    max_papers_for_topic: int = 5
    celebration_word: str = "Bingo!"  # Celebration for key insights
    send_telegram: bool = False  # Send to Telegram after generation
    generate_pptx: bool = True  # Generate PowerPoint in addition to PDF
    generate_html: bool = True  # Generate interactive HTML slides
    convert_pptx_to_pdf: bool = True  # Convert PPTX to PDF before Telegram (requires LibreOffice)
    use_lotus: bool = True  # Use LOTUS for semantic paper search/ranking (if available)
    lotus_model: str = "gpt-4o-mini"  # Model for LOTUS semantic operations

    # OPTIMIZATION OPTIONS
    # Modes:
    # - "parallel_deep": RECOMMENDED - parallel per-concept generation (30+ pages, ~90s)
    # - "unified": Fast - single LLM call for all content (~60s, less detailed)
    # - "parallel": Legacy - runs individual agents in parallel
    # - "sequential": Original - step-by-step generation (slowest)
    optimization_mode: str = "parallel_deep"  # Best quality + speed balance

    # Performance tuning
    max_concepts_quick: int = 2  # Limit concepts for QUICK depth
    max_concepts_standard: int = 4  # Limit concepts for STANDARD depth
    use_swarm_cache: bool = True  # Cache results at swarm level

    # SPEED OPTIMIZATIONS
    llm_model: str = "haiku"  # "haiku" (faster) or "sonnet" (higher quality)
    use_fast_predict: bool = True  # Use dspy.Predict (faster) instead of ChainOfThought
    llm_timeout: int = 0  # 0 → resolved in __post_init__ from LLM_TIMEOUT_SECONDS
    max_concurrent_llm: int = 5  # Concurrent LLM calls for parallel_deep mode

    def __post_init__(self) -> None:
        self.name = "ArxivLearningSwarm"
        if self.llm_timeout <= 0:
            from Jotty.core.infrastructure.foundation.config_defaults import LLM_TIMEOUT_SECONDS
            self.llm_timeout = LLM_TIMEOUT_SECONDS
        self.domain = "arxiv_learning"


@dataclass
class PaperInfo:
    """ArXiv paper information."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str
    arxiv_url: str


@dataclass
class Concept:
    """A concept from the paper."""
    name: str
    description: str
    why_it_matters: str
    prerequisites: List[str]
    difficulty: int  # 1-5
    math_required: bool


@dataclass
class LearningSection:
    """A section of learning content."""
    title: str
    content: str
    level: int  # 1=basics, 2=intuition, 3=math, 4=applications, 5=deep
    has_bingo_moment: bool = False
    code_example: str = ""
    visualization_desc: str = ""
    exercises: List[str] = field(default_factory=list)


@dataclass
class LearningContent:
    """Complete learning content for a paper."""
    paper: PaperInfo
    hook: str  # Why should you care?
    concepts: List[Concept]
    sections: List[LearningSection]
    key_insights: List[str]
    summary: str
    next_steps: List[str]
    total_words: int


@dataclass
class ArxivLearningResult(SwarmResult):
    """Result from ArxivLearningSwarm."""
    paper: Optional[PaperInfo] = None
    content: Optional[LearningContent] = None
    learning_time_estimate: str = ""
    concepts_covered: int = 0
    bingo_moments: int = 0
    difficulty_progression: List[int] = field(default_factory=list)
    pdf_path: Optional[str] = None
    pptx_path: Optional[str] = None
    pptx_pdf_path: Optional[str] = None  # PDF converted from PPTX (for Telegram)
    html_path: Optional[str] = None  # Interactive HTML slides


# =============================================================================
# DSPy SIGNATURES - Teaching Philosophy Embedded
# =============================================================================

