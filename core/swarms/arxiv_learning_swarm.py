"""
ArXiv Learning Swarm - World-Class Paper Understanding & Education
===================================================================

Production-grade swarm for:
- Fetching papers from ArXiv by ID or topic
- Creating engaging, intuitive explanations
- Building understanding from basics to advanced
- Making complex math accessible and fun
- Explaining the "why" behind every concept

Teaching Philosophy (Internal - NOT in outputs):
- Build intuition FIRST, then math
- Always explain WHY something matters before HOW it works
- Use analogies and real-world examples
- Celebrate understanding with "Bingo!" moments
- Progress: Basics → Intuition → Math → Applications → Deep Dive
- Every concept earns its place by solving a problem

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                      ARXIV LEARNING SWARM                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Paper        │  │   Concept      │  │   Intuition    │            │
│  │   Fetcher      │  │   Extractor    │  │   Builder      │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Math         │  │   Example      │  │   Progressive  │            │
│  │   Simplifier   │  │   Generator    │  │   Builder      │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     LEARNING CONTENT GENERATOR                   │   │
│  │   Creates engaging, progressive learning content                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.arxiv_learning_swarm import ArxivLearningSwarm, learn_paper

    # By ArXiv ID
    result = await learn_paper("2301.07041")  # Specific paper

    # By topic
    result = await learn_paper(topic="transformers attention mechanism")

    # With custom depth
    result = await learn_paper("1706.03762", depth="deep")  # Attention Is All You Need

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import re
import dspy
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from xml.etree import ElementTree

from .base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam
from ..agents.base import DomainAgent, DomainAgentConfig

# Import Telegram sender tools
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "skills" / "telegram-sender"))
    from tools import send_telegram_message_tool, send_telegram_file_tool
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    send_telegram_message_tool = None
    send_telegram_file_tool = None

# Import LOTUS for enhanced semantic operations
try:
    from ..skills.research.lotus_arxiv import LotusArxiv, LotusArxivConfig, is_lotus_available
    LOTUS_AVAILABLE = is_lotus_available()
except ImportError:
    LOTUS_AVAILABLE = False
    LotusArxiv = None
    LotusArxivConfig = None

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    llm_timeout: int = 120  # Timeout per LLM call in seconds
    max_concurrent_llm: int = 5  # Concurrent LLM calls for parallel_deep mode

    def __post_init__(self):
        self.name = "ArxivLearningSwarm"
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

class ConceptExtractionSignature(dspy.Signature):
    """Extract key concepts from a paper.

    You are extracting concepts that need to be TAUGHT, not just listed.
    For each concept, identify:
    1. What problem does it solve? (the WHY)
    2. What do you need to know first? (prerequisites)
    3. How hard is it to understand?

    Think like a teacher planning a lesson.
    """
    paper_title: str = dspy.InputField(desc="Paper title")
    abstract: str = dspy.InputField(desc="Paper abstract")
    full_text_summary: str = dspy.InputField(desc="Summary of full paper if available")

    concepts: str = dspy.OutputField(desc="JSON list of concepts with name, description, why_it_matters, prerequisites, difficulty(1-5)")
    learning_order: str = dspy.OutputField(desc="Recommended order to learn concepts, separated by |")
    key_innovation: str = dspy.OutputField(desc="The ONE key innovation in simple terms")


class IntuitionBuilderSignature(dspy.Signature):
    """Build intuition for a concept BEFORE diving into math.

    Your goal is to make the reader FEEL why this concept makes sense.

    RULES:
    1. Start with a real-world problem or analogy
    2. Build up step by step - no jumping ahead
    3. Use "imagine if..." scenarios
    4. Make the reader predict what comes next
    5. Celebrate understanding with enthusiasm!

    NO jargon until intuition is solid.
    """
    concept: str = dspy.InputField(desc="Concept to explain")
    why_it_matters: str = dspy.InputField(desc="Why this concept matters")
    audience_level: str = dspy.InputField(desc="Starting knowledge level")
    prerequisites: str = dspy.InputField(desc="What they should already know")

    hook: str = dspy.OutputField(desc="Opening hook that grabs attention - why should they care?")
    analogy: str = dspy.OutputField(desc="Real-world analogy that captures the essence")
    intuition_build: str = dspy.OutputField(desc="Step-by-step intuition building, each step on new line")
    aha_moment: str = dspy.OutputField(desc="The 'Bingo!' moment where it all clicks")
    check_understanding: str = dspy.OutputField(desc="Question to verify they got it")


class MathSimplifierSignature(dspy.Signature):
    """Make math accessible by building from basics.

    Your job is to make math feel INEVITABLE, not arbitrary.

    APPROACH:
    1. Start with what they know (basic algebra/calculus)
    2. Each new symbol EARNS its place by solving a problem
    3. Show WHY the math looks the way it does
    4. Connect equations to the intuition built earlier
    5. Use concrete numbers before variables

    Math should feel like a natural next step, not a wall.
    """
    concept: str = dspy.InputField(desc="Concept with its math")
    intuition: str = dspy.InputField(desc="The intuition already built")
    equations: str = dspy.InputField(desc="Key equations to explain")
    audience_level: str = dspy.InputField(desc="Math background")

    math_motivation: str = dspy.OutputField(desc="Why we need math here (the problem it solves)")
    building_blocks: str = dspy.OutputField(desc="Basic math building blocks needed, each on new line")
    step_by_step: str = dspy.OutputField(desc="Step-by-step derivation. Format: 'Step 1: [explanation]\\nStep 2: [explanation]\\n...' - EACH STEP MUST START ON A NEW LINE")
    concrete_example: str = dspy.OutputField(desc="Worked example with actual numbers, each calculation step on new line")
    connection_to_intuition: str = dspy.OutputField(desc="How this math connects to earlier intuition")


class ExampleGeneratorSignature(dspy.Signature):
    """Generate examples that reinforce understanding.

    Examples should:
    1. Start simple, get progressively harder
    2. Cover different angles of the concept
    3. Include "what if" variations
    4. Have clear, checkable answers

    Make examples that teach, not just test.
    """
    concept: str = dspy.InputField(desc="Concept to exemplify")
    intuition: str = dspy.InputField(desc="Intuition built")
    math_explanation: str = dspy.InputField(desc="Math explanation")

    simple_example: str = dspy.OutputField(desc="Simple example anyone can follow")
    intermediate_example: str = dspy.OutputField(desc="Example that tests understanding")
    challenging_example: str = dspy.OutputField(desc="Example that pushes boundaries")
    code_example: str = dspy.OutputField(desc="Python code demonstrating the concept")
    what_if_variations: str = dspy.OutputField(desc="What-if variations to explore, separated by |")


class ProgressiveBuilderSignature(dspy.Signature):
    """Build complete learning content progressively.

    Structure (ALWAYS in this order):
    1. THE HOOK - Why should anyone care? What problem does this solve?
    2. THE BASICS - What do we need to know first? (brief review)
    3. THE INTUITION - Build understanding without math
    4. THE MATH - Now that we get it, here's the precise formulation
    5. THE APPLICATION - See it in action
    6. THE DEEP DIVE - For those who want more

    Each section builds on the previous. No skipping!
    """
    paper_info: str = dspy.InputField(desc="Paper information")
    concepts: str = dspy.InputField(desc="Concepts to cover")
    intuitions: str = dspy.InputField(desc="Intuitions built")
    math_explanations: str = dspy.InputField(desc="Math explanations")
    examples: str = dspy.InputField(desc="Examples generated")
    celebration_word: str = dspy.InputField(desc="Word to celebrate insights")

    complete_content: str = dspy.OutputField(desc="Complete learning content with all sections")
    key_insights: str = dspy.OutputField(desc="Key insights (celebration moments), separated by |")
    summary: str = dspy.OutputField(desc="Concise summary of what was learned")
    next_steps: str = dspy.OutputField(desc="What to learn next, separated by |")


class ContentPolisherSignature(dspy.Signature):
    """Polish content to be engaging and clear.

    Make sure:
    1. Language is conversational, not academic
    2. Enthusiasm comes through (but not fake)
    3. Complex ideas have simple explanations
    4. Flow is smooth between sections
    5. Reader feels capable, not intimidated

    The reader should WANT to keep reading.
    """
    draft_content: str = dspy.InputField(desc="Draft learning content")
    style: str = dspy.InputField(desc="Desired style")
    audience: str = dspy.InputField(desc="Target audience")

    polished_content: str = dspy.OutputField(desc="Polished, engaging content")
    engagement_score: float = dspy.OutputField(desc="Estimated engagement 0-100")
    clarity_score: float = dspy.OutputField(desc="Clarity score 0-100")


# =============================================================================
# UNIFIED LEARNING SIGNATURE - MEGA-OPTIMIZATION
# =============================================================================
# This single signature replaces 8+ separate LLM calls with ONE comprehensive call.
# Benefits: 80% faster, better coherence, full context awareness, no huge prompts.

class UnifiedConceptLearningSignature(dspy.Signature):
    """Generate complete learning content for ALL concepts in ONE pass.

    You are a world-class educator creating an engaging learning experience.
    For each concept, provide intuition, math (if needed), and examples.

    TEACHING PHILOSOPHY:
    1. Hook first - why should they care?
    2. Intuition before math - make it FEEL right
    3. Math earns its place - solve real problems
    4. Examples reinforce - simple to challenging
    5. Celebrate insights with enthusiasm

    OUTPUT FORMAT (JSON):
    {
        "hook": "Opening hook - why this paper matters",
        "concepts": [
            {
                "name": "Concept Name",
                "analogy": "Real-world analogy",
                "intuition": "Detailed step-by-step intuition building",
                "aha_moment": "The key insight moment",
                "math_motivation": "Why we need math here",
                "math_steps": "Detailed step-by-step math explanation",
                "simple_example": "Easy example with walkthrough",
                "code_example": "Python code demonstrating concept"
            }
        ],
        "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
        "summary": "Comprehensive summary of what was learned",
        "next_steps": ["What to learn next 1", "What to learn next 2"]
    }
    """
    paper_title: str = dspy.InputField(desc="Paper title")
    paper_abstract: str = dspy.InputField(desc="Paper abstract (key content)")
    concepts_json: str = dspy.InputField(desc="JSON list of concepts with name, description, why_it_matters, difficulty")
    audience_level: str = dspy.InputField(desc="Target audience: beginner, intermediate, advanced")
    celebration_word: str = dspy.InputField(desc="Word to celebrate insights (e.g., Bingo!)")

    learning_content_json: str = dspy.OutputField(desc="Complete learning content as JSON (see format above)")


class SingleConceptDeepSignature(dspy.Signature):
    """Generate DEEP, comprehensive learning content for ONE concept.

    Create engaging, thorough educational content with:
    - Rich analogies and real-world connections
    - Step-by-step intuition building (multiple paragraphs)
    - Detailed math derivations with explanations
    - Multiple examples from simple to advanced
    - Working code with comments

    Be thorough - this will become multiple pages of content.
    """
    concept_name: str = dspy.InputField(desc="Concept name")
    concept_description: str = dspy.InputField(desc="What this concept is")
    why_it_matters: str = dspy.InputField(desc="Why this concept is important")
    paper_context: str = dspy.InputField(desc="Paper title and key context")
    audience_level: str = dspy.InputField(desc="beginner/intermediate/advanced")

    analogy: str = dspy.OutputField(desc="Rich real-world analogy (2-3 sentences)")
    intuition: str = dspy.OutputField(desc="Detailed intuition building (3-5 paragraphs, use newlines)")
    aha_moment: str = dspy.OutputField(desc="The key insight that makes it click")
    math_motivation: str = dspy.OutputField(desc="Why math is needed here")
    math_steps: str = dspy.OutputField(desc="Step-by-step math with explanations (use newlines between steps)")
    simple_example: str = dspy.OutputField(desc="Simple worked example with explanation")
    advanced_example: str = dspy.OutputField(desc="More challenging example")
    code_example: str = dspy.OutputField(desc="Python code with detailed comments (10-20 lines)")


# =============================================================================
# AGENTS
# =============================================================================

class BaseLearningAgent(DomainAgent):
    """Base class for learning agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 90):
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=None, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus
        self.learned_context = learned_context
        self.model = model
        self.use_fast_predict = use_fast_predict
        self.llm_timeout = llm_timeout
        self._lm = None

    def _get_lm(self):
        """Get or create LLM instance with configured model."""
        if self._lm is None:
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                self._lm = DirectClaudeCLI(model=self.model)
                dspy.configure(lm=self._lm)
            except Exception as e:
                logger.warning(f"Could not init LLM: {e}")
        return self._lm

    def _create_module(self, signature):
        """Create dspy module - Predict (fast) or ChainOfThought (reasoning)."""
        self._get_lm()
        if self.use_fast_predict:
            return dspy.Predict(signature)
        else:
            return dspy.ChainOfThought(signature)

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass


class PaperFetcherAgent(BaseLearningAgent):
    """Fetches papers from ArXiv with anti-rate-limit measures and caching."""

    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    CACHE_DIR = "/tmp/arxiv_cache"

    # Random user agents to rotate
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    ]

    def _get_random_headers(self) -> Dict[str, str]:
        """Get random browser-like headers."""
        import random
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "application/xml, text/xml, */*",
            "Accept-Language": random.choice(["en-US,en;q=0.9", "en-GB,en;q=0.8", "en;q=0.7"]),
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "DNT": "1",
        }

    def _get_proxy(self) -> Optional[str]:
        """Get VPN/proxy URL from environment."""
        import os
        # Check for proxy in environment variables
        proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or os.environ.get('ARXIV_PROXY')
        return proxy

    def _get_cache_path(self, arxiv_id: str) -> Path:
        """Get cache file path for an arxiv ID."""
        import os
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        safe_id = arxiv_id.replace("/", "_").replace(".", "_")
        return Path(self.CACHE_DIR) / f"{safe_id}.json"

    def _load_from_cache(self, arxiv_id: str) -> Optional[PaperInfo]:
        """Load paper info from cache if available."""
        cache_path = self._get_cache_path(arxiv_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"📦 Loaded from cache: {arxiv_id}")
                return PaperInfo(**data)
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")
        return None

    def _save_to_cache(self, arxiv_id: str, paper: PaperInfo):
        """Save paper info to cache."""
        cache_path = self._get_cache_path(arxiv_id)
        try:
            data = {
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'categories': paper.categories,
                'published': paper.published,
                'pdf_url': paper.pdf_url,
                'arxiv_url': paper.arxiv_url,
            }
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Cached: {arxiv_id}")
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    async def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Add random delay to avoid rate limiting."""
        import random
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)

    async def _fetch_with_retry(self, url: str, params: Dict, max_retries: int = 5) -> Optional[str]:
        """Fetch URL with retry logic, random headers, and optional proxy."""
        import random

        proxy = self._get_proxy()
        connector = None

        for attempt in range(max_retries):
            # Add longer random delay before each request (except first)
            if attempt > 0:
                # Exponential backoff with longer delays: 10s, 30s, 60s, 120s
                delay = (10 * (2 ** attempt)) + random.uniform(5, 15)
                logger.info(f"⏳ Retry {attempt + 1}/{max_retries} after {delay:.1f}s delay...")
                await asyncio.sleep(delay)
            else:
                # Initial delay of 2-5 seconds
                await self._random_delay(2.0, 5.0)

            headers = self._get_random_headers()

            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    kwargs = {"params": params, "timeout": aiohttp.ClientTimeout(total=30)}
                    if proxy:
                        kwargs["proxy"] = proxy
                        logger.debug(f"Using proxy: {proxy[:20]}...")

                    async with session.get(url, **kwargs) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        elif resp.status == 429:
                            logger.warning(f"⚠️ Rate limited (429), will retry...")
                            continue
                        else:
                            logger.error(f"ArXiv API error: {resp.status}")
                            if attempt < max_retries - 1:
                                continue
                            return None
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Request timeout, will retry...")
                continue
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    continue
                return None

        logger.error(f"❌ All {max_retries} attempts failed")
        return None

    async def fetch_by_id(self, arxiv_id: str) -> Optional[PaperInfo]:
        """Fetch paper by ArXiv ID with caching."""
        # Clean the ID
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]

        # Check cache first
        cached = self._load_from_cache(arxiv_id)
        if cached:
            return cached

        try:
            params = {"id_list": arxiv_id}
            xml_text = await self._fetch_with_retry(self.ARXIV_API_URL, params)
            if xml_text:
                paper = self._parse_arxiv_response(xml_text)
                if paper:
                    self._save_to_cache(arxiv_id, paper)
                return paper
            return None
        except Exception as e:
            logger.error(f"Paper fetch failed: {e}")
            return None

    async def search_by_topic(self, topic: str, max_results: int = 5) -> List[PaperInfo]:
        """Search papers by topic."""
        try:
            params = {
                "search_query": f"all:{topic}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            xml_text = await self._fetch_with_retry(self.ARXIV_API_URL, params)
            if xml_text:
                return self._parse_arxiv_search_response(xml_text)
            return []
        except Exception as e:
            logger.error(f"Topic search failed: {e}")
            return []

    async def search_and_rank_with_lotus(
        self,
        topic: str,
        max_results: int = 5,
        rank_by: str = "Which {abstract} is most exciting and impactful for learning?",
        lotus_model: str = "gpt-4o-mini"
    ) -> List[PaperInfo]:
        """
        Search papers using LOTUS with semantic ranking.

        Uses LOTUS sem_topk for intelligent paper ranking based on learning value,
        excitement, or custom criteria.

        Args:
            topic: Search topic
            max_results: Number of top papers to return
            rank_by: Natural language ranking criterion
            lotus_model: Model for LOTUS operations

        Returns:
            List of PaperInfo objects, semantically ranked
        """
        if not LOTUS_AVAILABLE:
            logger.warning("LOTUS not available, falling back to standard search")
            return await self.search_by_topic(topic, max_results)

        try:
            logger.info(f"🌸 LOTUS-powered search: {topic}")

            lotus_arxiv = LotusArxiv(LotusArxivConfig(model=lotus_model))

            # Search and rank with LOTUS
            ranked_df = await lotus_arxiv.search_and_rank(
                query=topic,
                rank_by=rank_by,
                limit=max_results * 3,  # Search more, rank to get best
                top_k=max_results
            )

            if ranked_df.empty:
                logger.warning("LOTUS search returned no results")
                return await self.search_by_topic(topic, max_results)

            # Convert DataFrame to PaperInfo objects
            papers = []
            for _, row in ranked_df.iterrows():
                # Extract arxiv_id from URL or id column
                arxiv_id = ""
                if "arxiv_id" in row:
                    arxiv_id = str(row["arxiv_id"])
                elif "url" in row:
                    url = str(row["url"])
                    if "/abs/" in url:
                        arxiv_id = url.split("/abs/")[-1]
                    elif "/pdf/" in url:
                        arxiv_id = url.split("/pdf/")[-1].replace(".pdf", "")

                papers.append(PaperInfo(
                    arxiv_id=arxiv_id,
                    title=str(row.get("title", "")),
                    authors=row.get("authors", []) if isinstance(row.get("authors"), list) else [],
                    abstract=str(row.get("abstract", "")),
                    categories=[],
                    published=str(row.get("published", ""))[:10] if row.get("published") else "",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
                    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
                ))

            logger.info(f"  Found {len(papers)} semantically ranked papers")
            return papers

        except Exception as e:
            logger.error(f"LOTUS search failed: {e}, falling back to standard search")
            return await self.search_by_topic(topic, max_results)

    def _parse_arxiv_response(self, xml_text: str) -> Optional[PaperInfo]:
        """Parse single paper from ArXiv XML response."""
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            root = ElementTree.fromstring(xml_text)

            entry = root.find('atom:entry', ns)
            if entry is None:
                return None

            # Extract ID
            id_elem = entry.find('atom:id', ns)
            arxiv_id = id_elem.text.split('/abs/')[-1] if id_elem is not None else ""

            # Extract title
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)

            # Extract abstract
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""

            # Extract categories
            categories = []
            for cat in entry.findall('arxiv:primary_category', ns):
                if cat.get('term'):
                    categories.append(cat.get('term'))
            for cat in entry.findall('atom:category', ns):
                if cat.get('term') and cat.get('term') not in categories:
                    categories.append(cat.get('term'))

            # Extract published date
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text[:10] if published_elem is not None else ""

            # Build URLs
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

            self._broadcast("paper_fetched", {'arxiv_id': arxiv_id, 'title': title[:50]})

            return PaperInfo(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=published,
                pdf_url=pdf_url,
                arxiv_url=arxiv_url
            )

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    def _parse_arxiv_search_response(self, xml_text: str) -> List[PaperInfo]:
        """Parse multiple papers from ArXiv search response."""
        papers = []
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            root = ElementTree.fromstring(xml_text)

            for entry in root.findall('atom:entry', ns):
                # Similar parsing as single paper
                id_elem = entry.find('atom:id', ns)
                arxiv_id = id_elem.text.split('/abs/')[-1] if id_elem is not None else ""

                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)

                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None else ""

                categories = []
                for cat in entry.findall('atom:category', ns):
                    if cat.get('term'):
                        categories.append(cat.get('term'))

                published_elem = entry.find('atom:published', ns)
                published = published_elem.text[:10] if published_elem is not None else ""

                papers.append(PaperInfo(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    categories=categories,
                    published=published,
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}"
                ))

            self._broadcast("papers_searched", {'count': len(papers)})

        except Exception as e:
            logger.error(f"Search parse error: {e}")

        return papers


class ConceptExtractorAgent(BaseLearningAgent):
    """Extracts concepts from papers."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._extractor = self._create_module(ConceptExtractionSignature)

    async def extract(self, paper: PaperInfo) -> List[Concept]:
        """Extract concepts from paper."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._extractor(
                paper_title=paper.title,
                abstract=paper.abstract,
                full_text_summary=f"Based on abstract only{context_suffix}"
            )

            try:
                # Strip markdown code blocks if present
                concepts_json = str(result.concepts)
                if concepts_json.startswith('```'):
                    # Remove ```json and ``` markers
                    lines = concepts_json.split('\n')
                    lines = [l for l in lines if not l.strip().startswith('```')]
                    concepts_json = '\n'.join(lines)
                concepts_data = json.loads(concepts_json)
            except Exception as e:
                logger.warning(f"Failed to parse concepts JSON: {e}")
                concepts_data = []

            concepts = []
            for c in concepts_data:
                # Ensure prerequisites is a list, not a string
                prereqs = c.get('prerequisites', [])
                if isinstance(prereqs, str):
                    # Convert string to list (split by comma or just wrap it)
                    prereqs = [p.strip() for p in prereqs.split(',') if p.strip()] if ',' in prereqs else [prereqs]
                elif not isinstance(prereqs, list):
                    prereqs = []

                concepts.append(Concept(
                    name=c.get('name', ''),
                    description=c.get('description', ''),
                    why_it_matters=c.get('why_it_matters', ''),
                    prerequisites=prereqs,
                    difficulty=int(c.get('difficulty', 3)),
                    math_required=c.get('math_required', True)
                ))

            self._broadcast("concepts_extracted", {'count': len(concepts)})

            return concepts

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []


class IntuitionBuilderAgent(BaseLearningAgent):
    """Builds intuition for concepts."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._builder = self._create_module(IntuitionBuilderSignature)

    async def build(
        self,
        concept: Concept,
        audience_level: str
    ) -> Dict[str, Any]:
        """Build intuition for a concept."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._builder(
                concept=f"{concept.name}: {concept.description}{context_suffix}",
                why_it_matters=concept.why_it_matters,
                audience_level=audience_level,
                prerequisites=", ".join(concept.prerequisites) if concept.prerequisites else "Basic math"
            )

            self._broadcast("intuition_built", {'concept': concept.name})

            return {
                'hook': str(result.hook),
                'analogy': str(result.analogy),
                'intuition_build': str(result.intuition_build),
                'aha_moment': str(result.aha_moment),
                'check_understanding': str(result.check_understanding)
            }

        except Exception as e:
            logger.error(f"Intuition building failed: {e}")
            return {}


class MathSimplifierAgent(BaseLearningAgent):
    """Simplifies math to be accessible."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._simplifier = self._create_module(MathSimplifierSignature)

    async def simplify(
        self,
        concept: Concept,
        intuition: Dict[str, Any],
        audience_level: str
    ) -> Dict[str, Any]:
        """Simplify math for a concept."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._simplifier(
                concept=f"{concept.name}: {concept.description}{context_suffix}",
                intuition=intuition.get('intuition_build', ''),
                equations="Key equations from the paper",
                audience_level=audience_level
            )

            self._broadcast("math_simplified", {'concept': concept.name})

            return {
                'math_motivation': str(result.math_motivation),
                'building_blocks': str(result.building_blocks),
                'step_by_step': str(result.step_by_step),
                'concrete_example': str(result.concrete_example),
                'connection_to_intuition': str(result.connection_to_intuition)
            }

        except Exception as e:
            logger.error(f"Math simplification failed: {e}")
            return {}


class ExampleGeneratorAgent(BaseLearningAgent):
    """Generates examples to reinforce learning."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._generator = self._create_module(ExampleGeneratorSignature)

    async def generate(
        self,
        concept: Concept,
        intuition: Dict[str, Any],
        math: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate examples for a concept."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._generator(
                concept=f"{concept.name}: {concept.description}{context_suffix}",
                intuition=intuition.get('intuition_build', ''),
                math_explanation=math.get('step_by_step', '')
            )

            what_ifs = [w.strip() for w in str(result.what_if_variations).split('|') if w.strip()]

            self._broadcast("examples_generated", {'concept': concept.name})

            return {
                'simple_example': str(result.simple_example),
                'intermediate_example': str(result.intermediate_example),
                'challenging_example': str(result.challenging_example),
                'code_example': str(result.code_example),
                'what_if_variations': what_ifs
            }

        except Exception as e:
            logger.error(f"Example generation failed: {e}")
            return {}


class ProgressiveBuilderAgent(BaseLearningAgent):
    """Builds progressive learning content."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._builder = self._create_module(ProgressiveBuilderSignature)

    async def build(
        self,
        paper: PaperInfo,
        concepts: List[Concept],
        intuitions: Dict[str, Dict],
        math_explanations: Dict[str, Dict],
        examples: Dict[str, Dict],
        celebration_word: str
    ) -> Dict[str, Any]:
        """Build complete progressive learning content."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._builder(
                paper_info=json.dumps({
                    'title': paper.title,
                    'abstract': paper.abstract[:500] + context_suffix,
                    'authors': paper.authors[:3]
                }),
                concepts=json.dumps([{'name': c.name, 'description': c.description} for c in concepts]),
                intuitions=json.dumps(intuitions),
                math_explanations=json.dumps(math_explanations),
                examples=json.dumps(examples),
                celebration_word=celebration_word
            )

            key_insights = [k.strip() for k in str(result.key_insights).split('|') if k.strip()]
            next_steps = [n.strip() for n in str(result.next_steps).split('|') if n.strip()]

            self._broadcast("content_built", {'paper': paper.title[:30]})

            return {
                'complete_content': str(result.complete_content),
                'key_insights': key_insights,
                'summary': str(result.summary),
                'next_steps': next_steps
            }

        except Exception as e:
            logger.error(f"Progressive building failed: {e}")
            return {}


class ContentPolisherAgent(BaseLearningAgent):
    """Polishes content to be engaging."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._polisher = self._create_module(ContentPolisherSignature)

    async def polish(
        self,
        draft_content: str,
        style: str,
        audience: str
    ) -> Dict[str, Any]:
        """Polish content for engagement."""
        try:
            context_suffix = f"\n\n{self.learned_context}" if self.learned_context else ""
            result = self._polisher(
                draft_content=draft_content,
                style=f"{style}{context_suffix}",
                audience=audience
            )

            self._broadcast("content_polished", {
                'engagement': float(result.engagement_score) if result.engagement_score else 0
            })

            return {
                'polished_content': str(result.polished_content),
                'engagement_score': float(result.engagement_score) if result.engagement_score else 75.0,
                'clarity_score': float(result.clarity_score) if result.clarity_score else 75.0
            }

        except Exception as e:
            logger.error(f"Content polishing failed: {e}")
            return {
                'polished_content': draft_content,
                'engagement_score': 50.0,
                'clarity_score': 50.0
            }


class UnifiedLearningAgent(BaseLearningAgent):
    """
    Optimized learning content generator with two modes:

    1. generate_all() - Single LLM call for all concepts (faster, less detailed)
    2. generate_parallel() - Parallel LLM calls per concept (slower, MUCH more detailed)

    For 30+ page documents, use generate_parallel() which runs concept generation
    concurrently for both speed AND quality.
    """

    # Cache for generated content (paper_id → content)
    _content_cache: Dict[str, Dict] = {}

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = "",
                 model: str = "haiku", use_fast_predict: bool = True, llm_timeout: int = 60):
        super().__init__(memory, context, bus, learned_context, model, use_fast_predict, llm_timeout)
        self._generator = self._create_module(UnifiedConceptLearningSignature)
        self._deep_generator = self._create_module(SingleConceptDeepSignature)

    async def generate_parallel(
        self,
        paper: 'PaperInfo',
        concepts: List['Concept'],
        audience_level: str,
        celebration_word: str = "Bingo!",
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Generate DEEP content for each concept IN PARALLEL.

        This produces 30+ page documents by running multiple LLM calls concurrently.
        Each concept gets thorough treatment with detailed intuition, math, and examples.
        """
        cache_key = f"parallel_{paper.arxiv_id}_{audience_level}_{len(concepts)}"
        if cache_key in self._content_cache:
            logger.info(f"📦 Using cached parallel content for {paper.arxiv_id}")
            return self._content_cache[cache_key]

        paper_context = f"{paper.title}: {paper.abstract[:300]}"

        # Semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_one_concept(concept: 'Concept') -> Dict[str, Any]:
            """Generate deep content for a single concept."""
            async with semaphore:
                try:
                    result = self._deep_generator(
                        concept_name=concept.name,
                        concept_description=concept.description,
                        why_it_matters=concept.why_it_matters,
                        paper_context=paper_context,
                        audience_level=audience_level
                    )
                    return {
                        'name': concept.name,
                        'analogy': str(result.analogy),
                        'intuition': str(result.intuition),
                        'aha_moment': str(result.aha_moment),
                        'math_motivation': str(result.math_motivation),
                        'math_steps': str(result.math_steps),
                        'simple_example': str(result.simple_example),
                        'advanced_example': str(result.advanced_example),
                        'code_example': str(result.code_example),
                    }
                except Exception as e:
                    logger.error(f"Failed to generate content for {concept.name}: {e}")
                    return {
                        'name': concept.name,
                        'analogy': '',
                        'intuition': concept.description,
                        'aha_moment': concept.why_it_matters,
                        'math_motivation': '',
                        'math_steps': '',
                        'simple_example': '',
                        'advanced_example': '',
                        'code_example': '',
                    }

        # Run all concept generations in parallel
        logger.info(f"🚀 Generating deep content for {len(concepts)} concepts in parallel (max {max_concurrent} concurrent)...")
        start = datetime.now()

        tasks = [generate_one_concept(c) for c in concepts]
        concept_results = await asyncio.gather(*tasks)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"✅ Parallel generation complete in {elapsed:.1f}s")

        # Build result structure
        intuitions = {}
        math_explanations = {}
        examples = {}

        for cr in concept_results:
            name = cr['name']
            intuitions[name] = {
                'hook': f"Let's understand {name}",
                'analogy': cr['analogy'],
                'intuition_build': cr['intuition'],
                'aha_moment': cr['aha_moment'],
                'check_understanding': ''
            }
            math_explanations[name] = {
                'math_motivation': cr['math_motivation'],
                'building_blocks': '',
                'step_by_step': cr['math_steps'],
                'concrete_example': cr['simple_example'],
                'connection_to_intuition': ''
            }
            examples[name] = {
                'simple_example': cr['simple_example'],
                'intermediate_example': cr['advanced_example'],
                'challenging_example': '',
                'code_example': cr['code_example'],
                'what_if_variations': []
            }

        result_data = {
            'hook': f"Let's explore the revolutionary ideas in {paper.title}",
            'concepts': concept_results,
            'key_insights': [f"{celebration_word}! {cr['aha_moment'][:100]}" for cr in concept_results if cr['aha_moment']],
            'summary': f"We explored {len(concepts)} key concepts from {paper.title}",
            'next_steps': ["Implement these concepts in code", "Read related papers", "Apply to your own projects"],
            'intuitions': intuitions,
            'math_explanations': math_explanations,
            'examples': examples,
            'complete_content': self._build_deep_content(concept_results, paper, celebration_word)
        }

        self._content_cache[cache_key] = result_data
        return result_data

    def _build_deep_content(self, concept_results: List[Dict], paper: 'PaperInfo', celebration_word: str) -> str:
        """Build comprehensive markdown content from parallel results."""
        sections = [f"# {paper.title}\n\n"]

        for cr in concept_results:
            sections.append(f"\n## {cr['name']}\n")

            if cr.get('analogy'):
                sections.append(f"\n**Think of it like this:** {cr['analogy']}\n")

            if cr.get('intuition'):
                sections.append(f"\n### Building Intuition\n\n{cr['intuition']}\n")

            if cr.get('aha_moment'):
                sections.append(f"\n💡 **{celebration_word}!** {cr['aha_moment']}\n")

            if cr.get('math_motivation') or cr.get('math_steps'):
                sections.append(f"\n### The Mathematics\n")
                if cr.get('math_motivation'):
                    sections.append(f"\n{cr['math_motivation']}\n")
                if cr.get('math_steps'):
                    sections.append(f"\n{cr['math_steps']}\n")

            if cr.get('simple_example'):
                sections.append(f"\n### Example\n\n{cr['simple_example']}\n")

            if cr.get('advanced_example'):
                sections.append(f"\n### Advanced Example\n\n{cr['advanced_example']}\n")

            if cr.get('code_example'):
                sections.append(f"\n### Code Implementation\n\n```python\n{cr['code_example']}\n```\n")

        return '\n'.join(sections)

    async def generate_all(
        self,
        paper: 'PaperInfo',
        concepts: List['Concept'],
        audience_level: str,
        celebration_word: str = "Bingo!"
    ) -> Dict[str, Any]:
        """
        Generate complete learning content for all concepts in ONE LLM call.

        Returns dict with:
        - hook: Opening hook
        - concepts: List of concept learning content
        - key_insights: List of key insights
        - summary: Summary text
        - next_steps: List of next steps
        - intuitions: Dict for backward compatibility
        - math_explanations: Dict for backward compatibility
        - examples: Dict for backward compatibility
        """
        # Check cache first
        cache_key = f"{paper.arxiv_id}_{audience_level}"
        if cache_key in self._content_cache:
            logger.info(f"📦 Using cached learning content for {paper.arxiv_id}")
            return self._content_cache[cache_key]

        try:
            # FULL context for quality content
            concepts_data = [
                {
                    'name': c.name,
                    'description': c.description,
                    'why_it_matters': c.why_it_matters,
                    'difficulty': c.difficulty,
                    'math_required': c.math_required
                }
                for c in concepts[:7]  # Up to 7 concepts for comprehensive coverage
            ]

            # Full context for high-quality generation
            result = self._generator(
                paper_title=paper.title,
                paper_abstract=paper.abstract,
                concepts_json=json.dumps(concepts_data),
                audience_level=audience_level,
                celebration_word=celebration_word
            )

            # Parse the JSON output
            try:
                content = json.loads(str(result.learning_content_json))
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                raw = str(result.learning_content_json)
                # Find JSON in response
                start = raw.find('{')
                end = raw.rfind('}') + 1
                if start >= 0 and end > start:
                    try:
                        content = json.loads(raw[start:end])
                    except:
                        content = self._fallback_content(paper, concepts, celebration_word)
                else:
                    content = self._fallback_content(paper, concepts, celebration_word)

            # Build backward-compatible structures
            intuitions = {}
            math_explanations = {}
            examples = {}

            for concept_data in content.get('concepts', []):
                name = concept_data.get('name', '')
                if not name:
                    continue

                intuitions[name] = {
                    'hook': content.get('hook', ''),
                    'analogy': concept_data.get('analogy', ''),
                    'intuition_build': concept_data.get('intuition', ''),
                    'aha_moment': concept_data.get('aha_moment', ''),
                    'check_understanding': ''
                }

                math_explanations[name] = {
                    'math_motivation': concept_data.get('math_motivation', ''),
                    'building_blocks': '',
                    'step_by_step': concept_data.get('math_steps', ''),
                    'concrete_example': concept_data.get('simple_example', ''),
                    'connection_to_intuition': ''
                }

                examples[name] = {
                    'simple_example': concept_data.get('simple_example', ''),
                    'intermediate_example': '',
                    'challenging_example': '',
                    'code_example': concept_data.get('code_example', ''),
                    'what_if_variations': []
                }

            result_data = {
                'hook': content.get('hook', f"Let's understand {paper.title}"),
                'concepts': content.get('concepts', []),
                'key_insights': content.get('key_insights', []),
                'summary': content.get('summary', ''),
                'next_steps': content.get('next_steps', []),
                # Backward compatibility
                'intuitions': intuitions,
                'math_explanations': math_explanations,
                'examples': examples,
                'complete_content': self._build_complete_content(content, paper)
            }

            # Cache the result
            self._content_cache[cache_key] = result_data

            self._broadcast("unified_learning_complete", {
                'paper': paper.title[:30],
                'concepts_count': len(content.get('concepts', [])),
                'insights_count': len(content.get('key_insights', []))
            })

            return result_data

        except Exception as e:
            logger.error(f"Unified learning generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_content(paper, concepts, celebration_word)

    def _fallback_content(self, paper: 'PaperInfo', concepts: List['Concept'], celebration_word: str) -> Dict:
        """Generate minimal fallback content if LLM fails."""
        return {
            'hook': f"Let's explore {paper.title}",
            'concepts': [{'name': c.name, 'analogy': '', 'intuition': c.description} for c in concepts[:3]],
            'key_insights': [f"Understanding {c.name} is key" for c in concepts[:3]],
            'summary': f"This paper introduces {concepts[0].name if concepts else 'new ideas'}",
            'next_steps': ["Explore related papers", "Try implementing the concepts"],
            'intuitions': {},
            'math_explanations': {},
            'examples': {},
            'complete_content': paper.abstract
        }

    def _build_complete_content(self, content: Dict, paper: 'PaperInfo') -> str:
        """Build complete markdown content from structured data."""
        sections = []

        # Hook
        sections.append(f"# {paper.title}\n\n{content.get('hook', '')}")

        # Concepts
        for c in content.get('concepts', []):
            name = c.get('name', 'Concept')
            sections.append(f"\n## {name}\n")

            if c.get('analogy'):
                sections.append(f"**Analogy:** {c['analogy']}\n")

            if c.get('intuition'):
                sections.append(f"\n### Understanding {name}\n{c['intuition']}\n")

            if c.get('aha_moment'):
                sections.append(f"\n💡 **{content.get('celebration_word', 'Bingo!')}!** {c['aha_moment']}\n")

            if c.get('math_steps'):
                sections.append(f"\n### The Math\n{c.get('math_motivation', '')}\n\n{c['math_steps']}\n")

            if c.get('code_example'):
                sections.append(f"\n### Code Example\n```python\n{c['code_example']}\n```\n")

        # Summary
        if content.get('summary'):
            sections.append(f"\n## Summary\n{content['summary']}\n")

        # Key Insights
        if content.get('key_insights'):
            sections.append("\n## Key Insights\n")
            for insight in content['key_insights']:
                sections.append(f"- {insight}\n")

        # Next Steps
        if content.get('next_steps'):
            sections.append("\n## What's Next?\n")
            for step in content['next_steps']:
                sections.append(f"- {step}\n")

        return '\n'.join(sections)


# =============================================================================
# ARXIV LEARNING SWARM
# =============================================================================

@register_swarm("arxiv_learning")
class ArxivLearningSwarm(DomainSwarm):
    """
    World-Class ArXiv Learning Swarm.

    Creates engaging, progressive learning content from academic papers.
    Builds understanding from basics to advanced, always explaining WHY.
    """

    AGENT_TEAM = AgentTeam.define(
        (PaperFetcherAgent, "PaperFetcher", "_paper_fetcher"),
        (ConceptExtractorAgent, "ConceptExtractor", "_concept_extractor"),
        (IntuitionBuilderAgent, "IntuitionBuilder", "_intuition_builder"),
        (MathSimplifierAgent, "MathSimplifier", "_math_simplifier"),
        (ExampleGeneratorAgent, "ExampleGenerator", "_example_generator"),
        (ProgressiveBuilderAgent, "ProgressiveBuilder", "_progressive_builder"),
        (ContentPolisherAgent, "ContentPolisher", "_content_polisher"),
        (UnifiedLearningAgent, "UnifiedLearner", "_unified_learner"),
    )

    def __init__(self, config: ArxivLearningConfig = None):
        super().__init__(config or ArxivLearningConfig())
        # Optimization mode from config: "unified" (fast, 2 LLM calls) or "sequential" (original, 10+ calls)
        self._optimization_mode = self.config.optimization_mode

    def set_optimization_mode(self, mode: str):
        """
        Switch optimization mode.

        Args:
            mode: "unified" (fast, ~2 LLM calls) or "sequential" (original, ~10 calls)
        """
        if mode not in ["unified", "sequential"]:
            raise ValueError(f"Invalid mode: {mode}. Use 'unified' or 'sequential'")
        self._optimization_mode = mode
        self._agents_initialized = False  # Force re-init
        logger.info(f"🔧 Optimization mode set to: {mode}")

    async def _execute_domain(
        self,
        paper_id: str = None,
        topic: str = None,
        **kwargs
    ) -> ArxivLearningResult:
        """Execute learning content generation."""
        return await self.learn(paper_id=paper_id, topic=topic, **kwargs)

    async def learn(
        self,
        paper_id: str = None,
        topic: str = None,
        depth: LearningDepth = None,
        send_telegram: bool = None
    ) -> ArxivLearningResult:
        """
        Create learning content from an ArXiv paper.

        Args:
            paper_id: ArXiv paper ID (e.g., "1706.03762")
            topic: Search topic (alternative to paper_id)
            depth: Learning depth (quick, standard, deep)
            send_telegram: Whether to send result to Telegram

        Returns:
            ArxivLearningResult with complete learning content
        """
        start_time = datetime.now()

        # Initialize agents before using them
        self._init_agents()

        config = self.config
        learning_depth = depth or config.depth
        # Convert string to enum if needed
        if isinstance(learning_depth, str):
            learning_depth = LearningDepth(learning_depth.lower())

        logger.info(f"📚 ArxivLearningSwarm starting...")

        try:
            # =================================================================
            # PHASE 1: FETCH PAPER
            # =================================================================
            logger.info("📄 Phase 1: Fetching paper...")

            paper = None
            if paper_id:
                paper = await self._paper_fetcher.fetch_by_id(paper_id)
            elif topic:
                papers = await self._paper_fetcher.search_by_topic(topic, 1)
                paper = papers[0] if papers else None

            if not paper:
                return ArxivLearningResult(
                    success=False,
                    swarm_name=self.config.name,
                    domain=self.config.domain,
                    output={},
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error="Could not fetch paper"
                )

            logger.info(f"  Found: {paper.title[:60]}...")

            self._trace_phase("PaperFetcher", AgentRole.ACTOR,
                {'paper_id': paper_id, 'topic': topic},
                {'title': paper.title if paper else None, 'found': paper is not None},
                success=paper is not None, phase_start=start_time, tools_used=['arxiv_fetch'])

            # =================================================================
            # PHASE 2: EXTRACT CONCEPTS (with swarm caching)
            # =================================================================
            logger.info("🧠 Phase 2: Extracting concepts...")

            # Check swarm cache first
            cache_key = f"concepts_{paper.arxiv_id}"
            concepts = None
            if config.use_swarm_cache:
                concepts = self._get_cached(cache_key)
                if concepts:
                    logger.info(f"  📦 Loaded {len(concepts)} concepts from cache")

            if not concepts:
                concepts = await self._concept_extractor.extract(paper)

                if not concepts:
                    # Create default concept from abstract
                    concepts = [Concept(
                        name="Main Contribution",
                        description=paper.abstract[:200],
                        why_it_matters="This is the paper's key innovation",
                        prerequisites=["Basic understanding of the field"],
                        difficulty=3,
                        math_required=True
                    )]

                # Cache for future use
                if config.use_swarm_cache:
                    self._cache_result(cache_key, concepts, ttl=7200)  # 2 hour TTL

            # Limit concepts based on depth to prevent timeouts
            if learning_depth == LearningDepth.QUICK:
                max_concepts = config.max_concepts_quick
            elif learning_depth == LearningDepth.STANDARD:
                max_concepts = config.max_concepts_standard
            else:
                max_concepts = 5  # DEEP

            if len(concepts) > max_concepts:
                logger.info(f"  Limiting concepts: {len(concepts)} → {max_concepts} (depth={learning_depth.value})")
                concepts = concepts[:max_concepts]

            logger.info(f"  Extracted {len(concepts)} concepts")

            self._trace_phase("ConceptExtractor", AgentRole.EXPERT,
                {'paper_title': paper.title},
                {'concepts_count': len(concepts)},
                success=len(concepts) > 0, phase_start=start_time, tools_used=['concept_extract'])

            # =================================================================
            # PHASE 3-7: CONTENT GENERATION
            # =================================================================
            # Modes: PARALLEL_DEEP (quality+speed), UNIFIED (fast), PARALLEL, SEQUENTIAL

            if self._optimization_mode == "parallel_deep" and self._unified_learner:
                # ═══════════════════════════════════════════════════════════════
                # 🚀 PARALLEL DEEP: Full quality with parallel per-concept generation
                # ═══════════════════════════════════════════════════════════════
                logger.info("🚀 Phase 3-7: Parallel deep content generation...")

                parallel_start = datetime.now()

                # Check cache first
                content_cache_key = f"parallel_deep_{paper.arxiv_id}_{config.audience.value}_{len(concepts)}"
                unified_result = None
                if config.use_swarm_cache:
                    unified_result = self._get_cached(content_cache_key)
                    if unified_result:
                        logger.info(f"  📦 Loaded parallel deep content from cache")

                if not unified_result:
                    # Run parallel generation for all concepts
                    unified_result = await self._unified_learner.generate_parallel(
                        paper=paper,
                        concepts=concepts,
                        audience_level=config.audience.value,
                        celebration_word=config.celebration_word,
                        max_concurrent=config.max_concurrent_llm
                    )
                    if config.use_swarm_cache and unified_result:
                        self._cache_result(content_cache_key, unified_result, ttl=3600)

                intuitions = unified_result.get('intuitions', {})
                math_explanations = unified_result.get('math_explanations', {})
                examples = unified_result.get('examples', {})
                draft_content = unified_result.get('complete_content', '')
                progressive_result = {
                    'complete_content': draft_content,
                    'key_insights': unified_result.get('key_insights', []),
                    'summary': unified_result.get('summary', ''),
                    'next_steps': unified_result.get('next_steps', [])
                }
                polished = {'polished_content': draft_content}

                logger.info(f"  ✅ Parallel deep generation complete in {(datetime.now() - parallel_start).total_seconds():.1f}s")

                self._trace_phase("ParallelDeepLearner", AgentRole.PLANNER,
                    {'concepts_count': len(concepts), 'audience': config.audience.value},
                    {'concepts_generated': len(unified_result.get('concepts', []))},
                    success=bool(draft_content), phase_start=parallel_start, tools_used=['parallel_deep_learning'])

                phase6_start = datetime.now()

            elif self._optimization_mode == "unified" and self._unified_learner:
                # ═══════════════════════════════════════════════════════════════
                # 🚀 UNIFIED PATH: Single LLM call (experimental, may timeout)
                # ═══════════════════════════════════════════════════════════════
                logger.info("🚀 Phase 3-7: Unified content generation...")

                unified_start = datetime.now()

                # Check swarm cache for unified content
                content_cache_key = f"unified_{paper.arxiv_id}_{config.audience.value}_{len(concepts)}"
                unified_result = None
                if config.use_swarm_cache:
                    unified_result = self._get_cached(content_cache_key)
                    if unified_result:
                        logger.info(f"  📦 Loaded unified content from cache")

                if not unified_result:
                    unified_result = await self._unified_learner.generate_all(
                        paper=paper,
                        concepts=concepts,
                        audience_level=config.audience.value,
                        celebration_word=config.celebration_word
                    )
                    # Cache for future use
                    if config.use_swarm_cache and unified_result:
                        self._cache_result(content_cache_key, unified_result, ttl=3600)

                intuitions = unified_result.get('intuitions', {})
                math_explanations = unified_result.get('math_explanations', {})
                examples = unified_result.get('examples', {})
                draft_content = unified_result.get('complete_content', '')
                progressive_result = {
                    'complete_content': draft_content,
                    'key_insights': unified_result.get('key_insights', []),
                    'summary': unified_result.get('summary', ''),
                    'next_steps': unified_result.get('next_steps', [])
                }
                polished = {'polished_content': draft_content}

                logger.info(f"  ✅ Unified generation complete in {(datetime.now() - unified_start).total_seconds():.1f}s")

                self._trace_phase("UnifiedLearner", AgentRole.PLANNER,
                    {'concepts_count': len(concepts), 'audience': config.audience.value},
                    {'concepts_generated': len(unified_result.get('concepts', []))},
                    success=bool(draft_content), phase_start=unified_start, tools_used=['unified_learning'])

                phase6_start = datetime.now()

            elif self._optimization_mode == "parallel":
                # ═══════════════════════════════════════════════════════════════
                # 🚀 PARALLEL PATH: Run concept operations with controlled concurrency
                # ═══════════════════════════════════════════════════════════════
                logger.info("🚀 Phase 3-5: Parallel content generation (max 2 concurrent)...")

                parallel_start = datetime.now()

                # Semaphore to limit concurrent LLM calls (Claude CLI can't handle many concurrent calls)
                llm_semaphore = asyncio.Semaphore(2)

                async def rate_limited_call(coro):
                    """Wrapper to limit concurrent LLM calls."""
                    async with llm_semaphore:
                        return await coro

                # Build intuitions for all concepts in parallel (limited to 2 concurrent)
                logger.info("💡 Building intuitions in parallel...")
                intuition_tasks = [
                    rate_limited_call(self._intuition_builder.build(c, config.audience.value))
                    for c in concepts[:3]
                ]
                intuition_results = await asyncio.gather(*intuition_tasks, return_exceptions=True)

                intuitions = {}
                for c, result in zip(concepts[:3], intuition_results):
                    if isinstance(result, dict):
                        intuitions[c.name] = result

                # Build math explanations in parallel (for concepts that need it)
                math_explanations = {}
                if learning_depth in [LearningDepth.STANDARD, LearningDepth.DEEP]:
                    logger.info("📐 Simplifying math in parallel...")
                    math_concepts = [c for c in concepts[:3] if c.math_required]
                    math_tasks = [
                        rate_limited_call(self._math_simplifier.simplify(c, intuitions.get(c.name, {}), config.audience.value))
                        for c in math_concepts
                    ]
                    math_results = await asyncio.gather(*math_tasks, return_exceptions=True)

                    for c, result in zip(math_concepts, math_results):
                        if isinstance(result, dict):
                            math_explanations[c.name] = result

                # Generate examples in parallel (for top 2 concepts)
                examples = {}
                if config.include_code_examples:
                    logger.info("💻 Generating examples in parallel...")
                    example_tasks = [
                        rate_limited_call(self._example_generator.generate(
                            c, intuitions.get(c.name, {}), math_explanations.get(c.name, {})))
                        for c in concepts[:2]
                    ]
                    example_results = await asyncio.gather(*example_tasks, return_exceptions=True)

                    for c, result in zip(concepts[:2], example_results):
                        if isinstance(result, dict):
                            examples[c.name] = result

                parallel_time = (datetime.now() - parallel_start).total_seconds()
                logger.info(f"  ✅ Parallel generation complete in {parallel_time:.1f}s")
                logger.info(f"     {len(intuitions)} intuitions, {len(math_explanations)} math, {len(examples)} examples")

                self._trace_phase("ParallelContentGen", AgentRole.ACTOR,
                    {'concepts_count': len(concepts[:3])},
                    {'intuitions': len(intuitions), 'math': len(math_explanations), 'examples': len(examples)},
                    success=len(intuitions) > 0, phase_start=parallel_start, tools_used=['parallel_generation'])

                # Phase 6: Build content directly (SKIP ProgressiveBuilder to avoid 76KB prompt)
                logger.info("🏗️ Phase 6: Building content directly (no extra LLM call)...")

                # Build content directly from sections - no need for another LLM call
                draft_parts = []
                draft_parts.append(f"# {paper.title}\n")

                # Add hook from first intuition
                for name, intuition in intuitions.items():
                    if intuition.get('hook'):
                        draft_parts.append(f"\n## Why Should You Care?\n{intuition['hook']}\n")
                        break

                # Add concept sections
                for concept in concepts[:3]:
                    intuition = intuitions.get(concept.name, {})
                    math = math_explanations.get(concept.name, {})
                    example = examples.get(concept.name, {})

                    draft_parts.append(f"\n## {concept.name}\n")
                    if intuition.get('analogy'):
                        draft_parts.append(f"**Analogy:** {intuition['analogy']}\n")
                    if intuition.get('intuition_build'):
                        draft_parts.append(f"\n{intuition['intuition_build']}\n")
                    if intuition.get('aha_moment'):
                        draft_parts.append(f"\n💡 **{config.celebration_word}!** {intuition['aha_moment']}\n")
                    if math.get('step_by_step'):
                        draft_parts.append(f"\n### The Math\n{math['step_by_step']}\n")
                    if example.get('code_example'):
                        draft_parts.append(f"\n### Code\n```python\n{example['code_example']}\n```\n")

                draft_content = '\n'.join(draft_parts)

                # Extract key insights from aha_moments
                key_insights = [
                    intuitions[c.name].get('aha_moment', '')
                    for c in concepts[:3] if c.name in intuitions and intuitions[c.name].get('aha_moment')
                ]

                progressive_result = {
                    'complete_content': draft_content,
                    'key_insights': key_insights,
                    'summary': f"This paper introduces {concepts[0].name if concepts else 'key concepts'} and related innovations.",
                    'next_steps': ['Explore related papers', 'Implement the concepts', 'Read the full paper']
                }
                polished = {'polished_content': draft_content}

                phase6_start = datetime.now()
                self._trace_phase("DirectContentBuild", AgentRole.PLANNER,
                    {'concepts_count': len(concepts)},
                    {'content_length': len(draft_content)},
                    success=bool(draft_content), phase_start=parallel_start, tools_used=['direct_build'])

            else:
                # ═══════════════════════════════════════════════════════════════
                # SEQUENTIAL PATH: Original multi-call approach
                # ═══════════════════════════════════════════════════════════════
                logger.info("💡 Phase 3: Building intuition...")

                intuitions = {}
                for concept in concepts[:3]:
                    try:
                        result = await self._intuition_builder.build(concept, config.audience.value)
                        if isinstance(result, dict):
                            intuitions[concept.name] = result
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Intuition building failed for {concept.name}: {e}")

                phase3_start = datetime.now()
                self._trace_phase("IntuitionBuilder", AgentRole.ACTOR,
                    {'concepts_count': len(concepts[:3])},
                    {'intuitions_built': len(intuitions)},
                    success=len(intuitions) > 0, phase_start=start_time, tools_used=['intuition_build'])

                # Phase 4: Math
                math_explanations = {}
                if learning_depth in [LearningDepth.STANDARD, LearningDepth.DEEP]:
                    logger.info("📐 Phase 4: Simplifying math...")
                    for concept in concepts[:3]:
                        if concept.math_required:
                            try:
                                result = await self._math_simplifier.simplify(
                                    concept, intuitions.get(concept.name, {}), config.audience.value)
                                if isinstance(result, dict):
                                    math_explanations[concept.name] = result
                                await asyncio.sleep(1)
                            except Exception as e:
                                logger.warning(f"Math simplification failed for {concept.name}: {e}")

                self._trace_phase("MathSimplifier", AgentRole.ACTOR,
                    {'math_concepts_count': sum(1 for c in concepts[:3] if c.math_required)},
                    {'math_explanations_count': len(math_explanations)},
                    success=True, phase_start=phase3_start, tools_used=['math_simplify'])

                # Phase 5: Examples
                examples = {}
                if config.include_code_examples:
                    logger.info("💻 Phase 5: Generating examples...")
                    for concept in concepts[:2]:
                        try:
                            result = await self._example_generator.generate(
                                concept, intuitions.get(concept.name, {}), math_explanations.get(concept.name, {}))
                            if isinstance(result, dict):
                                examples[concept.name] = result
                            await asyncio.sleep(1)
                        except Exception as e:
                            logger.warning(f"Example generation failed for {concept.name}: {e}")

                self._trace_phase("ExampleGenerator", AgentRole.ACTOR,
                    {'concepts_for_examples': min(3, len(concepts))},
                    {'examples_generated': len(examples)},
                    success=True, phase_start=phase3_start, tools_used=['example_generate'])

                # Phase 6: Progressive content
                logger.info("🏗️ Phase 6: Building progressive content...")
                progressive_result = await self._progressive_builder.build(
                    paper, concepts, intuitions, math_explanations, examples, config.celebration_word)

                phase6_start = datetime.now()
                self._trace_phase("ProgressiveBuilder", AgentRole.PLANNER,
                    {'concepts_count': len(concepts), 'has_math': bool(math_explanations), 'has_examples': bool(examples)},
                    {'has_content': bool(progressive_result.get('complete_content')), 'insights_count': len(progressive_result.get('key_insights', []))},
                    success=bool(progressive_result.get('complete_content')), phase_start=phase3_start, tools_used=['progressive_build'])

                # Phase 7: Polish
                logger.info("✨ Phase 7: Polishing content...")
                draft_content = progressive_result.get('complete_content', '')
                polished = await self._content_polisher.polish(draft_content, config.style.value, config.audience.value)

                self._trace_phase("ContentPolisher", AgentRole.REVIEWER,
                    {'draft_length': len(draft_content)},
                    {'polished': bool(polished.get('polished_content'))},
                    success=bool(polished.get('polished_content')), phase_start=phase6_start, tools_used=['content_polish'])

            # =================================================================
            # BUILD LEARNING SECTIONS
            # =================================================================
            sections = []

            # Hook section
            hook_text = ""
            for concept_name, intuition in intuitions.items():
                if intuition.get('hook'):
                    hook_text = intuition['hook']
                    break

            sections.append(LearningSection(
                title="Why Should You Care?",
                content=hook_text or f"Let's understand {paper.title}",
                level=1,
                has_bingo_moment=False
            ))

            # Intuition sections - apply step formatting post-processing
            for concept in concepts[:3]:
                intuition = intuitions.get(concept.name, {})
                if intuition:
                    # Format intuition_build content to ensure each step is on a new line
                    intuition_content = format_steps_on_newlines(intuition.get('intuition_build', ''))
                    sections.append(LearningSection(
                        title=f"Understanding {concept.name}",
                        content=f"{intuition.get('analogy', '')}\n\n{intuition_content}",
                        level=2,
                        has_bingo_moment=True if intuition.get('aha_moment') else False
                    ))

            # Math sections - apply step formatting post-processing
            for concept_name, math in math_explanations.items():
                # Format step-by-step content to ensure each step is on a new line
                step_content = format_steps_on_newlines(math.get('step_by_step', ''))
                sections.append(LearningSection(
                    title=f"The Math: {concept_name}",
                    content=f"{math.get('math_motivation', '')}\n\n{step_content}",
                    level=3,
                    has_bingo_moment=False
                ))

            # Example sections - apply step formatting post-processing
            for concept_name, ex in examples.items():
                # code_example is rendered separately by generate_learning_html,
                # so don't embed it in content (avoids duplication + "python" leak)
                raw_code = ex.get('code_example', '')
                # Strip any existing fence markers the LLM may have included
                raw_code = re.sub(r'^```\w*\n?', '', raw_code)
                raw_code = re.sub(r'\n?```$', '', raw_code).strip()
                # Format simple_example content to ensure steps are on new lines
                simple_example = format_steps_on_newlines(ex.get('simple_example', ''))
                sections.append(LearningSection(
                    title=f"See It In Action: {concept_name}",
                    content=f"**Simple Example:**\n{simple_example}",
                    level=4,
                    has_bingo_moment=False,
                    code_example=raw_code
                ))

            # Count bingo moments
            bingo_count = sum(1 for s in sections if s.has_bingo_moment)
            bingo_count += len(progressive_result.get('key_insights', []))

            # Build final content
            final_content = polished.get('polished_content', '') or draft_content

            learning_content = LearningContent(
                paper=paper,
                hook=hook_text,
                concepts=concepts,
                sections=sections,
                key_insights=progressive_result.get('key_insights', []),
                summary=progressive_result.get('summary', ''),
                next_steps=progressive_result.get('next_steps', []),
                total_words=len(final_content.split())
            )

            # =================================================================
            # PHASE 7.5-7.7: GENERATE OUTPUTS (PARALLEL)
            # =================================================================
            logger.info("📄 Generating outputs (PDF, PPTX, HTML in parallel)...")

            # OPTIMIZATION: Generate all outputs in PARALLEL
            async def gen_pdf():
                return await self._generate_pdf(paper, learning_content)

            async def gen_pptx():
                if self.config.generate_pptx:
                    return await self._generate_pptx(paper, learning_content)
                return (None, None)

            async def gen_html():
                if self.config.generate_html:
                    return await self._generate_html(paper, learning_content)
                return None

            # Run all generations in parallel
            output_results = await asyncio.gather(
                gen_pdf(),
                gen_pptx(),
                gen_html(),
                return_exceptions=True
            )

            # Extract results
            pdf_path = output_results[0] if not isinstance(output_results[0], Exception) else None
            pptx_result = output_results[1] if not isinstance(output_results[1], Exception) else (None, None)
            pptx_path, pptx_pdf_path = pptx_result if isinstance(pptx_result, tuple) else (None, None)
            html_path = output_results[2] if not isinstance(output_results[2], Exception) else None

            # Log any errors
            for i, result in enumerate(output_results):
                if isinstance(result, Exception):
                    logger.warning(f"Output generation {i} failed: {result}")

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            # Estimate learning time
            words = learning_content.total_words
            if learning_depth == LearningDepth.QUICK:
                learning_time = "5-10 minutes"
            elif learning_depth == LearningDepth.STANDARD:
                learning_time = "20-30 minutes"
            else:
                learning_time = "45-60 minutes"

            result = ArxivLearningResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={
                    'title': paper.title,
                    'arxiv_id': paper.arxiv_id,
                    'content': final_content
                },
                execution_time=exec_time,
                paper=paper,
                content=learning_content,
                learning_time_estimate=learning_time,
                concepts_covered=len(concepts),
                bingo_moments=bingo_count,
                difficulty_progression=[s.level for s in sections],
                pdf_path=pdf_path,
                pptx_path=pptx_path,
                pptx_pdf_path=pptx_pdf_path,
                html_path=html_path
            )

            logger.info(f"✅ ArxivLearningSwarm complete: {paper.title[:40]}...")
            logger.info(f"   {len(concepts)} concepts, {bingo_count} {config.celebration_word} moments")

            # =================================================================
            # SELF-IMPROVEMENT: Record trace + post-execution learning
            # =================================================================
            logger.info("📊 Running self-improvement evaluation...")

            # Record orchestrator-level execution trace
            self._record_trace(
                agent_name="ArxivLearningSwarm",
                agent_role=AgentRole.ORCHESTRATOR,
                input_data={'paper_id': paper.arxiv_id, 'topic': paper.title},
                output_data={
                    'concepts': len(concepts),
                    'sections': len(sections),
                    'bingo_moments': bingo_count,
                    'words': learning_content.total_words
                },
                execution_time=exec_time,
                success=True
            )

            # =================================================================
            # LOG LLM METRICS
            # =================================================================
            try:
                if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'get_metrics'):
                    metrics = dspy.settings.lm.get_metrics()
                    logger.info(f"📈 LLM Metrics: {metrics.get('successful_calls', 0)}/{metrics.get('total_calls', 0)} calls succeeded ({metrics.get('success_rate', 'N/A')})")
                    if metrics.get('retried_calls', 0) > 0:
                        logger.info(f"   🔄 Retried {metrics['retried_calls']} calls")
            except Exception:
                pass

            # =================================================================
            # AGENT0 + MORPHAGENT: POST-EXECUTION LEARNING
            # (evaluation + improvement cycle now handled centrally in base)
            # =================================================================
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['arxiv_fetch', 'concept_extract', 'content_generate']),
                task_type='paper_learning',
                output_data={
                    'concepts_count': len(concepts),
                    'bingo_moments': bingo_count,
                    'has_hook': bool(hook_text),
                    'has_summary': bool(learning_content.summary),
                    'has_examples': bool(examples),
                    'word_count': learning_content.total_words
                },
                input_data={'arxiv_id': paper.arxiv_id}
            )

            # =================================================================
            # PHASE 8: SEND TO TELEGRAM (if enabled)
            # =================================================================
            should_send = send_telegram if send_telegram is not None else config.send_telegram

            if should_send and TELEGRAM_AVAILABLE:
                logger.info("📱 Phase 8: Sending to Telegram...")
                await self._send_to_telegram(paper, learning_content, final_content, pdf_path=pdf_path, pptx_path=pptx_path, pptx_pdf_path=pptx_pdf_path, html_path=html_path)
            elif should_send and not TELEGRAM_AVAILABLE:
                logger.warning("⚠️ Telegram sending requested but tools not available")

            return result

        except Exception as e:
            logger.error(f"❌ ArxivLearningSwarm error: {e}")
            import traceback
            traceback.print_exc()

            # Agent0 + MorphAgent: Post-execution learning (failure path)
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['arxiv_fetch']),
                task_type='paper_learning'
            )

            return ArxivLearningResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=exec_time,
                error=str(e)
            )

    async def search_and_learn(
        self,
        topic: str,
        max_papers: int = 1,
        rank_by: str = "Which {abstract} is most exciting and impactful for learning?"
    ) -> List[ArxivLearningResult]:
        """
        Search for papers and create learning content.

        If LOTUS is enabled and available, uses semantic ranking to find
        the most valuable papers for learning. Otherwise falls back to
        standard relevance-based search.

        Args:
            topic: Search topic
            max_papers: Number of papers to process
            rank_by: Natural language ranking criterion (LOTUS only)

        Returns:
            List of ArxivLearningResult objects
        """
        self._init_agents()

        # Use LOTUS for semantic search if available and enabled
        if self.config.use_lotus and LOTUS_AVAILABLE:
            logger.info("🌸 Using LOTUS for semantic paper search...")
            papers = await self._paper_fetcher.search_and_rank_with_lotus(
                topic=topic,
                max_results=max_papers,
                rank_by=rank_by,
                lotus_model=self.config.lotus_model
            )
        else:
            papers = await self._paper_fetcher.search_by_topic(topic, max_papers)

        results = []
        for paper in papers:
            result = await self.learn(paper_id=paper.arxiv_id)
            results.append(result)

        return results

    async def _generate_pdf(
        self,
        paper: PaperInfo,
        content: LearningContent
    ) -> Optional[str]:
        """Generate professional PDF with visualizations. Returns path or None."""
        try:
            from ..skills.research.learning_pdf_template import (
                convert_learning_to_pdf,
                generate_concept_visualization
            )

            celebration = self.config.celebration_word

            # Prepare concepts for PDF
            concepts_data = [
                {
                    'name': c.name,
                    'description': c.description,
                    'difficulty': c.difficulty,
                    'why_it_matters': c.why_it_matters
                }
                for c in content.concepts
            ]

            # =============================================================
            # GENERATE VISUALIZATIONS FOR KEY CONCEPTS (PARALLEL)
            # =============================================================
            logger.info("🎨 Generating concept visualizations (parallel)...")
            visualization_paths = {}

            # Build a map of concept name -> intuition text from sections
            concept_intuitions = {}
            for s in content.sections:
                for c in content.concepts:
                    if c.name.lower() in s.title.lower() and c.name not in concept_intuitions:
                        concept_intuitions[c.name] = s.content[:500]

            # OPTIMIZATION: Generate visualizations in PARALLEL using asyncio.gather
            async def generate_viz_for_concept(concept):
                """Generate visualization for a single concept."""
                try:
                    description_parts = [concept.description, concept.why_it_matters]
                    if concept.name in concept_intuitions:
                        description_parts.append(concept_intuitions[concept.name])
                    rich_description = " ".join(filter(None, description_parts))

                    viz_path = await generate_concept_visualization(
                        concept_name=concept.name,
                        concept_description=rich_description,
                        output_dir="/tmp"
                    )
                    if viz_path:
                        return (concept.name, viz_path)
                except Exception as e:
                    logger.debug(f"Viz generation failed for {concept.name}: {e}")
                return None

            # Run all visualization generations in parallel
            viz_tasks = [generate_viz_for_concept(c) for c in content.concepts[:3]]  # Top 3 concepts
            viz_results = await asyncio.gather(*viz_tasks, return_exceptions=True)

            for result in viz_results:
                if result and isinstance(result, tuple):
                    concept_name, viz_path = result
                    visualization_paths[concept_name] = viz_path
                    logger.info(f"  Generated viz for: {concept_name}")

            # Prepare sections for PDF with visualizations
            used_visualizations = set()
            sections_data = []
            for s in content.sections:
                section_dict = {
                    'title': s.title,
                    'content': s.content,
                    'level': s.level,
                    'has_bingo_moment': s.has_bingo_moment,
                    'code_example': s.code_example
                }
                for concept_name, viz_path in visualization_paths.items():
                    if concept_name not in used_visualizations and concept_name.lower() in s.title.lower():
                        section_dict['visualization_path'] = viz_path
                        used_visualizations.add(concept_name)
                        break
                sections_data.append(section_dict)

            # Estimate learning time
            if content.total_words < 1000:
                learning_time = "10-15 min"
            elif content.total_words < 2000:
                learning_time = "20-30 min"
            else:
                learning_time = "45-60 min"

            # Generate PDF
            output_path = f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}_learning.pdf"

            pdf_path = await convert_learning_to_pdf(
                paper_title=paper.title,
                arxiv_id=paper.arxiv_id,
                authors=paper.authors,
                hook=content.hook,
                concepts=concepts_data,
                sections=sections_data,
                key_insights=content.key_insights,
                summary=content.summary,
                next_steps=content.next_steps,
                output_path=output_path,
                bingo_word=celebration,
                learning_time=learning_time,
                total_words=content.total_words
            )

            logger.info(f"✅ Generated PDF: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            return None

    async def _generate_pptx(
        self,
        paper: PaperInfo,
        content: LearningContent
    ) -> tuple:
        """Generate professional PowerPoint presentation.

        Returns:
            Tuple of (pptx_path, pptx_pdf_path). Either can be None if generation/conversion failed.
        """
        pptx_path = None
        pptx_pdf_path = None

        try:
            from ..skills.research.pptx_generator import (
                generate_learning_pptx,
                convert_pptx_to_pdf,
                is_libreoffice_available
            )

            celebration = self.config.celebration_word

            # Prepare concepts for PPTX
            concepts_data = [
                {
                    'name': c.name,
                    'description': c.description,
                    'difficulty': c.difficulty,
                    'why_it_matters': c.why_it_matters
                }
                for c in content.concepts
            ]

            # Prepare sections for PPTX
            sections_data = [
                {
                    'title': s.title,
                    'content': s.content,
                    'level': s.level,
                    'has_bingo_moment': s.has_bingo_moment,
                    'code_example': s.code_example
                }
                for s in content.sections
            ]

            # Estimate learning time
            if content.total_words < 1000:
                learning_time = "10-15 min"
            elif content.total_words < 2000:
                learning_time = "20-30 min"
            else:
                learning_time = "45-60 min"

            # Generate PPTX (use _presentation suffix to avoid conflict with _learning.pdf)
            output_path = f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}_presentation.pptx"

            pptx_path = await generate_learning_pptx(
                paper_title=paper.title,
                arxiv_id=paper.arxiv_id,
                authors=paper.authors,
                hook=content.hook,
                concepts=concepts_data,
                sections=sections_data,
                key_insights=content.key_insights,
                summary=content.summary,
                next_steps=content.next_steps,
                output_path=output_path,
                bingo_word=celebration,
                learning_time=learning_time,
                total_words=content.total_words
            )

            if pptx_path:
                logger.info(f"✅ Generated PPTX: {pptx_path}")

                # Convert PPTX to PDF if enabled (default: True)
                if self.config.convert_pptx_to_pdf:
                    if is_libreoffice_available():
                        logger.info("📄 Converting PPTX to PDF...")
                        pptx_pdf_path = await convert_pptx_to_pdf(pptx_path)
                        if pptx_pdf_path:
                            logger.info(f"✅ Converted PPTX to PDF: {pptx_pdf_path}")
                    else:
                        logger.warning("⚠️ PPTX-to-PDF conversion skipped (LibreOffice not installed)")

            return pptx_path, pptx_pdf_path

        except Exception as e:
            logger.warning(f"PPTX generation failed: {e}")
            return None, None

    async def _generate_html(
        self,
        paper: PaperInfo,
        content: LearningContent
    ) -> Optional[str]:
        """Generate interactive HTML slides.

        Returns:
            Path to generated HTML file, or None if generation failed.
        """
        try:
            from ..skills.research.pptx_generator import generate_learning_html

            celebration = self.config.celebration_word

            # Build comprehensive paper_data for HTML generator
            paper_data = {
                'paper_title': paper.title,
                'arxiv_id': paper.arxiv_id,
                'authors': paper.authors,
                'hook': content.hook,
                'summary': content.summary,
                'abstract': paper.abstract,
                'bingo_word': celebration,

                # Full concepts with all fields
                'concepts': [
                    {
                        'name': c.name,
                        'description': c.description,
                        'why_it_matters': c.why_it_matters,
                        'prerequisites': c.prerequisites,
                        'difficulty': c.difficulty,
                        'math_required': c.math_required,
                    }
                    for c in content.concepts
                ],

                # Full sections with all fields
                'sections': [
                    {
                        'title': s.title,
                        'content': s.content,
                        'level': s.level,
                        'has_bingo_moment': s.has_bingo_moment,
                        'code_example': s.code_example,
                        'visualization_desc': s.visualization_desc,
                        'exercises': s.exercises,
                    }
                    for s in content.sections
                ],

                'key_insights': content.key_insights,
                'next_steps': content.next_steps,

                # Estimate learning time
                'learning_time': "10-15 min" if content.total_words < 1000 else "20-30 min" if content.total_words < 2000 else "45-60 min",
            }

            # Generate HTML slides
            output_path = f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}_slides.html"

            html_path = await generate_learning_html(paper_data, output_path)

            if html_path:
                logger.info(f"✅ Generated HTML slides: {html_path}")

            return html_path

        except Exception as e:
            logger.warning(f"HTML slide generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _send_to_telegram(
        self,
        paper: PaperInfo,
        content: LearningContent,
        full_content: str,
        pdf_path: Optional[str] = None,
        pptx_path: Optional[str] = None,
        pptx_pdf_path: Optional[str] = None,
        html_path: Optional[str] = None
    ):
        """Send learning content summary, PDF, PPTX, and HTML to Telegram.

        Args:
            paper: Paper information
            content: Learning content
            full_content: Full text content
            pdf_path: Path to generated PDF (learning guide)
            pptx_path: Path to generated PPTX
            pptx_pdf_path: Path to PDF converted from PPTX (preferred for Telegram)
            html_path: Path to generated HTML slides
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram tools not available")
            return

        try:
            celebration = self.config.celebration_word

            # Determine what files we'll send
            has_pptx_pdf = pptx_pdf_path and Path(pptx_pdf_path).exists()
            has_pptx = pptx_path and Path(pptx_path).exists()
            has_html = html_path and Path(html_path).exists()
            presentation_label = "📊 Presentation PDF" if has_pptx_pdf else "📊 PPTX"
            html_label = " + 🌐 HTML Slides" if has_html else ""

            # =================================================================
            # SEND TELEGRAM MESSAGE (Summary)
            # =================================================================
            header = f"📚 *ArXiv Learning: {paper.title[:60]}*\n"
            header += f"📎 ID: `{paper.arxiv_id}`\n"
            header += f"👥 {', '.join(paper.authors[:3])}\n"
            header += f"🔗 {paper.arxiv_url}\n\n"

            hook_section = f"*🎯 Why Should You Care?*\n{content.hook[:400] if content.hook else 'Learn about cutting-edge research!'}\n\n"

            insights_section = ""
            if content.key_insights:
                insights_section = f"*✨ Key Insights ({celebration}!)*\n"
                for i, insight in enumerate(content.key_insights[:4], 1):
                    insights_section += f"{i}. {insight[:150]}\n"
                insights_section += "\n"

            concepts_section = f"*🧠 Concepts ({len(content.concepts)})*\n"
            for concept in content.concepts[:4]:
                concepts_section += f"• {concept.name}\n"
            concepts_section += "\n"

            stats = f"📊 {content.total_words} words | {len(content.concepts)} concepts | {len(content.key_insights)} insights\n"
            stats += f"📄 Learning PDF + {presentation_label}{html_label} attached below"

            message = header + hook_section + insights_section + concepts_section + stats

            if len(message) > 4000:
                message = message[:3950] + "\n..."

            result = await send_telegram_message_tool({
                'message': message,
                'parse_mode': 'Markdown'
            })

            if result.get('success'):
                logger.info(f"✅ Sent summary to Telegram: message_id {result.get('message_id')}")
            else:
                logger.error(f"❌ Telegram message failed: {result.get('error')}")

            # =================================================================
            # SEND PDF FILE (Learning Guide)
            # =================================================================
            if pdf_path and Path(pdf_path).exists():
                file_result = await send_telegram_file_tool({
                    'file_path': pdf_path,
                    'caption': f"📖 {paper.title[:50]} - Learning Guide"
                })

                if file_result.get('success'):
                    logger.info(f"✅ Sent Learning PDF to Telegram")
                else:
                    logger.error(f"❌ Learning PDF send failed: {file_result.get('error')}")

            # =================================================================
            # SEND PRESENTATION (prefer PDF from PPTX, fallback to raw PPTX)
            # =================================================================
            if has_pptx_pdf:
                # Send PDF converted from PPTX (better Telegram experience)
                file_result = await send_telegram_file_tool({
                    'file_path': pptx_pdf_path,
                    'caption': f"📊 {paper.title[:50]} - Presentation (PDF)"
                })

                if file_result.get('success'):
                    logger.info(f"✅ Sent Presentation PDF to Telegram")
                else:
                    logger.error(f"❌ Presentation PDF send failed: {file_result.get('error')}")
            elif has_pptx:
                # Fallback: Send raw PPTX (when LibreOffice not available)
                file_result = await send_telegram_file_tool({
                    'file_path': pptx_path,
                    'caption': f"📊 {paper.title[:50]} - Presentation (PPTX)"
                })

                if file_result.get('success'):
                    logger.info(f"✅ Sent PPTX to Telegram")
                else:
                    logger.error(f"❌ PPTX send failed: {file_result.get('error')}")

            # =================================================================
            # SEND HTML SLIDES
            # =================================================================
            if html_path and Path(html_path).exists():
                file_result = await send_telegram_file_tool({
                    'file_path': html_path,
                    'caption': f"🌐 {paper.title[:50]} - Interactive HTML Slides"
                })

                if file_result.get('success'):
                    logger.info(f"✅ Sent HTML slides to Telegram")
                else:
                    logger.error(f"❌ HTML slides send failed: {file_result.get('error')}")

            if not pdf_path or not Path(pdf_path).exists():
                # Fallback: Send markdown
                temp_path = Path(f"/tmp/arxiv_{paper.arxiv_id.replace('.', '_')}_learning.md")
                with open(temp_path, 'w') as f:
                    f.write(f"# Learning: {paper.title}\n\n")
                    f.write(f"**ArXiv ID:** {paper.arxiv_id}\n")
                    f.write(f"**Authors:** {', '.join(paper.authors)}\n\n")
                    f.write("---\n\n")
                    f.write(full_content)

                file_result = await send_telegram_file_tool({
                    'file_path': str(temp_path),
                    'caption': f"📖 Learning content for {paper.arxiv_id}"
                })

                if file_result.get('success'):
                    logger.info(f"✅ Sent markdown to Telegram")

                try:
                    temp_path.unlink()
                except:
                    pass

        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            import traceback
            traceback.print_exc()

    def seed_gold_standards(self):
        """
        Seed default gold standards for paper learning evaluation.

        This establishes baseline quality expectations for:
        - Concept extraction
        - Content quality
        - Engagement metrics
        """
        self._init_agents()

        # Gold standard for a well-learned paper
        self.add_gold_standard(
            task_type='paper_learning',
            input_data={'arxiv_id': 'any'},
            expected_output={
                'concepts_count': 5,  # At least 5 concepts
                'bingo_moments': 3,   # At least 3 key insights
                'has_hook': True,
                'has_summary': True,
                'has_examples': True,
                'word_count': 1000    # At least 1000 words
            },
            evaluation_criteria={
                'concepts_count': 0.25,
                'bingo_moments': 0.20,
                'has_hook': 0.15,
                'has_summary': 0.15,
                'has_examples': 0.15,
                'word_count': 0.10
            }
        )

        logger.info("✅ Seeded gold standards for ArxivLearningSwarm")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = {
            'evaluations_count': len(self._evaluations) if self._evaluations else 0,
            'traces_count': len(self._traces) if self._traces else 0,
            'improvements_suggested': 0,
            'avg_score': 0.0
        }

        if self._evaluations:
            stats['avg_score'] = sum(e.overall_score for e in self._evaluations) / len(self._evaluations)

        if self._improvement_history:
            pending = self._improvement_history.get_pending_suggestions()
            stats['improvements_suggested'] = len(pending)

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def learn_paper(
    paper_id: str = None,
    topic: str = None,
    depth: str = "standard",
    send_telegram: bool = False
) -> ArxivLearningResult:
    """
    One-liner paper learning.

    Usage:
        from core.swarms.arxiv_learning_swarm import learn_paper

        # By ID
        result = await learn_paper("1706.03762")  # Attention paper

        # By topic
        result = await learn_paper(topic="transformer attention")

        # With Telegram
        result = await learn_paper("2408.11574", send_telegram=True)
    """
    depth_enum = LearningDepth(depth) if isinstance(depth, str) else depth

    swarm = ArxivLearningSwarm()
    return await swarm.learn(paper_id=paper_id, topic=topic, depth=depth_enum, send_telegram=send_telegram)


def learn_paper_sync(
    paper_id: str = None,
    topic: str = None,
    depth: str = "standard",
    send_telegram: bool = False
) -> ArxivLearningResult:
    """Synchronous paper learning."""
    return asyncio.run(learn_paper(paper_id=paper_id, topic=topic, depth=depth, send_telegram=send_telegram))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ArxivLearningSwarm',
    'ArxivLearningConfig',
    'ArxivLearningResult',
    'LearningContent',
    'LearningSection',
    'Concept',
    'PaperInfo',
    'LearningDepth',
    'ContentStyle',
    'AudienceLevel',
    'learn_paper',
    'learn_paper_sync',
    # Agents
    'PaperFetcherAgent',
    'ConceptExtractorAgent',
    'IntuitionBuilderAgent',
    'MathSimplifierAgent',
    'ExampleGeneratorAgent',
    'ProgressiveBuilderAgent',
    'ContentPolisherAgent',
]
