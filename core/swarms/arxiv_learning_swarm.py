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

logger = logging.getLogger(__name__)


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
    step_by_step: str = dspy.OutputField(desc="Step-by-step derivation with WHY for each step")
    concrete_example: str = dspy.OutputField(desc="Worked example with actual numbers")
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
# AGENTS
# =============================================================================

class BaseLearningAgent:
    """Base class for learning agents."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        self.memory = memory
        self.context = context
        self.bus = bus
        self.learned_context = learned_context

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
    """Fetches papers from ArXiv."""

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    async def fetch_by_id(self, arxiv_id: str) -> Optional[PaperInfo]:
        """Fetch paper by ArXiv ID."""
        # Clean the ID
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]

        try:
            async with aiohttp.ClientSession() as session:
                params = {"id_list": arxiv_id}
                async with session.get(self.ARXIV_API_URL, params=params) as resp:
                    if resp.status == 200:
                        xml_text = await resp.text()
                        return self._parse_arxiv_response(xml_text)
                    else:
                        logger.error(f"ArXiv API error: {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"Paper fetch failed: {e}")
            return None

    async def search_by_topic(self, topic: str, max_results: int = 5) -> List[PaperInfo]:
        """Search papers by topic."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "search_query": f"all:{topic}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending"
                }
                async with session.get(self.ARXIV_API_URL, params=params) as resp:
                    if resp.status == 200:
                        xml_text = await resp.text()
                        return self._parse_arxiv_search_response(xml_text)
                    return []
        except Exception as e:
            logger.error(f"Topic search failed: {e}")
            return []

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

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._extractor = dspy.ChainOfThought(ConceptExtractionSignature)

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
                concepts_data = json.loads(result.concepts)
            except:
                concepts_data = []

            concepts = []
            for c in concepts_data:
                concepts.append(Concept(
                    name=c.get('name', ''),
                    description=c.get('description', ''),
                    why_it_matters=c.get('why_it_matters', ''),
                    prerequisites=c.get('prerequisites', []),
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

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._builder = dspy.ChainOfThought(IntuitionBuilderSignature)

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

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._simplifier = dspy.ChainOfThought(MathSimplifierSignature)

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

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._generator = dspy.ChainOfThought(ExampleGeneratorSignature)

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

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._builder = dspy.ChainOfThought(ProgressiveBuilderSignature)

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

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._polisher = dspy.ChainOfThought(ContentPolisherSignature)

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


# =============================================================================
# ARXIV LEARNING SWARM
# =============================================================================

@register_swarm("arxiv_learning")
class ArxivLearningSwarm(BaseSwarm):
    """
    World-Class ArXiv Learning Swarm.

    Creates engaging, progressive learning content from academic papers.
    Builds understanding from basics to advanced, always explaining WHY.
    """

    def __init__(self, config: ArxivLearningConfig = None):
        super().__init__(config or ArxivLearningConfig())
        self._agents_initialized = False

        # Agents
        self._paper_fetcher = None
        self._concept_extractor = None
        self._intuition_builder = None
        self._math_simplifier = None
        self._example_generator = None
        self._progressive_builder = None
        self._content_polisher = None

    def _init_agents(self):
        """Initialize all agents with per-agent learned context from prior executions."""
        if self._agents_initialized:
            return

        self._init_shared_resources()

        self._paper_fetcher = PaperFetcherAgent(self._memory, self._context, self._bus, self._agent_context("PaperFetcher"))
        self._concept_extractor = ConceptExtractorAgent(self._memory, self._context, self._bus, self._agent_context("ConceptExtractor"))
        self._intuition_builder = IntuitionBuilderAgent(self._memory, self._context, self._bus, self._agent_context("IntuitionBuilder"))
        self._math_simplifier = MathSimplifierAgent(self._memory, self._context, self._bus, self._agent_context("MathSimplifier"))
        self._example_generator = ExampleGeneratorAgent(self._memory, self._context, self._bus, self._agent_context("ExampleGenerator"))
        self._progressive_builder = ProgressiveBuilderAgent(self._memory, self._context, self._bus, self._agent_context("ProgressiveBuilder"))
        self._content_polisher = ContentPolisherAgent(self._memory, self._context, self._bus, self._agent_context("ContentPolisher"))

        self._agents_initialized = True
        has_ctx = self._agent_context("PaperFetcher") or self._agent_context("ConceptExtractor")
        if has_ctx:
            logger.info("ArxivLearningSwarm agents initialized (with per-agent learned context)")
        else:
            logger.info("ArxivLearningSwarm agents initialized")

    async def execute(
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

        # Pre-execution learning: load state, warmup, compute scores
        await self._pre_execute_learning()

        # Reset agents so they get fresh learned context
        self._agents_initialized = False
        self._init_agents()

        config = self.config
        learning_depth = depth or config.depth

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
            # PHASE 2: EXTRACT CONCEPTS
            # =================================================================
            logger.info("🧠 Phase 2: Extracting concepts...")

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

            logger.info(f"  Extracted {len(concepts)} concepts")

            self._trace_phase("ConceptExtractor", AgentRole.EXPERT,
                {'paper_title': paper.title},
                {'concepts_count': len(concepts)},
                success=len(concepts) > 0, phase_start=start_time, tools_used=['concept_extract'])

            # =================================================================
            # PHASE 3: BUILD INTUITION FOR EACH CONCEPT (parallel)
            # =================================================================
            logger.info("💡 Phase 3: Building intuition...")

            intuition_tasks = [
                self._intuition_builder.build(concept, config.audience.value)
                for concept in concepts[:5]  # Limit to top 5 concepts
            ]

            intuition_results = await asyncio.gather(*intuition_tasks, return_exceptions=True)

            intuitions = {}
            for i, result in enumerate(intuition_results):
                if isinstance(result, dict) and not isinstance(result, Exception):
                    intuitions[concepts[i].name] = result

            phase3_start = datetime.now()
            self._trace_phase("IntuitionBuilder", AgentRole.ACTOR,
                {'concepts_count': len(concepts[:5])},
                {'intuitions_built': len(intuitions)},
                success=len(intuitions) > 0, phase_start=start_time, tools_used=['intuition_build'])

            # =================================================================
            # PHASE 4: SIMPLIFY MATH (parallel)
            # =================================================================
            math_explanations = {}

            if learning_depth in [LearningDepth.STANDARD, LearningDepth.DEEP]:
                logger.info("📐 Phase 4: Simplifying math...")

                math_tasks = [
                    self._math_simplifier.simplify(
                        concept,
                        intuitions.get(concept.name, {}),
                        config.audience.value
                    )
                    for concept in concepts[:5]
                    if concept.math_required
                ]

                math_results = await asyncio.gather(*math_tasks, return_exceptions=True)

                math_idx = 0
                for concept in concepts[:5]:
                    if concept.math_required and math_idx < len(math_results):
                        result = math_results[math_idx]
                        if isinstance(result, dict) and not isinstance(result, Exception):
                            math_explanations[concept.name] = result
                        math_idx += 1

            self._trace_phase("MathSimplifier", AgentRole.ACTOR,
                {'math_concepts_count': sum(1 for c in concepts[:5] if c.math_required)},
                {'math_explanations_count': len(math_explanations)},
                success=True, phase_start=phase3_start, tools_used=['math_simplify'])

            # =================================================================
            # PHASE 5: GENERATE EXAMPLES (parallel)
            # =================================================================
            examples = {}

            if config.include_code_examples:
                logger.info("💻 Phase 5: Generating examples...")

                example_tasks = [
                    self._example_generator.generate(
                        concept,
                        intuitions.get(concept.name, {}),
                        math_explanations.get(concept.name, {})
                    )
                    for concept in concepts[:3]  # Top 3 for examples
                ]

                example_results = await asyncio.gather(*example_tasks, return_exceptions=True)

                for i, result in enumerate(example_results):
                    if isinstance(result, dict) and not isinstance(result, Exception):
                        examples[concepts[i].name] = result

            self._trace_phase("ExampleGenerator", AgentRole.ACTOR,
                {'concepts_for_examples': min(3, len(concepts))},
                {'examples_generated': len(examples)},
                success=True, phase_start=phase3_start, tools_used=['example_generate'])

            # =================================================================
            # PHASE 6: BUILD PROGRESSIVE CONTENT
            # =================================================================
            logger.info("🏗️ Phase 6: Building progressive content...")

            progressive_result = await self._progressive_builder.build(
                paper,
                concepts,
                intuitions,
                math_explanations,
                examples,
                config.celebration_word
            )

            phase6_start = datetime.now()
            self._trace_phase("ProgressiveBuilder", AgentRole.PLANNER,
                {'concepts_count': len(concepts), 'has_math': bool(math_explanations), 'has_examples': bool(examples)},
                {'has_content': bool(progressive_result.get('complete_content')), 'insights_count': len(progressive_result.get('key_insights', []))},
                success=bool(progressive_result.get('complete_content')), phase_start=phase3_start, tools_used=['progressive_build'])

            # =================================================================
            # PHASE 7: POLISH CONTENT
            # =================================================================
            logger.info("✨ Phase 7: Polishing content...")

            draft_content = progressive_result.get('complete_content', '')

            polished = await self._content_polisher.polish(
                draft_content,
                config.style.value,
                config.audience.value
            )

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

            # Intuition sections
            for concept in concepts[:3]:
                intuition = intuitions.get(concept.name, {})
                if intuition:
                    sections.append(LearningSection(
                        title=f"Understanding {concept.name}",
                        content=f"{intuition.get('analogy', '')}\n\n{intuition.get('intuition_build', '')}",
                        level=2,
                        has_bingo_moment=True if intuition.get('aha_moment') else False
                    ))

            # Math sections
            for concept_name, math in math_explanations.items():
                sections.append(LearningSection(
                    title=f"The Math: {concept_name}",
                    content=f"{math.get('math_motivation', '')}\n\n{math.get('step_by_step', '')}",
                    level=3,
                    has_bingo_moment=False
                ))

            # Example sections
            for concept_name, ex in examples.items():
                # code_example is rendered separately by generate_learning_html,
                # so don't embed it in content (avoids duplication + "python" leak)
                raw_code = ex.get('code_example', '')
                # Strip any existing fence markers the LLM may have included
                raw_code = re.sub(r'^```\w*\n?', '', raw_code)
                raw_code = re.sub(r'\n?```$', '', raw_code).strip()
                sections.append(LearningSection(
                    title=f"See It In Action: {concept_name}",
                    content=f"**Simple Example:**\n{ex.get('simple_example', '')}",
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
            # PHASE 7.5: GENERATE PDF (always, independent of Telegram)
            # =================================================================
            logger.info("📄 Generating PDF...")
            pdf_path = await self._generate_pdf(paper, learning_content)

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
                pdf_path=pdf_path
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
                await self._send_to_telegram(paper, learning_content, final_content, pdf_path=pdf_path)
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
        max_papers: int = 1
    ) -> List[ArxivLearningResult]:
        """Search for papers and create learning content."""
        self._init_agents()

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
            # GENERATE VISUALIZATIONS FOR KEY CONCEPTS
            # =============================================================
            logger.info("🎨 Generating concept visualizations...")
            visualization_paths = {}

            # Build a map of concept name -> intuition text from sections
            concept_intuitions = {}
            for s in content.sections:
                for c in content.concepts:
                    if c.name.lower() in s.title.lower() and c.name not in concept_intuitions:
                        concept_intuitions[c.name] = s.content[:500]

            # Generate visualizations for top 3 concepts using LLM-driven specs
            for concept in content.concepts[:3]:
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
                        visualization_paths[concept.name] = viz_path
                        logger.info(f"  Generated viz for: {concept.name}")
                except Exception as e:
                    logger.debug(f"Viz generation failed for {concept.name}: {e}")

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

    async def _send_to_telegram(
        self,
        paper: PaperInfo,
        content: LearningContent,
        full_content: str,
        pdf_path: Optional[str] = None
    ):
        """Send learning content summary and PDF to Telegram."""
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram tools not available")
            return

        try:
            celebration = self.config.celebration_word

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
            stats += f"📄 Full PDF attached below"

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
            # SEND PDF FILE
            # =================================================================
            if pdf_path and Path(pdf_path).exists():
                file_result = await send_telegram_file_tool({
                    'file_path': pdf_path,
                    'caption': f"📖 {paper.title[:50]} - Learning Guide"
                })

                if file_result.get('success'):
                    logger.info(f"✅ Sent PDF to Telegram")
                else:
                    logger.error(f"❌ PDF send failed: {file_result.get('error')}")
            else:
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
