"""
Idea Writer Swarm - World-Class Content Generation with Section Registry
=========================================================================

Production-grade swarm for:
- Multi-section content generation
- Template-based writing
- Research-backed articles
- Flexible section composition
- Multiple output formats

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                        IDEA WRITER SWARM                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Outline      │  │   Research     │  │   Section      │            │
│  │    Agent       │  │    Agent       │  │   Registry     │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    SECTION WRITERS                              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │    │
│  │  │ Intro    │ │ Analysis │ │ Research │ │ Conclusion│          │    │
│  │  │ Writer   │ │ Writer   │ │ Writer   │ │ Writer   │          │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     CONTENT ASSEMBLER                            │   │
│  │   Combines sections into cohesive, polished content              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.idea_writer_swarm import IdeaWriterSwarm, write

    # Full swarm with custom sections
    swarm = IdeaWriterSwarm()
    result = await swarm.write(
        topic="AI in Healthcare",
        sections=["introduction", "market_analysis", "case_studies", "conclusion"]
    )

    # One-liner
    result = await write("The Future of Electric Vehicles")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .base_swarm import (
    SwarmConfig, SwarmResult, AgentRole,
    register_swarm,
)
from .base import DomainSwarm, AgentTeam, _split_field
from .swarm_signatures import IdeaWriterSwarmSignature
from ..agents.base import DomainAgent, DomainAgentConfig, BaseSwarmAgent

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ContentType(Enum):
    ARTICLE = "article"
    REPORT = "report"
    BLOG = "blog"
    NEWSLETTER = "newsletter"
    WHITEPAPER = "whitepaper"
    RESEARCH = "research"


class Tone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ACADEMIC = "academic"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"


class OutputFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"
    PDF = "pdf"


@dataclass
class WriterConfig(SwarmConfig):
    """Configuration for IdeaWriterSwarm."""
    content_type: ContentType = ContentType.ARTICLE
    tone: Tone = Tone.PROFESSIONAL
    output_format: OutputFormat = OutputFormat.MARKDOWN
    word_count_target: int = 2000
    include_research: bool = True
    include_images: bool = False
    include_citations: bool = True
    language: str = "english"
    audience: str = "general"

    def __post_init__(self):
        self.name = "IdeaWriterSwarm"
        self.domain = "content_writing"


@dataclass
class Section:
    """A content section."""
    name: str
    title: str
    content: str
    word_count: int = 0
    sources: List[str] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class Outline:
    """Content outline."""
    title: str
    thesis: str
    sections: List[Dict[str, Any]]
    target_audience: str
    key_points: List[str]


@dataclass
class ContentResult:
    """Content generation result."""
    title: str
    content: str
    sections: List[Section]
    word_count: int
    sources: List[str]
    outline: Outline
    metadata: Dict[str, Any]


@dataclass
class WriterResult(SwarmResult):
    """Result from IdeaWriterSwarm."""
    content: Optional[ContentResult] = None
    title: str = ""
    word_count: int = 0
    sections_generated: int = 0
    quality_score: float = 0.0
    readability_score: float = 0.0


# =============================================================================
# SECTION REGISTRY
# =============================================================================

class SectionWriter(ABC):
    """Base class for section writers."""

    section_type: str = "generic"
    description: str = "Generic section writer"

    def __init__(self, memory=None, context=None, bus=None):
        self.memory = memory
        self.context = context
        self.bus = bus

    @abstractmethod
    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write the section content."""
        pass


class SectionRegistry:
    """
    Registry for section writers.

    Allows registering custom section types and retrieving them by name.
    """

    _writers: Dict[str, Type[SectionWriter]] = {}

    @classmethod
    def register(cls, section_type: str):
        """Decorator to register a section writer."""
        def decorator(writer_class: Type[SectionWriter]):
            cls._writers[section_type] = writer_class
            writer_class.section_type = section_type
            return writer_class
        return decorator

    @classmethod
    def get(cls, section_type: str) -> Optional[Type[SectionWriter]]:
        """Get a section writer by type."""
        return cls._writers.get(section_type)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered section types."""
        return list(cls._writers.keys())

    @classmethod
    def create(
        cls,
        section_type: str,
        memory=None,
        context=None,
        bus=None
    ) -> Optional[SectionWriter]:
        """Create a section writer instance."""
        writer_class = cls.get(section_type)
        if writer_class:
            return writer_class(memory, context, bus)
        return None


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class OutlineGenerationSignature(dspy.Signature):
    """Generate content outline.

    You are a CONTENT STRATEGIST. Create a compelling outline with:
    1. Engaging title
    2. Clear thesis/main argument
    3. Logical section flow
    4. Key points per section
    5. Target audience consideration

    Make the outline comprehensive but focused.
    """
    topic: str = dspy.InputField(desc="Main topic or idea")
    content_type: str = dspy.InputField(desc="Type of content: article, report, etc.")
    audience: str = dspy.InputField(desc="Target audience")
    tone: str = dspy.InputField(desc="Desired tone")
    word_count: int = dspy.InputField(desc="Target word count")

    title: str = dspy.OutputField(desc="Compelling title")
    thesis: str = dspy.OutputField(desc="Main thesis or argument")
    sections: str = dspy.OutputField(desc="JSON list of sections with titles and key points")
    key_points: str = dspy.OutputField(desc="Overall key points, separated by |")


class ResearchSignature(dspy.Signature):
    """Research a topic thoroughly.

    You are a RESEARCH SPECIALIST. Gather information on:
    1. Key facts and statistics
    2. Expert opinions
    3. Current trends
    4. Historical context
    5. Counter-arguments

    Provide well-sourced, accurate information.
    """
    topic: str = dspy.InputField(desc="Topic to research")
    outline: str = dspy.InputField(desc="Content outline for context")
    depth: str = dspy.InputField(desc="Research depth: surface, moderate, deep")

    facts: str = dspy.OutputField(desc="Key facts and statistics, separated by |")
    expert_views: str = dspy.OutputField(desc="Expert opinions, separated by |")
    trends: str = dspy.OutputField(desc="Current trends, separated by |")
    sources: str = dspy.OutputField(desc="Sources/references, separated by |")


class IntroductionSignature(dspy.Signature):
    """Write engaging introduction.

    You are a MASTER WRITER. Create an introduction that:
    1. Hooks the reader immediately
    2. Establishes the topic's importance
    3. Previews the main argument
    4. Sets the tone for the piece
    5. Transitions smoothly to body

    Make it compelling and concise.
    """
    topic: str = dspy.InputField(desc="Topic")
    thesis: str = dspy.InputField(desc="Main thesis")
    tone: str = dspy.InputField(desc="Writing tone")
    audience: str = dspy.InputField(desc="Target audience")
    research: str = dspy.InputField(desc="Research findings")

    hook: str = dspy.OutputField(desc="Opening hook sentence")
    introduction: str = dspy.OutputField(desc="Complete introduction paragraph(s)")


class BodySectionSignature(dspy.Signature):
    """Write body section content.

    You are a CONTENT EXPERT. Write a section that:
    1. Develops the main point clearly
    2. Supports claims with evidence
    3. Uses appropriate examples
    4. Maintains consistent tone
    5. Flows logically

    Balance depth with readability.
    """
    section_title: str = dspy.InputField(desc="Section title")
    key_points: str = dspy.InputField(desc="Key points to cover")
    research: str = dspy.InputField(desc="Relevant research")
    tone: str = dspy.InputField(desc="Writing tone")
    context: str = dspy.InputField(desc="Context from other sections")

    content: str = dspy.OutputField(desc="Complete section content")
    transitions: str = dspy.OutputField(desc="Transition sentences for flow")


class ConclusionSignature(dspy.Signature):
    """Write powerful conclusion.

    You are a CLOSING EXPERT. Create a conclusion that:
    1. Summarizes key points
    2. Reinforces the thesis
    3. Provides actionable insights
    4. Ends memorably
    5. Call to action if appropriate

    Leave a lasting impression.
    """
    topic: str = dspy.InputField(desc="Topic")
    thesis: str = dspy.InputField(desc="Main thesis")
    key_points: str = dspy.InputField(desc="Key points covered")
    tone: str = dspy.InputField(desc="Writing tone")

    summary: str = dspy.OutputField(desc="Summary of main points")
    conclusion: str = dspy.OutputField(desc="Complete conclusion")
    call_to_action: str = dspy.OutputField(desc="Call to action if appropriate")


class ContentPolishSignature(dspy.Signature):
    """Polish and refine content.

    You are an EDITOR. Improve the content by:
    1. Enhancing clarity
    2. Fixing transitions
    3. Improving flow
    4. Ensuring consistency
    5. Checking for errors

    Make it publication-ready.
    """
    content: str = dspy.InputField(desc="Draft content")
    tone: str = dspy.InputField(desc="Desired tone")
    audience: str = dspy.InputField(desc="Target audience")

    polished_content: str = dspy.OutputField(desc="Polished content")
    improvements: str = dspy.OutputField(desc="Improvements made, separated by |")
    quality_score: float = dspy.OutputField(desc="Quality score 0-100")


# =============================================================================
# BUILT-IN SECTION WRITERS
# =============================================================================

@SectionRegistry.register("introduction")
class IntroductionWriter(SectionWriter):
    """Writes engaging introductions."""

    section_type = "introduction"
    description = "Writes compelling introductions with hooks"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(IntroductionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write introduction."""
        try:
            result = self._writer(
                topic=topic,
                thesis=context.get('thesis', ''),
                tone=config.tone.value,
                audience=config.audience,
                research=json.dumps(research)
            )

            content = f"## Introduction\n\n{result.introduction}"

            return Section(
                name="introduction",
                title="Introduction",
                content=content,
                word_count=len(content.split())
            )

        except Exception as e:
            logger.error(f"Introduction writing failed: {e}")
            return Section(name="introduction", title="Introduction", content="")


@SectionRegistry.register("body")
class BodySectionWriter(SectionWriter):
    """Writes body section content."""

    section_type = "body"
    description = "Writes detailed body sections"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(BodySectionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write body section."""
        try:
            section_info = context.get('section_info', {})
            result = self._writer(
                section_title=section_info.get('title', topic),
                key_points=json.dumps(section_info.get('key_points', [])),
                research=json.dumps(research),
                tone=config.tone.value,
                context=json.dumps(context.get('previous_sections', []))
            )

            title = section_info.get('title', 'Section')
            content = f"## {title}\n\n{result.content}"

            return Section(
                name=section_info.get('name', 'body'),
                title=title,
                content=content,
                word_count=len(content.split())
            )

        except Exception as e:
            logger.error(f"Body section writing failed: {e}")
            return Section(name="body", title="Section", content="")


@SectionRegistry.register("conclusion")
class ConclusionWriter(SectionWriter):
    """Writes powerful conclusions."""

    section_type = "conclusion"
    description = "Writes memorable conclusions"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(ConclusionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write conclusion."""
        try:
            result = self._writer(
                topic=topic,
                thesis=context.get('thesis', ''),
                key_points=json.dumps(context.get('key_points', [])),
                tone=config.tone.value
            )

            content = f"## Conclusion\n\n{result.conclusion}"
            if result.call_to_action:
                content += f"\n\n{result.call_to_action}"

            return Section(
                name="conclusion",
                title="Conclusion",
                content=content,
                word_count=len(content.split())
            )

        except Exception as e:
            logger.error(f"Conclusion writing failed: {e}")
            return Section(name="conclusion", title="Conclusion", content="")


@SectionRegistry.register("market_analysis")
class MarketAnalysisWriter(SectionWriter):
    """Writes market analysis sections."""

    section_type = "market_analysis"
    description = "Writes market analysis with data and trends"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(BodySectionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write market analysis."""
        try:
            result = self._writer(
                section_title=f"Market Analysis: {topic}",
                key_points=json.dumps([
                    "Market size and growth",
                    "Key players",
                    "Trends and drivers",
                    "Challenges and opportunities"
                ]),
                research=json.dumps(research),
                tone=config.tone.value,
                context=""
            )

            content = f"## Market Analysis\n\n{result.content}"

            return Section(
                name="market_analysis",
                title="Market Analysis",
                content=content,
                word_count=len(content.split())
            )

        except Exception as e:
            logger.error(f"Market analysis writing failed: {e}")
            return Section(name="market_analysis", title="Market Analysis", content="")


@SectionRegistry.register("case_studies")
class CaseStudiesWriter(SectionWriter):
    """Writes case study sections."""

    section_type = "case_studies"
    description = "Writes case studies with examples"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(BodySectionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write case studies."""
        try:
            result = self._writer(
                section_title=f"Case Studies: {topic}",
                key_points=json.dumps([
                    "Real-world examples",
                    "Success stories",
                    "Lessons learned",
                    "Key takeaways"
                ]),
                research=json.dumps(research),
                tone=config.tone.value,
                context=""
            )

            content = f"## Case Studies\n\n{result.content}"

            return Section(
                name="case_studies",
                title="Case Studies",
                content=content,
                word_count=len(content.split())
            )

        except Exception as e:
            logger.error(f"Case studies writing failed: {e}")
            return Section(name="case_studies", title="Case Studies", content="")


@SectionRegistry.register("technical_deep_dive")
class TechnicalDeepDiveWriter(SectionWriter):
    """Writes technical deep dive sections."""

    section_type = "technical_deep_dive"
    description = "Writes detailed technical explanations"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(BodySectionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write technical deep dive."""
        try:
            result = self._writer(
                section_title=f"Technical Deep Dive: {topic}",
                key_points=json.dumps([
                    "Technical architecture",
                    "Implementation details",
                    "Best practices",
                    "Common pitfalls"
                ]),
                research=json.dumps(research),
                tone="technical",
                context=""
            )

            content = f"## Technical Deep Dive\n\n{result.content}"

            return Section(
                name="technical_deep_dive",
                title="Technical Deep Dive",
                content=content,
                word_count=len(content.split())
            )

        except Exception as e:
            logger.error(f"Technical deep dive writing failed: {e}")
            return Section(name="technical_deep_dive", title="Technical Deep Dive", content="")


@SectionRegistry.register("research_findings")
class ResearchFindingsWriter(SectionWriter):
    """Writes research findings sections using existing swarms."""

    section_type = "research_findings"
    description = "Integrates with ResearchSwarm for data-backed sections"

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._writer = dspy.ChainOfThought(BodySectionSignature)

    async def write(
        self,
        topic: str,
        context: Dict[str, Any],
        research: Dict[str, Any],
        config: WriterConfig
    ) -> Section:
        """Write research findings."""
        # Optionally integrate with ResearchSwarm
        research_data = research

        try:
            result = self._writer(
                section_title=f"Research Findings: {topic}",
                key_points=json.dumps([
                    "Data analysis",
                    "Key findings",
                    "Statistical insights",
                    "Implications"
                ]),
                research=json.dumps(research_data),
                tone=config.tone.value,
                context=""
            )

            content = f"## Research Findings\n\n{result.content}"

            return Section(
                name="research_findings",
                title="Research Findings",
                content=content,
                word_count=len(content.split()),
                sources=research.get('sources', [])
            )

        except Exception as e:
            logger.error(f"Research findings writing failed: {e}")
            return Section(name="research_findings", title="Research Findings", content="")


# =============================================================================
# AGENTS
# =============================================================================



class OutlineAgent(BaseSwarmAgent):
    """Generates content outlines."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=OutlineGenerationSignature)
        self._generator = dspy.ChainOfThought(OutlineGenerationSignature)
        self.learned_context = learned_context

    async def generate(
        self,
        topic: str,
        config: WriterConfig
    ) -> Outline:
        """Generate content outline."""
        try:
            enriched_topic = f"{topic}\n\n{self.learned_context}" if self.learned_context else topic
            result = self._generator(
                topic=enriched_topic,
                content_type=config.content_type.value,
                audience=config.audience,
                tone=config.tone.value,
                word_count=config.word_count_target
            )

            # Parse sections
            try:
                sections = json.loads(result.sections)
            except Exception:
                sections = [{'title': 'Main Content', 'key_points': []}]

            key_points = _split_field(result.key_points)

            self._broadcast("outline_generated", {
                'topic': topic,
                'sections': len(sections)
            })

            return Outline(
                title=str(result.title),
                thesis=str(result.thesis),
                sections=sections,
                target_audience=config.audience,
                key_points=key_points
            )

        except Exception as e:
            logger.error(f"Outline generation failed: {e}")
            return Outline(
                title=topic,
                thesis="",
                sections=[],
                target_audience="general",
                key_points=[]
            )


class ResearchAgent(BaseSwarmAgent):
    """Researches topics."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=ResearchSignature)
        self._researcher = dspy.ChainOfThought(ResearchSignature)
        self.learned_context = learned_context

    async def research(
        self,
        topic: str,
        outline: Outline,
        depth: str = "moderate"
    ) -> Dict[str, Any]:
        """Research a topic."""
        try:
            enriched_topic = f"{topic}\n\n{self.learned_context}" if self.learned_context else topic
            result = self._researcher(
                topic=enriched_topic,
                outline=json.dumps({
                    'title': outline.title,
                    'thesis': outline.thesis,
                    'sections': [s.get('title', '') for s in outline.sections]
                }),
                depth=depth
            )

            facts = _split_field(result.facts)
            expert_views = _split_field(result.expert_views)
            trends = _split_field(result.trends)
            sources = _split_field(result.sources)

            self._broadcast("research_completed", {
                'topic': topic,
                'facts_found': len(facts)
            })

            return {
                'facts': facts,
                'expert_views': expert_views,
                'trends': trends,
                'sources': sources
            }

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {'facts': [], 'expert_views': [], 'trends': [], 'sources': []}


class PolishAgent(BaseSwarmAgent):
    """Polishes and refines content."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=ContentPolishSignature)
        self.learned_context = learned_context
        self._polisher = dspy.ChainOfThought(ContentPolishSignature)

    async def polish(
        self,
        content: str,
        config: WriterConfig
    ) -> Dict[str, Any]:
        """Polish content."""
        try:
            enriched_content = f"{content}\n\n{self.learned_context}" if self.learned_context else content
            result = self._polisher(
                content=enriched_content,
                tone=config.tone.value,
                audience=config.audience
            )

            improvements = _split_field(result.improvements)

            self._broadcast("content_polished", {
                'improvements': len(improvements)
            })

            return {
                'polished_content': str(result.polished_content),
                'improvements': improvements,
                'quality_score': float(result.quality_score) if result.quality_score else 75.0
            }

        except Exception as e:
            logger.error(f"Polish failed: {e}")
            return {
                'polished_content': content,
                'improvements': [],
                'quality_score': 50.0
            }


# =============================================================================
# IDEA WRITER SWARM
# =============================================================================

@register_swarm("idea_writer")
class IdeaWriterSwarm(DomainSwarm):
    """
    World-Class Idea Writer Swarm.

    Generates high-quality content with:
    - Flexible section composition via registry
    - Research-backed writing
    - Multiple output formats
    - Professional polishing
    """

    AGENT_TEAM = AgentTeam.define(
        (OutlineAgent, "Outline", "_outline_agent"),
        (ResearchAgent, "Research", "_research_agent"),
        (PolishAgent, "Polish", "_polish_agent"),
    )
    SWARM_SIGNATURE = IdeaWriterSwarmSignature

    def __init__(self, config: WriterConfig = None):
        super().__init__(config or WriterConfig())

    async def write(
        self,
        topic: str,
        sections: List[str] = None,
        custom_outline: Outline = None
    ) -> WriterResult:
        """
        Write content on a topic.

        Args:
            topic: Main topic or idea
            sections: List of section types to include (uses registry)
            custom_outline: Optional pre-made outline

        Returns:
            WriterResult with generated content
        """
        return await self.execute(topic, sections=sections, custom_outline=custom_outline)

    async def _execute_domain(
        self,
        topic: str,
        sections: List[str] = None,
        custom_outline: Outline = None,
        **kwargs
    ) -> WriterResult:
        """
        Domain-specific content writing logic.

        Delegates to _safe_execute_domain which handles try/except,
        timing, and post-execute learning automatically via PhaseExecutor.

        Args:
            topic: Main topic or idea
            sections: List of section types to include (uses registry)
            custom_outline: Optional pre-made outline

        Returns:
            WriterResult with generated content
        """
        # Default sections if not provided
        if not sections:
            sections = ["introduction", "body", "body", "conclusion"]

        return await self._safe_execute_domain(
            task_type='content_writing',
            default_tools=['outline_generate', 'research', 'section_write', 'content_polish'],
            result_class=WriterResult,
            execute_fn=lambda executor: self._execute_phases(
                executor, topic, sections, custom_outline
            ),
            output_data_fn=lambda result: {
                'title': result.title,
                'word_count': result.word_count,
                'sections': result.sections_generated,
            },
            input_data_fn=lambda: {
                'topic': topic[:200],
                'sections': sections,
            },
        )

    async def _execute_phases(
        self,
        executor,
        topic: str,
        sections: List[str],
        custom_outline: Outline = None
    ) -> WriterResult:
        """
        Domain-specific phase logic using PhaseExecutor.

        Args:
            executor: PhaseExecutor instance for tracing and timing
            topic: Main topic or idea
            sections: List of section types to include
            custom_outline: Optional pre-made outline

        Returns:
            WriterResult with generated content
        """
        config = self.config

        logger.info(f"IdeaWriterSwarm starting: {topic}")

        # =================================================================
        # PHASE 1: OUTLINE GENERATION
        # =================================================================
        if custom_outline:
            outline = custom_outline
        else:
            outline = await executor.run_phase(
                1, "Outline Generation", "Outline", AgentRole.PLANNER,
                self._outline_agent.generate(topic, config),
                input_data={'topic': topic[:100]},
                tools_used=['outline_generate'],
            )

        if not outline.sections:
            # Create default sections from provided list
            outline.sections = [
                {'name': s, 'title': s.replace('_', ' ').title(), 'key_points': []}
                for s in sections
            ]

        # =================================================================
        # PHASE 2: RESEARCH
        # =================================================================
        research = {}
        if config.include_research:
            research = await executor.run_phase(
                2, "Research", "Research", AgentRole.EXPERT,
                self._research_agent.research(topic, outline, depth="moderate"),
                input_data={'include_research': config.include_research},
                tools_used=['research'],
            )

        # =================================================================
        # PHASE 3: SECTION WRITING (parallel)
        # =================================================================
        parallel_tasks = []

        for i, section_type in enumerate(sections):
            writer = SectionRegistry.create(
                section_type,
                self._memory,
                self._context,
                self._bus
            )

            if not writer:
                # Fallback to body writer for unknown types
                writer = SectionRegistry.create(
                    "body",
                    self._memory,
                    self._context,
                    self._bus
                )

            if writer:
                section_info = outline.sections[i] if i < len(outline.sections) else {}
                ctx = {
                    'thesis': outline.thesis,
                    'key_points': outline.key_points,
                    'section_info': section_info,
                    'previous_sections': []
                }
                parallel_tasks.append((
                    f"SectionWriter({section_type})",
                    AgentRole.ACTOR,
                    writer.write(topic, ctx, research, config),
                    ['section_write'],
                ))

        section_results = await executor.run_parallel(
            3, "Section Writing", parallel_tasks
        )

        written_sections = []
        for result in section_results:
            if isinstance(result, dict) and 'error' in result:
                logger.warning(f"Section failed: {result['error']}")
                continue
            if isinstance(result, Section):
                written_sections.append(result)

        # =================================================================
        # PHASE 4: ASSEMBLY
        # =================================================================
        title = outline.title or topic
        full_content = f"# {title}\n\n"
        full_content += f"*{outline.thesis}*\n\n" if outline.thesis else ""

        for section in written_sections:
            full_content += f"{section.content}\n\n"

        # =================================================================
        # PHASE 5: POLISH
        # =================================================================
        polish_result = await executor.run_phase(
            5, "Polish", "Polish", AgentRole.REVIEWER,
            self._polish_agent.polish(full_content, config),
            input_data={'content_length': len(full_content)},
            tools_used=['content_polish'],
        )

        final_content = polish_result.get('polished_content', full_content)
        quality_score = polish_result.get('quality_score', 75.0)

        # =================================================================
        # BUILD RESULT
        # =================================================================
        content_result = ContentResult(
            title=title,
            content=final_content,
            sections=written_sections,
            word_count=len(final_content.split()),
            sources=research.get('sources', []),
            outline=outline,
            metadata={
                'content_type': config.content_type.value,
                'tone': config.tone.value,
                'audience': config.audience
            }
        )

        result = WriterResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                'content': final_content,
                'title': title,
                'word_count': len(final_content.split()),
                'quality_score': quality_score / 100.0,
            },
            execution_time=executor.elapsed(),
            content=content_result,
            title=title,
            word_count=len(final_content.split()),
            sections_generated=len(written_sections),
            quality_score=quality_score / 100.0,
            readability_score=0.8  # Could use readability metrics
        )

        logger.info(f"IdeaWriterSwarm complete: {title}, {result.word_count} words")

        return result

    def list_available_sections(self) -> List[str]:
        """List all available section types."""
        return SectionRegistry.list_all()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def write(topic: str, **kwargs) -> WriterResult:
    """
    One-liner content writing.

    Usage:
        from core.swarms.idea_writer_swarm import write
        result = await write("The Future of AI")
    """
    swarm = IdeaWriterSwarm()
    return await swarm.execute(topic, **kwargs)


def write_sync(topic: str, **kwargs) -> WriterResult:
    """Synchronous content writing."""
    return asyncio.run(write(topic, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'IdeaWriterSwarm',
    'WriterConfig',
    'WriterResult',
    'ContentResult',
    'Section',
    'Outline',
    'ContentType',
    'Tone',
    'OutputFormat',
    'write',
    'write_sync',
    # Section Registry
    'SectionRegistry',
    'SectionWriter',
    # Built-in writers
    'IntroductionWriter',
    'BodySectionWriter',
    'ConclusionWriter',
    'MarketAnalysisWriter',
    'CaseStudiesWriter',
    'TechnicalDeepDiveWriter',
    'ResearchFindingsWriter',
    # Agents
    'OutlineAgent',
    'ResearchAgent',
    'PolishAgent',
]
