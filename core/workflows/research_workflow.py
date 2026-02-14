#!/usr/bin/env python3
"""
ResearchWorkflow - Intent-Based Research Pipeline
==================================================

Automatically decomposes research goals into stages and executes them
using ResearchSwarm and complementary agents.

Usage:
    # Simplest - just provide topic
    result = await research("AI safety challenges in 2026")

    # With depth control
    result = await research(
        topic="Quantum computing developments",
        depth="comprehensive",
        deliverables=["literature_review", "synthesis", "visualization"]
    )

    # Full customization
    workflow = ResearchWorkflow.from_intent(
        topic="AI safety",
        research_type="academic",
        deliverables=["literature_review", "analysis", "synthesis", "report"]
    )
    workflow.customize_stage("analysis", max_tokens=3000)
    result = await workflow.run()
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from ..orchestration import (
    MultiStagePipeline,
    SwarmAdapter,
    MergeStrategy,
    PipelineResult,
)


class ResearchDepth(Enum):
    """Research depth levels."""
    QUICK = "quick"              # 2-3 sources, surface-level
    STANDARD = "standard"        # 5-10 sources, moderate depth
    COMPREHENSIVE = "comprehensive"  # 15-25 sources, deep analysis
    EXHAUSTIVE = "exhaustive"    # 50+ sources, PhD-level


class ResearchType(Enum):
    """Types of research."""
    GENERAL = "general"          # General topic research
    ACADEMIC = "academic"        # Academic papers, citations
    MARKET = "market"            # Market/business research
    TECHNICAL = "technical"      # Technical/engineering research
    COMPETITIVE = "competitive"  # Competitive analysis
    TREND = "trend"             # Trend analysis


@dataclass
class ResearchIntent:
    """High-level research intent."""
    topic: str
    research_type: Optional[str] = "general"
    depth: Optional[str] = "standard"
    deliverables: Optional[List[str]] = None
    output_formats: Optional[List[str]] = None  # ["pdf", "html", "markdown"]
    send_telegram: bool = True
    max_sources: Optional[int] = None


class ResearchWorkflow:
    """
    Automatically decomposes research goals into stages and executes them.

    Research Pipeline Stages:
    1. Topic Analysis - Understand research question, identify sub-questions
    2. Literature Review - Find relevant sources (web, papers, books)
    3. Data Collection - Gather data from sources
    4. Analysis - Deep analysis of findings
    5. Synthesis - Combine findings into coherent narrative
    6. Validation - Cross-check claims, verify sources
    7. Visualization - Create charts, diagrams, infographics
    8. Documentation - Generate final report
    9. Bibliography - Compile citations
    """

    def __init__(self, intent: ResearchIntent):
        """Initialize research workflow."""
        self.intent = intent
        self.pipeline = None
        self.stage_configs = {}

    @classmethod
    def from_intent(
        cls,
        topic: str,
        research_type: Optional[str] = "general",
        depth: Optional[str] = "standard",
        deliverables: Optional[List[str]] = None,
        output_formats: Optional[List[str]] = None,
        send_telegram: bool = True,
        max_sources: Optional[int] = None
    ) -> "ResearchWorkflow":
        """
        Create research workflow from intent (without executing).

        Args:
            topic: Research topic/question
            research_type: Type of research (general/academic/market/technical)
            depth: Research depth (quick/standard/comprehensive/exhaustive)
            deliverables: What to generate (optional)
            output_formats: Output formats (pdf, html, markdown)
            send_telegram: Send results via Telegram
            max_sources: Max sources to use

        Returns:
            ResearchWorkflow instance (not yet executed)
        """
        intent = ResearchIntent(
            topic=topic,
            research_type=research_type,
            depth=depth,
            deliverables=deliverables,
            output_formats=output_formats or ["pdf", "markdown"],
            send_telegram=send_telegram,
            max_sources=max_sources
        )
        return cls(intent)

    @classmethod
    async def execute(
        cls,
        topic: str,
        research_type: Optional[str] = "general",
        depth: Optional[str] = "standard",
        deliverables: Optional[List[str]] = None,
        output_formats: Optional[List[str]] = None,
        send_telegram: bool = True,
        max_sources: Optional[int] = None,
        verbose: bool = True
    ) -> PipelineResult:
        """
        Execute research workflow from high-level intent (simplest API).

        Args:
            topic: Research topic/question
            research_type: Type of research
            depth: Research depth
            deliverables: What to generate
            output_formats: Output formats
            send_telegram: Send via Telegram
            max_sources: Max sources
            verbose: Print progress

        Returns:
            PipelineResult with all outputs
        """
        intent = ResearchIntent(
            topic=topic,
            research_type=research_type,
            depth=depth,
            deliverables=deliverables,
            output_formats=output_formats or ["pdf", "markdown"],
            send_telegram=send_telegram,
            max_sources=max_sources
        )

        workflow = cls(intent)
        workflow.build_pipeline()
        return await workflow.run(verbose=verbose)

    def build_pipeline(self):
        """Build pipeline from research intent."""
        deliverables = self.intent.deliverables or self._infer_deliverables()

        self.pipeline = MultiStagePipeline(task=f"Research: {self.intent.topic}")

        self._add_research_stages(deliverables)

    def _infer_deliverables(self) -> List[str]:
        """Infer deliverables from research type and depth."""
        research_type = self.intent.research_type
        depth = self.intent.depth

        # Map research type to deliverables
        if research_type == "academic":
            base = ["literature_review", "analysis", "synthesis", "bibliography"]
        elif research_type == "market":
            base = ["market_overview", "competitor_analysis", "trends", "recommendations"]
        elif research_type == "technical":
            base = ["technical_overview", "architecture_analysis", "comparison", "recommendations"]
        elif research_type == "competitive":
            base = ["competitor_profiles", "swot_analysis", "positioning", "recommendations"]
        elif research_type == "trend":
            base = ["trend_identification", "impact_analysis", "forecast", "recommendations"]
        else:  # general
            base = ["overview", "deep_dive", "synthesis", "summary"]

        # Add visualization and documentation based on depth
        if depth in ["comprehensive", "exhaustive"]:
            base.extend(["visualization", "documentation"])

        return base

    def _add_research_stages(self, deliverables: List[str]):
        """Add pipeline stages for research deliverables."""

        # Determine max sources based on depth
        depth_to_sources = {
            "quick": 5,
            "standard": 15,
            "comprehensive": 25,
            "exhaustive": 50
        }
        max_sources = self.intent.max_sources or depth_to_sources.get(self.intent.depth, 15)

        # Build base context
        base_context = f"""
Research Topic: {self.intent.topic}
Research Type: {self.intent.research_type}
Depth: {self.intent.depth}
Max Sources: {max_sources}

Guidelines:
- Cite all sources with URLs
- Cross-reference claims
- Identify knowledge gaps
- Note conflicting information
- Provide evidence for all claims
"""

        # Track previous stages for context chaining
        previous_stages = []

        # Stage mapping
        stage_prompts = {
            # General research
            "overview": ("Research Analyst", f"""Provide comprehensive overview of the topic.

{base_context}

Structure:
1. Executive Summary (3-5 key points)
2. Background & Context
3. Current State
4. Key Players/Stakeholders
5. Challenges & Opportunities

Use web search to find {max_sources} high-quality sources.
Output in markdown with citations."""),

            "deep_dive": ("Subject Matter Expert", f"""Deep dive analysis of the topic.

{base_context}

Required:
1. Technical Details
2. Mechanisms/Processes
3. Case Studies (3-5)
4. Data & Statistics
5. Expert Opinions

Cite all sources. Use tables and lists for clarity."""),

            "synthesis": ("Synthesis Analyst", f"""Synthesize all research findings into coherent narrative.

{base_context}

Combine:
- All gathered information
- Cross-cutting themes
- Contradictions/debates
- Emerging patterns
- Future directions

Create unified report with clear structure."""),

            # Academic research
            "literature_review": ("Academic Researcher", f"""Conduct systematic literature review.

{base_context}

Find and analyze {max_sources} academic papers, articles, and books.

Structure:
1. Search Strategy
2. Selection Criteria
3. Key Findings by Theme
4. Methodologies Used
5. Knowledge Gaps
6. Future Research Directions

Use proper academic citations (APA format)."""),

            "bibliography": ("Bibliographer", f"""Compile comprehensive bibliography.

Format all sources in APA citation format.
Organize by:
1. Primary Sources
2. Secondary Sources
3. Data Sources
4. Additional Reading

Include DOIs, URLs, and access dates where applicable."""),

            # Market research
            "market_overview": ("Market Analyst", f"""Provide market overview.

{base_context}

Include:
1. Market Size & Growth
2. Key Segments
3. Geographic Distribution
4. Regulatory Environment
5. Market Drivers & Barriers

Use recent data (last 2 years preferred)."""),

            "competitor_analysis": ("Competitive Analyst", f"""Analyze competitive landscape.

{base_context}

For each major competitor:
1. Company Profile
2. Market Share
3. Strengths/Weaknesses
4. Strategy & Positioning
5. Recent Developments

Create comparison matrix."""),

            "trends": ("Trend Analyst", f"""Identify and analyze key trends.

{base_context}

For each trend:
1. Description & Evidence
2. Drivers
3. Impact Assessment
4. Timeline (short/medium/long-term)
5. Uncertainty Factors

Prioritize by significance and likelihood."""),

            # Technical research
            "technical_overview": ("Technical Analyst", f"""Technical deep dive.

{base_context}

Cover:
1. Architecture/Design
2. Technologies Used
3. Implementation Details
4. Performance Characteristics
5. Limitations & Trade-offs

Include diagrams if applicable."""),

            "architecture_analysis": ("Solutions Architect", f"""Analyze architecture and design.

{base_context}

Evaluate:
1. System Components
2. Integration Patterns
3. Scalability Approach
4. Security Model
5. Technology Stack

Provide architecture diagrams in ASCII or describe in detail."""),

            "comparison": ("Comparison Analyst", f"""Compare alternative approaches/solutions.

{base_context}

For each alternative:
1. Description
2. Pros & Cons
3. Use Cases
4. Cost/Complexity
5. Maturity & Support

Create comparison matrix with scoring."""),

            # Competitive research
            "competitor_profiles": ("Intelligence Analyst", f"""Create detailed competitor profiles.

{base_context}

For each competitor:
1. Background & History
2. Products/Services
3. Market Position
4. Financial Performance
5. Strategy & Direction
6. Strengths/Weaknesses

Minimum 5 competitors."""),

            "swot_analysis": ("Strategic Analyst", f"""Conduct SWOT analysis.

{base_context}

For the subject and top 3 competitors:
1. Strengths (internal positives)
2. Weaknesses (internal negatives)
3. Opportunities (external positives)
4. Threats (external negatives)

Create matrix with specific, actionable points."""),

            "positioning": ("Positioning Analyst", f"""Analyze market positioning.

{base_context}

Create:
1. Positioning Map (2-3 dimensions)
2. Differentiation Analysis
3. Target Segments
4. Value Proposition Comparison
5. Positioning Gaps/Opportunities

Visual representation if possible."""),

            # Recommendations
            "recommendations": ("Strategy Consultant", f"""Provide strategic recommendations.

{base_context}

Based on all research:
1. Key Insights (5-7 points)
2. Strategic Recommendations (prioritized)
3. Action Plan (immediate/short/long-term)
4. Success Metrics
5. Risk Mitigation

Be specific and actionable."""),

            "forecast": ("Forecasting Analyst", f"""Forecast future developments.

{base_context}

Provide:
1. Base Case Scenario (most likely)
2. Optimistic Scenario
3. Pessimistic Scenario
4. Key Uncertainties
5. Leading Indicators to Watch
6. Timeline (1, 3, 5 years)

Justify all forecasts with data."""),

            # Cross-cutting stages
            "visualization": ("Data Visualizer", f"""Create data visualizations.

Based on all research data, create:
1. Key Charts (3-5)
   - Trend charts
   - Comparison charts
   - Market share pie/bar charts
2. Infographics (conceptual)
3. Diagrams (process/architecture)

Describe in detail what each visualization shows.
Use ASCII charts or describe charts in markdown."""),

            "documentation": ("Technical Writer", f"""Create comprehensive documentation.

{base_context}

Compile complete research report:
1. Executive Summary (1 page)
2. Table of Contents
3. Main Content (all stages)
4. Appendices
5. Bibliography

Output Format: {', '.join(self.intent.output_formats)}

Professional quality, publication-ready."""),

            "summary": ("Summarizer", f"""Create executive summary.

Distill all research into:
1. Key Findings (5-7 bullets)
2. Main Conclusions
3. Critical Insights
4. Recommendations (top 3)
5. Next Steps

Maximum 500 words, extremely clear and actionable."""),
        }

        # Add stages based on deliverables
        for deliverable in deliverables:
            stage_config = self.stage_configs.get(deliverable, {})

            # Check if stage is completely replaced
            if stage_config.get("replace"):
                swarms = stage_config["custom_swarms"]
                merge_strategy = stage_config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = stage_config.get("context_from", previous_stages.copy() if previous_stages else None)
            else:
                # Auto-generate with prompts
                if deliverable not in stage_prompts:
                    print(f"⚠️  Unknown deliverable: {deliverable}, skipping")
                    continue

                agent_name, prompt = stage_prompts[deliverable]

                # Apply customizations
                if stage_config.get("additional_context"):
                    prompt = f"{prompt}\n\nAdditional Context:\n{stage_config['additional_context']}"

                # Create swarms
                swarms = SwarmAdapter.quick_swarms(
                    [(agent_name, prompt)],
                    model=stage_config.get("model", "claude-3-5-haiku-20241022"),
                    max_tokens=stage_config.get("max_tokens", 2000)
                )

                merge_strategy = stage_config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = previous_stages.copy() if previous_stages else None

            # Add stage
            self.pipeline.add_stage(
                name=deliverable,
                swarms=swarms,
                merge_strategy=merge_strategy,
                context_from=context_from,
                max_context_chars=2000
            )

            previous_stages.append(deliverable)

        # Add any custom stages
        for stage_name, config in self.stage_configs.items():
            if config.get("custom_stage"):
                swarms = config["swarms"]
                merge_strategy = config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = config.get("context_from")

                self.pipeline.add_stage(
                    name=stage_name,
                    swarms=swarms,
                    merge_strategy=merge_strategy,
                    context_from=context_from,
                    max_context_chars=2000
                )

    def customize_stage(
        self,
        stage_name: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        merge_strategy: Optional[MergeStrategy] = None,
        additional_context: Optional[str] = None
    ):
        """Customize a specific stage."""
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        if model:
            self.stage_configs[stage_name]["model"] = model
        if max_tokens:
            self.stage_configs[stage_name]["max_tokens"] = max_tokens
        if merge_strategy:
            self.stage_configs[stage_name]["merge_strategy"] = merge_strategy
        if additional_context:
            self.stage_configs[stage_name]["additional_context"] = additional_context

    def replace_stage(
        self,
        stage_name: str,
        swarms: List[Any],
        merge_strategy: Optional[MergeStrategy] = None,
        context_from: Optional[List[str]] = None
    ):
        """Completely replace a stage with custom swarms."""
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        self.stage_configs[stage_name]["replace"] = True
        self.stage_configs[stage_name]["custom_swarms"] = swarms
        if merge_strategy:
            self.stage_configs[stage_name]["merge_strategy"] = merge_strategy
        if context_from:
            self.stage_configs[stage_name]["context_from"] = context_from

    def add_custom_stage(
        self,
        stage_name: str,
        swarms: List[Any],
        position: Optional[int] = None,
        merge_strategy: MergeStrategy = MergeStrategy.BEST_OF_N,
        context_from: Optional[List[str]] = None
    ):
        """Add a completely custom stage."""
        self.stage_configs[stage_name] = {
            "custom_stage": True,
            "swarms": swarms,
            "position": position,
            "merge_strategy": merge_strategy,
            "context_from": context_from
        }

    def show_pipeline(self, verbose: bool = True):
        """Print pipeline structure."""
        deliverables = self.intent.deliverables or self._infer_deliverables()

        if verbose:
            print("\n" + "="*70)
            print("RESEARCH PIPELINE INSPECTION")
            print("="*70)
            print(f"\n🎯 Topic: {self.intent.topic}")
            print(f"📊 Type: {self.intent.research_type}")
            print(f"📖 Depth: {self.intent.depth}")
            print(f"📄 Stages: {len(deliverables)}")
            print()

            for i, stage in enumerate(deliverables, 1):
                status = ""
                if self.stage_configs.get(stage, {}).get("replace"):
                    status = "🔧 REPLACED (custom swarms)"
                elif stage in self.stage_configs:
                    status = "⚙️  CUSTOMIZED (tweaked)"
                else:
                    status = "🤖 AUTO (built-in prompts)"

                print(f"{i}. {stage:<25} {status}")

            print("\n" + "="*70 + "\n")

    async def run(self, verbose: bool = True) -> PipelineResult:
        """Execute the research pipeline."""
        if self.pipeline is None:
            self.build_pipeline()

        return await self.pipeline.execute(auto_trace=True, verbose=verbose)

    async def run_with_outputs(
        self,
        output_formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute pipeline and generate multiple output formats.

        Args:
            output_formats: List of formats (pdf, epub, html, docx, presentation)
            output_dir: Output directory (default: ~/jotty/outputs)
            verbose: Print progress

        Returns:
            Dict with 'pipeline_result' and 'outputs' (dict of OutputSinkResults)
        """
        # Execute pipeline
        pipeline_result = await self.run(verbose=verbose)

        # Use configured output formats if not specified
        formats = output_formats or self.intent.output_formats or ["pdf", "markdown"]

        # Generate outputs from documentation stage if it exists
        documentation_stage = None
        for stage in pipeline_result.stages:
            if stage.stage_name == "documentation":
                documentation_stage = stage
                break

        if not documentation_stage:
            # No documentation stage, just return pipeline result
            return {
                'pipeline_result': pipeline_result,
                'outputs': {},
                'note': 'No documentation stage found - outputs not generated'
            }

        # Save markdown content to file
        import tempfile
        from pathlib import Path
        import os

        output_path = Path(output_dir or os.path.expanduser("~/jotty/outputs"))
        output_path.mkdir(parents=True, exist_ok=True)

        # Create markdown file
        safe_topic = "".join(c for c in self.intent.topic if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        markdown_path = str(output_path / f"research_{safe_topic}.md")

        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.intent.topic}\n\n")
            f.write(documentation_stage.output)

        if verbose:
            print(f"\n📄 Saved markdown: {markdown_path}")

        # Generate outputs using OutputSinkManager
        try:
            from .output_sinks import OutputSinkManager

            manager = OutputSinkManager(output_dir=str(output_path))
            outputs = manager.generate_all(
                markdown_path=markdown_path,
                formats=[fmt for fmt in formats if fmt != "markdown"],
                title=self.intent.topic,
                author="Jotty Research Workflow",
                n_slides=12,  # For presentations
                tone="professional"
            )

            if verbose:
                summary = manager.get_summary(outputs)
                print(f"\n📦 Generated {summary['successful']}/{summary['total']} output formats:")
                for fmt in summary['successful_formats']:
                    print(f"   ✅ {fmt}: {summary['file_paths'][fmt]}")
                if summary['failed_formats']:
                    print(f"\n❌ Failed formats: {', '.join(summary['failed_formats'])}")

            return {
                'pipeline_result': pipeline_result,
                'outputs': outputs,
                'markdown_path': markdown_path
            }

        except Exception as e:
            if verbose:
                print(f"\n⚠️  Output generation failed: {e}")
            return {
                'pipeline_result': pipeline_result,
                'outputs': {},
                'markdown_path': markdown_path,
                'error': str(e)
            }


# Convenience functions
async def research(
    topic: str,
    **kwargs
) -> PipelineResult:
    """
    Simplest API - research a topic.

    Usage:
        result = await research("AI safety challenges in 2026")
        result = await research("Quantum computing", depth="comprehensive")
    """
    return await ResearchWorkflow.execute(topic, **kwargs)
