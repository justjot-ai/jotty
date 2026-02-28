"""Topic Research Swarm — Main swarm orchestrator.

A generic, topic-agnostic research swarm that codifies the pattern:
    Phase 0: Topic Decomposition
    Phase 1: Parallel Multi-Source Web Research
    Phase 2: Data Synthesis
    Phase 3: Comparative Analysis (if applicable)
    Phase 4: Fact Checking
    Phase 5: Report Compilation

Inspired by the Paytm MYE / PhonePe DRHP research workflow.

Usage:
    # Quick research
    result = await topic_research("Impact of AI on healthcare")

    # Deep comparison
    result = await topic_research(
        "PhonePe vs Paytm",
        mode="comparison",
        entities=["PhonePe", "Paytm"],
        dimensions=["Revenue", "Market Share", "Profitability", "Moat"],
        depth="exhaustive",
    )

    # Landscape mapping
    result = await topic_research(
        "Indian Digital Payments Market 2026",
        mode="landscape",
        depth="deep",
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._base.registry import register_swarm
from .._base.swarm_types import AgentRole
from ..base import SwarmTemplate, TeamCoordinator
from .agents import (
    ComparativeAnalystAgent,
    DataSynthesizerAgent,
    FactCheckerAgent,
    ReportCompilerAgent,
    TopicDecomposerAgent,
    WebResearcherAgent,
)
from .pptx_generator import generate_research_pptx
from .types import (
    ComparisonDimension,
    OutputFormat,
    ResearchDepth,
    ResearchFinding,
    ResearchMode,
    SourceReference,
    TopicResearchConfig,
    TopicResearchResult,
)

logger = logging.getLogger(__name__)


@register_swarm("topic_research")
class TopicResearchSwarm(SwarmTemplate):
    """
    Generic Topic Research Swarm — researches ANY topic through
    multi-source web research, synthesis, comparison, and reporting.

    Agent Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    TopicResearchSwarm                            │
    │                                                                  │
    │  Phase 0: ┌──────────────────┐                                  │
    │           │ TopicDecomposer  │  → sub-topics + search queries    │
    │           └────────┬─────────┘                                  │
    │                    ▼                                             │
    │  Phase 1: ┌──────────────────┐ × N (parallel per sub-topic)     │
    │           │  WebResearcher   │  → raw search results             │
    │           └────────┬─────────┘                                  │
    │                    ▼                                             │
    │  Phase 2: ┌──────────────────┐ × N (parallel per sub-topic)     │
    │           │ DataSynthesizer  │  → structured findings            │
    │           └────────┬─────────┘                                  │
    │                    ▼                                             │
    │  Phase 3: ┌──────────────────┐ (if comparison mode)             │
    │           │ComparativeAnalyst│  → comparison matrix + verdict    │
    │           └────────┬─────────┘                                  │
    │                    ▼                                             │
    │  Phase 4: ┌──────────────────┐                                  │
    │           │   FactChecker    │  → validated claims               │
    │           └────────┬─────────┘                                  │
    │                    ▼                                             │
    │  Phase 5: ┌──────────────────┐                                  │
    │           │ ReportCompiler   │  → Markdown / PPTX report         │
    │           └──────────────────┘                                  │
    └──────────────────────────────────────────────────────────────────┘

    Coordination Patterns:
    - Phase 0: Sequential (need decomposition before search)
    - Phase 1: Parallel (all sub-topics researched concurrently)
    - Phase 2: Parallel (all sub-topics synthesized concurrently)
    - Phase 3: Sequential (needs all synthesis results)
    - Phase 4: Sequential (needs all findings)
    - Phase 5: Sequential (needs everything)
    """

    AGENT_TEAM = TeamCoordinator.define(
        (TopicDecomposerAgent, "TopicDecomposer", "_decomposer"),
        (WebResearcherAgent, "WebResearcher", "_researcher"),
        (DataSynthesizerAgent, "DataSynthesizer", "_synthesizer"),
        (ComparativeAnalystAgent, "ComparativeAnalyst", "_comparator"),
        (ReportCompilerAgent, "ReportCompiler", "_compiler"),
        (FactCheckerAgent, "FactChecker", "_fact_checker"),
    )

    TASK_TYPE = "topic_research"
    DEFAULT_TOOLS = ["web_search", "data_synthesis", "report_generation"]
    RESULT_CLASS = TopicResearchResult

    def __init__(self, config: Optional[TopicResearchConfig] = None) -> None:
        super().__init__(config or TopicResearchConfig())

    async def _execute_domain(self, query: str, **kwargs: Any) -> TopicResearchResult:
        """Main execution: research any topic end-to-end."""
        return await self.research(query, **kwargs)

    async def research(
        self,
        topic: str,
        mode: Optional[str] = None,
        depth: Optional[str] = None,
        entities: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
        output_format: Optional[str] = None,
        **kwargs: Any,
    ) -> TopicResearchResult:
        """Research any topic with configurable depth and mode.

        Args:
            topic: The research topic or question.
            mode: "single_topic", "comparison", "landscape", "deep_dive".
            depth: "quick", "standard", "deep", "exhaustive".
            entities: Entities to compare (for comparison mode).
            dimensions: Comparison dimensions (for comparison mode).
            output_format: "markdown", "pptx", "pdf", "json".

        Returns:
            TopicResearchResult with findings, report, and sources.
        """
        start_time = time.time()
        self._init_agents()

        # Apply overrides
        _mode = mode or self.config.mode.value
        _depth = depth or self.config.depth.value
        _entities = entities or self.config.comparison_entities
        _dimensions = dimensions or self.config.comparison_dimensions
        is_comparison = _mode == "comparison" and len(_entities) >= 2

        executor = self._phase_executor()

        try:
            # ── Phase 0: Topic Decomposition ───────────────────────
            decomposition = await executor.run_phase(
                phase_num=0,
                phase_name="Topic Decomposition",
                agent_name="TopicDecomposer",
                agent_role=AgentRole.ACTOR,
                coro=self._decomposer.decompose(
                    topic=topic,
                    mode=_mode,
                    depth=_depth,
                    entities=_entities,
                    dimensions=_dimensions,
                ),
                tools_used=["topic_decomposition"],
            )

            sub_topics = decomposition.get("sub_topics", [topic])
            search_queries = decomposition.get("search_queries", {topic: [topic]})
            key_questions = decomposition.get("key_questions", [])

            # Limit sub-topics based on depth
            max_st = {"quick": 3, "standard": 5, "deep": 8, "exhaustive": 12}
            sub_topics = sub_topics[: max_st.get(_depth, 8)]

            # ── Phase 1: Parallel Web Research ─────────────────────
            max_results = {
                "quick": 5,
                "standard": 8,
                "deep": 10,
                "exhaustive": 15,
            }

            research_tasks = []
            for st in sub_topics:
                queries = search_queries.get(st, [st])
                research_tasks.append(
                    (
                        f"WebResearcher-{st[:30]}",
                        AgentRole.ACTOR,
                        self._researcher.research_sub_topic(
                            sub_topic=st,
                            queries=queries,
                            max_results=max_results.get(_depth, 10),
                        ),
                        ["web_search"],
                    )
                )

            research_results = await executor.run_parallel(
                phase_num=1,
                phase_name="Multi-Source Web Research",
                tasks=research_tasks,
            )

            # Collect all sources
            all_sources: List[SourceReference] = []
            all_raw_text_by_topic: Dict[str, str] = {}
            for i, result in enumerate(research_results):
                st = sub_topics[i] if i < len(sub_topics) else f"sub_{i}"
                if isinstance(result, dict):
                    all_raw_text_by_topic[st] = result.get("raw_text", "")
                    for src in result.get("sources", []):
                        all_sources.append(
                            SourceReference(
                                url=src.get("url", ""),
                                title=src.get("title", ""),
                            )
                        )

            # ── Phase 2: Parallel Data Synthesis ───────────────────
            synthesis_tasks = []
            for st in sub_topics:
                raw_text = all_raw_text_by_topic.get(st, "")
                if raw_text:
                    synthesis_tasks.append(
                        (
                            f"DataSynthesizer-{st[:30]}",
                            AgentRole.ACTOR,
                            self._synthesizer.synthesize(
                                topic=topic,
                                sub_topic=st,
                                raw_data=raw_text,
                                key_questions=key_questions,
                            ),
                            ["data_synthesis"],
                        )
                    )

            synthesis_results = await executor.run_parallel(
                phase_num=2,
                phase_name="Data Synthesis",
                tasks=synthesis_tasks,
            )

            # Collect all findings and key numbers
            all_findings: List[ResearchFinding] = []
            all_key_numbers: Dict[str, str] = {}
            all_data_gaps: List[str] = []
            findings_text_parts: List[str] = []

            for result in synthesis_results:
                if isinstance(result, dict):
                    for f in result.get("findings", []):
                        all_findings.append(
                            ResearchFinding(
                                claim=f.get("claim", ""),
                                evidence=f.get("evidence", ""),
                                confidence=f.get("confidence", 0.5),
                                sub_topic=f.get("sub_topic", ""),
                            )
                        )
                        findings_text_parts.append(
                            f"CLAIM: {f.get('claim', '')} | "
                            f"EVIDENCE: {f.get('evidence', '')} | "
                            f"CONFIDENCE: {f.get('confidence', 0.5)}"
                        )
                    all_key_numbers.update(result.get("key_numbers", {}))
                    all_data_gaps.extend(result.get("data_gaps", []))

            findings_text = "\n".join(findings_text_parts)

            # ── Phase 3: Comparative Analysis (if applicable) ──────
            comparison_data: Dict[str, Any] = {}
            comparison_text = ""
            if is_comparison:
                comparison_data = await executor.run_phase(
                    phase_num=3,
                    phase_name="Comparative Analysis",
                    agent_name="ComparativeAnalyst",
                    agent_role=AgentRole.ACTOR,
                    coro=self._comparator.compare(
                        entities=_entities,
                        dimensions=_dimensions
                        or [
                            "Revenue",
                            "Market Share",
                            "Profitability",
                            "Growth",
                            "Moat",
                        ],
                        research_data=findings_text,
                    ),
                    tools_used=["comparative_analysis"],
                )
                # Format comparison for report
                comp_parts = []
                for dim in comparison_data.get("comparison_matrix", []):
                    scores = " | ".join(
                        f"{k}: {v}" for k, v in dim.get("entity_scores", {}).items()
                    )
                    comp_parts.append(
                        f"DIMENSION: {dim.get('dimension', '')} | "
                        f"{scores} | WINNER: {dim.get('winner', '')} | "
                        f"ANALYSIS: {dim.get('analysis', '')}"
                    )
                comparison_text = "\n".join(comp_parts)
                if comparison_data.get("verdict"):
                    comparison_text += f"\n\nVERDICT: {comparison_data['verdict']}"

            # ── Phase 4: Fact Checking ─────────────────────────────
            fact_check_result: Dict[str, Any] = {"overall_reliability": 0.5}
            if self.config.enable_fact_checking and all_findings:
                claims = [f.claim for f in all_findings if f.claim][:20]
                evidence_text = "\n".join(f"[{s.title}] {s.url}" for s in all_sources[:30])
                evidence_text += "\n\n" + findings_text[:8000]

                fact_check_result = await executor.run_phase(
                    phase_num=4,
                    phase_name="Fact Checking",
                    agent_name="FactChecker",
                    agent_role=AgentRole.ACTOR,
                    coro=self._fact_checker.check(
                        claims=claims,
                        evidence=evidence_text,
                    ),
                    tools_used=["fact_checking"],
                )

            # ── Phase 5: Report Compilation ────────────────────────
            sources_text = "\n".join(f"- [{s.title}]({s.url})" for s in all_sources[:50] if s.url)
            key_numbers_text = "\n".join(f"- {k}: {v}" for k, v in all_key_numbers.items())

            report_data = await executor.run_phase(
                phase_num=5,
                phase_name="Report Compilation",
                agent_name="ReportCompiler",
                agent_role=AgentRole.ACTOR,
                coro=self._compiler.compile_report(
                    topic=topic,
                    findings=findings_text,
                    comparison=comparison_text,
                    key_numbers=key_numbers_text,
                    sources=sources_text,
                ),
                tools_used=["report_compilation"],
            )

            report_markdown = report_data.get("report", "")

            # ── Save report to disk ────────────────────────────────
            md_path = ""
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic[:60]).strip()
            md_path = str(output_dir / f"{safe_name}.md")
            try:
                with open(md_path, "w") as f:
                    f.write(report_markdown)
                logger.info(f"Report saved to {md_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
                md_path = ""

            # ── Phase 6: PPTX Generation (if requested) ────────────
            pptx_path = ""
            if self.config.generate_pptx or self.config.output_format == OutputFormat.PPTX:
                try:
                    # Build a partial result for the generator
                    _partial = TopicResearchResult(
                        success=True,
                        swarm_name="TopicResearchSwarm",
                        domain="topic_research",
                        output={
                            "report": report_markdown,
                            "executive_summary": report_data.get("executive_summary", ""),
                            "one_liner": report_data.get("one_liner", ""),
                        },
                        execution_time=time.time() - start_time,
                        topic=topic,
                        sub_topics=sub_topics,
                        mode=_mode,
                        depth=_depth,
                        summary=report_data.get("executive_summary", ""),
                        key_findings=[f.claim for f in all_findings if f.confidence >= 0.6],
                        detailed_findings=all_findings,
                        comparison_entities=(_entities if is_comparison else []),
                        comparison_matrix=[
                            ComparisonDimension(
                                dimension=d.get("dimension", ""),
                                entity_scores=d.get("entity_scores", {}),
                                winner=d.get("winner", ""),
                                analysis=d.get("analysis", ""),
                            )
                            for d in comparison_data.get("comparison_matrix", [])
                        ],
                        comparison_verdict=comparison_data.get("verdict", ""),
                        sources=all_sources,
                        total_sources_consulted=len(all_sources),
                        unique_domains=len(
                            set(s.url.split("/")[2] for s in all_sources if s.url and "/" in s.url)
                        ),
                        report_markdown=report_markdown,
                        fact_check_score=fact_check_result.get("overall_reliability", 0.5),
                        completeness_score=min(
                            1.0,
                            len(all_findings) / max(len(sub_topics) * 3, 1),
                        ),
                        agent_contributions={
                            "decomposer": 0.1,
                            "researcher": 0.3,
                            "synthesizer": 0.25,
                            "comparator": 0.15 if is_comparison else 0.0,
                            "fact_checker": 0.05,
                            "compiler": 0.15,
                        },
                    )
                    pptx_path = generate_research_pptx(
                        _partial,
                        output_dir=self.config.output_dir,
                    )
                    logger.info(f"PPTX generated: {pptx_path}")
                except Exception as e:
                    logger.error(f"PPTX generation failed: {e}")

            # ── Build result ───────────────────────────────────────
            execution_time = time.time() - start_time

            # Build comparison dimensions for result
            comp_dims = []
            for dim in comparison_data.get("comparison_matrix", []):
                comp_dims.append(
                    ComparisonDimension(
                        dimension=dim.get("dimension", ""),
                        entity_scores=dim.get("entity_scores", {}),
                        winner=dim.get("winner", ""),
                        analysis=dim.get("analysis", ""),
                    )
                )

            unique_domains = len(
                set(s.url.split("/")[2] for s in all_sources if s.url and "/" in s.url)
            )

            return TopicResearchResult(
                success=True,
                swarm_name="TopicResearchSwarm",
                domain="topic_research",
                output={
                    "report": report_markdown,
                    "executive_summary": report_data.get("executive_summary", ""),
                    "one_liner": report_data.get("one_liner", ""),
                },
                execution_time=execution_time,
                # Research identity
                topic=topic,
                sub_topics=sub_topics,
                mode=_mode,
                depth=_depth,
                # Findings
                summary=report_data.get("executive_summary", ""),
                key_findings=[f.claim for f in all_findings if f.confidence >= 0.6],
                detailed_findings=all_findings,
                # Comparison
                comparison_entities=_entities if is_comparison else [],
                comparison_matrix=comp_dims,
                comparison_verdict=comparison_data.get("verdict", ""),
                # Sources
                sources=all_sources,
                total_sources_consulted=len(all_sources),
                unique_domains=unique_domains,
                # Outputs
                report_markdown=report_markdown,
                md_path=md_path,
                pptx_path=pptx_path,
                # Quality
                fact_check_score=fact_check_result.get("overall_reliability", 0.5),
                completeness_score=min(1.0, len(all_findings) / max(len(sub_topics) * 3, 1)),
                # Agent contributions
                agent_contributions={
                    "decomposer": 0.1,
                    "researcher": 0.3,
                    "synthesizer": 0.25,
                    "comparator": 0.15 if is_comparison else 0.0,
                    "fact_checker": 0.05,
                    "compiler": 0.15,
                },
            )

        except Exception as e:
            logger.error(f"TopicResearchSwarm failed: {e}", exc_info=True)
            return TopicResearchResult(
                success=False,
                swarm_name="TopicResearchSwarm",
                domain="topic_research",
                output={},
                execution_time=time.time() - start_time,
                error=str(e),
                topic=topic,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


async def topic_research(
    topic: str,
    mode: str = "single_topic",
    depth: str = "deep",
    entities: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    **kwargs: Any,
) -> TopicResearchResult:
    """Research any topic with a single function call.

    Examples:
        # Simple research
        result = await topic_research("Impact of AI on healthcare")

        # Comparison
        result = await topic_research(
            "PhonePe vs Paytm",
            mode="comparison",
            entities=["PhonePe", "Paytm"],
            dimensions=["Revenue", "Market Share", "Profitability"],
        )

        # Landscape mapping
        result = await topic_research(
            "Indian SaaS Market 2026",
            mode="landscape",
            depth="exhaustive",
        )
    """
    config = TopicResearchConfig(
        depth=ResearchDepth(depth),
        mode=ResearchMode(mode),
        comparison_entities=entities or [],
        comparison_dimensions=dimensions or [],
    )
    swarm = TopicResearchSwarm(config=config)
    return await swarm.execute(topic, **kwargs)


def topic_research_sync(
    topic: str,
    mode: str = "single_topic",
    depth: str = "deep",
    entities: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    **kwargs: Any,
) -> TopicResearchResult:
    """Synchronous wrapper for topic_research."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                topic_research(topic, mode, depth, entities, dimensions, **kwargs),
            )
            return future.result()
    else:
        return asyncio.run(topic_research(topic, mode, depth, entities, dimensions, **kwargs))
