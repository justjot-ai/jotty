"""Topic Research Swarm — Agent implementations.

Five specialized agents that work together to research any topic:
1. TopicDecomposerAgent — breaks topic into searchable sub-topics
2. WebResearcherAgent — performs parallel web searches
3. DataSynthesizerAgent — synthesizes raw data into findings
4. ComparativeAnalystAgent — structured comparison (when applicable)
5. ReportCompilerAgent — compiles final report
6. FactCheckerAgent — validates claims against evidence
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import dspy

from Jotty.core.intelligence.reasoning.agents.swarm_agent import SwarmLearningAgent

from .parsers import (
    parse_comparison_matrix,
    parse_findings,
    parse_key_numbers,
    parse_verified_claims,
)
from .signatures import (
    ComparativeAnalysisSignature,
    DataSynthesisSignature,
    FactCheckSignature,
    ReportCompilationSignature,
    TopicDecompositionSignature,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1: TOPIC DECOMPOSER
# ═══════════════════════════════════════════════════════════════════════════════


class TopicDecomposerAgent(SwarmLearningAgent):
    """Breaks a research topic into focused, searchable sub-topics.

    Uses LLM to understand the topic and generate:
    - 4-8 sub-topics for parallel investigation
    - 2-3 search queries per sub-topic
    - Key questions the research should answer
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__(memory=memory, context=context, bus=bus)
        self._decomposer = dspy.ChainOfThought(TopicDecompositionSignature)

    async def decompose(
        self,
        topic: str,
        mode: str = "single_topic",
        depth: str = "deep",
        entities: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Decompose topic into sub-topics and search queries."""
        try:
            context_parts = [f"Mode: {mode}", f"Depth: {depth}"]
            if entities:
                context_parts.append(f"Entities to compare: {', '.join(entities)}")
            if dimensions:
                context_parts.append(f"Comparison dimensions: {', '.join(dimensions)}")
            context_str = ". ".join(context_parts)

            result = self._decomposer(topic=topic, context=context_str)

            sub_topics = [s.strip() for s in result.sub_topics.split("|") if s.strip()]
            queries_raw = result.search_queries.split("||")
            search_queries: Dict[str, List[str]] = {}
            for i, query_group in enumerate(queries_raw):
                st = sub_topics[i] if i < len(sub_topics) else f"sub_topic_{i}"
                search_queries[st] = [q.strip() for q in query_group.split("|") if q.strip()]
            key_questions = [q.strip() for q in result.key_questions.split("|") if q.strip()]

            self._broadcast(
                "topic_decomposed",
                {
                    "topic": topic,
                    "sub_topics": len(sub_topics),
                    "queries": sum(len(v) for v in search_queries.values()),
                },
            )

            return {
                "sub_topics": sub_topics,
                "search_queries": search_queries,
                "key_questions": key_questions,
            }
        except Exception as e:
            logger.error(f"TopicDecomposerAgent error: {e}")
            # Fallback: use topic itself as the only sub-topic
            return {
                "sub_topics": [topic],
                "search_queries": {topic: [topic]},
                "key_questions": [f"What is {topic}?"],
            }

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.decompose(
            topic=kwargs.get("topic", ""),
            mode=kwargs.get("mode", "single_topic"),
            depth=kwargs.get("depth", "deep"),
            entities=kwargs.get("entities"),
            dimensions=kwargs.get("dimensions"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2: WEB RESEARCHER
# ═══════════════════════════════════════════════════════════════════════════════


class WebResearcherAgent(SwarmLearningAgent):
    """Performs web searches and extracts content.

    Uses the web-search skill from the skills registry to perform
    searches. Falls back to a basic search if skill is unavailable.
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__(memory=memory, context=context, bus=bus)
        self._search_tool = None
        self._initialized = False

    def _init_search(self) -> None:
        """Lazy-init the web search tool."""
        if self._initialized:
            return
        self._initialized = True
        try:
            from Jotty.core.capabilities.registry.skills_registry import (
                get_skills_registry,
            )

            registry = get_skills_registry()
            registry.init()
            web_skill = registry.get_skill("web-search")
            if web_skill:
                self._search_tool = web_skill.tools.get("search_web_tool")
        except Exception as e:
            logger.warning(f"Could not init web search tool: {e}")

    async def search(
        self,
        queries: List[str],
        max_results_per_query: int = 10,
    ) -> Dict[str, Any]:
        """Execute web searches for multiple queries."""
        self._init_search()

        all_results: List[Dict[str, Any]] = []
        sources: List[Dict[str, str]] = []
        raw_texts: List[str] = []

        for query in queries:
            try:
                if self._search_tool:
                    result = await asyncio.to_thread(
                        self._search_tool,
                        {"query": query, "max_results": max_results_per_query},
                    )
                    if isinstance(result, dict):
                        items = result.get("results", [])
                    elif isinstance(result, list):
                        items = result
                    elif isinstance(result, str):
                        items = [{"title": query, "snippet": result, "url": ""}]
                    else:
                        items = []
                else:
                    # Fallback: no search tool available
                    items = []
                    logger.warning(f"No web search tool available for query: {query}")

                for item in items[:max_results_per_query]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", item.get("description", ""))
                    url = item.get("url", item.get("link", ""))

                    all_results.append(
                        {
                            "query": query,
                            "title": title,
                            "snippet": snippet,
                            "url": url,
                        }
                    )
                    sources.append({"title": title, "url": url})
                    if snippet:
                        raw_texts.append(f"[{title}] {snippet}")

            except Exception as e:
                logger.error(f"Search error for '{query}': {e}")
                continue

        self._broadcast(
            "web_research_complete",
            {
                "queries": len(queries),
                "results": len(all_results),
                "sources": len(sources),
            },
        )

        return {
            "results": all_results,
            "sources": sources,
            "raw_text": "\n\n".join(raw_texts),
            "total_results": len(all_results),
        }

    async def research_sub_topic(
        self,
        sub_topic: str,
        queries: List[str],
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Research a single sub-topic with multiple queries."""
        result = await self.search(queries, max_results_per_query=max_results)
        result["sub_topic"] = sub_topic
        return result

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        queries = kwargs.get("queries", [])
        if not queries:
            queries = [kwargs.get("query", "")]
        return await self.search(queries)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3: DATA SYNTHESIZER
# ═══════════════════════════════════════════════════════════════════════════════


class DataSynthesizerAgent(SwarmLearningAgent):
    """Synthesizes raw web data into structured findings.

    Takes raw search results and uses LLM to extract:
    - Key findings with confidence scores
    - Important numbers/data points
    - Data gaps and contradictions
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__(memory=memory, context=context, bus=bus)
        self._synthesizer = dspy.ChainOfThought(DataSynthesisSignature)

    async def synthesize(
        self,
        topic: str,
        sub_topic: str,
        raw_data: str,
        key_questions: List[str],
    ) -> Dict[str, Any]:
        """Synthesize raw data into structured findings."""
        try:
            # Truncate raw data to fit context
            max_chars = 12000
            if len(raw_data) > max_chars:
                raw_data = raw_data[:max_chars] + "\n[... truncated]"

            result = self._synthesizer(
                topic=topic,
                sub_topic=sub_topic,
                raw_data=raw_data,
                key_questions=" | ".join(key_questions[:10]),
            )

            # Parse findings (robust — handles pipe, JSON, numbered lists)
            findings = parse_findings(result.findings, sub_topic)

            # Parse key numbers (robust)
            key_numbers = parse_key_numbers(result.key_numbers)

            # Parse data gaps
            data_gaps_raw = result.data_gaps
            if isinstance(data_gaps_raw, list):
                data_gaps = [str(g).strip() for g in data_gaps_raw if g]
            elif isinstance(data_gaps_raw, str):
                data_gaps = [g.strip() for g in data_gaps_raw.split("|") if g.strip()]
            else:
                data_gaps = []

            self._broadcast(
                "data_synthesized",
                {
                    "sub_topic": sub_topic,
                    "findings": len(findings),
                    "key_numbers": len(key_numbers),
                },
            )

            return {
                "sub_topic": sub_topic,
                "findings": findings,
                "key_numbers": key_numbers,
                "data_gaps": data_gaps,
            }
        except Exception as e:
            logger.error(f"DataSynthesizerAgent error for '{sub_topic}': {e}")
            return {
                "sub_topic": sub_topic,
                "findings": [],
                "key_numbers": {},
                "data_gaps": [f"Synthesis failed: {e}"],
            }

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.synthesize(
            topic=kwargs.get("topic", ""),
            sub_topic=kwargs.get("sub_topic", ""),
            raw_data=kwargs.get("raw_data", ""),
            key_questions=kwargs.get("key_questions", []),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4: COMPARATIVE ANALYST
# ═══════════════════════════════════════════════════════════════════════════════


class ComparativeAnalystAgent(SwarmLearningAgent):
    """Performs structured head-to-head comparison between entities.

    Given research data about multiple entities, produces a structured
    comparison matrix with scoring, winner determination, and verdict.
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__(memory=memory, context=context, bus=bus)
        self._comparator = dspy.ChainOfThought(ComparativeAnalysisSignature)

    async def compare(
        self,
        entities: List[str],
        dimensions: List[str],
        research_data: str,
    ) -> Dict[str, Any]:
        """Compare entities across dimensions."""
        try:
            # Truncate research data
            max_chars = 15000
            if len(research_data) > max_chars:
                research_data = research_data[:max_chars] + "\n[... truncated]"

            result = self._comparator(
                entities=" | ".join(entities),
                dimensions=" | ".join(dimensions),
                research_data=research_data,
            )

            # Parse comparison matrix (robust)
            matrix = parse_comparison_matrix(result.comparison, entities)

            # Parse key asymmetries
            ka_raw = result.key_asymmetries
            if isinstance(ka_raw, list):
                key_asymmetries = [str(a).strip() for a in ka_raw if a]
            elif isinstance(ka_raw, str):
                key_asymmetries = [a.strip() for a in ka_raw.split("|") if a.strip()]
            else:
                key_asymmetries = []

            self._broadcast(
                "comparison_complete",
                {
                    "entities": entities,
                    "dimensions": len(matrix),
                },
            )

            return {
                "comparison_matrix": matrix,
                "verdict": result.verdict,
                "key_asymmetries": key_asymmetries,
            }
        except Exception as e:
            logger.error(f"ComparativeAnalystAgent error: {e}")
            return {
                "comparison_matrix": [],
                "verdict": f"Comparison failed: {e}",
                "key_asymmetries": [],
            }

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.compare(
            entities=kwargs.get("entities", []),
            dimensions=kwargs.get("dimensions", []),
            research_data=kwargs.get("research_data", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 5: REPORT COMPILER
# ═══════════════════════════════════════════════════════════════════════════════


class ReportCompilerAgent(SwarmLearningAgent):
    """Compiles all findings into a structured, professional report.

    Produces executive-quality markdown report with:
    - Executive summary
    - Key findings per sub-topic
    - Data tables
    - Comparison matrix (if applicable)
    - Sources
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__(memory=memory, context=context, bus=bus)
        self._compiler = dspy.ChainOfThought(ReportCompilationSignature)

    async def compile_report(
        self,
        topic: str,
        findings: str,
        comparison: str,
        key_numbers: str,
        sources: str,
    ) -> Dict[str, Any]:
        """Compile findings into a structured report."""
        try:
            # Truncate inputs to fit context
            max_per_field = 8000
            if len(findings) > max_per_field:
                findings = findings[:max_per_field] + "\n[... truncated]"
            if len(comparison) > max_per_field:
                comparison = comparison[:max_per_field] + "\n[... truncated]"

            result = self._compiler(
                topic=topic,
                findings=findings,
                comparison=comparison or "N/A — single topic research",
                key_numbers=key_numbers,
                sources=sources,
            )

            self._broadcast(
                "report_compiled",
                {
                    "topic": topic,
                    "report_length": len(result.report),
                },
            )

            return {
                "report": result.report,
                "executive_summary": result.executive_summary,
                "one_liner": result.one_liner,
            }
        except Exception as e:
            logger.error(f"ReportCompilerAgent error: {e}")
            return {
                "report": f"# {topic}\n\nReport compilation failed: {e}",
                "executive_summary": f"Error: {e}",
                "one_liner": "",
            }

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.compile_report(
            topic=kwargs.get("topic", ""),
            findings=kwargs.get("findings", ""),
            comparison=kwargs.get("comparison", ""),
            key_numbers=kwargs.get("key_numbers", ""),
            sources=kwargs.get("sources", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 6: FACT CHECKER
# ═══════════════════════════════════════════════════════════════════════════════


class FactCheckerAgent(SwarmLearningAgent):
    """Validates research claims against evidence.

    Reviews findings for:
    - Unsupported claims (no evidence)
    - Contradictions between sources
    - Outdated data
    - Logical inconsistencies
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
    ) -> None:
        super().__init__(memory=memory, context=context, bus=bus)
        self._checker = dspy.ChainOfThought(FactCheckSignature)

    async def check(
        self,
        claims: List[str],
        evidence: str,
    ) -> Dict[str, Any]:
        """Fact-check a list of claims against evidence."""
        try:
            claims_str = " || ".join(claims[:20])  # Limit to 20 claims

            max_chars = 10000
            if len(evidence) > max_chars:
                evidence = evidence[:max_chars] + "\n[... truncated]"

            result = self._checker(
                claims=claims_str,
                evidence=evidence,
            )

            # Parse verified claims (robust)
            verified_raw = result.verified_claims
            verified, _, _ = parse_verified_claims(verified_raw)

            # Parse reliability
            try:
                reliability = float(result.overall_reliability)
                if reliability > 1.0:
                    reliability = reliability / 100.0
            except (ValueError, TypeError):
                reliability = 0.5

            # Parse issues
            issues_raw = result.issues
            if isinstance(issues_raw, list):
                issues = [str(i).strip() for i in issues_raw if i]
            elif isinstance(issues_raw, str):
                issues = [i.strip() for i in issues_raw.split("|") if i.strip()]
            else:
                issues = []

            self._broadcast(
                "fact_check_complete",
                {
                    "claims_checked": len(verified),
                    "reliability": reliability,
                    "issues": len(issues),
                },
            )

            return {
                "verified_claims": verified,
                "overall_reliability": reliability,
                "issues": issues,
            }
        except Exception as e:
            logger.error(f"FactCheckerAgent error: {e}")
            return {
                "verified_claims": [],
                "overall_reliability": 0.5,
                "issues": [f"Fact-checking failed: {e}"],
            }

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.check(
            claims=kwargs.get("claims", []),
            evidence=kwargs.get("evidence", ""),
        )
