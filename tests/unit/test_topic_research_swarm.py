"""Tests for TopicResearchSwarm — generic topic-agnostic research swarm.

All tests use mocks — NEVER call real LLM providers or web services.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestImports:
    """Verify all public exports are importable."""

    def test_swarm_import(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchSwarm,
        )

        assert TopicResearchSwarm is not None

    def test_config_import(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchConfig,
        )

        assert TopicResearchConfig is not None

    def test_result_import(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchResult,
        )

        assert TopicResearchResult is not None

    def test_agent_imports(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ComparativeAnalystAgent,
            DataSynthesizerAgent,
            FactCheckerAgent,
            ReportCompilerAgent,
            TopicDecomposerAgent,
            WebResearcherAgent,
        )

        assert TopicDecomposerAgent is not None
        assert WebResearcherAgent is not None
        assert DataSynthesizerAgent is not None
        assert ComparativeAnalystAgent is not None
        assert ReportCompilerAgent is not None
        assert FactCheckerAgent is not None

    def test_enum_imports(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            OutputFormat,
            ResearchDepth,
            ResearchMode,
        )

        assert ResearchDepth.DEEP.value == "deep"
        assert ResearchMode.COMPARISON.value == "comparison"
        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_data_class_imports(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ComparisonDimension,
            ResearchFinding,
            SourceReference,
        )

        src = SourceReference(url="https://example.com", title="Test")
        assert src.url == "https://example.com"

        finding = ResearchFinding(claim="Test claim", confidence=0.9)
        assert finding.confidence == 0.9

        dim = ComparisonDimension(dimension="Revenue", winner="A")
        assert dim.winner == "A"

    def test_convenience_function_import(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            topic_research,
            topic_research_sync,
        )

        assert callable(topic_research)
        assert callable(topic_research_sync)

    def test_lazy_import_from_swarms_package(self):
        """Test that lazy imports work from the parent swarms package."""
        from Jotty.core.intelligence.orchestration.swarms import (
            TopicResearchSwarm,
        )

        assert TopicResearchSwarm is not None

    def test_lazy_import_config(self):
        from Jotty.core.intelligence.orchestration.swarms import (
            TopicResearchConfig,
        )

        assert TopicResearchConfig is not None

    def test_lazy_import_convenience(self):
        from Jotty.core.intelligence.orchestration.swarms import (
            topic_research,
        )

        assert callable(topic_research)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicResearchConfig:
    """Test configuration defaults and post_init."""

    def test_default_config(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ResearchDepth,
            ResearchMode,
            TopicResearchConfig,
        )

        config = TopicResearchConfig()
        assert config.name == "TopicResearchSwarm"
        assert config.domain == "topic_research"
        assert config.depth == ResearchDepth.DEEP
        assert config.mode == ResearchMode.SINGLE_TOPIC
        assert config.max_sub_topics == 8
        assert config.enable_fact_checking is True

    def test_comparison_config(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ResearchDepth,
            ResearchMode,
            TopicResearchConfig,
        )

        config = TopicResearchConfig(
            depth=ResearchDepth.EXHAUSTIVE,
            mode=ResearchMode.COMPARISON,
            comparison_entities=["PhonePe", "Paytm"],
            comparison_dimensions=["Revenue", "Market Share"],
        )
        assert config.mode == ResearchMode.COMPARISON
        assert len(config.comparison_entities) == 2
        assert len(config.comparison_dimensions) == 2

    def test_output_dir_set(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchConfig,
        )

        config = TopicResearchConfig()
        assert "jotty/research" in config.output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicResearchResult:
    """Test result dataclass."""

    def test_default_result(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchResult,
        )

        result = TopicResearchResult(
            success=True,
            swarm_name="TopicResearchSwarm",
            domain="topic_research",
            output={},
            execution_time=1.0,
            topic="Test Topic",
        )
        assert result.success is True
        assert result.topic == "Test Topic"
        assert result.key_findings == []
        assert result.sources == []

    def test_result_with_findings(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ResearchFinding,
            TopicResearchResult,
        )

        findings = [
            ResearchFinding(claim="Claim 1", confidence=0.9),
            ResearchFinding(claim="Claim 2", confidence=0.7),
        ]
        result = TopicResearchResult(
            success=True,
            swarm_name="TopicResearchSwarm",
            domain="topic_research",
            output={},
            execution_time=1.0,
            topic="Test",
            detailed_findings=findings,
        )
        assert len(result.detailed_findings) == 2
        assert result.detailed_findings[0].confidence == 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TESTS (with mocked DSPy)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicDecomposerAgent:
    """Test topic decomposition agent."""

    @pytest.mark.asyncio
    async def test_decompose_basic(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicDecomposerAgent,
        )

        agent = TopicDecomposerAgent()

        # Mock the DSPy module
        mock_result = MagicMock()
        mock_result.sub_topics = "Sub-topic A | Sub-topic B | Sub-topic C"
        mock_result.search_queries = "query A1 | query A2 || query B1 | query B2 || query C1"
        mock_result.key_questions = "Q1? | Q2? | Q3?"
        agent._decomposer = MagicMock(return_value=mock_result)

        result = await agent.decompose("Test Topic", mode="single_topic")

        assert len(result["sub_topics"]) == 3
        assert "Sub-topic A" in result["sub_topics"]
        assert len(result["key_questions"]) == 3
        assert "Sub-topic A" in result["search_queries"]

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_error(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicDecomposerAgent,
        )

        agent = TopicDecomposerAgent()
        agent._decomposer = MagicMock(side_effect=RuntimeError("LLM error"))

        result = await agent.decompose("Fallback Topic")

        # Should return the topic itself as fallback
        assert result["sub_topics"] == ["Fallback Topic"]
        assert "Fallback Topic" in result["search_queries"]


class TestWebResearcherAgent:
    """Test web research agent."""

    @pytest.mark.asyncio
    async def test_search_with_mock_tool(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            WebResearcherAgent,
        )

        agent = WebResearcherAgent()
        agent._initialized = True

        # Mock the search tool
        agent._search_tool = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Test Result",
                        "snippet": "Some info about the topic",
                        "url": "https://example.com",
                    }
                ]
            }
        )

        result = await agent.search(["test query"], max_results_per_query=5)

        assert result["total_results"] == 1
        assert len(result["sources"]) == 1
        assert "Test Result" in result["raw_text"]

    @pytest.mark.asyncio
    async def test_search_without_tool(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            WebResearcherAgent,
        )

        agent = WebResearcherAgent()
        agent._initialized = True
        agent._search_tool = None

        result = await agent.search(["query without tool"])

        assert result["total_results"] == 0


class TestDataSynthesizerAgent:
    """Test data synthesis agent."""

    @pytest.mark.asyncio
    async def test_synthesize_basic(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            DataSynthesizerAgent,
        )

        agent = DataSynthesizerAgent()

        mock_result = MagicMock()
        mock_result.findings = (
            "CLAIM: Revenue grew 40% | EVIDENCE: FY25 report | CONFIDENCE: 0.9 || "
            "CLAIM: Market share is 48% | EVIDENCE: NPCI data | CONFIDENCE: 0.85"
        )
        mock_result.data_gaps = "No profitability data available"
        mock_result.key_numbers = "Revenue: ₹7,115 Cr | Growth: 40% YoY"
        agent._synthesizer = MagicMock(return_value=mock_result)

        result = await agent.synthesize(
            topic="PhonePe Analysis",
            sub_topic="Revenue",
            raw_data="Some raw data...",
            key_questions=["What is the revenue?"],
        )

        assert len(result["findings"]) == 2
        assert result["findings"][0]["claim"] == "Revenue grew 40%"
        assert result["key_numbers"]["Revenue"] == "₹7,115 Cr"
        assert len(result["data_gaps"]) == 1


class TestComparativeAnalystAgent:
    """Test comparative analysis agent."""

    @pytest.mark.asyncio
    async def test_compare_basic(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ComparativeAnalystAgent,
        )

        agent = ComparativeAnalystAgent()

        mock_result = MagicMock()
        mock_result.comparison = (
            "DIMENSION: Revenue | PHONEPE: ₹7,115 Cr | PAYTM: ₹6,900 Cr | "
            "WINNER: PhonePe | ANALYSIS: Slightly higher"
        )
        mock_result.verdict = "PhonePe leads on scale, Paytm on monetization"
        mock_result.key_asymmetries = "UPI share gap | Device count gap"
        agent._comparator = MagicMock(return_value=mock_result)

        result = await agent.compare(
            entities=["PhonePe", "Paytm"],
            dimensions=["Revenue"],
            research_data="Some research data...",
        )

        assert len(result["comparison_matrix"]) == 1
        assert result["comparison_matrix"][0]["dimension"] == "Revenue"
        assert "PhonePe" in result["verdict"]


class TestFactCheckerAgent:
    """Test fact checking agent."""

    @pytest.mark.asyncio
    async def test_check_basic(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            FactCheckerAgent,
        )

        agent = FactCheckerAgent()

        mock_result = MagicMock()
        mock_result.verified_claims = (
            "CLAIM: Revenue is ₹7,115 Cr | STATUS: verified | "
            "CONFIDENCE: 0.95 | NOTE: Confirmed in annual report"
        )
        mock_result.overall_reliability = 0.9
        mock_result.issues = "No major issues"
        agent._checker = MagicMock(return_value=mock_result)

        result = await agent.check(
            claims=["Revenue is ₹7,115 Cr"],
            evidence="Annual report data...",
        )

        assert result["overall_reliability"] == 0.9
        assert len(result["verified_claims"]) == 1
        assert result["verified_claims"][0]["status"] == "verified"


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM INTEGRATION TESTS (with mocked agents)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicResearchSwarm:
    """Test the swarm orchestrator with mocked agents."""

    def _make_swarm(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchConfig,
            TopicResearchSwarm,
        )

        config = TopicResearchConfig(enable_fact_checking=False)
        swarm = TopicResearchSwarm(config=config)
        return swarm

    def _mock_agents(self, swarm):
        """Replace all agents with mocks."""
        # Decomposer
        swarm._decomposer = AsyncMock()
        swarm._decomposer.decompose = AsyncMock(
            return_value={
                "sub_topics": ["Market Size", "Key Players"],
                "search_queries": {
                    "Market Size": ["market size query"],
                    "Key Players": ["key players query"],
                },
                "key_questions": ["How big is the market?"],
            }
        )

        # Researcher
        swarm._researcher = AsyncMock()
        swarm._researcher.research_sub_topic = AsyncMock(
            return_value={
                "results": [{"title": "Source 1", "snippet": "Data 1", "url": "https://a.com"}],
                "sources": [{"title": "Source 1", "url": "https://a.com"}],
                "raw_text": "Source 1: Data about the market",
                "total_results": 1,
                "sub_topic": "Market Size",
            }
        )

        # Synthesizer
        swarm._synthesizer = AsyncMock()
        swarm._synthesizer.synthesize = AsyncMock(
            return_value={
                "sub_topic": "Market Size",
                "findings": [
                    {
                        "claim": "Market is growing at 25%",
                        "evidence": "Industry report",
                        "confidence": 0.8,
                        "sub_topic": "Market Size",
                    }
                ],
                "key_numbers": {"Growth Rate": "25%"},
                "data_gaps": [],
            }
        )

        # Comparator (not used in single_topic mode)
        swarm._comparator = AsyncMock()

        # Fact checker (disabled in config)
        swarm._fact_checker = AsyncMock()

        # Compiler
        swarm._compiler = AsyncMock()
        swarm._compiler.compile_report = AsyncMock(
            return_value={
                "report": "# Research Report\n\nMarket is growing at 25%",
                "executive_summary": "Market growing at 25%",
                "one_liner": "Strong growth market",
            }
        )

    @pytest.mark.asyncio
    async def test_single_topic_research(self, tmp_path):
        swarm = self._make_swarm()
        swarm.config.output_dir = str(tmp_path)
        self._mock_agents(swarm)

        # Skip _init_agents since we already have mocks
        swarm._init_agents = MagicMock()

        result = await swarm.research("Indian Digital Payments Market")

        assert result.success is True
        assert result.topic == "Indian Digital Payments Market"
        assert len(result.sub_topics) == 2
        assert len(result.key_findings) >= 1
        assert "25%" in result.report_markdown
        assert result.mode == "single_topic"

    @pytest.mark.asyncio
    async def test_comparison_research(self, tmp_path):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            ResearchMode,
            TopicResearchConfig,
            TopicResearchSwarm,
        )

        config = TopicResearchConfig(
            mode=ResearchMode.COMPARISON,
            comparison_entities=["PhonePe", "Paytm"],
            comparison_dimensions=["Revenue", "Market Share"],
            enable_fact_checking=False,
            output_dir=str(tmp_path),
        )
        swarm = TopicResearchSwarm(config=config)
        self._mock_agents(swarm)
        swarm._init_agents = MagicMock()

        # Add comparator mock
        swarm._comparator.compare = AsyncMock(
            return_value={
                "comparison_matrix": [
                    {
                        "dimension": "Revenue",
                        "entity_scores": {
                            "PhonePe": "₹7,115 Cr",
                            "Paytm": "₹6,900 Cr",
                        },
                        "winner": "PhonePe",
                        "analysis": "PhonePe slightly ahead",
                    }
                ],
                "verdict": "PhonePe leads on consumer scale",
                "key_asymmetries": ["UPI share gap"],
            }
        )

        result = await swarm.research(
            "PhonePe vs Paytm",
            mode="comparison",
            entities=["PhonePe", "Paytm"],
            dimensions=["Revenue", "Market Share"],
        )

        assert result.success is True
        assert result.mode == "comparison"
        assert len(result.comparison_entities) == 2
        assert len(result.comparison_matrix) == 1
        assert result.comparison_verdict != ""

    @pytest.mark.asyncio
    async def test_error_handling(self, tmp_path):
        swarm = self._make_swarm()
        swarm.config.output_dir = str(tmp_path)
        swarm._init_agents = MagicMock()

        # Decomposer raises exception
        swarm._decomposer = AsyncMock()
        swarm._decomposer.decompose = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        result = await swarm.research("Test topic")

        assert result.success is False
        assert "LLM unavailable" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Test swarm registry integration."""

    def test_swarm_registered(self):
        from Jotty.core.intelligence.orchestration.swarms._base.registry import (
            SwarmRegistry,
        )

        # Force import to trigger registration
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchSwarm,
        )

        swarm_class = SwarmRegistry.get("topic_research")
        assert swarm_class is TopicResearchSwarm

    def test_registry_list_includes_topic_research(self):
        from Jotty.core.intelligence.orchestration.swarms._base.registry import (
            SwarmRegistry,
        )
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            TopicResearchSwarm,
        )

        all_swarms = SwarmRegistry.list_all()
        assert "topic_research" in all_swarms


# ═══════════════════════════════════════════════════════════════════════════════
# PPTX GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPptxGenerator:
    """Test PPTX generation from research results."""

    def test_import_generator(self):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            generate_research_pptx,
        )

        assert callable(generate_research_pptx)

    def test_generate_pptx_basic(self, tmp_path):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            generate_research_pptx,
        )
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm.types import (
            ComparisonDimension,
            ResearchFinding,
            SourceReference,
            TopicResearchResult,
        )

        result = TopicResearchResult(
            success=True,
            swarm_name="TopicResearchSwarm",
            domain="topic_research",
            output={
                "report": "# Test\n\nTest report",
                "executive_summary": "- Point 1\n- Point 2",
                "one_liner": "Test thesis statement",
            },
            execution_time=42.0,
            topic="Test Topic Research",
            sub_topics=["Sub A", "Sub B", "Sub C"],
            mode="single_topic",
            depth="deep",
            summary="- Point 1\n- Point 2",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            detailed_findings=[
                ResearchFinding(
                    claim="Test claim",
                    evidence="Test evidence",
                    confidence=0.85,
                    sub_topic="Sub A",
                ),
            ],
            sources=[
                SourceReference(url="https://example.com/1", title="Source 1"),
                SourceReference(url="https://example.com/2", title="Source 2"),
            ],
            total_sources_consulted=2,
            unique_domains=1,
            report_markdown="# Test\n\n## Findings\n\n- Finding 1\n- Finding 2",
            fact_check_score=0.8,
            completeness_score=0.7,
            agent_contributions={"researcher": 0.3, "synthesizer": 0.25},
        )

        pptx_path = generate_research_pptx(
            result,
            output_dir=str(tmp_path),
            filename="test_output",
        )
        assert pptx_path.endswith(".pptx")
        assert Path(pptx_path).exists()
        assert Path(pptx_path).stat().st_size > 10000  # At least 10KB

    def test_generate_pptx_comparison(self, tmp_path):
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
            generate_research_pptx,
        )
        from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm.types import (
            ComparisonDimension,
            TopicResearchResult,
        )

        result = TopicResearchResult(
            success=True,
            swarm_name="TopicResearchSwarm",
            domain="topic_research",
            output={"report": "comparison report", "one_liner": "A vs B"},
            execution_time=60.0,
            topic="A vs B",
            sub_topics=["Revenue", "Growth"],
            mode="comparison",
            depth="deep",
            key_findings=["A leads in revenue", "B leads in growth"],
            comparison_entities=["Company A", "Company B"],
            comparison_matrix=[
                ComparisonDimension(
                    dimension="Revenue",
                    entity_scores={"Company A": "$10B", "Company B": "$8B"},
                    winner="Company A",
                    analysis="A has 25% higher revenue",
                ),
                ComparisonDimension(
                    dimension="Growth",
                    entity_scores={"Company A": "15%", "Company B": "30%"},
                    winner="Company B",
                    analysis="B growing 2x faster",
                ),
            ],
            comparison_verdict="Company A wins on fundamentals, B on momentum",
            report_markdown="# A vs B\n\n## Comparison\n\nA vs B analysis",
            fact_check_score=0.75,
            completeness_score=0.65,
        )

        pptx_path = generate_research_pptx(
            result,
            output_dir=str(tmp_path),
        )
        assert Path(pptx_path).exists()

        from pptx import Presentation

        prs = Presentation(pptx_path)
        # Should have: title, exec summary, scope, findings, comparison, sections, sources, scorecard
        assert len(prs.slides) >= 5
