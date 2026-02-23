"""Tests for OlympiadLearningSwarm research agents.

All tests use mocked LLM and network — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from Jotty.core.intelligence.memory.vector_store import DocumentChunk, VectorStore


# =============================================================================
# FIXTURES
# =============================================================================


class FakeEmbedder:
    """Mock EmbeddingService."""

    available = True
    dim = 384

    def embed(self, text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % 10000
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts: list) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


@pytest.fixture
def vector_store():
    """Fresh VectorStore with fake embedder."""
    s = VectorStore()
    s._embedder = FakeEmbedder()
    s._use_chroma = False
    s._initialized = True
    return s


class MockDSPyResult:
    """Mock DSPy prediction result."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# =============================================================================
# SOURCE FINDER TESTS
# =============================================================================


class TestSourceFinderAgent:
    """Tests for SourceFinderAgent."""

    @pytest.mark.unit
    def test_build_queries_foundation(self):
        """Builds grade-appropriate queries for foundation level."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            SourceFinderAgent,
        )

        agent = SourceFinderAgent.__new__(SourceFinderAgent)
        queries = agent._build_queries("Fractions", "mathematics", "foundation")
        assert len(queries) >= 3
        # Should include elementary/kids keywords for foundation
        assert any("kid" in q.lower() or "elementary" in q.lower() for q in queries)

    @pytest.mark.unit
    def test_build_queries_olympiad(self):
        """Builds competition-focused queries for olympiad level."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            SourceFinderAgent,
        )

        agent = SourceFinderAgent.__new__(SourceFinderAgent)
        queries = agent._build_queries("Number Theory", "mathematics", "olympiad")
        assert len(queries) >= 3
        assert any("olympiad" in q.lower() or "competition" in q.lower() for q in queries)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contribute_dedup(self):
        """contribute() deduplicates against existing blackboard URLs."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            SourceFinderAgent,
        )

        agent = SourceFinderAgent.__new__(SourceFinderAgent)
        agent.learned_context = ""
        agent._lm = MagicMock()
        agent._searcher = MagicMock(return_value=MockDSPyResult(search_queries="q1 | q2"))
        agent._call_with_own_lm = MagicMock(return_value=MockDSPyResult(search_queries="q1 | q2"))

        # Mock _search_web to return known URLs
        async def mock_search(query):
            return ["https://example.com/a", "https://example.com/b"]

        agent._search_web = mock_search

        blackboard = {
            "task": {
                "topic": "Fractions",
                "subject": "math",
                "target_level": "foundation",
                "max_sources": 10,
            },
            "sources": [{"url": "https://example.com/a"}],  # Already exists
        }

        result = await agent.contribute(blackboard, {})
        assert result is not None
        # Should not re-add the existing URL
        new_urls = [s["url"] for s in result.get("new_sources", [])]
        assert "https://example.com/a" not in new_urls

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contribute_respects_max_sources(self):
        """contribute() returns None when max_sources reached."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            SourceFinderAgent,
        )

        agent = SourceFinderAgent.__new__(SourceFinderAgent)

        blackboard = {
            "task": {
                "topic": "X",
                "subject": "math",
                "target_level": "foundation",
                "max_sources": 2,
            },
            "sources": [{"url": "u1"}, {"url": "u2"}],
        }

        result = await agent.contribute(blackboard, {})
        assert result is None


# =============================================================================
# CONTENT EXTRACTOR TESTS
# =============================================================================


class TestContentExtractorAgent:
    """Tests for ContentExtractorAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contribute_extracts_unprocessed(self, vector_store):
        """contribute() processes unextracted sources."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentExtractorAgent,
        )

        agent = ContentExtractorAgent.__new__(ContentExtractorAgent)
        agent.learned_context = ""
        agent._lm = None

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_url = AsyncMock(return_value=5)
        agent._pipeline = mock_pipeline

        blackboard = {
            "task": {"topic": "Fractions", "subject": "math", "target_level": "foundation"},
            "sources": [{"url": "https://example.com/1"}, {"url": "https://example.com/2"}],
            "extracted": set(),
        }

        result = await agent.contribute(blackboard, {})
        assert result is not None
        assert result["total_chunks"] == 10  # 5 per URL
        assert len(blackboard["extracted"]) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contribute_skips_already_extracted(self, vector_store):
        """contribute() returns None when all sources already extracted."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentExtractorAgent,
        )

        agent = ContentExtractorAgent.__new__(ContentExtractorAgent)
        agent.learned_context = ""

        blackboard = {
            "task": {"topic": "X", "subject": "math", "target_level": "foundation"},
            "sources": [{"url": "https://example.com/1"}],
            "extracted": {"https://example.com/1"},
        }

        result = await agent.contribute(blackboard, {})
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_single_url(self, vector_store):
        """execute() extracts from a single URL (ROUND_ROBIN interface)."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentExtractorAgent,
        )

        agent = ContentExtractorAgent.__new__(ContentExtractorAgent)
        agent.learned_context = ""

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_url = AsyncMock(return_value=3)
        agent._pipeline = mock_pipeline

        result = await agent.execute(input="https://example.com/test", context={"topic": "math"})
        assert result["url"] == "https://example.com/test"
        assert result["chunks"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_empty_url(self):
        """execute() handles empty URL gracefully."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentExtractorAgent,
        )

        agent = ContentExtractorAgent.__new__(ContentExtractorAgent)
        agent.learned_context = ""

        result = await agent.execute(input="", context={})
        assert result["chunks"] == 0
        assert "error" in result


# =============================================================================
# CONTENT CURATOR TESTS
# =============================================================================


class TestContentCuratorAgent:
    """Tests for ContentCuratorAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contribute_scores_content(self, vector_store):
        """contribute() scores unscored sources."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentCuratorAgent,
        )

        # Pre-populate vector store with some chunks
        chunks = [
            DocumentChunk(
                "c1", "Fractions are cool", "https://example.com/1", "webpage", {"subject": "math"}
            ),
        ]
        vector_store.add_chunks(chunks)

        agent = ContentCuratorAgent.__new__(ContentCuratorAgent)
        agent.learned_context = ""
        agent._lm = MagicMock()
        agent._curator = MagicMock()
        agent._call_with_own_lm = MagicMock(
            return_value=MockDSPyResult(quality_score="0.8", reasoning="Good content")
        )

        blackboard = {
            "task": {
                "topic": "Fractions",
                "subject": "math",
                "target_level": "foundation",
                "min_quality_sources": 1,
            },
            "extracted": {"https://example.com/1"},
            "quality_scores": {},
        }

        # Patch get_vector_store to return our test store
        with patch(
            "Jotty.core.intelligence.memory.vector_store.VectorStore.get_instance",
            return_value=vector_store,
        ):
            result = await agent.contribute(blackboard, {})

        assert result is not None
        assert "scores" in result
        # Should mark done since we have >= 1 quality source (threshold 0.6)
        assert blackboard.get("done") is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_contribute_marks_done(self, vector_store):
        """contribute() marks blackboard done when enough quality sources."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentCuratorAgent,
        )

        agent = ContentCuratorAgent.__new__(ContentCuratorAgent)
        agent.learned_context = ""

        blackboard = {
            "task": {
                "topic": "Fractions",
                "subject": "math",
                "target_level": "foundation",
                "min_quality_sources": 2,
            },
            "extracted": set(),
            "quality_scores": {"url1": 0.9, "url2": 0.7},  # Already scored, both good
        }

        result = await agent.contribute(blackboard, {})
        # No unscored URLs -> check if done
        assert result is not None
        assert blackboard["done"] is True


# =============================================================================
# BLACKBOARD INTEGRATION TEST
# =============================================================================


class TestResearchBlackboard:
    """Integration test: 2-round BLACKBOARD cycle with mocked agents."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_blackboard_2_rounds(self, vector_store):
        """Full blackboard cycle: find -> extract -> curate."""
        from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.research_agents import (
            ContentCuratorAgent,
            ContentExtractorAgent,
            SourceFinderAgent,
        )

        # Create agents
        finder = SourceFinderAgent.__new__(SourceFinderAgent)
        finder.learned_context = ""
        finder._lm = MagicMock()
        finder._searcher = MagicMock()
        finder._call_with_own_lm = MagicMock(return_value=MockDSPyResult(search_queries="q1"))

        async def mock_search(q):
            return ["https://example.com/math1"]

        finder._search_web = mock_search

        extractor = ContentExtractorAgent.__new__(ContentExtractorAgent)
        extractor.learned_context = ""
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_url = AsyncMock(return_value=3)
        extractor._pipeline = mock_pipeline

        curator = ContentCuratorAgent.__new__(ContentCuratorAgent)
        curator.learned_context = ""
        curator._lm = MagicMock()
        curator._curator = MagicMock()
        curator._call_with_own_lm = MagicMock(
            return_value=MockDSPyResult(quality_score="0.85", reasoning="Good")
        )

        # Simulate blackboard
        blackboard = {
            "task": {
                "topic": "Fractions",
                "subject": "math",
                "target_level": "foundation",
                "max_sources": 5,
                "min_quality_sources": 1,
            },
            "contributions": {},
            "done": False,
        }

        # Round 1: finder finds, extractor extracts
        r1 = await finder.contribute(blackboard, {})
        assert r1 is not None
        assert len(blackboard.get("sources", [])) > 0

        r2 = await extractor.contribute(blackboard, {})
        assert r2 is not None
        assert r2["total_chunks"] > 0

        # Round 2: curator scores
        # Need to add chunks to vector_store for curator to find
        chunks = [
            DocumentChunk("c1", "Fraction content", "https://example.com/math1", "webpage"),
        ]
        vector_store.add_chunks(chunks)

        with patch(
            "Jotty.core.intelligence.memory.vector_store.VectorStore.get_instance",
            return_value=vector_store,
        ):
            r3 = await curator.contribute(blackboard, {})

        assert r3 is not None
        assert blackboard.get("done") is True
