"""Tests for SwarmTemplate general BLACKBOARD/ROUND_ROBIN helpers.

Verifies that run_blackboard(), run_round_robin(), and search_documents()
work as thin wrappers over TeamCoordinator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Jotty.core.intelligence.memory.vector_store import DocumentChunk, VectorStore
from Jotty.core.intelligence.orchestration.swarms.base.swarm_template import SwarmTemplate
from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import TeamResult


# =============================================================================
# FIXTURES
# =============================================================================


class FakeEmbedder:
    available = True
    dim = 384

    def embed(self, text):
        seed = sum(ord(c) for c in text) % 10000
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


class MockBlackboardAgent:
    """Agent that contributes to blackboard."""

    def __init__(self, name: str, contribution: dict = None):
        self.name = name
        self._contribution = contribution or {"data": f"from_{name}"}

    async def contribute(self, blackboard=None, context=None, **kwargs):
        if blackboard.get("done"):
            return None
        blackboard["contributions"][self.name] = self._contribution
        return self._contribution


class MockDoneAgent:
    """Agent that marks blackboard done on second round."""

    def __init__(self):
        self._call_count = 0

    async def contribute(self, blackboard=None, context=None, **kwargs):
        self._call_count += 1
        if self._call_count >= 2:
            blackboard["done"] = True
        return {"round": self._call_count}


class MockExecuteAgent:
    """Agent that implements execute() for ROUND_ROBIN."""

    def __init__(self, name: str):
        self.name = name

    async def execute(self, input=None, context=None, **kwargs):
        return {"processed_by": self.name, "input": input}


class ConcreteSwarm(SwarmTemplate):
    """Minimal concrete swarm for testing."""

    TASK_TYPE = "test"

    async def _execute_domain(self, *args, **kwargs):
        pass


@pytest.fixture
def swarm():
    """Create a minimal SwarmTemplate subclass for testing helpers."""
    from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import SwarmBaseConfig

    config = SwarmBaseConfig()
    config.name = "test_swarm"
    config.domain = "test"
    s = ConcreteSwarm(config)
    s._agents_initialized = True
    return s


@pytest.fixture
def vector_store():
    """Fresh VectorStore with fake embedder."""
    s = VectorStore()
    s._embedder = FakeEmbedder()
    s._use_chroma = False
    s._initialized = True
    return s


# =============================================================================
# RUN_BLACKBOARD TESTS
# =============================================================================


class TestRunBlackboard:
    """Tests for SwarmTemplate.run_blackboard()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_basic_blackboard(self, swarm):
        """run_blackboard runs agents in BLACKBOARD pattern."""
        agent1 = MockBlackboardAgent("agent1")
        agent2 = MockBlackboardAgent("agent2")

        result = await swarm.run_blackboard(
            agents=[(agent1, "Agent1"), (agent2, "Agent2")],
            task={"topic": "test"},
            max_rounds=2,
        )

        assert isinstance(result, TeamResult)
        assert result.pattern.value == "blackboard"
        # Both agents should have contributed
        bb = result.merged_output
        assert "contributions" in bb

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_blackboard_respects_done(self, swarm):
        """run_blackboard stops when blackboard['done'] is True."""
        agent1 = MockBlackboardAgent("agent1")
        done_agent = MockDoneAgent()

        result = await swarm.run_blackboard(
            agents=[(agent1, "Worker"), (done_agent, "Stopper")],
            task={"topic": "test"},
            max_rounds=5,
        )

        assert isinstance(result, TeamResult)
        # Should have stopped before max_rounds
        bb = result.merged_output
        assert bb.get("done") is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_blackboard_with_context(self, swarm):
        """run_blackboard passes context to agents."""
        received_context = {}

        class ContextCapture:
            async def contribute(self, blackboard=None, context=None, **kwargs):
                received_context.update(context or {})
                blackboard["done"] = True
                return {"captured": True}

        agent = ContextCapture()
        await swarm.run_blackboard(
            agents=[(agent, "Capture")],
            task={"topic": "test"},
            context={"key": "value"},
            max_rounds=1,
        )

        assert received_context.get("key") == "value"


# =============================================================================
# RUN_ROUND_ROBIN TESTS
# =============================================================================


class TestRunRoundRobin:
    """Tests for SwarmTemplate.run_round_robin()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_basic_round_robin(self, swarm):
        """run_round_robin distributes tasks across agents."""
        w1 = MockExecuteAgent("w1")
        w2 = MockExecuteAgent("w2")

        result = await swarm.run_round_robin(
            agents=[(w1, "W1"), (w2, "W2")],
            tasks=["task_a", "task_b", "task_c", "task_d"],
        )

        assert isinstance(result, TeamResult)
        assert result.pattern.value == "round_robin"
        # 4 tasks distributed across 2 agents
        assert len(result.outputs) == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_round_robin_single_task(self, swarm):
        """run_round_robin handles single task."""
        w1 = MockExecuteAgent("w1")

        result = await swarm.run_round_robin(
            agents=[(w1, "W1")],
            tasks=["only_task"],
        )

        assert result.success
        assert len(result.outputs) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_round_robin_with_context(self, swarm):
        """run_round_robin passes context to agents."""
        received_context = {}

        class ContextAgent:
            async def execute(self, input=None, context=None, **kwargs):
                received_context.update(context or {})
                return {"done": True}

        agent = ContextAgent()
        await swarm.run_round_robin(
            agents=[(agent, "A1")],
            tasks=["t1"],
            context={"subject": "math"},
        )

        assert received_context.get("subject") == "math"


# =============================================================================
# SEARCH_DOCUMENTS TESTS
# =============================================================================


class TestSearchDocuments:
    """Tests for SwarmTemplate.search_documents()."""

    @pytest.mark.unit
    def test_returns_empty_when_no_store(self, swarm):
        """search_documents returns [] when vector store unavailable."""
        swarm._vector_store = None
        results = swarm.search_documents("anything")
        assert results == []

    @pytest.mark.unit
    def test_returns_results_with_store(self, swarm, vector_store):
        """search_documents delegates to vector store."""
        # Add some chunks
        chunks = [
            DocumentChunk(
                "c1", "Fractions are parts of a whole", "src", "text", {"subject": "math"}
            ),
        ]
        vector_store.add_chunks(chunks)

        swarm._vector_store = vector_store
        results = swarm.search_documents("fractions", top_k=5)
        assert len(results) > 0

    @pytest.mark.unit
    def test_search_with_filters(self, swarm, vector_store):
        """search_documents passes filters to vector store."""
        chunks = [
            DocumentChunk("c1", "Math content", "s1", "text", {"subject": "math"}),
            DocumentChunk("c2", "Physics content", "s2", "text", {"subject": "physics"}),
        ]
        vector_store.add_chunks(chunks)

        swarm._vector_store = vector_store
        results = swarm.search_documents("content", top_k=5, subject="math")
        # Should only return math chunks
        for r in results:
            if r.similarity > 0:
                assert r.chunk.metadata.get("subject") == "math"


# =============================================================================
# VECTOR STORE PROPERTY TEST
# =============================================================================


class TestVectorStoreProperty:
    """Tests for SwarmTemplate.vector_store property."""

    @pytest.mark.unit
    def test_lazy_init(self, swarm):
        """vector_store property lazy-initializes."""
        # Clear any cached store
        if hasattr(swarm, "_vector_store"):
            delattr(swarm, "_vector_store")

        with patch(
            "Jotty.core.intelligence.memory.vector_store.VectorStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            store = swarm.vector_store
            assert store is mock_store
            # Second access should not call get_instance again
            store2 = swarm.vector_store
            assert store2 is mock_store
            mock_get.assert_called_once()

    @pytest.mark.unit
    def test_returns_none_on_error(self, swarm):
        """vector_store returns None if import fails."""
        if hasattr(swarm, "_vector_store"):
            delattr(swarm, "_vector_store")

        with patch(
            "Jotty.core.intelligence.memory.vector_store.VectorStore.get_instance",
            side_effect=ImportError("no chromadb"),
        ):
            store = swarm.vector_store
            assert store is None
