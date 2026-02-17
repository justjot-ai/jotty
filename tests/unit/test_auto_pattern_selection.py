"""Tests for AUTO pattern selection and _record_pattern_success.

Covers previously untested code paths:
- PatternSelector.select_pattern() with all 5 layers
- PatternSelector._analyze_task() keyword matching
- PatternSelector._fallback_heuristic() agent-count rules
- SwarmTemplate._record_pattern_success() for memory, TD-Lambda, and SI
- SwarmTemplate.execute_team() with CoordinationPattern.AUTO
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Jotty.core.infrastructure.foundation.types.execution_types import (
    CoordinationPattern,
)
from Jotty.core.intelligence.orchestration.swarms._base.pattern_selector import (
    PatternSelector,
)
from Jotty.core.intelligence.orchestration.swarms._base.swarm_types import (
    SwarmBaseConfig,
    SwarmResult,
)
from Jotty.core.intelligence.orchestration.swarms.base.swarm_template import (
    SwarmTemplate,
)
from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import (
    TeamCoordinator,
    TeamResult,
)


# =========================================================================
# PATTERN SELECTOR TESTS
# =========================================================================


class TestPatternSelectorAnalyzeTask:
    """Tests for keyword-based task analysis (Layer 4)."""

    @pytest.mark.unit
    def test_parallel_keywords(self) -> None:
        selector = PatternSelector()
        # PARALLEL requires list-like structure (numbered/bulleted) or commas with 3+ agents
        assert (
            selector._analyze_task("Compare each item: A, B, C", agent_count=3)
            == CoordinationPattern.PARALLEL
        )

    @pytest.mark.unit
    def test_sequential_keywords(self) -> None:
        selector = PatternSelector()
        assert (
            selector._analyze_task("First do X, then do Y", agent_count=2)
            == CoordinationPattern.SEQUENTIAL
        )

    @pytest.mark.unit
    def test_consensus_keywords(self) -> None:
        selector = PatternSelector()
        assert (
            selector._analyze_task("Review and evaluate the code", agent_count=3)
            == CoordinationPattern.CONSENSUS
        )

    @pytest.mark.unit
    def test_debate_keywords(self) -> None:
        selector = PatternSelector()
        assert (
            selector._analyze_task("Debate the pros and cons", agent_count=2)
            == CoordinationPattern.DEBATE
        )

    @pytest.mark.unit
    def test_iterative_keywords(self) -> None:
        selector = PatternSelector()
        assert (
            selector._analyze_task("Improve and refine the output", agent_count=1)
            == CoordinationPattern.ITERATIVE
        )

    @pytest.mark.unit
    def test_hierarchical_keywords(self) -> None:
        selector = PatternSelector()
        assert (
            selector._analyze_task("Manage and delegate the complex project", agent_count=3)
            == CoordinationPattern.HIERARCHICAL
        )

    @pytest.mark.unit
    def test_blackboard_keywords(self) -> None:
        selector = PatternSelector()
        assert (
            selector._analyze_task("Collaborate to build incrementally", agent_count=2)
            == CoordinationPattern.BLACKBOARD
        )

    @pytest.mark.unit
    def test_no_match_returns_none(self) -> None:
        selector = PatternSelector()
        assert selector._analyze_task("random nonsense xyz", agent_count=1) is None


class TestPatternSelectorFallback:
    """Tests for fallback heuristic (Layer 5)."""

    @pytest.mark.unit
    def test_single_agent_returns_sequential(self) -> None:
        selector = PatternSelector()
        assert selector._fallback_heuristic(1) == CoordinationPattern.SEQUENTIAL

    @pytest.mark.unit
    def test_two_agents_returns_sequential(self) -> None:
        selector = PatternSelector()
        assert selector._fallback_heuristic(2) == CoordinationPattern.SEQUENTIAL

    @pytest.mark.unit
    def test_three_plus_agents_returns_parallel(self) -> None:
        selector = PatternSelector()
        assert selector._fallback_heuristic(3) == CoordinationPattern.PARALLEL
        assert selector._fallback_heuristic(5) == CoordinationPattern.PARALLEL


class TestPatternSelectorSelectPattern:
    """Tests for the full select_pattern() pipeline."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_learning_falls_through_to_analysis(self) -> None:
        """With no memory/TD/SI, falls through to task analysis or fallback."""
        selector = PatternSelector()
        pattern = await selector.select_pattern(task="Improve the code quality", agent_count=2)
        # "improve" triggers ITERATIVE
        assert pattern == CoordinationPattern.ITERATIVE

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_learning_falls_through_to_fallback(self) -> None:
        """With no keywords matching, falls through to heuristic fallback."""
        selector = PatternSelector()
        pattern = await selector.select_pattern(task="Do something generic", agent_count=4)
        assert pattern == CoordinationPattern.PARALLEL

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_td_lambda_overrides_analysis(self) -> None:
        """TD-Lambda with learned Q-values overrides keyword analysis."""
        mock_td = MagicMock()

        # Return high Q-value for DEBATE
        def get_q(state, action):
            if action.get("pattern") == "debate":
                return 0.9
            return 0.0

        mock_td.get_q_value = get_q

        selector = PatternSelector(td_learner=mock_td, swarm_name="test")
        # "improve" would normally trigger ITERATIVE, but TD says DEBATE
        pattern = await selector.select_pattern(task="Improve the code quality", agent_count=2)
        assert pattern == CoordinationPattern.DEBATE

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_si_pattern_learner(self) -> None:
        """Swarm Intelligence pattern learner provides recommendation."""
        mock_si = MagicMock()
        mock_si.pattern_learner.get_best_pattern.return_value = "consensus"

        selector = PatternSelector(swarm_intelligence=mock_si, swarm_name="test")
        pattern = await selector.select_pattern(task="Do something generic", agent_count=3)
        assert pattern == CoordinationPattern.CONSENSUS


class TestPatternSelectorHelpers:
    """Tests for helper methods."""

    @pytest.mark.unit
    def test_extract_task_type(self) -> None:
        selector = PatternSelector()
        assert selector._extract_task_type("research AI startups") == "research"
        assert selector._extract_task_type("analyze sales data") == "analysis"
        assert selector._extract_task_type("implement login feature") == "coding"
        assert selector._extract_task_type("review the PR") == "review"
        assert selector._extract_task_type("write a report") == "generation"
        assert selector._extract_task_type("test the API") == "testing"
        assert selector._extract_task_type("do stuff") == "general"

    @pytest.mark.unit
    def test_hash_task_deterministic(self) -> None:
        selector = PatternSelector()
        h1 = selector._hash_task("research AI")
        h2 = selector._hash_task("research AI")
        assert h1 == h2
        assert len(h1) == 8


# =========================================================================
# RECORD PATTERN SUCCESS TESTS
# =========================================================================


class _DummyAgent:
    def __init__(self, **kwargs):
        pass


class AutoSwarm(SwarmTemplate):
    """Swarm with AUTO pattern for testing _record_pattern_success."""

    AGENT_TEAM = TeamCoordinator.define(
        (_DummyAgent, "AgentA"),
        (_DummyAgent, "AgentB"),
        pattern=CoordinationPattern.AUTO,
    )

    async def _execute_domain(self, *args, **kwargs) -> SwarmResult:
        return SwarmResult(
            success=True,
            swarm_name="auto",
            domain="test",
            output={},
            execution_time=0.1,
        )


class TestRecordPatternSuccess:
    """Tests for _record_pattern_success() — verifies learning is stored."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stores_in_memory(self) -> None:
        """Pattern result is stored in procedural memory."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = AutoSwarm(config)

        mock_memory = MagicMock()
        swarm._memory = mock_memory

        await swarm._record_pattern_success(
            task="test task",
            pattern=CoordinationPattern.PARALLEL,
            success=True,
        )

        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args[1]
        stored_data = json.loads(call_kwargs["content"])
        assert stored_data["pattern"] == "parallel"
        assert stored_data["success"] == 1.0
        assert call_kwargs["goal"] == "pattern_selection"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_updates_td_lambda(self) -> None:
        """Pattern result updates TD-Lambda Q-values."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = AutoSwarm(config)
        swarm._memory = None  # Skip memory

        mock_td = MagicMock()
        swarm._td_learner = mock_td

        await swarm._record_pattern_success(
            task="test task",
            pattern=CoordinationPattern.DEBATE,
            success=False,
        )

        mock_td.update.assert_called_once()
        call_kwargs = mock_td.update.call_args[1]
        assert call_kwargs["action"]["pattern"] == "debate"
        assert call_kwargs["reward"] == 0.0  # Failed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_updates_si_pattern_learner(self) -> None:
        """Pattern result is reported to SwarmIntelligence pattern_learner."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = AutoSwarm(config)
        swarm._memory = None

        mock_si = MagicMock()
        mock_si.pattern_learner.record_pattern_result = MagicMock()
        swarm._swarm_intelligence = mock_si

        await swarm._record_pattern_success(
            task="test task",
            pattern=CoordinationPattern.PARALLEL,
            success=True,
        )

        mock_si.pattern_learner.record_pattern_result.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_all_components_missing(self) -> None:
        """Does not raise when memory, TD, and SI are all None."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = AutoSwarm(config)
        swarm._memory = None
        swarm._td_learner = None
        swarm._swarm_intelligence = None

        # Should not raise
        await swarm._record_pattern_success(
            task="test", pattern=CoordinationPattern.SEQUENTIAL, success=True
        )


class TestExecuteTeamAuto:
    """Tests for execute_team() with CoordinationPattern.AUTO."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_selects_and_restores_pattern(self) -> None:
        """AUTO selects a pattern, executes, and restores AUTO afterward."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = AutoSwarm(config)
        swarm._agents_initialized = True
        swarm._agent_a = _DummyAgent()
        swarm._agent_b = _DummyAgent()
        swarm.AGENT_TEAM.set_instances({"_agent_a": swarm._agent_a, "_agent_b": swarm._agent_b})

        # No learning — falls through to fallback
        swarm._memory = None
        swarm._td_learner = None
        swarm._swarm_intelligence = None

        mock_result = TeamResult(
            success=True,
            outputs={"AgentA": "ok", "AgentB": "ok"},
            merged_output="result",
        )

        with patch.object(swarm.AGENT_TEAM, "execute", return_value=mock_result):
            result = await swarm.execute_team(task="Do something generic")

        # Result should be the team result
        assert result.success is True
        # Pattern should be restored to AUTO
        assert swarm.AGENT_TEAM.pattern == CoordinationPattern.AUTO
