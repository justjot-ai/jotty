"""
Tests for ClawWork-inspired features: LearningStore.get_episode_count
and Orchestrator._attempt_recovery.

Covers:
- get_episode_count: no episodes, with domain filter, without filter
- _attempt_recovery: no output, trajectory salvage, tagged_outputs salvage,
  already has output, empty trajectory, exception safety
"""

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from Jotty.core.intelligence.learning.learning_store import (
    EpisodeRecord,
    LearningStore,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_episode(domain: str = "general", success: bool = True) -> EpisodeRecord:
    """Create a minimal EpisodeRecord for testing."""
    return EpisodeRecord(
        episode_id=f"ep-{time.time_ns()}",
        unit_type="orchestrator",
        unit_name="test",
        domain=domain,
        task_type="test",
        context={},
        action={},
        outcome={},
        success=success,
        quality=0.8 if success else 0.2,
        execution_time=1.0,
        cost=0.01,
    )


@dataclass
class FakeTaggedOutput:
    name: str = "output"
    tag: str = "useful"
    why_useful: str = "test"
    content: Any = None


@dataclass
class FakeEpisodeResult:
    """Minimal EpisodeResult stand-in for testing _attempt_recovery."""

    output: Any = ""
    success: bool = False
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    tagged_outputs: list = field(default_factory=list)
    episode: int = 1
    execution_time: float = 10.0
    architect_results: list = field(default_factory=list)
    auditor_results: list = field(default_factory=list)
    agent_contributions: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


# =========================================================================
# TestLearningStoreEpisodeCount
# =========================================================================


@pytest.mark.unit
class TestLearningStoreEpisodeCount:
    """Tests for LearningStore.get_episode_count()."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a LearningStore backed by a temp SQLite file."""
        return LearningStore(db_path=str(tmp_path / "test_learning.db"))

    def test_empty_store_returns_zero(self, store):
        assert store.get_episode_count() == 0

    def test_empty_store_with_domain_returns_zero(self, store):
        assert store.get_episode_count(domain="coding") == 0

    def test_count_after_inserts(self, store):
        store.save_episode(_make_episode(domain="coding"))
        store.save_episode(_make_episode(domain="coding"))
        store.save_episode(_make_episode(domain="research"))

        assert store.get_episode_count() == 3

    def test_count_filtered_by_domain(self, store):
        store.save_episode(_make_episode(domain="coding"))
        store.save_episode(_make_episode(domain="coding"))
        store.save_episode(_make_episode(domain="research"))

        assert store.get_episode_count(domain="coding") == 2
        assert store.get_episode_count(domain="research") == 1
        assert store.get_episode_count(domain="nonexistent") == 0


# =========================================================================
# TestAttemptRecovery
# =========================================================================


@pytest.mark.unit
class TestAttemptRecovery:
    """Tests for Orchestrator._attempt_recovery()."""

    @pytest.fixture
    def orchestrator(self):
        """Create a minimal Orchestrator just for _attempt_recovery."""
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = object.__new__(Orchestrator)
        return orch

    def test_salvages_from_trajectory(self, orchestrator):
        """Recovery extracts partial outputs from trajectory steps."""
        result = FakeEpisodeResult(
            output="",
            success=False,
            trajectory=[
                {"action": "search", "output": "Found 3 relevant documents about AI."},
                {"action": "analyze", "result": "The documents show growth in AI adoption."},
            ],
        )
        recovered = orchestrator._attempt_recovery(result, "Research AI trends")

        assert "Partial result" in recovered.output
        assert "relevant documents" in recovered.output
        assert len(recovered.alerts) == 1
        assert "2 partial fragments" in recovered.alerts[0]

    def test_salvages_from_tagged_outputs(self, orchestrator):
        """Recovery extracts from tagged_outputs when trajectory is empty."""
        result = FakeEpisodeResult(
            output="",
            success=False,
            tagged_outputs=[
                FakeTaggedOutput(content="Detailed analysis of market trends."),
            ],
        )
        recovered = orchestrator._attempt_recovery(result, "Analyze market")

        assert "Partial result" in recovered.output
        assert "market trends" in recovered.output

    def test_skips_when_output_already_substantial(self, orchestrator):
        """If output already has >100 chars, recovery is a no-op."""
        original_output = "A" * 200
        result = FakeEpisodeResult(
            output=original_output,
            success=False,
            trajectory=[
                {"action": "step", "output": "This should NOT be appended."},
            ],
        )
        recovered = orchestrator._attempt_recovery(result, "Some goal")

        assert recovered.output == original_output
        assert len(recovered.alerts) == 0

    def test_returns_unchanged_when_nothing_to_salvage(self, orchestrator):
        """No trajectory or tagged outputs → result unchanged."""
        result = FakeEpisodeResult(output="", success=False)
        recovered = orchestrator._attempt_recovery(result, "Some goal")

        assert recovered.output == ""
        assert len(recovered.alerts) == 0

    def test_skips_short_fragments(self, orchestrator):
        """Fragments shorter than 20 chars are ignored."""
        result = FakeEpisodeResult(
            output="",
            success=False,
            trajectory=[
                {"action": "step", "output": "tiny"},  # < 20 chars
                {"action": "step", "output": "This is a substantial fragment for recovery."},
            ],
        )
        recovered = orchestrator._attempt_recovery(result, "Goal")

        assert "substantial fragment" in recovered.output
        assert "1 partial fragments" in recovered.alerts[0]

    def test_caps_at_five_fragments(self, orchestrator):
        """Recovery should cap at 5 fragments to avoid bloat."""
        trajectory = [
            {"action": f"step-{i}", "output": f"Fragment number {i} with enough content."}
            for i in range(10)
        ]
        result = FakeEpisodeResult(output="", success=False, trajectory=trajectory)
        recovered = orchestrator._attempt_recovery(result, "Goal")

        # Should salvage 10 fragments but cap displayed at 5
        fragment_count = recovered.output.count("Fragment number")
        assert fragment_count == 5

    def test_success_flag_unchanged(self, orchestrator):
        """Recovery should NOT change the success flag — result is still a failure."""
        result = FakeEpisodeResult(
            output="",
            success=False,
            trajectory=[
                {"action": "step", "output": "Some partial work was done here."},
            ],
        )
        recovered = orchestrator._attempt_recovery(result, "Goal")
        assert recovered.success is False

    def test_exception_safety(self, orchestrator):
        """If recovery itself fails, original result is returned."""
        result = FakeEpisodeResult(output="", success=False)
        # Make trajectory a non-iterable to trigger an exception
        result.trajectory = None
        recovered = orchestrator._attempt_recovery(result, "Goal")
        # Should not raise — just return the result
        assert recovered is result
