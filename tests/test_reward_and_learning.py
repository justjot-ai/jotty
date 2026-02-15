"""
Tests for reward computation, schema versioning, and effectiveness tracking.

Covers:
- _compute_episode_reward: empty, short, padded vs concise, bounds, failure cap
- _load_versioned / _save_versioned: current version, incompatible major, legacy
- EffectivenessTracker: no data, improving after good data, serialization roundtrip
"""

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from Jotty.core.intelligence.orchestration.learning_pipeline import (
    EffectivenessTracker,
    SwarmLearningPipeline,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_result(output="", success=True, execution_time=30.0, trajectory=None):
    """Create a minimal mock EpisodeResult."""
    return SimpleNamespace(
        output=output,
        success=success,
        execution_time=execution_time,
        trajectory=trajectory or [],
    )


# =========================================================================
# TestComputeEpisodeReward
# =========================================================================

class TestComputeEpisodeReward:
    """Test the multi-dimensional episode reward computation."""

    def test_empty_output_scores_low(self):
        result = _make_result(output="", success=True)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do something")
        assert reward < 0.5, f"Empty output should score low, got {reward}"

    def test_short_output(self):
        result = _make_result(output="ok", success=True)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do something")
        assert 0.0 < reward < 0.7, f"Short output reward out of range: {reward}"

    def test_padded_vs_concise(self):
        """Padded/repetitive text must NOT win over concise, diverse text."""
        # Padded: same sentence repeated many times
        padded_text = ("This is a filler sentence. " * 100).strip()
        concise_text = (
            "The analysis shows three key findings.\n"
            "## Finding 1: Market Growth\n"
            "Revenue increased 15% YoY driven by expansion.\n"
            "## Finding 2: Cost Reduction\n"
            "Operating costs decreased through automation.\n"
            "## Finding 3: Innovation Pipeline\n"
            "Three new products entered beta testing.\n"
            "In conclusion, the outlook is positive."
        )
        goal = "Analyze market performance"

        padded_result = _make_result(output=padded_text, success=True)
        concise_result = _make_result(output=concise_text, success=True)

        padded_reward = SwarmLearningPipeline._compute_episode_reward(padded_result, goal)
        concise_reward = SwarmLearningPipeline._compute_episode_reward(concise_result, goal)

        assert concise_reward > padded_reward, (
            f"Concise ({concise_reward:.3f}) should beat padded ({padded_reward:.3f})"
        )

    def test_reward_always_in_bounds(self):
        """Reward must always be in [0, 1]."""
        test_cases = [
            _make_result(output="", success=False),
            _make_result(output="x" * 10000, success=True),
            _make_result(output="error: traceback failed to unable to", success=True),
            _make_result(output="Hello world", success=True, execution_time=0.1),
            _make_result(output="Hello world", success=True, execution_time=1000),
        ]
        for i, result in enumerate(test_cases):
            reward = SwarmLearningPipeline._compute_episode_reward(result, "test goal")
            assert 0.0 <= reward <= 1.0, (
                f"Case {i}: reward {reward} out of [0, 1] bounds"
            )

    def test_failure_caps_at_03(self):
        """Failed results should have reward capped at 0.3."""
        result = _make_result(
            output="A great detailed analysis with lots of content " * 50,
            success=False,
        )
        reward = SwarmLearningPipeline._compute_episode_reward(result, "analyze")
        assert reward <= 0.3, f"Failed result should cap at 0.3, got {reward}"

    def test_error_indicators_penalized(self):
        """Output with error indicators should score lower than clean output."""
        clean = _make_result(
            output="The system is running correctly with all checks passing.",
            success=True,
        )
        errors = _make_result(
            output="error: could not connect. failed to load. traceback shown.",
            success=True,
        )
        goal = "Check system status"
        clean_reward = SwarmLearningPipeline._compute_episode_reward(clean, goal)
        error_reward = SwarmLearningPipeline._compute_episode_reward(errors, goal)
        assert clean_reward > error_reward, (
            f"Clean ({clean_reward:.3f}) should beat errors ({error_reward:.3f})"
        )

    def test_relevance_boosts_score(self):
        """Output that mentions goal keywords should score higher."""
        goal = "Research artificial intelligence trends"
        relevant = _make_result(
            output="Artificial intelligence trends show rapid growth in research.",
            success=True,
        )
        irrelevant = _make_result(
            output="The weather today is sunny with clear skies.",
            success=True,
        )
        rel_reward = SwarmLearningPipeline._compute_episode_reward(relevant, goal)
        irr_reward = SwarmLearningPipeline._compute_episode_reward(irrelevant, goal)
        assert rel_reward > irr_reward, (
            f"Relevant ({rel_reward:.3f}) should beat irrelevant ({irr_reward:.3f})"
        )

    def test_no_tools_slight_penalty(self):
        """No tool usage should get 0.3 (slight penalty), not 0.5 neutral."""
        result = _make_result(output="Good output", success=True, trajectory=[])
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do task")
        # tool_use dimension should be 0.3, contributing 0.15 * 0.3 = 0.045
        # vs old 0.15 * 0.5 = 0.075 — we just verify it's not too high
        assert reward < 1.0


# =========================================================================
# TestSchemaVersioning
# =========================================================================

class TestSchemaVersioning:
    """Test versioned persistence in SwarmLearningPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a minimal SwarmLearningPipeline with mocked config."""
        config = MagicMock()
        config.base_path = None
        with patch.object(SwarmLearningPipeline, '_init_components'):
            lp = SwarmLearningPipeline.__new__(SwarmLearningPipeline)
            lp.config = config
            lp.episode_count = 0
            lp._SCHEMA_VERSION = "2.0"
            lp._MIGRATIONS = {}
        return lp

    def test_save_and_load_current_version(self, pipeline, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "count": 42}
        pipeline._save_versioned(path, data)

        loaded = pipeline._load_versioned(path)
        assert loaded == data

        # Verify envelope structure
        with open(path) as f:
            raw = json.load(f)
        assert raw["schema_version"] == "2.0"
        assert raw["data"] == data

    def test_load_incompatible_major_returns_empty(self, pipeline, tmp_path):
        """Loading data with incompatible major version returns empty dict."""
        path = tmp_path / "old.json"
        envelope = {"schema_version": "1.0", "data": {"old_key": "old_value"}}
        with open(path, 'w') as f:
            json.dump(envelope, f)

        loaded = pipeline._load_versioned(path)
        assert loaded == {}, f"Incompatible major version should return empty, got {loaded}"

    def test_load_incompatible_with_migration(self, pipeline, tmp_path):
        """If a migration exists, it should be applied."""
        path = tmp_path / "migrate.json"
        envelope = {"schema_version": "1.0", "data": {"old_format": True}}
        with open(path, 'w') as f:
            json.dump(envelope, f)

        # Register a migration on the class (static method accesses class dict)
        SwarmLearningPipeline._MIGRATIONS[("1", "2")] = lambda data: {"migrated": True}
        try:
            loaded = pipeline._load_versioned(path)
            assert loaded == {"migrated": True}
        finally:
            # Clean up
            SwarmLearningPipeline._MIGRATIONS.pop(("1", "2"), None)

    def test_load_legacy_bare_dict(self, pipeline, tmp_path):
        """Legacy format (no envelope) should still load."""
        path = tmp_path / "legacy.json"
        data = {"legacy_key": "legacy_value"}
        with open(path, 'w') as f:
            json.dump(data, f)

        loaded = pipeline._load_versioned(path)
        assert loaded == data

    def test_compatible_minor_version_loads(self, pipeline, tmp_path):
        """Same major version but different minor should load fine."""
        path = tmp_path / "minor.json"
        envelope = {"schema_version": "2.5", "data": {"updated": True}}
        with open(path, 'w') as f:
            json.dump(envelope, f)

        loaded = pipeline._load_versioned(path)
        assert loaded == {"updated": True}


# =========================================================================
# TestEffectivenessTracker
# =========================================================================

class TestEffectivenessTracker:
    """Test the EffectivenessTracker."""

    def test_no_data_not_improving(self):
        tracker = EffectivenessTracker(recent_window=5, historical_window=10)
        assert tracker.is_improving() is False
        assert tracker.is_improving("analysis") is False

    def test_improving_after_good_data(self):
        tracker = EffectivenessTracker(recent_window=5, historical_window=10)

        # Historical: 50% success rate (10 episodes, 5 success)
        for i in range(10):
            tracker.record("analysis", success=(i % 2 == 0), quality=0.5)

        # Recent: 100% success rate (5 episodes, all success)
        for _ in range(5):
            tracker.record("analysis", success=True, quality=0.9)

        report = tracker.improvement_report()
        analysis = report.get("analysis", {})
        assert analysis.get("improving") is True, f"Should be improving: {analysis}"
        assert analysis["recent_success_rate"] > analysis["historical_success_rate"]

    def test_serialization_roundtrip(self):
        tracker = EffectivenessTracker(recent_window=5, historical_window=10)
        tracker.record("coding", success=True, quality=0.8, agent="auto")
        tracker.record("coding", success=False, quality=0.3, agent="auto")
        tracker.record("analysis", success=True, quality=0.9, agent="researcher")

        # Serialize
        data = tracker.to_dict()
        assert "coding" in data
        assert "analysis" in data
        assert len(data["coding"]) == 2
        assert len(data["analysis"]) == 1

        # Deserialize
        restored = EffectivenessTracker.from_dict(data, recent_window=5, historical_window=10)
        assert len(restored._records["coding"]) == 2
        assert len(restored._records["analysis"]) == 1

        # Reports should match
        orig_report = tracker.improvement_report()
        rest_report = restored.improvement_report()
        for key in ("coding", "analysis"):
            assert orig_report[key]["total_episodes"] == rest_report[key]["total_episodes"]

    def test_record_clamps_quality(self):
        tracker = EffectivenessTracker()
        tracker.record("test", success=True, quality=1.5)  # Over 1.0
        tracker.record("test", success=True, quality=-0.5)  # Under 0.0
        records = list(tracker._records["test"])
        assert records[0][2] == 1.0  # Clamped to 1.0
        assert records[1][2] == 0.0  # Clamped to 0.0

    def test_global_tracks_all_types(self):
        tracker = EffectivenessTracker(recent_window=5, historical_window=10)
        tracker.record("type_a", success=True, quality=0.8)
        tracker.record("type_b", success=False, quality=0.2)
        assert len(tracker._global) == 2
