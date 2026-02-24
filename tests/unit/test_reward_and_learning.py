"""
Tests for reward computation, schema versioning, and effectiveness tracking.

Covers:
- _compute_episode_reward: empty, short, padded vs concise, bounds, failure cap
- _load_versioned / _save_versioned: current version, incompatible major, legacy
- EffectivenessTracker: no data, improving after good data, serialization roundtrip
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from Jotty.core.intelligence.orchestration.learning.swarm_learning_pipeline import (
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
        # Empty output triggers quality cliff → -0.1
        assert reward < 0.5, f"Empty output should score low, got {reward}"

    def test_short_output(self):
        result = _make_result(output="ok", success=True)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do something")
        # Short output may trigger quality cliff → -0.1
        assert -0.1 <= reward < 0.7, f"Short output reward out of range: {reward}"

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

        assert (
            concise_reward > padded_reward
        ), f"Concise ({concise_reward:.3f}) should beat padded ({padded_reward:.3f})"

    def test_reward_always_in_bounds(self):
        """Reward must always be in [-0.1, 1] (quality cliff allows negative)."""
        test_cases = [
            _make_result(output="", success=False),
            _make_result(output="x" * 10000, success=True),
            _make_result(output="error: traceback failed to unable to", success=True),
            _make_result(output="Hello world", success=True, execution_time=0.1),
            _make_result(output="Hello world", success=True, execution_time=1000),
        ]
        for i, result in enumerate(test_cases):
            reward = SwarmLearningPipeline._compute_episode_reward(result, "test goal")
            assert -0.1 <= reward <= 1.0, f"Case {i}: reward {reward} out of [-0.1, 1] bounds"

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
        assert (
            clean_reward > error_reward
        ), f"Clean ({clean_reward:.3f}) should beat errors ({error_reward:.3f})"

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
        assert (
            rel_reward > irr_reward
        ), f"Relevant ({rel_reward:.3f}) should beat irrelevant ({irr_reward:.3f})"

    def test_no_tools_slight_penalty(self):
        """No tool usage should get 0.3 (slight penalty), not 0.5 neutral."""
        result = _make_result(output="Good output", success=True, trajectory=[])
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do task")
        # tool_use dimension should be 0.3, contributing w["tool_use"] * 0.3
        assert reward < 1.0


# =========================================================================
# TestQualityCliff (ClawWork-inspired)
# =========================================================================


@pytest.mark.unit
class TestQualityCliff:
    """Test the quality cliff: low-quality outputs get negative reward."""

    def test_cliff_triggers_on_empty_failed_output(self):
        """Empty + failed output should be capped at failure ceiling (0.3).

        After quality cliff threshold was lowered from 0.35 to 0.20,
        empty+failed no longer falls below the cliff — the failure cap
        of 0.3 is above the 0.20 threshold, so we get 0.3 (not -0.1).
        """
        result = _make_result(output="", success=False)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do something")
        assert reward == 0.3, f"Expected 0.3 (failure cap, above cliff threshold), got {reward}"

    def test_cliff_triggers_on_garbage_output(self):
        """Very short, irrelevant, failed output is capped at failure ceiling.

        After quality cliff threshold was lowered from 0.35 to 0.20,
        garbage+failed is capped at 0.3 (failure cap) which is above
        the 0.20 cliff threshold, so the cliff does not trigger.
        """
        result = _make_result(output="x", success=False)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "analyze data")
        assert reward == 0.3, f"Expected 0.3 (failure cap, above cliff threshold), got {reward}"

    def test_empty_success_stays_above_cliff(self):
        """Empty output with success=True may stay above cliff due to
        non-content dimensions (no_errors=1.0, efficiency, relevance=0.5)."""
        result = _make_result(output="", success=True)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do something")
        # Non-content dimensions can keep score above 0.35 threshold
        assert (
            reward >= SwarmLearningPipeline.QUALITY_CLIFF_THRESHOLD
        ), f"Expected above cliff (non-content dims compensate), got {reward}"

    def test_cliff_does_not_trigger_on_good_output(self):
        """Substantial, relevant output should NOT trigger the cliff."""
        result = _make_result(
            output=(
                "## Analysis Results\n\n"
                "The data analysis shows strong growth patterns.\n"
                "Key findings include increased revenue and market share.\n"
                "In conclusion, the outlook is positive for the next quarter."
            ),
            success=True,
        )
        reward = SwarmLearningPipeline._compute_episode_reward(result, "analyze data")
        assert (
            reward > SwarmLearningPipeline.QUALITY_CLIFF_THRESHOLD
        ), f"Good output should be above cliff threshold, got {reward}"

    def test_cliff_threshold_is_configurable(self):
        """QUALITY_CLIFF_THRESHOLD is a class attribute that can be overridden."""
        original = SwarmLearningPipeline.QUALITY_CLIFF_THRESHOLD
        try:
            SwarmLearningPipeline.QUALITY_CLIFF_THRESHOLD = 0.0
            # With threshold at 0, even empty output should NOT get -0.1
            # (0 is the borderline — only strictly below triggers)
            result = _make_result(output="Some output here", success=True)
            reward = SwarmLearningPipeline._compute_episode_reward(result, "test")
            assert reward >= 0.0, f"Expected non-negative with 0 threshold, got {reward}"
        finally:
            SwarmLearningPipeline.QUALITY_CLIFF_THRESHOLD = original

    def test_failure_with_cliff_gives_capped_value(self):
        """Failed + low quality = capped at failure ceiling (0.3).

        After quality cliff threshold was lowered from 0.35 to 0.20,
        the failure cap of 0.3 is above the cliff threshold, so the
        cliff no longer overrides. We get 0.3 (the failure cap).
        """
        result = _make_result(output="", success=False)
        reward = SwarmLearningPipeline._compute_episode_reward(result, "do task")
        # success=False caps at 0.3, which is above the 0.20 cliff threshold
        assert reward == 0.3, f"Expected 0.3 for failed+empty (above cliff threshold), got {reward}"


# =========================================================================
# TestDomainWeights (ClawWork-inspired)
# =========================================================================


@pytest.mark.unit
class TestDomainWeights:
    """Test domain-specific reward weight profiles."""

    def test_get_domain_weights_coding(self):
        w = SwarmLearningPipeline._get_domain_weights("coding")
        assert (
            w["no_errors"] > w["substance"]
        ), "Coding should weight no_errors higher than substance"
        assert abs(sum(w.values()) - 1.0) < 0.01, f"Weights should sum to ~1.0: {sum(w.values())}"

    def test_get_domain_weights_research(self):
        w = SwarmLearningPipeline._get_domain_weights("research")
        assert (
            w["substance"] > w["efficiency"]
        ), "Research should weight substance higher than efficiency"
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_get_domain_weights_writing(self):
        w = SwarmLearningPipeline._get_domain_weights("writing")
        assert (
            w["structure"] > w["tool_use"]
        ), "Writing should weight structure higher than tool_use"
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_get_domain_weights_analysis(self):
        w = SwarmLearningPipeline._get_domain_weights("analysis")
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_get_domain_weights_unknown_uses_default(self):
        w = SwarmLearningPipeline._get_domain_weights("unknown_domain_xyz")
        assert w == SwarmLearningPipeline._DEFAULT_WEIGHTS

    def test_get_domain_weights_case_insensitive(self):
        """Domain matching should be case-insensitive."""
        w = SwarmLearningPipeline._get_domain_weights("Coding Task")
        assert w == SwarmLearningPipeline._DOMAIN_WEIGHTS["coding"]

    def test_domain_affects_reward(self):
        """Same output scored under different domains should differ."""
        result = _make_result(
            output=(
                "```python\ndef solve(): return 42\n```\n"
                "The code runs without errors and produces correct output."
            ),
            success=True,
            trajectory=[{"action": "tool_call", "tool_name": "run_code", "output": "42"}],
        )
        goal = "Write code to solve the problem"

        coding_reward = SwarmLearningPipeline._compute_episode_reward(
            result, goal, task_type="coding"
        )
        research_reward = SwarmLearningPipeline._compute_episode_reward(
            result, goal, task_type="research"
        )
        # Coding weights tool_use and no_errors more; research weights substance more.
        # The exact difference depends on the output, but they should differ.
        assert coding_reward != research_reward, (
            f"Domain should affect reward: coding={coding_reward:.3f}, "
            f"research={research_reward:.3f}"
        )

    def test_default_weights_sum_to_one(self):
        assert abs(sum(SwarmLearningPipeline._DEFAULT_WEIGHTS.values()) - 1.0) < 0.01

    def test_all_domain_weights_sum_to_one(self):
        for domain, w in SwarmLearningPipeline._DOMAIN_WEIGHTS.items():
            total = sum(w.values())
            assert (
                abs(total - 1.0) < 0.01
            ), f"Domain '{domain}' weights sum to {total}, expected ~1.0"


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
        with patch.object(SwarmLearningPipeline, "_init_components"):
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
        with open(path, "w") as f:
            json.dump(envelope, f)

        loaded = pipeline._load_versioned(path)
        assert loaded == {}, f"Incompatible major version should return empty, got {loaded}"

    def test_load_incompatible_with_migration(self, pipeline, tmp_path):
        """If a migration exists, it should be applied."""
        path = tmp_path / "migrate.json"
        envelope = {"schema_version": "1.0", "data": {"old_format": True}}
        with open(path, "w") as f:
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
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = pipeline._load_versioned(path)
        assert loaded == data

    def test_compatible_minor_version_loads(self, pipeline, tmp_path):
        """Same major version but different minor should load fine."""
        path = tmp_path / "minor.json"
        envelope = {"schema_version": "2.5", "data": {"updated": True}}
        with open(path, "w") as f:
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
