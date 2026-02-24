"""
Tests for advanced learning components.

Tests: UCB in SkillQTable, Reflexion, FewShotCurator, VoyagerSkillLib.

All tests are offline — no real LLM calls. DSPy modules are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# UCB in SkillQTable
# =============================================================================


class TestSkillQTableUCB:
    """Test UCB1 exploration in SkillQTable."""

    @pytest.mark.unit
    def test_cold_start_returns_all_skills(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()
        result = q.select("research", ["web-search", "calculator", "pdf-gen"])
        assert len(result) == 3
        assert set(result) == {"web-search", "calculator", "pdf-gen"}

    @pytest.mark.unit
    def test_unvisited_skill_explored_first(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()
        q.update("research", "web-search", 0.8)
        q.update("research", "web-search", 0.9)
        q.update("research", "calculator", 0.3)

        result = q.select("research", ["web-search", "calculator", "pdf-gen"])
        # pdf-gen has 0 visits → UCB = inf → should be first
        assert result[0] == "pdf-gen"

    @pytest.mark.unit
    def test_ucb_balances_exploitation_and_exploration(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(ucb_c=1.41)
        # Give web-search many visits and high Q
        for _ in range(20):
            q.update("research", "web-search", 0.9)
        # Give calculator few visits and moderate Q
        for _ in range(2):
            q.update("research", "calculator", 0.7)

        result = q.select("research", ["web-search", "calculator"])
        # calculator has fewer visits, UCB exploration term is larger
        # With c=1.41, sqrt(ln(22)/2) ≈ 1.24, so calculator UCB ≈ 0.7 + 1.41*1.24 ≈ 2.45
        # web-search UCB ≈ 0.9 + 1.41*sqrt(ln(22)/20) ≈ 0.9 + 0.55 ≈ 1.45
        assert result[0] == "calculator"

    @pytest.mark.unit
    def test_serialization_preserves_ucb_c(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(ucb_c=2.0)
        q.update("coding", "python", 0.8)
        data = q.to_dict()
        restored = SkillQTable.from_dict(data)
        assert restored.ucb_c == 2.0
        assert restored.get_q("coding", "python") == q.get_q("coding", "python")


# =============================================================================
# Reflexion
# =============================================================================


class TestReflexion:
    """Test Reflexion failure reflection."""

    @pytest.mark.unit
    def test_reflect_returns_none_when_no_lm(self):
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        r = Reflexion()
        r._init_attempts = 10
        result = r.reflect_on_failure("ep1", "agent", "goal", "output")
        assert result is None

    @pytest.mark.unit
    def test_reflect_with_mocked_dspy(self):
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        r = Reflexion()
        mock_module = MagicMock()
        mock_module.return_value = MagicMock(
            observation="The API endpoint returned 500",
            analysis="Missing error handling for null input",
            adjustment="Add input validation before API call",
        )
        r._reflect_module = mock_module
        r._lm = MagicMock()

        with (
            patch("dspy.context"),
            patch(
                "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
            ) as mock_get,
        ):
            mock_store = MagicMock()
            mock_get.return_value = mock_store

            result = r.reflect_on_failure(
                "ep1", "CodingSwarm", "Build REST API", "Error 500", "timeout"
            )

        assert result is not None
        assert "validation" in result["adjustment"].lower()
        mock_store.save_reflection.assert_called_once()

    @pytest.mark.unit
    def test_get_relevant_reflections(self):
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        r = Reflexion()

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.get_reflections.return_value = [
                MagicMock(
                    observation="Test failed",
                    adjustment="Add edge case handling",
                ),
            ]

            results = r.get_relevant_reflections("TestAgent", limit=3)

        assert len(results) == 1
        assert "edge case" in results[0]


# =============================================================================
# FewShotCurator
# =============================================================================


class TestFewShotCurator:
    """Test few-shot episode curation."""

    @pytest.mark.unit
    def test_returns_empty_on_no_episodes(self):
        from Jotty.core.intelligence.learning.advanced_learning import FewShotCurator

        curator = FewShotCurator()

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.query_episodes.return_value = []

            examples = curator.get_examples(domain="coding", n=5)

        assert examples == []

    @pytest.mark.unit
    def test_curates_high_quality_episodes(self):
        from Jotty.core.intelligence.learning.advanced_learning import FewShotCurator

        curator = FewShotCurator()

        mock_ep = MagicMock()
        mock_ep.quality = 0.9
        mock_ep.context = {"task": "Build REST API"}
        mock_ep.action = {"paradigm": "pipeline"}
        mock_ep.outcome = {"content": "Here is the code..."}
        mock_ep.task_type = "coding"

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.query_episodes.return_value = [mock_ep]

            examples = curator.get_examples(domain="coding", n=5, min_quality=0.7)

        assert len(examples) == 1


# =============================================================================
# VoyagerSkillLib
# =============================================================================


class TestVoyagerSkillLib:
    """Test Voyager-style skill library."""

    @pytest.mark.unit
    def test_skip_low_quality_episodes(self):
        from Jotty.core.intelligence.learning.advanced_learning import VoyagerSkillLib

        lib = VoyagerSkillLib()
        result = lib.extract_skill_pattern(
            "ep1", "coding", "api", "Build API", "pipeline", quality=0.5
        )
        assert result is None

    @pytest.mark.unit
    def test_extract_high_quality_pattern(self):
        from Jotty.core.intelligence.learning.advanced_learning import VoyagerSkillLib

        lib = VoyagerSkillLib()

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.get_patterns.return_value = []

            result = lib.extract_skill_pattern(
                "ep1", "coding", "api", "Build REST API", "pipeline + test", quality=0.9
            )

        assert result is not None
        assert result.startswith("skill_")
        mock_store.save_pattern.assert_called_once()

    @pytest.mark.unit
    def test_boosts_existing_pattern(self):
        from Jotty.core.intelligence.learning.advanced_learning import VoyagerSkillLib

        lib = VoyagerSkillLib()

        mock_pattern = MagicMock()
        mock_pattern.description = "Build REST API"
        mock_pattern.confidence = 0.8
        mock_pattern.evidence_count = 3
        mock_pattern.pattern_id = "skill_existing"

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            mock_store.get_patterns.return_value = [mock_pattern]

            result = lib.extract_skill_pattern(
                "ep2", "coding", "api", "Build REST API", "pipeline", quality=0.85
            )

        assert result == "skill_existing"
        assert abs(mock_pattern.confidence - 0.85) < 1e-9
        assert mock_pattern.evidence_count == 4
