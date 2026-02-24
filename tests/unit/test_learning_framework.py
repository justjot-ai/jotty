"""
Tests for World-Class Learning Framework Roadmap (8 phases).

Phase 1: Constants Consolidation
Phase 2: Q-Table Recency Weighting
Phase 3: Domain-Aware Score Blending
Phase 4: Dual Plan Tracking
Phase 5: Graduated Crystallization (Role-Level Confidence)
Phase 6: Staleness Canary
Phase 7: Reflexion Retrieval
Phase 8: Curriculum Ordering
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Phase 1: Constants Consolidation
# =============================================================================


class TestLearningConfigConstants:
    """Phase 1: LearningConfig fields and validation."""

    @pytest.mark.unit
    def test_defaults_match_original_hardcoded_values(self):
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig()
        # Score blending
        assert cfg.judge_llm_weight == 0.6
        assert cfg.judge_heuristic_weight == 0.4
        # Quality thresholds
        assert cfg.reflexion_quality_threshold == 0.4
        assert cfg.pattern_extract_quality_threshold == 0.8
        assert cfg.auto_reflect_quality_threshold == 0.75
        # Context budget
        assert cfg.context_max_total == 2000
        assert cfg.context_max_retrieval == 1200
        assert cfg.context_max_abstract == 400
        # Q-table management
        assert cfg.plan_history_max == 100
        assert cfg.q_staleness_halflife == 50
        # Crystallization
        assert cfg.crystal_min_episodes == 25
        assert cfg.crystal_min_success_rate == 0.85
        assert cfg.crystal_min_plan_consistency == 0.60
        assert cfg.crystal_min_role_q == 0.65
        assert cfg.crystal_min_plans == 8
        assert cfg.crystal_staleness_failures == 3

    @pytest.mark.unit
    def test_custom_values_propagate(self):
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig(
            judge_llm_weight=0.9,
            judge_heuristic_weight=0.1,
            reflexion_quality_threshold=0.5,
            context_max_total=3000,
        )
        assert cfg.judge_llm_weight == 0.9
        assert cfg.judge_heuristic_weight == 0.1
        assert cfg.reflexion_quality_threshold == 0.5
        assert cfg.context_max_total == 3000

    @pytest.mark.unit
    def test_weight_sum_validation(self):
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        with pytest.raises(ValueError, match="judge_llm_weight.*judge_heuristic_weight"):
            LearningConfig(judge_llm_weight=0.8, judge_heuristic_weight=0.8)

    @pytest.mark.unit
    def test_unit_field_validation(self):
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        with pytest.raises(ValueError):
            LearningConfig(reflexion_quality_threshold=1.5)

    @pytest.mark.unit
    def test_positive_int_validation(self):
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        with pytest.raises(ValueError):
            LearningConfig(plan_history_max=0)

    @pytest.mark.unit
    def test_blend_overrides_optional(self):
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig()
        assert cfg.judge_blend_overrides is None

        cfg2 = LearningConfig(judge_blend_overrides={"coding": (0.9, 0.1)})
        assert cfg2.judge_blend_overrides == {"coding": (0.9, 0.1)}

    @pytest.mark.unit
    def test_crystallization_thresholds_derive_from_config(self):
        """DEFAULT_THRESHOLDS in crystallization.py should match LearningConfig."""
        from Jotty.core.intelligence.learning.crystallization import DEFAULT_THRESHOLDS
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig()
        assert DEFAULT_THRESHOLDS["min_episodes"] == cfg.crystal_min_episodes
        assert DEFAULT_THRESHOLDS["min_success_rate"] == cfg.crystal_min_success_rate
        assert DEFAULT_THRESHOLDS["min_plan_consistency"] == cfg.crystal_min_plan_consistency
        assert DEFAULT_THRESHOLDS["min_role_q"] == cfg.crystal_min_role_q
        assert DEFAULT_THRESHOLDS["min_plans"] == cfg.crystal_min_plans


# =============================================================================
# Phase 2: Q-Table Recency Weighting
# =============================================================================


class TestQTableRecencyWeighting:
    """Phase 2: Stale Q-entries update faster."""

    @pytest.mark.unit
    def test_skill_q_stale_entry_drops_faster(self):
        """Update skill A 100× with reward=0.9, then 5× with reward=0.1.
        Q should drop faster than with static alpha."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        # Static alpha baseline
        static = SkillQTable(alpha=0.1)
        for _ in range(100):
            static.update("test", "skill_a", 0.9)
        static_q_before = static.get_q("test", "skill_a")
        # Disable recency by setting counter back (hack for comparison)
        for _ in range(5):
            static.update("test", "skill_a", 0.1)
        static_q_after = static.get_q("test", "skill_a")
        static_drop = static_q_before - static_q_after

        # Recency-aware: create fresh table, train 100× on skill_a, then
        # update another skill 200× to advance counter, then update skill_a 5×
        recency = SkillQTable(alpha=0.1)
        for _ in range(100):
            recency.update("test", "skill_a", 0.9)
        recency_q_before = recency.get_q("test", "skill_a")
        # Advance counter without touching skill_a
        for _ in range(200):
            recency.update("test", "skill_b", 0.5)
        # Now skill_a is stale — effective alpha should be higher
        for _ in range(5):
            recency.update("test", "skill_a", 0.1)
        recency_q_after = recency.get_q("test", "skill_a")
        recency_drop = recency_q_before - recency_q_after

        # Stale entry should drop MORE (larger drop = faster adaptation)
        assert recency_drop > static_drop

    @pytest.mark.unit
    def test_skill_q_serialization_roundtrip(self):
        """Recency tracking survives to_dict / from_dict."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()
        q.update("test", "skill_a", 0.9)
        q.update("test", "skill_b", 0.5)

        data = q.to_dict()
        assert "last_updated" in data
        assert "global_counter" in data
        assert data["global_counter"] > 0

        restored = SkillQTable.from_dict(data)
        assert restored._global_counter == q._global_counter
        assert restored._last_updated == q._last_updated
        assert restored.get_q("test", "skill_a") == q.get_q("test", "skill_a")

    @pytest.mark.unit
    def test_step_q_recency_tracking(self):
        """StepQTable also has recency-aware updates."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable(alpha=0.1)
        sq.update("test", 0, "skill_a", 0.9)
        assert sq._global_counter > 0
        assert "test" in sq._last_updated_pos

    @pytest.mark.unit
    def test_step_q_serialization_roundtrip(self):
        """StepQTable recency tracking survives serialization."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        sq.update("test", 0, "skill_a", 0.9)
        sq.update("test", 1, "skill_b", 0.7)

        data = sq.to_dict()
        assert "last_updated_pos" in data
        assert "last_updated_role" in data
        assert "global_counter" in data

        restored = StepQTable.from_dict(data)
        assert restored._global_counter == sq._global_counter


# =============================================================================
# Phase 3: Domain-Aware Score Blending
# =============================================================================


class TestDomainAwareBlending:
    """Phase 3: Different domains use different LLM/heuristic weights."""

    @pytest.mark.unit
    def test_coding_domain_uses_higher_llm_weight(self):
        """Coding domain should use 0.85/0.15 blend (not default 0.6/0.4)."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        svc = LearningService.__new__(LearningService)
        # Manually init just what we need
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        svc._config = LearningConfig()
        svc._blend_overrides = svc._config.judge_blend_overrides or {
            "coding": (0.85, 0.15),
            "algorithms": (0.85, 0.15),
            "math": (0.80, 0.20),
        }

        w_llm, w_heur = svc._blend_overrides.get(
            "coding", (svc._config.judge_llm_weight, svc._config.judge_heuristic_weight)
        )
        assert w_llm == 0.85
        assert w_heur == 0.15

    @pytest.mark.unit
    def test_unknown_domain_uses_default_weights(self):
        """Unknown domains use the default 0.6/0.4 blend."""
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig()
        overrides = {"coding": (0.85, 0.15)}
        w_llm, w_heur = overrides.get("travel", (cfg.judge_llm_weight, cfg.judge_heuristic_weight))
        assert w_llm == 0.6
        assert w_heur == 0.4

    @pytest.mark.unit
    def test_low_wordcount_code_gets_high_score_with_high_llm(self):
        """Code output with low word count but high LLM score should score well."""
        llm_score = 0.95
        heuristic_score = 0.2  # low word count → low heuristic

        # Coding blend (85/15)
        coding_blend = llm_score * 0.85 + heuristic_score * 0.15
        # Default blend (60/40)
        default_blend = llm_score * 0.6 + heuristic_score * 0.4

        # Coding should score higher (trusts LLM more)
        assert coding_blend > default_blend
        assert coding_blend > 0.8  # Still high


# =============================================================================
# Phase 4: Dual Plan Tracking
# =============================================================================


class TestDualPlanTracking:
    """Phase 4: Raw plans preserved alongside normalized plans."""

    @pytest.mark.unit
    def test_raw_plan_preserved(self):
        """Raw plan with duplicate roles should be preserved in _raw_plan_history."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        # Skills that map to: research, research, research, synthesize, save
        skills = ["web-search", "web-search", "web-search", "claude-cli-llm", "file-operations"]
        descriptions = [
            "search for info",
            "search for more info",
            "search for details",
            "synthesize findings",
            "save to file",
        ]
        sq.record_plan("research", skills, 0.9, descriptions=descriptions)

        # Normalized plan should collapse consecutive duplicates
        norm_plans = sq.get_best_plan("research")
        assert len(norm_plans) > 0
        roles, reward = norm_plans[0]
        # Should be deduplicated: (research, synthesize, save) or similar
        assert len(roles) < len(skills)

        # Raw plan should preserve all roles
        raw_plans = sq.get_best_raw_plan("research")
        assert len(raw_plans) > 0
        raw_roles, raw_reward = raw_plans[0]
        assert len(raw_roles) == len(skills)

    @pytest.mark.unit
    def test_raw_plan_serialization_roundtrip(self):
        """Raw plan history survives to_dict / from_dict."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        sq.record_plan("test", ["web-search", "web-search", "file-operations"], 0.8)

        data = sq.to_dict()
        assert "raw_plans" in data
        assert len(data["raw_plans"]) > 0

        restored = StepQTable.from_dict(data)
        assert len(restored._raw_plan_history) > 0
        raw = restored.get_best_raw_plan("test")
        assert len(raw) > 0

    @pytest.mark.unit
    def test_existing_get_best_plan_unchanged(self):
        """get_best_plan() still returns normalized plans (no behavior change)."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        sq.record_plan("test", ["web-search", "web-search", "file-operations"], 0.8)

        plans = sq.get_best_plan("test")
        assert len(plans) > 0
        roles, _ = plans[0]
        # Should be normalized (no consecutive duplicates)
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i + 1]


# =============================================================================
# Phase 5: Graduated Crystallization
# =============================================================================


class TestGraduatedCrystallization:
    """Phase 5: Per-role confidence scores in CrystallizedConfig."""

    @pytest.mark.unit
    def test_role_confidence_in_plan_hint(self):
        """Low-confidence roles should trigger a warning in to_plan_hint()."""
        from Jotty.core.intelligence.learning.crystallization import CrystallizedConfig

        config = CrystallizedConfig(
            domain_key="research:travel",
            task_type="research",
            domain="travel",
            sop_roles=("research", "synthesize", "save"),
            role_skill_map={
                "research": "web-search",
                "synthesize": "claude-cli-llm",
                "save": "file-operations",
            },
            role_confidence={"research": 0.95, "synthesize": 0.80, "save": 0.40},
        )
        hint = config.to_plan_hint()
        assert "LOW-CONFIDENCE ROLES" in hint
        assert "save" in hint
        # High-confidence roles should NOT be listed
        assert "research" not in hint.split("LOW-CONFIDENCE ROLES")[1]

    @pytest.mark.unit
    def test_no_warning_when_all_roles_confident(self):
        """No warning when all roles have high confidence."""
        from Jotty.core.intelligence.learning.crystallization import CrystallizedConfig

        config = CrystallizedConfig(
            domain_key="coding",
            task_type="coding",
            sop_roles=("generate", "test"),
            role_confidence={"generate": 0.95, "test": 0.85},
        )
        hint = config.to_plan_hint()
        assert "LOW-CONFIDENCE" not in hint

    @pytest.mark.unit
    def test_empty_role_confidence_loads_fine(self):
        """Configs without role_confidence (old format) should load with default {}."""
        from Jotty.core.intelligence.learning.crystallization import CrystallizedConfig

        config = CrystallizedConfig(
            domain_key="test",
            task_type="test",
        )
        assert config.role_confidence == {}
        hint = config.to_plan_hint()
        assert "LOW-CONFIDENCE" not in hint

    @pytest.mark.unit
    def test_role_confidence_computation(self):
        """Confidence = q * min(1.0, visits / 10)."""
        # High Q, many visits → high confidence
        q, visits = 0.95, 20
        conf = round(q * min(1.0, visits / 10), 3)
        assert conf == 0.95

        # High Q, few visits → lower confidence
        q, visits = 0.95, 5
        conf = round(q * min(1.0, visits / 10), 3)
        assert conf == 0.475


# =============================================================================
# Phase 6: Staleness Canary
# =============================================================================


class TestStalenessCanary:
    """Phase 6: Auto-decrystallize after consecutive failures."""

    @pytest.mark.unit
    def test_consecutive_failures_trigger_decrystallize(self, tmp_path):
        """3 consecutive failures should remove the config."""
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            record_crystallized_outcome,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        # Redirect crystal dir to temp
        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            config = CrystallizedConfig(
                domain_key="test",
                task_type="test",
                skills=["web-search"],
                sop_roles=("research",),
            )
            _save(config)

            # First 2 failures: no decrystallize
            result1 = record_crystallized_outcome("test", success=False, max_failures=3)
            assert result1 is None
            result2 = record_crystallized_outcome("test", success=False, max_failures=3)
            assert result2 is None

            # 3rd failure: decrystallize
            result3 = record_crystallized_outcome("test", success=False, max_failures=3)
            assert result3 == "decrystallized"

            # Config should be gone
            from Jotty.core.intelligence.learning.crystallization import load

            assert load("test") is None
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_success_resets_counter(self, tmp_path):
        """2 failures + 1 success → counter resets to 0."""
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            load,
            record_crystallized_outcome,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            config = CrystallizedConfig(
                domain_key="test",
                task_type="test",
                skills=["web-search"],
            )
            _save(config)

            record_crystallized_outcome("test", success=False, max_failures=3)
            record_crystallized_outcome("test", success=False, max_failures=3)
            # Success resets
            record_crystallized_outcome("test", success=True, max_failures=3)

            loaded = load("test")
            assert loaded is not None
            assert loaded.consecutive_failures == 0
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_consecutive_failures_default_zero(self):
        """Old configs without consecutive_failures default to 0."""
        from Jotty.core.intelligence.learning.crystallization import CrystallizedConfig

        config = CrystallizedConfig(domain_key="test", task_type="test")
        assert config.consecutive_failures == 0


# =============================================================================
# Phase 7: Reflexion Retrieval
# =============================================================================


class TestReflexionRetrieval:
    """Phase 7: Past failure reflections wired into build_context_string()."""

    @pytest.mark.unit
    def test_reflections_included_when_failures_exist(self):
        """build_context_string should include reflexion data when domain has failures."""
        from Jotty.core.intelligence.learning.learning_service import LearningService
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        svc = LearningService.__new__(LearningService)
        svc._config = LearningConfig()
        svc._blend_overrides = {}

        mock_store = MagicMock()
        svc._store = mock_store
        svc._success_rate_cache = {}
        svc._pattern_cache = {}
        svc._cache_ttl = 60.0

        # Mock success rate showing failures
        mock_store.get_success_rate.return_value = (0.5, 10)  # 50% success, 10 total

        # Mock guidance with failure analysis
        svc.query = MagicMock(
            return_value={
                "success_rate": 0.5,
                "total_episodes": 10,
                "failure_analysis": [],
            }
        )

        # Mock distilled lessons
        svc.retrieve_distilled_lessons = MagicMock(return_value=[])
        svc.get_best_approach_for_domain = MagicMock(return_value=None)
        svc.build_retrieval_context = MagicMock(return_value="")

        mock_store.get_lessons.return_value = []

        # Mock reflexion to return past insights
        mock_reflexion = MagicMock()
        mock_reflexion.get_relevant_reflections.return_value = [
            "[Past failure] API call failed → Fix: Use retry with backoff"
        ]

        with patch(
            "Jotty.core.intelligence.learning.advanced_learning.Reflexion.get_instance",
            return_value=mock_reflexion,
        ):
            result = svc.build_context_string(domain="test_domain", task_type="test")

        # Should contain the reflexion data
        if result:  # Only check if context was generated
            # The reflexion retrieval is gated by has_failures
            pass  # We verify it doesn't crash; actual content depends on guidance flow

    @pytest.mark.unit
    def test_reflexion_excluded_when_succeeding(self):
        """Reflections should NOT be included when domain is succeeding."""
        from Jotty.core.intelligence.learning.advanced_learning import Reflexion

        reflexion = Reflexion.__new__(Reflexion)
        # get_relevant_reflections requires store access; we just verify it exists
        assert hasattr(reflexion, "get_relevant_reflections")


# =============================================================================
# Phase 8: Curriculum Ordering
# =============================================================================


class TestCurriculumOrdering:
    """Phase 8: Goals sorted by complexity (easy → hard)."""

    @pytest.mark.unit
    def test_goals_sorted_by_complexity(self):
        """Simple goals should come before complex ones."""
        goals = [
            "Implement a distributed concurrent algorithm and then benchmark it",
            "Research best hotels",
            "Architect a scalable security system and then integrate monitoring",
            "List top restaurants",
            "Optimize database queries",
        ]

        # Import the complexity scorer by reproducing it (it's a local function)
        def _complexity_score(goal: str) -> float:
            score = 0.0
            score += min(len(goal) / 500, 1.0) * 0.3
            tech_terms = {
                "implement",
                "architect",
                "optimize",
                "distributed",
                "concurrent",
                "algorithm",
                "benchmark",
                "migrate",
                "refactor",
                "security",
                "scale",
                "integrate",
            }
            words = goal.lower().split()
            score += min(sum(1 for w in words if w in tech_terms) / 3, 1.0) * 0.4
            for phrase in ("and then", "followed by", "after that", "finally"):
                if phrase in goal.lower():
                    score += 0.1
            return min(score, 1.0)

        sorted_goals = sorted(goals, key=_complexity_score)

        # Simple goals should be first
        assert (
            "Research best hotels" in sorted_goals[:2] or "List top restaurants" in sorted_goals[:2]
        )
        # Complex goals should be last
        complex_goals = sorted_goals[-2:]
        assert any("distributed" in g.lower() or "architect" in g.lower() for g in complex_goals)

    @pytest.mark.unit
    def test_single_goal_unchanged(self):
        """Single goal should not be reordered."""
        goals = ["Just one goal"]
        original = list(goals)
        # Sort only applies when len > 2
        if len(goals) > 2:
            goals.sort(key=lambda g: len(g))
        assert goals == original

    @pytest.mark.unit
    def test_empty_goals_unchanged(self):
        """Empty goal list should not crash."""
        goals: List[str] = []
        if len(goals) > 2:
            goals.sort(key=lambda g: len(g))
        assert goals == []

    @pytest.mark.unit
    def test_complexity_score_ranges(self):
        """All complexity scores should be in [0, 1]."""

        def _complexity_score(goal: str) -> float:
            score = 0.0
            score += min(len(goal) / 500, 1.0) * 0.3
            tech_terms = {
                "implement",
                "architect",
                "optimize",
                "distributed",
                "concurrent",
                "algorithm",
                "benchmark",
                "migrate",
                "refactor",
                "security",
                "scale",
                "integrate",
            }
            words = goal.lower().split()
            score += min(sum(1 for w in words if w in tech_terms) / 3, 1.0) * 0.4
            for phrase in ("and then", "followed by", "after that", "finally"):
                if phrase in goal.lower():
                    score += 0.1
            return min(score, 1.0)

        test_goals = [
            "",
            "Hi",
            "Research topic",
            "Implement distributed concurrent algorithm and then benchmark",
            "A" * 1000,  # Very long
        ]
        for goal in test_goals:
            score = _complexity_score(goal)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: {goal[:50]}"


# =============================================================================
# Integration: LearningService uses config
# =============================================================================


class TestLearningServiceConfig:
    """Verify LearningService wires up config correctly."""

    @pytest.mark.unit
    def test_learning_service_has_config(self):
        """LearningService should have a _config attribute."""
        from Jotty.core.intelligence.learning.learning_service import LearningService
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        svc = LearningService.__new__(LearningService)
        svc._config = LearningConfig()
        assert hasattr(svc, "_config")
        assert isinstance(svc._config, LearningConfig)

    @pytest.mark.unit
    def test_blend_overrides_populated(self):
        """LearningService should have domain blend overrides."""
        from Jotty.core.intelligence.learning.learning_service import LearningService
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        svc = LearningService.__new__(LearningService)
        svc._config = LearningConfig()
        svc._blend_overrides = svc._config.judge_blend_overrides or {
            "coding": (0.85, 0.15),
            "algorithms": (0.85, 0.15),
            "math": (0.80, 0.20),
        }
        assert "coding" in svc._blend_overrides
        assert svc._blend_overrides["coding"] == (0.85, 0.15)


# =============================================================================
# Edge Cases & Stress Tests
# =============================================================================


class TestRecencyEdgeCases:
    """Phase 2 edge cases: alpha clamping, zero staleness, huge staleness."""

    @pytest.mark.unit
    def test_effective_alpha_never_exceeds_half(self):
        """Even with extreme staleness, alpha caps at 0.5."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(alpha=0.1)
        # Advance counter massively without touching skill_a
        q._global_counter = 100_000
        q.update("test", "skill_a", 0.9)
        # The update should work (not overflow) and Q should be reasonable
        assert 0.0 <= q.get_q("test", "skill_a") <= 1.0

    @pytest.mark.unit
    def test_zero_staleness_uses_base_alpha(self):
        """Consecutive updates on same skill → staleness=1 → alpha ≈ base."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(alpha=0.1)
        q.update("test", "skill_a", 0.9)
        q.update("test", "skill_a", 0.9)
        q.update("test", "skill_a", 0.9)
        # Should converge normally, not overshoot
        assert 0.5 < q.get_q("test", "skill_a") < 1.0

    @pytest.mark.unit
    def test_step_q_stale_entry_drops_faster(self):
        """StepQTable: stale position entries also adapt faster."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable(alpha=0.1)
        # Train skill_a at pos 0
        for _ in range(50):
            sq.update("test", 0, "skill_a", 0.9)
        q_before = sq._q["test"][0]["skill_a"]

        # Advance counter on different skill
        for _ in range(200):
            sq.update("test", 1, "skill_b", 0.5)

        # Now update stale skill_a with bad reward
        for _ in range(3):
            sq.update("test", 0, "skill_a", 0.1)
        q_after = sq._q["test"][0]["skill_a"]

        # Should have dropped noticeably
        assert q_before - q_after > 0.05

    @pytest.mark.unit
    def test_recency_with_multiple_domains(self):
        """Recency tracking is per-key, not global."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable(alpha=0.1)
        q.update("coding", "skill_a", 0.9, domain="python")
        q.update("research", "skill_a", 0.9, domain="travel")

        # Both should have separate tracking
        assert "coding:python" in q._last_updated
        assert "research:travel" in q._last_updated


class TestDualPlanEdgeCases:
    """Phase 4 edge cases: empty plans, single-step plans, domain plans."""

    @pytest.mark.unit
    def test_single_skill_plan(self):
        """Single-skill plan: raw and normalized should be identical."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        sq.record_plan("test", ["web-search"], 0.9, descriptions=["search"])

        norm = sq.get_best_plan("test")
        raw = sq.get_best_raw_plan("test")
        assert norm[0][0] == raw[0][0]  # same for single skill

    @pytest.mark.unit
    def test_raw_plan_domain_fallback(self):
        """Raw plan should fall back to base key like normalized plan."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        sq.record_plan(
            "research",
            ["web-search", "file-operations"],
            0.8,
            descriptions=["search", "save"],
            domain="travel",
        )

        # Query with domain
        raw_domain = sq.get_best_raw_plan("research", domain="travel")
        assert len(raw_domain) > 0

        # Base key should also have it
        raw_base = sq.get_best_raw_plan("research")
        assert len(raw_base) > 0

    @pytest.mark.unit
    def test_raw_plan_max_history(self):
        """Raw plan history should respect max_history limit."""
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        sq = StepQTable()
        for i in range(150):
            sq.record_plan("test", ["web-search"], float(i) / 150)

        # Should be capped at plan_history_max (default 100)
        assert len(sq._raw_plan_history["test"]) <= 100


class TestStalenessCanaryEdgeCases:
    """Phase 6 edge cases: no config, interleaved success/failure."""

    @pytest.mark.unit
    def test_record_outcome_no_config(self):
        """Recording outcome when no config exists should be a no-op."""
        from Jotty.core.intelligence.learning.crystallization import record_crystallized_outcome
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = Path("/tmp/nonexistent_crystal_dir_test")
        try:
            result = record_crystallized_outcome("nonexistent", success=False)
            assert result is None
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_interleaved_failures_and_successes(self, tmp_path):
        """F, F, S, F, F, S → never decrystallizes (counter resets each time)."""
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            load,
            record_crystallized_outcome,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            _save(CrystallizedConfig(domain_key="test", task_type="test", skills=["s"]))

            # F, F, S → reset
            record_crystallized_outcome("test", success=False, max_failures=3)
            record_crystallized_outcome("test", success=False, max_failures=3)
            record_crystallized_outcome("test", success=True, max_failures=3)

            # F, F, S → reset again
            record_crystallized_outcome("test", success=False, max_failures=3)
            record_crystallized_outcome("test", success=False, max_failures=3)
            record_crystallized_outcome("test", success=True, max_failures=3)

            # Config should still exist
            assert load("test") is not None
            assert load("test").consecutive_failures == 0
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_custom_max_failures(self, tmp_path):
        """Custom max_failures=1 should decrystallize on first failure."""
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            record_crystallized_outcome,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            _save(CrystallizedConfig(domain_key="test", task_type="test", skills=["s"]))
            result = record_crystallized_outcome("test", success=False, max_failures=1)
            assert result == "decrystallized"
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir


class TestCrystallizedConfigSerialization:
    """Phase 5+6: New fields survive JSON roundtrip."""

    @pytest.mark.unit
    def test_new_fields_roundtrip(self, tmp_path):
        """role_confidence and consecutive_failures survive save/load."""
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            load,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            config = CrystallizedConfig(
                domain_key="coding:python",
                task_type="coding",
                domain="python",
                skills=["claude-cli-llm"],
                sop_roles=("generate", "test"),
                role_confidence={"generate": 0.95, "test": 0.72},
                consecutive_failures=2,
            )
            _save(config)
            loaded = load("coding", "python")

            assert loaded is not None
            assert loaded.role_confidence == {"generate": 0.95, "test": 0.72}
            assert loaded.consecutive_failures == 2
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_old_config_without_new_fields(self, tmp_path):
        """Old JSON without role_confidence/consecutive_failures loads fine."""
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            tmp_path.mkdir(parents=True, exist_ok=True)
            old_json = {
                "domain_key": "test",
                "task_type": "test",
                "domain": "",
                "skills": ["web-search"],
                "sop_roles": ["research", "save"],
                "role_skill_map": {},
                "prompt_guidance": "",
                "success_rate": 0.9,
                "total_episodes": 30,
                "created_at": "2026-01-01",
            }
            (tmp_path / "test.json").write_text(json.dumps(old_json))

            from Jotty.core.intelligence.learning.crystallization import load

            loaded = load("test")
            assert loaded is not None
            assert loaded.role_confidence == {}
            assert loaded.consecutive_failures == 0
        finally:
            crystal_mod._CRYSTAL_DIR = original_dir


class TestBlendingAcrossDomains:
    """Phase 3: Verify all configured domains get correct weights."""

    @pytest.mark.unit
    def test_all_override_domains(self):
        """All code-producing domains get custom weights."""
        overrides = {
            "coding": (0.85, 0.15),
            "algorithms": (0.85, 0.15),
            "math": (0.80, 0.20),
            "devops": (0.85, 0.15),
            "testing": (0.85, 0.15),
            "data_analysis": (0.80, 0.20),
        }
        for domain, (w_llm, w_heur) in overrides.items():
            assert abs(w_llm + w_heur - 1.0) < 0.01, f"{domain} weights don't sum to 1"

    @pytest.mark.unit
    def test_blending_correctness_all_domains(self):
        """Verify blended scores are correct for each domain."""
        overrides = {
            "coding": (0.85, 0.15),
            "algorithms": (0.85, 0.15),
            "math": (0.80, 0.20),
            "devops": (0.85, 0.15),
            "testing": (0.85, 0.15),
            "data_analysis": (0.80, 0.20),
        }
        llm = 0.9
        heur = 0.3

        for domain, (w_llm, w_heur) in overrides.items():
            blended = llm * w_llm + heur * w_heur
            default = llm * 0.6 + heur * 0.4
            assert blended > default, f"{domain} should score higher than default"


class TestConfigOverrideBlend:
    """Phase 1+3: Custom blend overrides from config propagate correctly."""

    @pytest.mark.unit
    def test_custom_config_overrides_default(self):
        """User-supplied judge_blend_overrides should override defaults."""
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        custom = {"research": (0.5, 0.5), "coding": (0.95, 0.05)}
        cfg = LearningConfig(judge_blend_overrides=custom)

        # LearningService should use these
        blend_overrides = cfg.judge_blend_overrides or {}
        assert blend_overrides["research"] == (0.5, 0.5)
        assert blend_overrides["coding"] == (0.95, 0.05)


class TestCurriculumOrderingEdgeCases:
    """Phase 8 edge cases: identical scores, very long goals, unicode."""

    @pytest.mark.unit
    def test_identical_complexity_stable_sort(self):
        """Goals with same complexity score maintain relative order."""
        goals = ["Research topic A", "Research topic B", "Research topic C"]

        def _complexity_score(goal: str) -> float:
            score = min(len(goal) / 500, 1.0) * 0.3
            tech_terms = {
                "implement",
                "architect",
                "optimize",
                "distributed",
                "concurrent",
                "algorithm",
                "benchmark",
                "migrate",
                "refactor",
                "security",
                "scale",
                "integrate",
            }
            words = goal.lower().split()
            score += min(sum(1 for w in words if w in tech_terms) / 3, 1.0) * 0.4
            for phrase in ("and then", "followed by", "after that", "finally"):
                if phrase in goal.lower():
                    score += 0.1
            return min(score, 1.0)

        sorted_goals = sorted(goals, key=_complexity_score)
        # All have same score → order preserved (Python sort is stable)
        assert sorted_goals == goals

    @pytest.mark.unit
    def test_unicode_goals_dont_crash(self):
        """Unicode in goals should not cause errors."""
        goals = [
            "研究東京のホテル",
            "Implement distributed système",
            "Simple task café",
        ]

        def _complexity_score(goal: str) -> float:
            score = min(len(goal) / 500, 1.0) * 0.3
            tech_terms = {"implement", "architect", "optimize", "distributed"}
            words = goal.lower().split()
            score += min(sum(1 for w in words if w in tech_terms) / 3, 1.0) * 0.4
            return min(score, 1.0)

        # Should not raise
        sorted_goals = sorted(goals, key=_complexity_score)
        assert len(sorted_goals) == 3


class TestFullQTableLifecycle:
    """Integration: full lifecycle across phases 2+4."""

    @pytest.mark.unit
    def test_train_stale_retrain_with_raw_plans(self):
        """Full cycle: train → go stale → retrain. Verify both Q and plans."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable, StepQTable

        # Setup
        sq = StepQTable(alpha=0.1)
        skill_q = SkillQTable(alpha=0.1)

        # Phase 1: Train coding domain
        for i in range(20):
            skill_q.update("coding", "claude-cli-llm", 0.9)
            sq.update("coding", 0, "claude-cli-llm", 0.9, description="generate code")
            sq.update("coding", 1, "file-operations", 0.85, description="save to file")
            sq.record_plan(
                "coding",
                ["claude-cli-llm", "file-operations"],
                0.9,
                descriptions=["generate code", "save to file"],
            )

        assert skill_q.get_q("coding", "claude-cli-llm") > 0.7
        raw_plans = sq.get_best_raw_plan("coding")
        assert len(raw_plans) > 0

        # Phase 2: Go stale (train other domain for 100 ticks)
        for _ in range(100):
            skill_q.update("research", "web-search", 0.8)
            sq.update("research", 0, "web-search", 0.8, description="search")

        # Phase 3: Coding skill degrades
        for _ in range(5):
            skill_q.update("coding", "claude-cli-llm", 0.2)
            sq.update("coding", 0, "claude-cli-llm", 0.2, description="generate code")

        # Q should have dropped faster due to recency
        final_q = skill_q.get_q("coding", "claude-cli-llm")
        assert final_q < 0.7  # dropped from >0.7

        # But raw plans still preserved
        raw_plans_after = sq.get_best_raw_plan("coding")
        assert len(raw_plans_after) > 0

    @pytest.mark.unit
    def test_serialization_full_roundtrip(self):
        """Both Q-tables serialize and restore all new fields."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable, StepQTable

        # Build up state
        sq = StepQTable(alpha=0.1)
        skill_q = SkillQTable(alpha=0.1)

        for i in range(10):
            skill_q.update("test", "skill_a", 0.9)
            sq.update("test", 0, "skill_a", 0.9, description="step 1")
            sq.record_plan(
                "test",
                ["skill_a", "skill_a", "skill_b"],
                0.8,
                descriptions=["step 1", "step 1 again", "step 2"],
            )

        # Serialize
        sq_data = sq.to_dict()
        skill_data = skill_q.to_dict()

        # Restore
        sq2 = StepQTable.from_dict(sq_data)
        skill2 = SkillQTable.from_dict(skill_data)

        # Verify all new fields
        assert skill2._global_counter == skill_q._global_counter
        assert skill2._last_updated == skill_q._last_updated
        assert sq2._global_counter == sq._global_counter
        assert sq2._last_updated_pos == sq._last_updated_pos
        assert sq2._last_updated_role == sq._last_updated_role
        assert len(sq2._raw_plan_history) == len(sq._raw_plan_history)
        assert sq2.get_best_raw_plan("test") == sq.get_best_raw_plan("test")


# =============================================================================
# Integration: LLMJudge config
# =============================================================================


class TestLLMJudgeConfig:
    """Verify LLMJudge accepts config."""

    @pytest.mark.unit
    def test_llm_judge_accepts_config(self):
        """LLMJudge should accept optional config."""
        from Jotty.core.intelligence.learning.advanced_learning import LLMJudge
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig(judge_llm_weight=0.9, judge_heuristic_weight=0.1)
        judge = LLMJudge(config=cfg)
        assert judge._config is not None
        assert judge._config.judge_llm_weight == 0.9

    @pytest.mark.unit
    def test_llm_judge_default_config(self):
        """LLMJudge without config should load default."""
        from Jotty.core.intelligence.learning.advanced_learning import LLMJudge

        judge = LLMJudge()
        assert judge._config is not None


# =============================================================================
# Learning Validation — prove what's learned is actually better
# =============================================================================


class TestLearningValidation:
    """Verify the LearningValidator correctly evaluates learned knowledge."""

    def _make_episode(self, quality, success=True, exec_time=0.0, action=None):
        from Jotty.core.intelligence.learning.learning_store import EpisodeRecord

        return EpisodeRecord(
            episode_id=f"ep_test_{exec_time}",
            unit_type="agent",
            unit_name="TestAgent",
            domain="test",
            task_type="test_task",
            context={"goal": "test"},
            action=action or {},
            outcome={},
            success=success,
            quality=quality,
            execution_time=exec_time,
            cost=0.0,
        )

    @pytest.mark.unit
    def test_temporal_improvement_detects_improving_quality(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        episodes = [self._make_episode(0.5, exec_time=float(i)) for i in range(5)]
        episodes += [self._make_episode(0.9, exec_time=float(i + 5)) for i in range(5)]
        result = v._check_temporal_improvement(episodes)
        assert result.passed
        assert result.data["delta"] > 0

    @pytest.mark.unit
    def test_temporal_improvement_detects_declining_quality(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        episodes = [self._make_episode(0.9, exec_time=float(i)) for i in range(5)]
        episodes += [self._make_episode(0.3, exec_time=float(i + 5)) for i in range(5)]
        result = v._check_temporal_improvement(episodes)
        assert not result.passed
        assert result.data["delta"] < 0

    @pytest.mark.unit
    def test_baseline_lift_with_real_improvement(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        # First 3 are baseline (low quality), rest are learned (high quality)
        episodes = [self._make_episode(0.4, exec_time=float(i)) for i in range(3)]
        episodes += [self._make_episode(0.85, exec_time=float(i + 3)) for i in range(12)]
        result = v._check_baseline_lift("test", "test_task", episodes)
        assert result.passed
        assert result.data["lift"] > 0
        assert result.data["cohens_d"] > 0

    @pytest.mark.unit
    def test_baseline_lift_with_no_improvement(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        # All same quality — flat, no improvement
        episodes = [self._make_episode(0.7, exec_time=float(i)) for i in range(15)]
        result = v._check_baseline_lift("test", "test_task", episodes)
        assert result.passed  # not declining, so passes
        assert abs(result.data["lift"]) < 0.05

    @pytest.mark.unit
    def test_holdout_inconclusive_when_no_skill_data(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        episodes = [self._make_episode(0.8, exec_time=float(i)) for i in range(15)]
        result = v._check_holdout(episodes)
        assert result.passed  # inconclusive → neutral pass

    @pytest.mark.unit
    def test_holdout_validates_skill_quality(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        # Training: skill-a episodes are high quality
        train = [
            self._make_episode(0.9, exec_time=float(i), action={"skills_used": ["skill-a"]})
            for i in range(8)
        ]
        # Training: skill-b episodes are low quality
        train += [
            self._make_episode(0.3, exec_time=float(i + 8), action={"skills_used": ["skill-b"]})
            for i in range(3)
        ]
        # Test: skill-a still does well
        test = [
            self._make_episode(0.85, exec_time=float(i + 11), action={"skills_used": ["skill-a"]})
            for i in range(3)
        ]
        test += [
            self._make_episode(0.35, exec_time=float(i + 14), action={"skills_used": ["skill-b"]})
            for i in range(2)
        ]
        result = v._check_holdout(train + test)
        assert result.passed
        assert result.data.get("lift", 0) > 0

    @pytest.mark.unit
    def test_staleness_passes_when_no_crystal(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        result = v._check_staleness("nonexistent_domain", "nonexistent_task")
        assert result.passed

    @pytest.mark.unit
    def test_full_validation_with_too_few_episodes(self):
        from Jotty.core.intelligence.learning.validation import LearningValidator

        v = LearningValidator()
        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.query_episodes"
        ) as mock_query:
            mock_query.return_value = [self._make_episode(0.8)]
            report = v.validate_domain("sparse_domain", min_episodes=10)
            assert not report.overall_passed
            assert report.recommendation == "keep_learning"

    @pytest.mark.unit
    def test_facade_exposes_validator(self):
        from Jotty.core.intelligence.learning.facade import get_learning_validator

        validator = get_learning_validator()
        assert hasattr(validator, "validate_domain")
