"""
Comprehensive unit tests for the Jotty learning layer.

Covers:
- core/learning/td_lambda.py (TDLambdaLearner, GroupedValueBaseline, SkillQTable, COMACredit)
- core/learning/adaptive_components.py (AdaptiveLearningRate, IntermediateRewardCalculator, AdaptiveExploration)
- core/learning/rl_components.py (RLComponents)
"""

import time
from unittest.mock import Mock

import pytest

try:
    from core.infrastructure.foundation.data_structures import (
        GoalValue,
        MemoryEntry,
        MemoryLevel,
        SwarmConfig,
    )
    from core.intelligence.learning.adaptive_components import (
        AdaptiveExploration,
        AdaptiveLearningRate,
        IntermediateRewardCalculator,
    )
    from core.intelligence.learning.td_lambda import (
        COMACredit,
        GroupedValueBaseline,
        SkillQTable,
        TDLambdaLearner,
        get_learned_context,
    )

    HAS_LEARNING = True
except ImportError:
    HAS_LEARNING = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not HAS_LEARNING, reason="Learning modules not available"),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a SwarmConfig with sensible test defaults."""
    cfg = SwarmConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_memory_entry(
    key="mem_1", content="test content", level=MemoryLevel.EPISODIC, goal="test_goal", value=0.5
):
    """Create a lightweight MemoryEntry for tests."""
    entry = MemoryEntry(
        key=key,
        content=content,
        level=level,
        context={"domain": "test"},
    )
    entry.goal_values[goal] = GoalValue(value=value)
    return entry


# ============================================================================
# 1. GroupedValueBaseline
# ============================================================================


class TestGroupedValueBaseline:
    """Tests for GroupedValueBaseline (HRPO-inspired)."""

    @pytest.mark.unit
    def test_default_baseline(self):
        gvb = GroupedValueBaseline()
        assert gvb.get_baseline("unknown_type") == 0.5

    @pytest.mark.unit
    def test_update_group_changes_baseline(self):
        gvb = GroupedValueBaseline(ema_alpha=0.5)
        gvb.update_group("analysis", 0.9)
        expected = (1 - 0.5) * 0.5 + 0.5 * 0.9  # 0.7
        assert abs(gvb.group_baselines["analysis"] - expected) < 1e-6

    @pytest.mark.unit
    def test_baseline_requires_min_samples(self):
        gvb = GroupedValueBaseline()
        gvb.update_group("type_a", 0.9)
        gvb.update_group("type_a", 0.8)
        assert gvb.get_baseline("type_a") == 0.5

    @pytest.mark.unit
    def test_baseline_returned_after_enough_samples(self):
        gvb = GroupedValueBaseline(ema_alpha=0.1)
        for _ in range(5):
            gvb.update_group("type_b", 0.8)
        assert gvb.get_baseline("type_b") != 0.5

    @pytest.mark.unit
    def test_domain_baseline_fallback(self):
        gvb = GroupedValueBaseline(ema_alpha=0.5)
        for _ in range(5):
            gvb.update_group("type_x", 0.9, domain="ml")
        baseline = gvb.get_baseline("type_y", domain="ml")
        assert baseline != 0.5

    @pytest.mark.unit
    def test_compute_relative_advantage(self):
        gvb = GroupedValueBaseline()
        assert gvb.compute_relative_advantage("any", 0.8) == pytest.approx(0.3)

    @pytest.mark.unit
    def test_get_group_variance_insufficient_samples(self):
        gvb = GroupedValueBaseline()
        assert gvb.get_group_variance("empty") == 0.25

    @pytest.mark.unit
    def test_get_group_variance_zero_for_identical(self):
        gvb = GroupedValueBaseline()
        gvb.group_samples["t"] = [0.5, 0.5, 0.5]
        assert gvb.get_group_variance("t") == 0.0

    @pytest.mark.unit
    def test_get_statistics_structure(self):
        gvb = GroupedValueBaseline()
        gvb.update_group("a", 0.5)
        stats = gvb.get_statistics()
        assert stats["num_groups"] >= 1
        assert stats["total_samples"] >= 1

    @pytest.mark.unit
    def test_max_samples_cap(self):
        gvb = GroupedValueBaseline()
        gvb.max_samples_per_group = 5
        for i in range(10):
            gvb.update_group("t", float(i))
        assert len(gvb.group_samples["t"]) <= 5

    @pytest.mark.unit
    def test_to_dict_from_dict_roundtrip(self):
        gvb = GroupedValueBaseline(ema_alpha=0.2)
        gvb.update_group("research", 0.7)
        gvb.update_group("analysis", 0.3)
        data = gvb.to_dict()
        restored = GroupedValueBaseline.from_dict(data)
        assert restored.ema_alpha == 0.2
        assert "research" in restored.group_baselines
        assert "analysis" in restored.group_baselines

    @pytest.mark.unit
    def test_from_dict_defaults(self):
        restored = GroupedValueBaseline.from_dict({})
        assert restored.ema_alpha == 0.1
        assert len(restored.group_baselines) == 0

    @pytest.mark.unit
    def test_update_group_with_domain(self):
        gvb = GroupedValueBaseline(ema_alpha=0.5)
        gvb.update_group("type_a", 0.8, domain="finance")
        assert "finance" in gvb.domain_baselines
        assert gvb.group_counts.get("domain:finance", 0) == 1


# ============================================================================
# 2. TDLambdaLearner
# ============================================================================


class TestTDLambdaLearner:
    """Tests for TDLambdaLearner."""

    def _make_learner(self, **overrides):
        cfg = _make_config(**overrides)
        return TDLambdaLearner(cfg)

    @pytest.mark.unit
    def test_init_defaults(self):
        learner = self._make_learner()
        assert learner.gamma == 0.99
        assert learner.lambda_trace == 0.95
        assert learner.traces == {}

    @pytest.mark.unit
    def test_start_episode_clears_state(self):
        learner = self._make_learner()
        learner.traces["old"] = 1.0
        learner.values_at_access["old"] = 0.5
        learner.access_sequence.append("old")
        learner.start_episode("new goal")
        assert learner.traces == {}
        assert learner.values_at_access == {}
        assert learner.access_sequence == []
        assert learner.current_goal == "new goal"

    @pytest.mark.unit
    def test_start_episode_infers_task_type(self):
        learner = self._make_learner()
        learner.start_episode("analyze customer trends")
        assert learner.current_task_type == "analysis"

    @pytest.mark.unit
    def test_start_episode_explicit_task_type(self):
        learner = self._make_learner()
        learner.start_episode("something", task_type="aggregation")
        assert learner.current_task_type == "aggregation"

    @pytest.mark.unit
    def test_start_episode_domain(self):
        learner = self._make_learner()
        learner.start_episode("g", domain="finance")
        assert learner.current_domain == "finance"

    @pytest.mark.unit
    def test_infer_task_type_general(self):
        learner = self._make_learner()
        assert learner._infer_task_type("do stuff") == "general"

    @pytest.mark.unit
    def test_infer_task_type_validation(self):
        learner = self._make_learner()
        assert learner._infer_task_type("validate the outputs") == "validation"

    @pytest.mark.unit
    def test_infer_task_type_filtering(self):
        learner = self._make_learner()
        assert learner._infer_task_type("filter top results") == "filtering"

    @pytest.mark.unit
    def test_infer_task_type_planning(self):
        learner = self._make_learner()
        assert learner._infer_task_type("plan the deployment") == "planning"

    @pytest.mark.unit
    def test_infer_task_type_transformation(self):
        learner = self._make_learner()
        assert learner._infer_task_type("transform the data") == "transformation"

    @pytest.mark.unit
    def test_update_td0_changes_baseline(self):
        learner = self._make_learner(alpha=0.5)
        learner.start_episode("test goal")
        state = {"goal": "test goal"}
        action = {"type": "execute"}
        next_state = {"completed": True}
        learner.update(state, action, 1.0, next_state)
        key = f"{learner.current_task_type}:execute"
        assert learner.grouped_baseline.group_baselines.get(key, 0.5) > 0.5

    @pytest.mark.unit
    def test_update_non_terminal(self):
        learner = self._make_learner(alpha=0.5)
        learner.start_episode("test goal")
        state = {"goal": "test goal"}
        action = {"type": "step"}
        next_state = {"completed": False}
        learner.update(state, action, 0.0, next_state)
        # Should not raise and should update state

    @pytest.mark.unit
    def test_update_switches_goal(self):
        learner = self._make_learner()
        learner.start_episode("goal_a")
        learner.update({"goal": "goal_b"}, {"type": "x"}, 0.5, {})
        assert learner.current_goal == "goal_b"

    @pytest.mark.unit
    def test_record_access_accumulating_trace(self):
        learner = self._make_learner()
        learner.start_episode("g")
        entry = _make_memory_entry(key="k1", goal="g")
        t1 = learner.record_access(entry)
        assert t1 == 1.0
        t2 = learner.record_access(entry)
        assert t2 > 1.0

    @pytest.mark.unit
    def test_record_access_decays_other_traces(self):
        learner = self._make_learner()
        learner.start_episode("g")
        e1 = _make_memory_entry(key="k1", goal="g")
        e2 = _make_memory_entry(key="k2", goal="g")
        learner.record_access(e1)
        learner.record_access(e2)
        assert learner.traces["k1"] < 1.0

    @pytest.mark.unit
    def test_record_access_tracks_sequence(self):
        learner = self._make_learner()
        learner.start_episode("g")
        e1 = _make_memory_entry(key="a", goal="g")
        e2 = _make_memory_entry(key="b", goal="g")
        learner.record_access(e1)
        learner.record_access(e2)
        assert learner.access_sequence == ["a", "b"]

    @pytest.mark.unit
    def test_record_access_no_duplicate_sequence(self):
        learner = self._make_learner()
        learner.start_episode("g")
        entry = _make_memory_entry(key="k", goal="g")
        learner.record_access(entry)
        learner.record_access(entry)
        assert learner.access_sequence == ["k"]

    @pytest.mark.unit
    def test_record_access_step_reward(self):
        learner = self._make_learner()
        learner.start_episode("g")
        entry = _make_memory_entry(key="k", goal="g")
        learner.record_access(entry, step_reward=0.1)
        assert learner.intermediate_calc.step_rewards == [0.1]

    @pytest.mark.unit
    def test_end_episode_updates_values(self):
        learner = self._make_learner(alpha=0.5)
        learner.start_episode("goal_x")
        entry = _make_memory_entry(key="m1", goal="goal_x", value=0.3)
        learner.record_access(entry)
        updates = learner.end_episode(1.0, {"m1": entry})
        assert len(updates) == 1
        key, old_v, new_v = updates[0]
        assert key == "m1"
        assert new_v != old_v

    @pytest.mark.unit
    def test_end_episode_clips_values(self):
        learner = self._make_learner(alpha=1.0)
        learner.start_episode("g")
        entry = _make_memory_entry(key="m", goal="g", value=0.9)
        learner.record_access(entry)
        updates = learner.end_episode(5.0, {"m": entry})
        _, _, new_v = updates[0]
        assert 0.0 <= new_v <= 1.0

    @pytest.mark.unit
    def test_end_episode_skips_missing_memories(self):
        learner = self._make_learner()
        learner.start_episode("g")
        entry = _make_memory_entry(key="present", goal="g")
        learner.record_access(entry)
        updates = learner.end_episode(1.0, {"other": entry})
        assert len(updates) == 0

    @pytest.mark.unit
    def test_end_episode_with_adaptive_lr(self):
        cfg = _make_config()
        adaptive_lr = AdaptiveLearningRate(cfg)
        learner = TDLambdaLearner(cfg, adaptive_lr=adaptive_lr)
        learner.start_episode("g")
        entry = _make_memory_entry(key="m", goal="g", value=0.5)
        learner.record_access(entry)
        learner.end_episode(0.8, {"m": entry})
        assert len(adaptive_lr.td_errors) > 0

    @pytest.mark.unit
    def test_get_grouped_learning_stats(self):
        learner = self._make_learner()
        stats = learner.get_grouped_learning_stats()
        assert "num_groups" in stats
        assert "total_samples" in stats

    @pytest.mark.unit
    def test_trace_pruning_below_threshold(self):
        learner = self._make_learner(gamma=0.01, lambda_trace=0.01)
        learner.start_episode("g")
        e1 = _make_memory_entry(key="old", goal="g")
        learner.record_access(e1)
        for i in range(100):
            e = _make_memory_entry(key=f"new_{i}", goal="g")
            learner.record_access(e)
        assert "old" not in learner.traces

    @pytest.mark.unit
    def test_update_with_adaptive_lr(self):
        cfg = _make_config()
        adaptive_lr = AdaptiveLearningRate(cfg)
        learner = TDLambdaLearner(cfg, adaptive_lr=adaptive_lr)
        learner.start_episode("test")
        learner.update({"goal": "test"}, {"type": "x"}, 0.8, {"completed": True})
        assert len(adaptive_lr.td_errors) > 0


# ============================================================================
# 3. SkillQTable
# ============================================================================


class TestSkillQTable:
    """Tests for SkillQTable."""

    @pytest.mark.unit
    def test_default_q_value(self):
        q = SkillQTable()
        assert q.get_q("research", "web-search") == 0.5

    @pytest.mark.unit
    def test_update_moves_q(self):
        q = SkillQTable(alpha=0.5)
        td = q.update("research", "web-search", 1.0)
        assert q.get_q("research", "web-search") == pytest.approx(0.75)
        assert td == pytest.approx(0.5)

    @pytest.mark.unit
    def test_update_clips_upper(self):
        q = SkillQTable(alpha=1.0)
        q.update("t", "s", 5.0)
        assert q.get_q("t", "s") <= 1.0

    @pytest.mark.unit
    def test_update_clips_lower(self):
        q = SkillQTable(alpha=1.0)
        q.update("t", "s", -5.0)
        assert q.get_q("t", "s") >= 0.0

    @pytest.mark.unit
    def test_update_increments_count(self):
        q = SkillQTable()
        q.update("t", "s", 0.5)
        q.update("t", "s", 0.6)
        assert q._counts["t"]["s"] == 2

    @pytest.mark.unit
    def test_select_exploit(self):
        q = SkillQTable(epsilon=0.0)
        q.update("t", "best", 1.0)
        q.update("t", "worst", 0.0)
        result = q.select("t", ["worst", "best"])
        assert result[0] == "best"

    @pytest.mark.unit
    def test_select_empty_skills(self):
        assert SkillQTable().select("t", []) == []

    @pytest.mark.unit
    def test_select_explore_returns_all(self):
        q = SkillQTable(epsilon=1.0)
        result = q.select("t", ["a", "b", "c"])
        assert set(result) == {"a", "b", "c"}

    @pytest.mark.unit
    def test_get_top_skills_sorted(self):
        q = SkillQTable(alpha=1.0)
        q.update("t", "a", 0.9)
        q.update("t", "b", 0.3)
        q.update("t", "c", 0.7)
        top = q.get_top_skills("t", n=2)
        assert len(top) == 2
        assert top[0][0] == "a"

    @pytest.mark.unit
    def test_get_top_skills_empty(self):
        assert SkillQTable().get_top_skills("nonexistent") == []

    @pytest.mark.unit
    def test_to_dict_from_dict_roundtrip(self):
        q = SkillQTable(alpha=0.2, gamma=0.8, epsilon=0.1)
        q.update("research", "web-search", 0.9)
        q.update("research", "calculator", 0.3)
        data = q.to_dict()
        q2 = SkillQTable.from_dict(data)
        assert q2.alpha == 0.2
        assert q2.gamma == 0.8
        assert q2.epsilon == 0.1
        assert q2.get_q("research", "web-search") == q.get_q("research", "web-search")
        assert q2._counts == q._counts

    @pytest.mark.unit
    def test_from_dict_defaults(self):
        q = SkillQTable.from_dict({})
        assert q.alpha == 0.1
        assert q._q == {}

    @pytest.mark.unit
    def test_to_dict_contains_all_keys(self):
        q = SkillQTable()
        data = q.to_dict()
        assert "q" in data and "counts" in data
        assert "alpha" in data and "gamma" in data and "epsilon" in data


# ============================================================================
# 4. COMACredit
# ============================================================================


class TestCOMACredit:
    """Tests for COMACredit (counterfactual credit assignment)."""

    @pytest.mark.unit
    def test_get_credit_unknown_agent(self):
        assert COMACredit().get_credit("ghost") == 0.0

    @pytest.mark.unit
    def test_record_episode_and_get_credit(self):
        coma = COMACredit()
        coma.record_episode(0.8, {"researcher": 0.4, "writer": 0.4})
        assert coma.get_credit("researcher") == pytest.approx(0.8 - 0.5)

    @pytest.mark.unit
    def test_counterfactual_baseline_builds(self):
        coma = COMACredit()
        coma.record_episode(0.9, {"A": 0.5})
        coma.record_episode(0.3, {"B": 0.5})
        assert coma.get_credit("A") == pytest.approx(0.9 - 0.3)

    @pytest.mark.unit
    def test_get_all_credits(self):
        coma = COMACredit()
        coma.record_episode(0.8, {"A": 0.3, "B": 0.5})
        credits = coma.get_all_credits()
        assert "A" in credits and "B" in credits

    @pytest.mark.unit
    def test_history_bounded_at_200(self):
        coma = COMACredit()
        for i in range(300):
            coma.record_episode(float(i) / 300, {"agent": 0.5})
        assert len(coma._history["agent"]) <= 200

    @pytest.mark.unit
    def test_counterfactual_bounded_at_200(self):
        coma = COMACredit()
        coma._history["X"] = []
        coma._counterfactual["X"] = []
        for i in range(300):
            coma.record_episode(0.5, {"Y": 0.5})
        assert len(coma._counterfactual["X"]) <= 200

    @pytest.mark.unit
    def test_to_dict_from_dict_roundtrip(self):
        coma = COMACredit()
        coma.record_episode(0.7, {"A": 0.3, "B": 0.4})
        coma.record_episode(0.5, {"A": 0.5})
        data = coma.to_dict()
        restored = COMACredit.from_dict(data)
        assert restored.get_credit("A") == pytest.approx(coma.get_credit("A"))
        assert restored.get_credit("B") == pytest.approx(coma.get_credit("B"))

    @pytest.mark.unit
    def test_from_dict_empty(self):
        restored = COMACredit.from_dict({})
        assert restored._history == {} and restored._counterfactual == {}

    @pytest.mark.unit
    def test_negative_credit(self):
        coma = COMACredit()
        # Agent present when team does badly
        coma.record_episode(0.2, {"bad_agent": 0.5})
        # Team does great without bad_agent
        coma.record_episode(0.9, {"other": 0.5})
        credit = coma.get_credit("bad_agent")
        assert credit < 0


# ============================================================================
# 5. get_learned_context (module-level function)
# ============================================================================


class TestGetLearnedContext:
    """Tests for the get_learned_context function in td_lambda."""

    def _make_td_learner(self):
        return TDLambdaLearner(_make_config())

    @pytest.mark.unit
    def test_empty_context_no_data(self):
        assert get_learned_context(self._make_td_learner()) == ""

    @pytest.mark.unit
    def test_context_with_task_type_baseline(self):
        td = self._make_td_learner()
        for _ in range(5):
            td.grouped_baseline.update_group("research", 0.8)
        ctx = get_learned_context(td, task_type="research")
        assert "LEARNED CONTEXT" in ctx
        assert "research" in ctx

    @pytest.mark.unit
    def test_context_with_skill_q(self):
        td = self._make_td_learner()
        for _ in range(5):
            td.grouped_baseline.update_group("research", 0.8)
        sq = SkillQTable(alpha=1.0)
        sq.update("research", "web-search", 0.95)
        ctx = get_learned_context(td, skill_q=sq, task_type="research")
        assert "web-search" in ctx

    @pytest.mark.unit
    def test_context_with_coma_credits(self):
        td = self._make_td_learner()
        coma = COMACredit()
        for _ in range(5):
            coma.record_episode(0.9, {"researcher": 0.5})
        for _ in range(5):
            coma.record_episode(0.2, {"writer": 0.5})
        ctx = get_learned_context(td, coma=coma)
        if coma.get_credit("researcher") > 0.05:
            assert "researcher" in ctx

    @pytest.mark.unit
    def test_max_lines_limit(self):
        td = self._make_td_learner()
        for _ in range(5):
            td.grouped_baseline.update_group("t", 0.7)
        ctx = get_learned_context(td, task_type="t", max_lines=1)
        lines = [l for l in ctx.split("\n") if l.strip()]
        assert len(lines) <= 2

    @pytest.mark.unit
    def test_context_with_transfer_insights(self):
        td = self._make_td_learner()
        # Manually set up transfer matrix
        td.grouped_baseline.transfer_matrix["research"] = {"analysis": 0.8}
        for _ in range(5):
            td.grouped_baseline.update_group("analysis", 0.7)
        ctx = get_learned_context(td, task_type="research")
        # Should mention similar task type
        if ctx:
            assert "analysis" in ctx or ctx == ""


# ============================================================================
# 6. AdaptiveLearningRate
# ============================================================================


class TestAdaptiveLearningRate:
    """Tests for AdaptiveLearningRate."""

    def _make_alr(self, **overrides):
        return AdaptiveLearningRate(_make_config(**overrides))

    @pytest.mark.unit
    def test_initial_alpha(self):
        alr = self._make_alr()
        assert alr.alpha == alr.config.alpha

    @pytest.mark.unit
    def test_get_adapted_alpha_disabled(self):
        alr = self._make_alr(enable_adaptive_alpha=False)
        alr.td_errors = [0.1] * 20
        assert alr.get_adapted_alpha() == alr.config.alpha

    @pytest.mark.unit
    def test_get_adapted_alpha_few_errors(self):
        alr = self._make_alr()
        alr.td_errors = [0.1] * 5
        assert alr.get_adapted_alpha() == alr.alpha

    @pytest.mark.unit
    def test_record_td_error_stores_abs(self):
        alr = self._make_alr()
        alr.record_td_error(-0.3)
        assert alr.td_errors == [0.3]

    @pytest.mark.unit
    def test_record_td_error_window_cap(self):
        # Pruning triggers when len > window_size * 2, keeping last window_size
        alr = self._make_alr(adaptive_window_size=5)
        for i in range(20):
            alr.record_td_error(float(i))
        assert len(alr.td_errors) <= 5 * 2

    @pytest.mark.unit
    def test_record_success_values(self):
        alr = self._make_alr()
        alr.record_success(True)
        alr.record_success(False)
        assert alr.success_rates == [1.0, 0.0]

    @pytest.mark.unit
    def test_record_success_window_cap(self):
        # Pruning triggers when len > window_size * 2, keeping last window_size
        alr = self._make_alr(adaptive_window_size=5)
        for _ in range(20):
            alr.record_success(True)
        assert len(alr.success_rates) <= 5 * 2

    @pytest.mark.unit
    def test_alpha_bounded_min_max(self):
        alr = self._make_alr(
            enable_adaptive_alpha=True,
            alpha_min=0.01,
            alpha_max=0.1,
        )
        for _ in range(50):
            alr.record_td_error(0.001)
        adapted = alr.get_adapted_alpha()
        assert alr.config.alpha_min <= adapted <= alr.config.alpha_max

    @pytest.mark.unit
    def test_reset_clears_everything(self):
        alr = self._make_alr()
        alr.record_td_error(0.5)
        alr.record_success(True)
        alr.alpha = 999.0
        alr.reset()
        assert alr.alpha == alr.config.alpha
        assert alr.td_errors == []
        assert alr.success_rates == []

    @pytest.mark.unit
    def test_high_variance_tends_to_decrease_alpha(self):
        alr = self._make_alr(
            enable_adaptive_alpha=True,
            alpha=0.05,
            alpha_min=0.001,
            alpha_max=0.1,
            adaptive_window_size=10,
        )
        for i in range(20):
            alr.record_td_error(10.0 if i % 2 == 0 else 0.0)
        adapted = alr.get_adapted_alpha()
        assert adapted <= alr.config.alpha_max


# ============================================================================
# 7. IntermediateRewardCalculator
# ============================================================================


class TestIntermediateRewardCalculator:
    """Tests for IntermediateRewardCalculator."""

    def _make_calc(self, **overrides):
        return IntermediateRewardCalculator(_make_config(**overrides))

    @pytest.mark.unit
    def test_reset_clears(self):
        calc = self._make_calc()
        calc.step_rewards = [0.1, 0.2]
        calc.reset()
        assert calc.step_rewards == []

    @pytest.mark.unit
    def test_reward_architect_proceed_disabled(self):
        calc = self._make_calc(enable_intermediate_rewards=False)
        assert calc.reward_architect_proceed(0.9) == 0.0
        assert calc.step_rewards == []

    @pytest.mark.unit
    def test_reward_architect_proceed_enabled(self):
        calc = self._make_calc(
            enable_intermediate_rewards=True,
            architect_proceed_reward=0.1,
        )
        r = calc.reward_architect_proceed(0.8)
        assert r == pytest.approx(0.08)
        assert len(calc.step_rewards) == 1

    @pytest.mark.unit
    def test_reward_tool_success_true(self):
        calc = self._make_calc(
            enable_intermediate_rewards=True,
            tool_success_reward=0.05,
        )
        assert calc.reward_tool_success("web-search", True) == pytest.approx(0.05)

    @pytest.mark.unit
    def test_reward_tool_success_false(self):
        calc = self._make_calc(
            enable_intermediate_rewards=True,
            tool_success_reward=0.05,
        )
        assert calc.reward_tool_success("web-search", False) == pytest.approx(-0.025)

    @pytest.mark.unit
    def test_reward_tool_disabled(self):
        calc = self._make_calc(enable_intermediate_rewards=False)
        assert calc.reward_tool_success("t", True) == 0.0

    @pytest.mark.unit
    def test_reward_partial_completion(self):
        calc = self._make_calc(enable_intermediate_rewards=True)
        assert calc.reward_partial_completion(0.5) == pytest.approx(0.15)

    @pytest.mark.unit
    def test_reward_partial_completion_disabled(self):
        calc = self._make_calc(enable_intermediate_rewards=False)
        assert calc.reward_partial_completion(0.5) == 0.0

    @pytest.mark.unit
    def test_reward_partial_completion_full(self):
        calc = self._make_calc(enable_intermediate_rewards=True)
        assert calc.reward_partial_completion(1.0) == pytest.approx(0.3)

    @pytest.mark.unit
    def test_get_total_intermediate_reward(self):
        calc = self._make_calc(enable_intermediate_rewards=True, tool_success_reward=0.1)
        calc.reward_tool_success("a", True)
        calc.reward_tool_success("b", True)
        assert calc.get_total_intermediate_reward() == pytest.approx(0.2)

    @pytest.mark.unit
    def test_get_discounted_intermediate_reward(self):
        calc = self._make_calc()
        calc.step_rewards = [0.1, 0.1]
        expected = 0.1 * 1.0 + 0.1 * 0.9
        assert calc.get_discounted_intermediate_reward(0.9) == pytest.approx(expected)

    @pytest.mark.unit
    def test_discounted_reward_empty(self):
        calc = self._make_calc()
        assert calc.get_discounted_intermediate_reward(0.99) == 0.0


# ============================================================================
# 8. AdaptiveExploration
# ============================================================================


class TestAdaptiveExploration:
    """Tests for AdaptiveExploration."""

    def _make_expl(self, **overrides):
        return AdaptiveExploration(_make_config(**overrides))

    @pytest.mark.unit
    def test_get_epsilon_new_goal_boost(self):
        expl = self._make_expl(epsilon_start=0.3, epsilon_end=0.05, epsilon_decay_episodes=100)
        eps = expl.get_epsilon("new_goal", episode=0)
        # New goal (< 5 visits) gets 1.5x boost capped at 0.5
        assert eps == pytest.approx(min(0.5, 0.3 * 1.5))

    @pytest.mark.unit
    def test_get_epsilon_decays(self):
        expl = self._make_expl(epsilon_start=0.3, epsilon_end=0.05, epsilon_decay_episodes=100)
        for _ in range(10):
            expl.record_goal_visit("g")
        eps0 = expl.get_epsilon("g", episode=0)
        eps100 = expl.get_epsilon("g", episode=100)
        assert eps100 < eps0

    @pytest.mark.unit
    def test_get_epsilon_at_end_of_decay(self):
        expl = self._make_expl(epsilon_start=0.3, epsilon_end=0.05, epsilon_decay_episodes=100)
        for _ in range(10):
            expl.record_goal_visit("g")
        eps = expl.get_epsilon("g", episode=1000)
        assert eps == pytest.approx(0.05)

    @pytest.mark.unit
    def test_record_goal_visit_increments(self):
        expl = self._make_expl()
        expl.record_goal_visit("g")
        expl.record_goal_visit("g")
        assert expl.goal_visit_counts["g"] == 2

    @pytest.mark.unit
    def test_stall_detection_activates(self):
        expl = self._make_expl(stall_detection_window=100, stall_threshold=0.01)
        for _ in range(60):
            expl.record_value_change(0.0001)
        assert expl.stall_boost_active is True

    @pytest.mark.unit
    def test_stall_detection_not_active_with_large_changes(self):
        expl = self._make_expl(stall_detection_window=100, stall_threshold=0.01)
        for _ in range(60):
            expl.record_value_change(1.0)
        assert expl.stall_boost_active is False

    @pytest.mark.unit
    def test_stall_boost_increases_epsilon(self):
        expl = self._make_expl(
            epsilon_start=0.1,
            epsilon_end=0.05,
            exploration_boost_on_stall=0.1,
            stall_detection_window=100,
            stall_threshold=0.01,
        )
        for _ in range(10):
            expl.record_goal_visit("g")
        for _ in range(60):
            expl.record_value_change(0.0001)
        eps = expl.get_epsilon("g", episode=500)
        assert eps > 0.05

    @pytest.mark.unit
    def test_record_value_change_window_cap(self):
        expl = self._make_expl(stall_detection_window=10)
        for i in range(30):
            expl.record_value_change(float(i))
        assert len(expl.recent_values) <= 10
