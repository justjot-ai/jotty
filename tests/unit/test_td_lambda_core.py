"""
Unit tests for TD(λ) core formulas and Q-table operations.

Tests the mathematical correctness of:
- GroupedValueBaseline: EMA updates, hierarchical lookup, variance, transfer
- TDLambdaLearner: Eligibility traces, TD error, value updates, episode lifecycle
- SkillQTable: Q-value updates, UCB1 selection, domain fallback, staleness
- StepQTable: Role guidance, plan recording, convergence detection
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from Jotty.core.infrastructure.foundation.data_structures import (
    GoalValue,
    MemoryEntry,
    MemoryLevel,
    SwarmConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a SwarmConfig with sensible test defaults."""
    return SwarmConfig(**overrides)


def _make_memory_entry(key: str, goal: str = "test_goal", value: float = 0.5):
    """Create a MemoryEntry with a preset goal value."""
    entry = MemoryEntry(
        key=key,
        content=f"memory for {key}",
        level=MemoryLevel.EPISODIC,
        context={},
    )
    entry.goal_values[goal] = GoalValue(value=value)
    return entry


# ---------------------------------------------------------------------------
# GroupedValueBaseline Tests
# ---------------------------------------------------------------------------


class TestGroupedValueBaseline:
    """Tests for HRPO-inspired grouped baselines."""

    @pytest.mark.unit
    def test_default_baseline_is_0_5(self):
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline()
        assert gvb.get_baseline("unseen_task") == 0.5

    @pytest.mark.unit
    def test_ema_update_formula(self):
        """Verify: baseline_new = (1 - α) * baseline_old + α * reward."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        alpha = 0.2
        gvb = GroupedValueBaseline(ema_alpha=alpha)

        # Manually populate enough samples to pass the min-3 threshold
        for _ in range(3):
            gvb.update_group("coding", 1.0)

        # After 3 updates starting from 0.5 with alpha=0.2 and reward=1.0:
        # b1 = 0.8*0.5 + 0.2*1.0 = 0.6
        # b2 = 0.8*0.6 + 0.2*1.0 = 0.68
        # b3 = 0.8*0.68 + 0.2*1.0 = 0.744
        expected = 0.5
        for _ in range(3):
            expected = (1 - alpha) * expected + alpha * 1.0

        actual = gvb.get_baseline("coding")
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"

    @pytest.mark.unit
    def test_domain_fallback(self):
        """Task-type baseline falls back to domain baseline when unseen."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline(ema_alpha=0.5)

        # Build up domain baseline with enough samples
        for _ in range(5):
            gvb.update_group("other_task", 0.9, domain="ml")

        # New task type in same domain should get domain baseline
        baseline = gvb.get_baseline("new_task", domain="ml")
        # Should be domain baseline (not 0.5 default) since domain has >= 3 samples
        assert baseline != 0.5

    @pytest.mark.unit
    def test_group_variance_computation(self):
        """Variance = Σ(x-μ)²/n."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline()
        samples = [0.2, 0.4, 0.6, 0.8, 1.0]
        for s in samples:
            gvb.update_group("test", s)

        variance = gvb.get_group_variance("test")
        mean = sum(samples) / len(samples)
        expected_var = sum((s - mean) ** 2 for s in samples) / len(samples)
        assert abs(variance - expected_var) < 1e-6

    @pytest.mark.unit
    def test_relative_advantage(self):
        """Advantage = reward - baseline."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline()
        # Force a baseline value
        gvb.group_baselines["coding"] = 0.7
        gvb.group_counts["coding"] = 10

        advantage = gvb.compute_relative_advantage("coding", 0.9)
        assert abs(advantage - 0.2) < 1e-6

    @pytest.mark.unit
    def test_tier_tracking(self):
        """Tier success tracking (CogRouter)."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline(ema_alpha=0.5)
        gvb.update_tier("DIRECT", "coding", 1.0)
        gvb.update_tier("DIRECT", "coding", 0.0)

        result = gvb.get_tier_success("DIRECT", "coding")
        assert result is not None
        rate, count = result
        assert count == 2
        # EMA: 0.5*0.5 + 0.5*1.0 = 0.75, then 0.5*0.75 + 0.5*0.0 = 0.375
        assert abs(rate - 0.375) < 1e-6

    @pytest.mark.unit
    def test_agent_proficiency_tracking(self):
        """Agent proficiency tracking (Team of Thoughts)."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline(ema_alpha=0.5)
        gvb.update_agent("researcher", "analysis", 0.8)

        result = gvb.get_agent_proficiency("researcher", "analysis")
        assert result is not None
        proficiency, count = result
        assert count == 1
        # EMA: 0.5*0.5 + 0.5*0.8 = 0.65
        assert abs(proficiency - 0.65) < 1e-6

    @pytest.mark.unit
    def test_action_type_composite_key(self):
        """MAS-Bench: composite task_type:action_type keys."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline(ema_alpha=0.5)
        for _ in range(5):
            gvb.update_group("automation", 0.9, action_type="api")
            gvb.update_group("automation", 0.3, action_type="gui")

        best = gvb.get_best_action_type("automation", ["api", "gui"])
        assert best == "api"

    @pytest.mark.unit
    def test_serialization_roundtrip(self):
        """to_dict → from_dict preserves state."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        gvb = GroupedValueBaseline(ema_alpha=0.3)
        gvb.update_group("coding", 0.8)
        gvb.update_group("research", 0.6)
        gvb.update_tier("DIRECT", "coding", 1.0)
        gvb.update_agent("coder", "coding", 0.9)

        data = gvb.to_dict()
        restored = GroupedValueBaseline.from_dict(data)

        assert restored.ema_alpha == 0.3
        assert restored.group_baselines == gvb.group_baselines
        assert restored.group_counts == gvb.group_counts


# ---------------------------------------------------------------------------
# TDLambdaLearner Tests
# ---------------------------------------------------------------------------


class TestTDLambdaLearner:
    """Tests for the core TD(λ) learner."""

    @pytest.fixture
    def learner(self):
        """Create a TDLambdaLearner with mocked persistence."""
        from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner

        config = _make_config()
        with (
            patch.object(TDLambdaLearner, "_try_restore_baseline"),
            patch.object(TDLambdaLearner, "_try_restore_skill_q"),
            patch.object(TDLambdaLearner, "_try_restore_step_q"),
        ):
            td = TDLambdaLearner(config=config)
        return td

    @pytest.mark.unit
    def test_eligibility_trace_accumulating(self, learner):
        """Traces should accumulate (add 1.0) not replace."""
        learner.start_episode("test goal")
        mem = _make_memory_entry("mem_a", "test goal")

        # First access: trace = 0 * γλ + 1.0 = 1.0
        t1 = learner.record_access(mem)
        assert abs(t1 - 1.0) < 1e-6

        # Second access: trace = 1.0 * γλ + 1.0 = γλ + 1
        t2 = learner.record_access(mem)
        expected = learner.gamma * learner.lambda_trace + 1.0
        assert abs(t2 - expected) < 1e-6

    @pytest.mark.unit
    def test_trace_decay(self, learner):
        """Other traces decay by γλ when a new memory is accessed."""
        learner.start_episode("test goal")
        mem_a = _make_memory_entry("mem_a", "test goal")
        mem_b = _make_memory_entry("mem_b", "test goal")

        learner.record_access(mem_a)  # trace_a = 1.0
        learner.record_access(mem_b)  # trace_b = 1.0, trace_a decays to γλ

        assert abs(learner.traces["mem_a"] - learner.gamma * learner.lambda_trace) < 1e-6
        assert abs(learner.traces["mem_b"] - 1.0) < 1e-6

    @pytest.mark.unit
    def test_td_error_terminal_state(self, learner):
        """At terminal: δ = (R - baseline) - V(s)."""
        learner.start_episode("test goal", task_type="coding")

        mem = _make_memory_entry("mem_a", "test goal", value=0.3)
        memories = {"mem_a": mem}
        learner.record_access(mem)

        # Set known baseline
        learner.grouped_baseline.group_baselines["coding"] = 0.5
        learner.grouped_baseline.group_counts["coding"] = 10

        updates = learner.end_episode(final_reward=0.9, memories=memories)

        # Verify we got an update
        assert len(updates) == 1
        key, old_v, new_v = updates[0]
        assert key == "mem_a"

        # TD error: (0.9 - 0.5) - 0.3 = 0.1
        # Value update: V(s) + α * δ * trace = 0.3 + α * 0.1 * 1.0
        expected_td = 0.9 - 0.5 - 0.3
        expected_new = 0.3 + learner.alpha * expected_td * 1.0
        assert abs(new_v - expected_new) < 1e-4

    @pytest.mark.unit
    def test_td0_update_terminal(self, learner):
        """Single-step TD(0): V(s) ← V(s) + α(R - V(s)) at terminal."""
        learner.start_episode("test goal", task_type="testing")

        state = {"goal": "test goal"}
        action = {"type": "execute"}
        next_state = {"completed": True}

        learner.update(state, action, reward=1.0, next_state=next_state)

        # Check that value was updated
        state_key = "testing:execute"
        assert state_key in learner.grouped_baseline.group_baselines

    @pytest.mark.unit
    def test_episode_clears_traces(self, learner):
        """start_episode should clear all traces."""
        learner.traces = {"old": 0.5}
        learner.values_at_access = {"old": 0.3}
        learner.access_sequence = ["old"]

        learner.start_episode("new goal")

        assert len(learner.traces) == 0
        assert len(learner.values_at_access) == 0
        assert len(learner.access_sequence) == 0

    @pytest.mark.unit
    def test_value_clipped_to_0_1(self, learner):
        """Values should be clipped to [0, 1]."""
        learner.start_episode("test", task_type="coding")

        # Memory with very high value + positive reward should stay ≤ 1.0
        mem = _make_memory_entry("mem_a", "test", value=0.99)
        memories = {"mem_a": mem}
        learner.record_access(mem)
        learner.grouped_baseline.group_baselines["coding"] = 0.0
        learner.grouped_baseline.group_counts["coding"] = 10

        updates = learner.end_episode(final_reward=1.0, memories=memories)
        _, _, new_v = updates[0]
        assert 0.0 <= new_v <= 1.0

    @pytest.mark.unit
    def test_tool_reliability_tracking(self, learner):
        """Tool reliability transitions through LEARNING → HEALTHY/DEGRADED/FAILING."""
        # First 4 calls: LEARNING status
        for _ in range(4):
            result = learner.update_tool_reliability("search-api", True)
        assert result["status"] == "LEARNING"

        # 5th call: enough data, should be HEALTHY
        result = learner.update_tool_reliability("search-api", True)
        assert result["status"] == "HEALTHY"

        # Add failures to degrade
        for _ in range(20):
            learner.update_tool_reliability("bad-api", False)
        result = learner.update_tool_reliability("bad-api", False)
        assert result["status"] == "FAILING"
        assert learner.should_avoid_tool("bad-api")

    @pytest.mark.unit
    def test_infer_task_type(self, learner):
        """Task type inference from goal keywords."""
        assert learner._infer_task_type("count all users") == "aggregation"
        assert learner._infer_task_type("analyze trends") == "analysis"
        assert learner._infer_task_type("validate output") == "validation"
        assert learner._infer_task_type("hello world") == "general"


# ---------------------------------------------------------------------------
# SkillQTable Tests
# ---------------------------------------------------------------------------


class TestSkillQTable:
    """Tests for Q-learning skill selection."""

    @pytest.fixture
    def q(self):
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        return SkillQTable(alpha=0.1, gamma=0.9, epsilon=0.15, ucb_c=1.41)

    @pytest.mark.unit
    def test_default_q_value(self, q):
        """Unseen skill has optimistic default of 0.5."""
        assert q.get_q("coding", "web-search") == 0.5

    @pytest.mark.unit
    def test_q_update_td_error(self, q):
        """Q-update returns correct TD error: reward - old_Q."""
        td_error = q.update("coding", "web-search", reward=0.9)
        # old_Q = 0.5, reward = 0.9 → TD error = 0.4
        assert abs(td_error - 0.4) < 1e-4

    @pytest.mark.unit
    def test_q_value_converges(self, q):
        """Q-value converges toward consistent reward."""
        for _ in range(100):
            q.update("coding", "web-search", reward=0.9)

        final_q = q.get_q("coding", "web-search")
        assert abs(final_q - 0.9) < 0.05  # Should converge near 0.9

    @pytest.mark.unit
    def test_domain_specific_q(self, q):
        """Domain-specific Q-values are tracked separately."""
        q.update("coding", "web-search", reward=0.9, domain="finance")
        q.update("coding", "web-search", reward=0.3, domain="gaming")

        q_finance = q.get_q("coding", "web-search", domain="finance")
        q_gaming = q.get_q("coding", "web-search", domain="gaming")

        # After 1 update from 0.5: finance = 0.5 + α*(0.9-0.5), gaming = 0.5 + α*(0.3-0.5)
        assert q_finance > q_gaming

    @pytest.mark.unit
    def test_domain_fallback(self, q):
        """Unknown domain falls back to base task_type Q-value."""
        q.update("coding", "web-search", reward=0.8)
        q_unknown = q.get_q("coding", "web-search", domain="unknown_domain")
        q_base = q.get_q("coding", "web-search")
        # Should fall back to base since domain has no data
        assert abs(q_unknown - q_base) < 1e-6

    @pytest.mark.unit
    def test_ucb1_exploration(self, q):
        """UCB1: less-visited skills get exploration bonus."""
        # Give one skill many visits, another few
        for _ in range(50):
            q.update("coding", "popular-skill", reward=0.6)
        q.update("coding", "rare-skill", reward=0.6)

        # Both have same reward history mean, but rare-skill should rank higher
        # due to UCB exploration bonus
        ranking = q.select("coding", ["popular-skill", "rare-skill"])
        assert ranking[0] == "rare-skill"  # UCB bonus should push it up

    @pytest.mark.unit
    def test_select_returns_all_skills(self, q):
        """select() returns all available skills, ranked."""
        skills = ["a", "b", "c"]
        ranking = q.select("coding", skills)
        assert set(ranking) == set(skills)
        assert len(ranking) == 3

    @pytest.mark.unit
    def test_q_clipped_to_0_1(self, q):
        """Q-values stay in [0, 1]."""
        q.update("coding", "skill_a", reward=0.0)
        q.update("coding", "skill_a", reward=0.0)
        assert q.get_q("coding", "skill_a") >= 0.0

        q.update("coding", "skill_b", reward=1.0)
        q.update("coding", "skill_b", reward=1.0)
        assert q.get_q("coding", "skill_b") <= 1.0

    @pytest.mark.unit
    def test_serialization_roundtrip(self, q):
        """to_dict → from_dict preserves Q-values."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q.update("coding", "web-search", reward=0.9)
        q.update("research", "math-toolkit", reward=0.7)

        data = q.to_dict()
        restored = SkillQTable.from_dict(data)

        assert abs(restored.get_q("coding", "web-search") - q.get_q("coding", "web-search")) < 1e-6
        assert (
            abs(restored.get_q("research", "math-toolkit") - q.get_q("research", "math-toolkit"))
            < 1e-6
        )


# ---------------------------------------------------------------------------
# StepQTable Tests
# ---------------------------------------------------------------------------


class TestStepQTable:
    """Tests for plan/step Q-table and role guidance."""

    @pytest.fixture
    def sq(self):
        from Jotty.core.intelligence.learning.td_lambda import StepQTable

        return StepQTable()

    @pytest.mark.unit
    def test_step_update_returns_td_error(self, sq):
        """update() returns TD error."""
        td = sq.update("coding", step_pos=0, skill="architect", reward=0.8)
        # Starting from 0.5 default: td = 0.8 - 0.5 = 0.3
        assert abs(td - 0.3) < 0.1  # Approximate due to alpha

    @pytest.mark.unit
    def test_plan_recording(self, sq):
        """record_plan stores plan template with reward."""
        sq.record_plan("coding", ["architect", "developer", "tester"], reward=0.9)
        plans = sq._plan_history.get("coding", [])
        assert len(plans) == 1

    @pytest.mark.unit
    def test_role_guidance_returns_entries(self, sq):
        """get_role_guidance returns guidance after enough updates."""
        # Populate with data
        for _ in range(5):
            sq.update("coding", 0, "web-search", 0.9, description="research phase")
            sq.update("coding", 1, "code-gen", 0.8, description="implementation")

        guidance = sq.get_role_guidance("coding")
        assert isinstance(guidance, list)


# ---------------------------------------------------------------------------
# Algorithmic Credit (Shapley) Tests
# ---------------------------------------------------------------------------


class TestShapleyGlobalStateFix:
    """Verify that Shapley estimation uses instance-based RNG."""

    @pytest.mark.unit
    def test_shapley_does_not_pollute_global_random(self):
        """Shapley estimation should not call random.seed() on global state."""
        import random

        from Jotty.core.intelligence.learning.algorithmic_credit import ShapleyValueEstimator

        estimator = ShapleyValueEstimator(num_samples=5)

        # Set global random state
        random.seed(999)
        before = random.random()

        # Reset and run Shapley (synchronous mock since it's async)
        random.seed(999)
        # The key test: calling estimate_shapley_values should NOT affect
        # global random state. We verify by checking the source code uses
        # rng = random.Random(42) instead of random.seed(42 + i)
        import inspect

        source = inspect.getsource(estimator.estimate_shapley_values)
        assert "random.seed(" not in source, "Shapley should use instance RNG, not random.seed()"
        assert "rng.sample" in source or "rng = random.Random" in source


# ---------------------------------------------------------------------------
# Shaped Rewards Batching Tests
# ---------------------------------------------------------------------------


class TestShapedRewardsBatching:
    """Verify batched evaluation in shaped rewards."""

    @pytest.mark.unit
    def test_evaluator_has_batch_method(self):
        """AgenticRewardEvaluator should have batch evaluation."""
        from Jotty.core.intelligence.learning.shaped_rewards import AgenticRewardEvaluator

        evaluator = AgenticRewardEvaluator()
        assert hasattr(evaluator, "evaluate_conditions_batch")

    @pytest.mark.unit
    def test_batch_returns_dict(self):
        """evaluate_conditions_batch returns a dict of results."""
        from Jotty.core.intelligence.learning.shaped_rewards import (
            AgenticRewardEvaluator,
            RewardCondition,
        )

        evaluator = AgenticRewardEvaluator()

        conditions = [
            RewardCondition(name="test_a", description="Test A", reward_value=0.1),
            RewardCondition(name="test_b", description="Test B", reward_value=0.2),
        ]

        # Without DSPy, falls back to individual evaluation which returns (False, 0.0, "No evaluator available")
        results = evaluator.evaluate_conditions_batch(conditions, {}, [])
        assert isinstance(results, dict)
        assert "test_a" in results
        assert "test_b" in results

    @pytest.mark.unit
    def test_empty_conditions_returns_empty_dict(self):
        """Empty conditions list returns empty dict."""
        from Jotty.core.intelligence.learning.shaped_rewards import AgenticRewardEvaluator

        evaluator = AgenticRewardEvaluator()
        results = evaluator.evaluate_conditions_batch([], {}, [])
        assert results == {}

    @pytest.mark.unit
    def test_manager_uses_batch(self):
        """ShapedRewardManager.check_rewards uses batched evaluation."""
        from Jotty.core.intelligence.learning.shaped_rewards import ShapedRewardManager

        mgr = ShapedRewardManager()

        # Force evaluators off so the test is deterministic regardless
        # of whether earlier tests have initialized DSPy with a mock LM.
        mgr.evaluator.evaluator = None
        mgr.evaluator.batch_evaluator = None

        # Without evaluator, all conditions return (False, 0.0) so reward = 0
        reward = mgr.check_rewards("actor_complete", {"output": "test"}, [])
        assert reward == 0.0  # No evaluator → no rewards triggered
