"""
Unit tests for health_budget module (LearningHealthMonitor, DynamicBudgetManager).

The original transfer_learning.py, reasoning_credit.py tests were removed
because those modules were dead code (replaced by algorithmic_credit.py,
LearningService, and td_lambda hierarchical keys).
"""

from unittest.mock import MagicMock, Mock

import pytest

try:
    from Jotty.core.intelligence.learning.health_budget import (
        DynamicBudgetManager,
        LearningHealthMonitor,
    )

    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False

try:
    from Jotty.core.infrastructure.foundation.data_structures import (
        GoalValue,
        MemoryEntry,
        MemoryLevel,
        SwarmConfig,
    )

    HAS_DATA_STRUCTURES = True
except ImportError:
    HAS_DATA_STRUCTURES = False


def _make_swarm_config(**overrides):
    """Create a SwarmConfig with sensible test defaults, applying overrides."""
    if HAS_DATA_STRUCTURES:
        return SwarmConfig(**overrides)
    cfg = MagicMock()
    defaults = dict(
        max_context_tokens=100000,
        system_prompt_budget=5000,
        current_input_budget=15000,
        trajectory_budget=20000,
        tool_output_budget=15000,
        memory_budget=45000,
        enable_dynamic_budget=False,
        min_memory_budget=10000,
        max_memory_budget=60000,
        max_entry_tokens=2000,
        suspicion_threshold=0.95,
        min_rejection_rate=0.05,
        stall_threshold=0.001,
        reasoning_weight=0.3,
        evidence_weight=0.2,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_memory_entry(key, content, token_count=100, goal=None, value=0.5):
    """Create a MemoryEntry for budget tests."""
    if HAS_DATA_STRUCTURES:
        entry = MemoryEntry(
            key=key,
            content=content,
            level=MemoryLevel.EPISODIC,
            context={},
            token_count=token_count,
        )
        if goal:
            entry.goal_values[goal] = GoalValue(value=value)
        return entry
    mock = MagicMock()
    mock.key = key
    mock.content = content
    mock.token_count = token_count
    mock.get_value = Mock(return_value=value)
    return mock


# =============================================================================
# LearningHealthMonitor tests
# =============================================================================


@pytest.mark.skipif(not HAS_HEALTH, reason="health_budget not importable")
class TestLearningHealthMonitor:
    """Tests for LearningHealthMonitor."""

    def _monitor(self, **overrides):
        cfg = _make_swarm_config(**overrides)
        return LearningHealthMonitor(cfg)

    @pytest.mark.unit
    def test_init(self):
        m = self._monitor()
        assert m.metrics.episode_count == 0

    @pytest.mark.unit
    def test_record_episode_increments_count(self):
        m = self._monitor()
        m.record_episode(
            success=True,
            goal="goal1",
            architect_decisions=[True],
            auditor_decisions=[True],
            value_updates=[("k", 0.5, 0.6)],
        )
        assert m.metrics.episode_count == 1

    @pytest.mark.unit
    def test_record_episode_tracks_success(self):
        m = self._monitor()
        m.record_episode(
            success=True,
            goal="g",
            architect_decisions=[True],
            auditor_decisions=[],
            value_updates=[],
        )
        assert m.metrics.success_count == 1

    @pytest.mark.unit
    def test_detect_reward_hacking_below_threshold(self):
        m = self._monitor()
        assert m._detect_reward_hacking() is False

    @pytest.mark.unit
    def test_detect_reward_hacking_triggers(self):
        m = self._monitor(suspicion_threshold=0.9)
        m.metrics.recent_successes = [True] * 60
        assert m._detect_reward_hacking() is True

    @pytest.mark.unit
    def test_detect_reward_hacking_not_triggered_below_rate(self):
        m = self._monitor(suspicion_threshold=0.95)
        m.metrics.recent_successes = [True] * 45 + [False] * 15
        assert m._detect_reward_hacking() is False

    @pytest.mark.unit
    def test_detect_conservative_collapse_not_enough_episodes(self):
        m = self._monitor()
        assert m._detect_conservative_collapse(0.01) is False

    @pytest.mark.unit
    def test_detect_conservative_collapse_triggers(self):
        m = self._monitor(min_rejection_rate=0.05)
        m.metrics.episode_count = 25
        assert m._detect_conservative_collapse(0.01) is True

    @pytest.mark.unit
    def test_detect_conservative_collapse_not_triggered(self):
        m = self._monitor(min_rejection_rate=0.05)
        m.metrics.episode_count = 25
        assert m._detect_conservative_collapse(0.5) is False

    @pytest.mark.unit
    def test_detect_learning_stall_not_enough_data(self):
        m = self._monitor()
        assert m._detect_learning_stall() is False

    @pytest.mark.unit
    def test_detect_learning_stall_triggers(self):
        m = self._monitor(stall_threshold=0.001)
        m.metrics.value_changes = [0.0001] * 110
        assert m._detect_learning_stall() is True

    @pytest.mark.unit
    def test_detect_learning_stall_not_triggered(self):
        m = self._monitor(stall_threshold=0.001)
        m.metrics.value_changes = [0.1] * 110
        assert m._detect_learning_stall() is False

    @pytest.mark.unit
    def test_detect_goal_drift_no_drift(self):
        m = self._monitor()
        result = m._detect_goal_drift("goal_a")
        assert result is None

    @pytest.mark.unit
    def test_detect_goal_drift_single_goal_dominating(self):
        m = self._monitor()
        for _ in range(55):
            m._detect_goal_drift("same_goal")
        result = m._detect_goal_drift("same_goal")
        assert result is not None
        assert "Single goal dominating" in result

    @pytest.mark.unit
    def test_get_health_summary(self):
        m = self._monitor()
        m.record_episode(
            success=True,
            goal="g",
            architect_decisions=[True],
            auditor_decisions=[],
            value_updates=[("k", 0.5, 0.6)],
        )
        summary = m.get_health_summary()
        assert summary["episode_count"] == 1
        assert "success_rate" in summary
        assert "learning_velocity" in summary

    @pytest.mark.unit
    def test_record_episode_returns_alerts(self):
        m = self._monitor(suspicion_threshold=0.9, min_rejection_rate=0.5)
        m.metrics.episode_count = 25
        m.metrics.recent_successes = [True] * 55
        alerts = m.record_episode(
            success=True,
            goal="g",
            architect_decisions=[False, False],
            auditor_decisions=[],
            value_updates=[],
        )
        assert isinstance(alerts, list)

    @pytest.mark.unit
    def test_record_episode_tracks_value_changes(self):
        m = self._monitor()
        m.record_episode(
            success=True,
            goal="g",
            architect_decisions=[True],
            auditor_decisions=[],
            value_updates=[("k1", 0.5, 0.7), ("k2", 0.3, 0.4)],
        )
        assert len(m.metrics.value_changes) == 2
        assert m.metrics.value_changes[0] == pytest.approx(0.2)

    @pytest.mark.unit
    def test_record_episode_goals_seen(self):
        m = self._monitor()
        m.record_episode(
            success=True,
            goal="goal_a",
            architect_decisions=[True],
            auditor_decisions=[],
            value_updates=[],
        )
        m.record_episode(
            success=True,
            goal="goal_b",
            architect_decisions=[True],
            auditor_decisions=[],
            value_updates=[],
        )
        assert "goal_a" in m.metrics.goals_seen
        assert "goal_b" in m.metrics.goals_seen


# =============================================================================
# DynamicBudgetManager tests
# =============================================================================


@pytest.mark.skipif(not HAS_HEALTH, reason="health_budget not importable")
class TestDynamicBudgetManager:
    """Tests for DynamicBudgetManager."""

    def _manager(self, **overrides):
        cfg = _make_swarm_config(**overrides)
        return DynamicBudgetManager(cfg)

    @pytest.mark.unit
    def test_init(self):
        mgr = self._manager()
        assert mgr.total_budget == 100000

    @pytest.mark.unit
    def test_static_allocation(self):
        mgr = self._manager(enable_dynamic_budget=False)
        alloc = mgr.compute_allocation(
            system_prompt_tokens=3000,
            input_tokens=5000,
            trajectory_tokens=10000,
            tool_output_tokens=8000,
        )
        assert alloc["system_prompt"] == 5000
        assert alloc["current_input"] == 15000
        assert alloc["trajectory"] == 20000
        assert alloc["tool_output"] == 15000
        assert "memory" in alloc

    @pytest.mark.unit
    def test_dynamic_allocation_basic(self):
        mgr = self._manager(enable_dynamic_budget=True)
        alloc = mgr.compute_allocation(
            system_prompt_tokens=3000,
            input_tokens=5000,
            trajectory_tokens=10000,
            tool_output_tokens=8000,
        )
        used = 3000 + 5000 + 10000 + 8000
        expected_memory = min(60000, max(10000, 100000 - used))
        assert alloc["memory"] == expected_memory
        assert alloc["system_prompt"] == 3000

    @pytest.mark.unit
    def test_dynamic_allocation_min_memory(self):
        mgr = self._manager(
            enable_dynamic_budget=True,
            max_context_tokens=50000,
            min_memory_budget=10000,
        )
        alloc = mgr.compute_allocation(
            system_prompt_tokens=10000,
            input_tokens=15000,
            trajectory_tokens=15000,
            tool_output_tokens=10000,
        )
        assert alloc["memory"] >= 10000

    @pytest.mark.unit
    def test_dynamic_allocation_max_memory(self):
        mgr = self._manager(
            enable_dynamic_budget=True,
            max_context_tokens=200000,
            max_memory_budget=60000,
        )
        alloc = mgr.compute_allocation(
            system_prompt_tokens=100,
            input_tokens=100,
            trajectory_tokens=100,
            tool_output_tokens=100,
        )
        assert alloc["memory"] <= 60000

    @pytest.mark.unit
    def test_dynamic_allocation_trajectory_reduction_on_overflow(self):
        mgr = self._manager(
            enable_dynamic_budget=True,
            max_context_tokens=50000,
            min_memory_budget=20000,
        )
        alloc = mgr.compute_allocation(
            system_prompt_tokens=10000,
            input_tokens=10000,
            trajectory_tokens=15000,
            tool_output_tokens=10000,
        )
        assert alloc["trajectory"] <= 15000

    @pytest.mark.unit
    def test_select_within_budget_basic(self):
        mgr = self._manager()
        items = [
            _make_memory_entry("k1", "content1", token_count=100, goal="g", value=0.9),
            _make_memory_entry("k2", "content2", token_count=100, goal="g", value=0.5),
            _make_memory_entry("k3", "content3", token_count=100, goal="g", value=0.7),
        ]
        selected = mgr.select_within_budget(items, budget=250, goal="g")
        assert len(selected) == 2

    @pytest.mark.unit
    def test_select_within_budget_respects_max_items(self):
        mgr = self._manager()
        items = [
            _make_memory_entry(f"k{i}", f"c{i}", token_count=10, goal="g", value=0.5)
            for i in range(100)
        ]
        selected = mgr.select_within_budget(items, budget=10000, goal="g", max_items=5)
        assert len(selected) <= 5

    @pytest.mark.unit
    def test_select_within_budget_skips_oversized(self):
        mgr = self._manager(max_entry_tokens=500)
        items = [
            _make_memory_entry("k1", "small", token_count=100, goal="g", value=0.9),
            _make_memory_entry("k2", "huge", token_count=1000, goal="g", value=1.0),
        ]
        selected = mgr.select_within_budget(items, budget=5000, goal="g")
        keys = [s.key for s in selected]
        assert "k1" in keys
        assert "k2" not in keys

    @pytest.mark.unit
    def test_select_within_budget_empty(self):
        mgr = self._manager()
        selected = mgr.select_within_budget([], budget=1000, goal="g")
        assert selected == []

    @pytest.mark.unit
    def test_select_within_budget_zero_budget(self):
        mgr = self._manager()
        items = [
            _make_memory_entry("k1", "c", token_count=100, goal="g", value=0.9),
        ]
        selected = mgr.select_within_budget(items, budget=0, goal="g")
        assert selected == []

    @pytest.mark.unit
    def test_select_within_budget_priority_order(self):
        mgr = self._manager()
        items = [
            _make_memory_entry("low", "c", token_count=100, goal="g", value=0.1),
            _make_memory_entry("high", "c", token_count=100, goal="g", value=0.9),
            _make_memory_entry("mid", "c", token_count=100, goal="g", value=0.5),
        ]
        selected = mgr.select_within_budget(items, budget=150, goal="g")
        assert len(selected) == 1
        assert selected[0].key == "high"
