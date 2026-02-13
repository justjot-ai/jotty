"""
Tests for Learning Core Module
================================
Tests for TDLambdaLearner, GroupedValueBaseline, and LearningManager.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any


# =============================================================================
# GroupedValueBaseline Tests
# =============================================================================

class TestGroupedValueBaseline:
    """Tests for HRPO-inspired grouped baselines."""

    @pytest.mark.unit
    def test_default_baseline(self):
        """Default baseline for unknown task type is 0.5."""
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        baseline = GroupedValueBaseline()
        assert baseline.get_baseline("unknown_task") == 0.5

    @pytest.mark.unit
    def test_baseline_update_ema(self):
        """Baseline updates via EMA."""
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        baseline = GroupedValueBaseline(ema_alpha=0.5)

        # Need min 3 samples for baseline to activate
        for reward in [0.8, 0.9, 1.0]:
            baseline.update_group("research", reward)

        result = baseline.get_baseline("research")
        assert 0.5 < result <= 1.0  # Should be above default

    @pytest.mark.unit
    def test_baseline_requires_min_samples(self):
        """Baseline returns default until min samples met."""
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        baseline = GroupedValueBaseline()

        baseline.update_group("research", 0.9)
        baseline.update_group("research", 0.8)
        # Only 2 samples, need 3
        result = baseline.get_baseline("research")
        assert result == 0.5  # Still default

    @pytest.mark.unit
    def test_relative_advantage(self):
        """compute_relative_advantage returns reward - baseline."""
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        baseline = GroupedValueBaseline()

        for _ in range(5):
            baseline.update_group("test", 0.5)

        advantage = baseline.compute_relative_advantage("test", 0.8)
        # Should be positive (0.8 > baseline ~0.5)
        assert advantage > 0

    @pytest.mark.unit
    def test_group_variance_default(self):
        """Variance defaults to 0.25 with insufficient samples."""
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        baseline = GroupedValueBaseline()
        assert baseline.get_group_variance("unknown") == 0.25

    @pytest.mark.unit
    def test_statistics(self):
        """get_statistics returns correct structure."""
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        baseline = GroupedValueBaseline()

        for _ in range(3):
            baseline.update_group("research", 0.7)
            baseline.update_group("coding", 0.8)

        stats = baseline.get_statistics()
        assert stats['num_groups'] == 2
        assert stats['total_samples'] == 6


# =============================================================================
# TDLambdaLearner Tests
# =============================================================================

class TestTDLambdaLearner:
    """Tests for TDLambdaLearner core algorithm."""

    @pytest.fixture
    def learner(self, minimal_jotty_config):
        """Create a TDLambdaLearner with test config."""
        from Jotty.core.learning.td_lambda import TDLambdaLearner
        return TDLambdaLearner(minimal_jotty_config)

    @pytest.mark.unit
    def test_creation(self, learner):
        """TDLambdaLearner creates with config values."""
        assert learner.gamma > 0
        assert learner.lambda_trace > 0
        assert learner.alpha > 0

    @pytest.mark.unit
    def test_start_episode_resets_state(self, learner):
        """start_episode resets traces and state."""
        learner.start_episode("test goal", task_type="research")
        assert learner.current_goal == "test goal"
        assert learner.current_task_type == "research"
        assert len(learner.traces) == 0

    @pytest.mark.unit
    def test_record_access_creates_trace(self, learner, minimal_jotty_config):
        """record_access creates eligibility trace for memory."""
        from Jotty.core.foundation.data_structures import MemoryLevel, MemoryEntry
        learner.start_episode("test goal")

        entry = MemoryEntry(
            content="test content",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test goal",
            initial_value=0.5,
        )
        entry.key = "test:research:abc123"

        trace_value = learner.record_access(entry)
        assert trace_value > 0
        assert len(learner.traces) == 1

    @pytest.mark.unit
    def test_trace_accumulation(self, learner, minimal_jotty_config):
        """Multiple accesses accumulate traces (not replace)."""
        from Jotty.core.foundation.data_structures import MemoryLevel, MemoryEntry
        learner.start_episode("test")

        entry = MemoryEntry(
            content="test",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test",
            initial_value=0.5,
        )
        entry.key = "test:key:abc"

        v1 = learner.record_access(entry)
        v2 = learner.record_access(entry)
        # Second access should accumulate
        assert v2 > v1

    @pytest.mark.unit
    def test_trace_decay(self, learner, minimal_jotty_config):
        """Traces decay with gamma * lambda on each step."""
        from Jotty.core.foundation.data_structures import MemoryLevel, MemoryEntry
        learner.start_episode("test")

        entry1 = MemoryEntry(content="first", level=MemoryLevel.EPISODIC,
                              context={}, goal="test", initial_value=0.5)
        entry1.key = "key1"
        entry2 = MemoryEntry(content="second", level=MemoryLevel.EPISODIC,
                              context={}, goal="test", initial_value=0.5)
        entry2.key = "key2"

        learner.record_access(entry1)
        initial_trace = learner.traces.get("key1", 0)

        # Recording access to another entry decays existing traces
        learner.record_access(entry2)
        decayed_trace = learner.traces.get("key1", 0)

        assert decayed_trace < initial_trace

    @pytest.mark.unit
    def test_update_single_step(self, learner):
        """update() performs single-step TD(0) update."""
        state = {"task_type": "research", "action_type": "search"}
        action = {"type": "search", "tool": "web-search"}
        reward = 0.8
        next_state = {"task_type": "research", "action_type": "summarize"}

        learner.update(state, action, reward, next_state)
        # Should not raise, and internal state should be updated
        stats = learner.get_grouped_learning_stats()
        assert stats is not None

    @pytest.mark.unit
    def test_end_episode_returns_updates(self, learner, minimal_jotty_config):
        """end_episode returns list of (key, old_val, new_val) tuples."""
        from Jotty.core.foundation.data_structures import MemoryLevel, MemoryEntry
        learner.start_episode("test goal", task_type="research")

        entry = MemoryEntry(content="test", level=MemoryLevel.EPISODIC,
                            context={}, goal="test goal", initial_value=0.5)
        entry.key = "research:test:abc"
        entry.goal_values = {"test goal": 0.5}

        learner.record_access(entry)
        memories = {"research:test:abc": entry}

        updates = learner.end_episode(
            final_reward=0.9,
            memories=memories,
        )
        assert isinstance(updates, list)
        if len(updates) > 0:
            key, old_val, new_val = updates[0]
            assert isinstance(key, str)
            assert isinstance(old_val, (int, float))
            assert isinstance(new_val, (int, float))

    @pytest.mark.unit
    def test_value_clipping(self, learner, minimal_jotty_config):
        """Values are clipped to [0, 1]."""
        from Jotty.core.foundation.data_structures import MemoryLevel, MemoryEntry
        learner.start_episode("test", task_type="test")

        entry = MemoryEntry(content="test", level=MemoryLevel.EPISODIC,
                            context={}, goal="test", initial_value=0.99)
        entry.key = "test:key:abc"
        entry.goal_values = {"test": 0.99}

        learner.record_access(entry)
        memories = {"test:key:abc": entry}

        updates = learner.end_episode(final_reward=1.0, memories=memories)
        for key, old_val, new_val in updates:
            assert 0 <= new_val <= 1


# =============================================================================
# LearningManager Tests
# =============================================================================

class TestLearningManager:
    """Tests for LearningManager lifecycle."""

    @pytest.mark.unit
    def test_creation(self, tmp_path):
        """LearningManager creates with config."""
        from Jotty.core.learning.learning_coordinator import LearningManager
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            output_base_dir=str(tmp_path),
            create_run_folder=False,
            enable_beautified_logs=False,
        )
        manager = LearningManager(config, base_dir=str(tmp_path / "learning"))
        assert manager is not None
        assert manager.session_id is not None

    @pytest.mark.unit
    def test_initialize(self, tmp_path):
        """initialize() creates session directory."""
        from Jotty.core.learning.learning_coordinator import LearningManager
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            output_base_dir=str(tmp_path),
            create_run_folder=False,
            enable_beautified_logs=False,
        )
        manager = LearningManager(config, base_dir=str(tmp_path / "learning"))
        result = manager.initialize(auto_load=False)
        # First session, no previous learning
        assert result is False

    @pytest.mark.unit
    def test_record_experience(self, tmp_path):
        """record_experience stores agent experience."""
        from Jotty.core.learning.learning_coordinator import LearningManager
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            output_base_dir=str(tmp_path),
            create_run_folder=False,
            enable_beautified_logs=False,
        )
        manager = LearningManager(config, base_dir=str(tmp_path / "learning"))
        manager.initialize(auto_load=False)

        update = manager.record_experience(
            agent_name="TestAgent",
            state={"query": "test"},
            action={"tool": "web-search"},
            reward=0.8,
        )
        assert update is not None
        assert update.actor == "TestAgent"
        assert update.reward == 0.8

    @pytest.mark.unit
    def test_get_learning_summary(self, tmp_path):
        """get_learning_summary returns correct structure."""
        from Jotty.core.learning.learning_coordinator import LearningManager
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            output_base_dir=str(tmp_path),
            create_run_folder=False,
            enable_beautified_logs=False,
        )
        manager = LearningManager(config, base_dir=str(tmp_path / "learning"))
        manager.initialize(auto_load=False)

        summary = manager.get_learning_summary()
        assert isinstance(summary, dict)
        assert 'session_id' in summary
        assert 'total_sessions' in summary

    @pytest.mark.unit
    def test_save_and_load_session(self, tmp_path):
        """save_all and load_session round-trip."""
        from Jotty.core.learning.learning_coordinator import LearningManager
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            output_base_dir=str(tmp_path),
            create_run_folder=False,
            enable_beautified_logs=False,
        )
        manager = LearningManager(config, base_dir=str(tmp_path / "learning"))
        manager.initialize(auto_load=False)

        # Record some experience
        manager.record_experience(
            agent_name="Agent1",
            state={"query": "test"},
            action={"tool": "search"},
            reward=0.7,
        )

        # Save
        manager.save_all(episode_count=1, avg_reward=0.7, domains=["test"])

        # Load in new manager
        manager2 = LearningManager(config, base_dir=str(tmp_path / "learning"))
        loaded = manager2.initialize(auto_load=True)
        # Should find and load the saved session
        assert loaded is True

    @pytest.mark.unit
    def test_singleton_factory(self):
        """get_learning_coordinator returns singleton."""
        from Jotty.core.learning.learning_coordinator import (
            get_learning_coordinator, reset_learning_coordinator,
        )
        reset_learning_coordinator()
        lm1 = get_learning_coordinator()
        lm2 = get_learning_coordinator()
        assert lm1 is lm2
        reset_learning_coordinator()

    @pytest.mark.unit
    def test_predict_q_value(self, tmp_path):
        """predict_q_value returns tuple of (q, confidence, suggestion)."""
        from Jotty.core.learning.learning_coordinator import LearningManager
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            output_base_dir=str(tmp_path),
            create_run_folder=False,
            enable_beautified_logs=False,
        )
        manager = LearningManager(config, base_dir=str(tmp_path / "learning"))
        manager.initialize(auto_load=False)

        q_value, confidence, suggestion = manager.predict_q_value(
            state={"query": "test"},
            action={"tool": "search"},
        )
        # May return None/low values for untrained model
        assert q_value is None or isinstance(q_value, float)
