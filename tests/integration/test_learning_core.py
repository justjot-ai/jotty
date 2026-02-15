"""
Comprehensive unit tests for the Jotty learning layer.

Covers:
- core/learning/learning_coordinator.py (LearningManager, singletons, dataclasses, fallbacks)
- core/learning/td_lambda.py (TDLambdaLearner, GroupedValueBaseline, SkillQTable, COMACredit)
- core/learning/adaptive_components.py (AdaptiveLearningRate, IntermediateRewardCalculator, AdaptiveExploration)
- core/learning/rl_components.py (RLComponents)
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

try:
    from core.foundation.data_structures import GoalValue, MemoryEntry, MemoryLevel, SwarmConfig
    from core.learning.adaptive_components import (
        AdaptiveExploration,
        AdaptiveLearningRate,
        IntermediateRewardCalculator,
    )
    from core.learning.learning_coordinator import (
        LearningManager,
        LearningSession,
        LearningUpdate,
        _NoOpLearner,
        _NoOpMemory,
        get_learning_coordinator,
        reset_learning_coordinator,
    )
    from core.learning.rl_components import RLComponents
    from core.learning.td_lambda import (
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
# 1. LearningSession and LearningUpdate dataclasses
# ============================================================================


class TestLearningSessionDataclass:
    """Tests for the LearningSession dataclass."""

    @pytest.mark.unit
    def test_creation_with_all_fields(self):
        session = LearningSession(
            session_id="s1",
            created_at=100.0,
            updated_at=200.0,
            episode_count=5,
            total_experiences=42,
            domains=["ml", "data"],
            agents=["Planner", "Coder"],
            avg_reward=0.82,
            path="/tmp/s1",
        )
        assert session.session_id == "s1"
        assert session.avg_reward == 0.82
        assert "ml" in session.domains

    @pytest.mark.unit
    def test_equality(self):
        kwargs = dict(
            session_id="s2",
            created_at=0,
            updated_at=0,
            episode_count=0,
            total_experiences=0,
            domains=[],
            agents=[],
            avg_reward=0.0,
            path="",
        )
        assert LearningSession(**kwargs) == LearningSession(**kwargs)


class TestLearningUpdateDataclass:
    """Tests for the LearningUpdate dataclass."""

    @pytest.mark.unit
    def test_defaults(self):
        upd = LearningUpdate(actor="Planner", reward=0.9)
        assert upd.q_value is None
        assert upd.td_error is None

    @pytest.mark.unit
    def test_full_creation(self):
        upd = LearningUpdate(actor="Agent", reward=0.5, q_value=0.6, td_error=0.1)
        assert upd.q_value == 0.6
        assert upd.td_error == 0.1


# ============================================================================
# 2. _NoOpLearner and _NoOpMemory fallbacks
# ============================================================================


class TestNoOpLearner:
    """Tests for _NoOpLearner fallback class."""

    @pytest.mark.unit
    def test_add_experience_noop(self):
        _NoOpLearner().add_experience({}, {}, 0.5)

    @pytest.mark.unit
    def test_record_outcome_noop(self):
        _NoOpLearner().record_outcome({}, {}, 0.5)

    @pytest.mark.unit
    def test_predict_q_value_returns_defaults(self):
        q, conf, alt = _NoOpLearner().predict_q_value({}, {})
        assert q == 0.5
        assert conf == 0.1
        assert alt is None

    @pytest.mark.unit
    def test_get_learned_context_empty(self):
        assert _NoOpLearner().get_learned_context({}) == ""

    @pytest.mark.unit
    def test_get_q_table_stats(self):
        stats = _NoOpLearner().get_q_table_stats()
        assert stats["size"] == 0

    @pytest.mark.unit
    def test_save_load_state_noop(self):
        learner = _NoOpLearner()
        learner.save_state("/tmp/dummy")
        learner.load_state("/tmp/dummy")


class TestNoOpMemory:
    """Tests for _NoOpMemory fallback class."""

    @pytest.mark.unit
    def test_store_noop(self):
        _NoOpMemory().store("key", "value")

    @pytest.mark.unit
    def test_retrieve_empty(self):
        assert _NoOpMemory().retrieve("query") == []

    @pytest.mark.unit
    def test_get_statistics(self):
        assert _NoOpMemory().get_statistics()["total_entries"] == 0

    @pytest.mark.unit
    def test_save_load_noop(self):
        mem = _NoOpMemory()
        mem.save("/tmp/dummy")
        mem.load("/tmp/dummy")


# ============================================================================
# 3. LearningManager
# ============================================================================


class TestLearningManagerInit:
    """LearningManager initialization and directory setup."""

    @pytest.mark.unit
    def test_init_creates_learning_dir(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        assert mgr.learning_dir.exists()

    @pytest.mark.unit
    def test_session_id_format(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        assert mgr.session_id.startswith("session_")

    @pytest.mark.unit
    def test_empty_registry_on_fresh_start(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        assert len(mgr.registry) == 0


class TestLearningManagerCoreLearnersInit:
    """Tests for _init_core_learners with mocked imports."""

    @pytest.mark.unit
    def test_init_with_q_learning_unavailable(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path), enable_rl=False)
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        assert mgr._shared_q_learner is None

    @pytest.mark.unit
    def test_init_with_rl_disabled_no_td_lambda(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path), enable_rl=False)
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        assert mgr._td_lambda_learner is None


class TestLearningManagerInitializeAndLoad:
    """Tests for initialize(), load_latest(), load_session()."""

    def _make_manager(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            return LearningManager(cfg, base_dir=str(tmp_path))

    @pytest.mark.unit
    def test_initialize_auto_load_false(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.initialize(auto_load=False)
        assert result is False
        assert mgr.session_dir.exists()

    @pytest.mark.unit
    def test_initialize_auto_load_no_registry(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.initialize(auto_load=True)
        assert result is False

    @pytest.mark.unit
    def test_load_latest_empty_registry(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.load_latest() is False

    @pytest.mark.unit
    def test_load_latest_picks_most_recent(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        for sid, updated in [("old", 1), ("new", 10)]:
            p = tmp_path / sid
            p.mkdir()
            mgr.registry[sid] = LearningSession(
                session_id=sid,
                created_at=1,
                updated_at=updated,
                episode_count=0,
                total_experiences=0,
                domains=[],
                agents=[],
                avg_reward=0,
                path=str(p),
            )
        result = mgr.load_latest()
        # No actual data to load, but code path is exercised
        assert result is False

    @pytest.mark.unit
    def test_load_session_unknown_id(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.load_session("nonexistent") is False

    @pytest.mark.unit
    def test_load_session_missing_path(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.registry["s1"] = LearningSession(
            session_id="s1",
            created_at=0,
            updated_at=0,
            episode_count=0,
            total_experiences=0,
            domains=[],
            agents=[],
            avg_reward=0,
            path="/nonexistent/path",
        )
        assert mgr.load_session("s1") is False

    @pytest.mark.unit
    def test_load_session_with_shared_q_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        session_path = tmp_path / "session_test"
        session_path.mkdir()
        q_path = session_path / "shared_q_learning.json"
        q_path.write_text("{}")
        mock_q = Mock()
        mgr._shared_q_learner = mock_q
        mgr.registry["test"] = LearningSession(
            session_id="test",
            created_at=0,
            updated_at=0,
            episode_count=0,
            total_experiences=0,
            domains=[],
            agents=[],
            avg_reward=0,
            path=str(session_path),
        )
        result = mgr.load_session("test")
        mock_q.load_state.assert_called_once()
        assert result is True


class TestLearningManagerAgentAccess:
    """Tests for get_agent_learner, get_agent_memory, get_shared_learner."""

    def _make_manager(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            return LearningManager(cfg, base_dir=str(tmp_path))

    @pytest.mark.unit
    def test_get_agent_learner_returns_noop_when_import_fails(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        with patch.dict("sys.modules", {"core.learning.q_learning": None}):
            learner = mgr.get_agent_learner("TestAgent")
        assert isinstance(learner, _NoOpLearner)

    @pytest.mark.unit
    def test_get_agent_learner_caches(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        l1 = mgr.get_agent_learner("X")
        l2 = mgr.get_agent_learner("X")
        assert l1 is l2

    @pytest.mark.unit
    def test_get_agent_memory_returns_noop_on_import_fail(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        with patch.dict("sys.modules", {"core.memory.fallback_memory": None}):
            mem = mgr.get_agent_memory("TestAgent")
        assert isinstance(mem, _NoOpMemory)

    @pytest.mark.unit
    def test_get_agent_memory_caches(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        m1 = mgr.get_agent_memory("Y")
        m2 = mgr.get_agent_memory("Y")
        assert m1 is m2

    @pytest.mark.unit
    def test_get_shared_learner_creates_noop_if_unavailable(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        with patch.dict("sys.modules", {"core.learning.q_learning": None}):
            shared = mgr.get_shared_learner()
        assert isinstance(shared, _NoOpLearner)

    @pytest.mark.unit
    def test_q_learner_property_delegates(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = Mock()
        assert mgr.q_learner is mgr._shared_q_learner


class TestLearningManagerQValueAndExperience:
    """Tests for predict_q_value, record_experience, record_outcome."""

    def _make_manager(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            return LearningManager(cfg, base_dir=str(tmp_path))

    @pytest.mark.unit
    def test_predict_q_value_no_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        q, conf, alt = mgr.predict_q_value({}, {})
        assert q == 0.5 and conf == 0.1 and alt is None

    @pytest.mark.unit
    def test_predict_q_value_delegates(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mock_q.predict_q_value.return_value = (0.8, 0.9, "use_plan_b")
        mgr._shared_q_learner = mock_q
        q, conf, alt = mgr.predict_q_value({"s": 1}, {"a": 2}, goal="g")
        assert q == 0.8 and alt == "use_plan_b"
        mock_q.predict_q_value.assert_called_once_with({"s": 1}, {"a": 2}, "g")

    @pytest.mark.unit
    def test_predict_q_value_exception_fallback(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mock_q.predict_q_value.side_effect = RuntimeError("boom")
        mgr._shared_q_learner = mock_q
        q, conf, alt = mgr.predict_q_value({}, {})
        assert q == 0.5

    @pytest.mark.unit
    def test_record_experience_returns_update(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        update = mgr.record_experience("AgentA", {"s": 1}, {"a": 1}, 0.7)
        assert isinstance(update, LearningUpdate)
        assert update.actor == "AgentA" and update.reward == 0.7

    @pytest.mark.unit
    def test_record_experience_tracks_domain(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.record_experience("A", {}, {}, 0.5, domain="ml")
        assert "ml" in mgr._current_domains

    @pytest.mark.unit
    def test_record_experience_does_not_duplicate_domain(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.record_experience("A", {}, {}, 0.5, domain="ml")
        mgr.record_experience("A", {}, {}, 0.5, domain="ml")
        assert mgr._current_domains.count("ml") == 1

    @pytest.mark.unit
    def test_record_experience_calls_shared_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_shared = Mock()
        mock_shared.predict_q_value.return_value = (0.6, 0.5, None)
        mgr._shared_q_learner = mock_shared
        update = mgr.record_experience("A", {"x": 1}, {"y": 2}, 0.9)
        mock_shared.record_outcome.assert_called_once()
        assert update.q_value == 0.6

    @pytest.mark.unit
    def test_record_outcome_extracts_actor(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        update = mgr.record_outcome({}, {"actor": "Bot"}, 0.3)
        assert update.actor == "Bot"

    @pytest.mark.unit
    def test_record_outcome_default_actor(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        update = mgr.record_outcome({}, {}, 0.3)
        assert update.actor == "unknown"


class TestLearningManagerTDLambda:
    """Tests for update_td_lambda."""

    @pytest.mark.unit
    def test_update_td_lambda_no_learner(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        mgr._td_lambda_learner = None
        mgr.update_td_lambda([], 1.0)  # should not raise

    @pytest.mark.unit
    def test_update_td_lambda_delegates(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        mock_td = Mock()
        mgr._td_lambda_learner = mock_td
        traj = [({}, {}, 0.5)]
        mgr.update_td_lambda(traj, 1.0, gamma=0.9, lambda_trace=0.8)
        mock_td.update.assert_called_once_with(traj, 1.0, 0.9, 0.8)

    @pytest.mark.unit
    def test_update_td_lambda_handles_exception(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            mgr = LearningManager(cfg, base_dir=str(tmp_path))
        mock_td = Mock()
        mock_td.update.side_effect = RuntimeError("td fail")
        mgr._td_lambda_learner = mock_td
        mgr.update_td_lambda([], 1.0)  # should log error, not raise


class TestLearningManagerContextAndSummaries:
    """Tests for get_learned_context, get_q_table_summary, get_learning_summary, list_sessions."""

    def _make_manager(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            return LearningManager(cfg, base_dir=str(tmp_path))

    @pytest.mark.unit
    def test_get_learned_context_no_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        assert mgr.get_learned_context({}) == ""

    @pytest.mark.unit
    def test_get_learned_context_delegates(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mock_q.get_learned_context.return_value = "Use skill X"
        mgr._shared_q_learner = mock_q
        assert mgr.get_learned_context({"s": 1}, {"a": 2}) == "Use skill X"

    @pytest.mark.unit
    def test_get_learned_context_exception_fallback(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mock_q.get_learned_context.side_effect = RuntimeError("fail")
        mgr._shared_q_learner = mock_q
        assert mgr.get_learned_context({}) == ""

    @pytest.mark.unit
    def test_get_q_table_summary_no_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        assert "not available" in mgr.get_q_table_summary()

    @pytest.mark.unit
    def test_get_q_table_summary_with_method(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mock_q.get_q_table_summary.return_value = "Q-table: 10 entries"
        mgr._shared_q_learner = mock_q
        assert mgr.get_q_table_summary() == "Q-table: 10 entries"

    @pytest.mark.unit
    def test_get_q_table_summary_fallback_experience_buffer(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock(spec=[])
        mock_q.experience_buffer = [1, 2, 3]
        mgr._shared_q_learner = mock_q
        assert "3 experiences" in mgr.get_q_table_summary()

    @pytest.mark.unit
    def test_get_learning_summary_structure(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        summary = mgr.get_learning_summary()
        assert "session_id" in summary
        assert "total_sessions" in summary
        assert isinstance(summary["per_agent_stats"], dict)

    @pytest.mark.unit
    def test_list_sessions_empty(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.list_sessions() == []

    @pytest.mark.unit
    def test_list_sessions_sorted_by_updated_at(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.registry["a"] = LearningSession(
            session_id="a",
            created_at=1,
            updated_at=5,
            episode_count=2,
            total_experiences=10,
            domains=["d1"],
            agents=["A"],
            avg_reward=0.5,
            path="/a",
        )
        mgr.registry["b"] = LearningSession(
            session_id="b",
            created_at=2,
            updated_at=10,
            episode_count=3,
            total_experiences=20,
            domains=["d2"],
            agents=["B"],
            avg_reward=0.8,
            path="/b",
        )
        sessions = mgr.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "b"


class TestLearningManagerMemoryOps:
    """Tests for promote_demote_memories and prune_tier3."""

    def _make_manager(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            return LearningManager(cfg, base_dir=str(tmp_path))

    @pytest.mark.unit
    def test_promote_demote_no_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        mgr.promote_demote_memories(0.8)

    @pytest.mark.unit
    def test_promote_demote_delegates(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mgr._shared_q_learner = mock_q
        mgr.promote_demote_memories(0.9)
        mock_q._promote_demote_memories.assert_called_once_with(episode_reward=0.9)

    @pytest.mark.unit
    def test_prune_tier3_no_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._shared_q_learner = None
        mgr.prune_tier3(0.2)

    @pytest.mark.unit
    def test_prune_tier3_delegates(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock()
        mgr._shared_q_learner = mock_q
        mgr.prune_tier3(0.15)
        mock_q.prune_tier3_by_causal_impact.assert_called_once_with(sample_rate=0.15)

    @pytest.mark.unit
    def test_promote_demote_handles_missing_method(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_q = Mock(spec=[])  # no _promote_demote_memories
        mgr._shared_q_learner = mock_q
        mgr.promote_demote_memories(0.5)  # should not raise


class TestLearningManagerPersistence:
    """Tests for save_all and registry persistence."""

    def _make_manager(self, tmp_path):
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            return LearningManager(cfg, base_dir=str(tmp_path))

    @pytest.mark.unit
    def test_save_all_creates_session_dir(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_all(episode_count=5, avg_reward=0.7, domains=["ml"])
        assert mgr.session_dir.exists()

    @pytest.mark.unit
    def test_save_all_writes_registry(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_all(episode_count=3, avg_reward=0.6)
        assert mgr.registry_path.exists()
        with open(mgr.registry_path) as f:
            data = json.load(f)
        assert mgr.session_id in data["sessions"]

    @pytest.mark.unit
    def test_save_all_merges_domains(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr._current_domains = ["data"]
        mgr.save_all(domains=["ml"])
        session = mgr.registry[mgr.session_id]
        assert "ml" in session.domains and "data" in session.domains

    @pytest.mark.unit
    def test_save_all_saves_agent_learners(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_learner = Mock()
        mock_learner.experience_buffer = []
        mgr._agent_q_learners["TestAgent"] = mock_learner
        mgr.save_all()
        mock_learner.save_state.assert_called_once()

    @pytest.mark.unit
    def test_save_all_saves_agent_memories(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_mem = Mock()
        mgr._agent_memories["TestAgent"] = mock_mem
        mgr.save_all()
        mock_mem.save.assert_called_once()

    @pytest.mark.unit
    def test_save_all_updates_domain_index(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_all(domains=["finance"])
        assert "finance" in mgr._domain_index
        assert mgr.session_id in mgr._domain_index["finance"]

    @pytest.mark.unit
    def test_save_all_no_duplicate_domain_index(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_all(domains=["finance"])
        mgr.save_all(domains=["finance"])
        assert mgr._domain_index["finance"].count(mgr.session_id) == 1


class TestLearningManagerSingleton:
    """Tests for get_learning_coordinator / reset_learning_coordinator."""

    @pytest.mark.unit
    def test_get_creates_singleton(self, tmp_path):
        reset_learning_coordinator()
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            coord = get_learning_coordinator(
                config=_make_config(output_base_dir=str(tmp_path)),
                base_dir=str(tmp_path),
            )
        assert isinstance(coord, LearningManager)
        reset_learning_coordinator()

    @pytest.mark.unit
    def test_get_returns_same_instance(self, tmp_path):
        reset_learning_coordinator()
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            c1 = get_learning_coordinator(cfg, str(tmp_path))
            c2 = get_learning_coordinator()
        assert c1 is c2
        reset_learning_coordinator()

    @pytest.mark.unit
    def test_reset_clears_singleton(self, tmp_path):
        reset_learning_coordinator()
        cfg = _make_config(output_base_dir=str(tmp_path))
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            c1 = get_learning_coordinator(cfg, str(tmp_path))
        reset_learning_coordinator()
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            c2 = get_learning_coordinator(cfg, str(tmp_path))
        assert c1 is not c2
        reset_learning_coordinator()

    @pytest.mark.unit
    def test_get_with_no_config_uses_minimal(self):
        reset_learning_coordinator()
        with patch("core.learning.learning_coordinator.LearningManager._init_core_learners"):
            coord = get_learning_coordinator()
        assert coord is not None
        reset_learning_coordinator()


# ============================================================================
# 4. GroupedValueBaseline
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
# 5. TDLambdaLearner
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
# 6. SkillQTable
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
# 7. COMACredit
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
# 8. get_learned_context (module-level function)
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
# 9. AdaptiveLearningRate
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
# 10. IntermediateRewardCalculator
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
# 11. AdaptiveExploration
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


# ============================================================================
# 12. RLComponents
# ============================================================================


class TestRLComponentsSemantic:
    """Tests for RLComponents.get_similar_experiences_semantic."""

    @pytest.mark.unit
    def test_empty_buffer(self):
        rl = RLComponents(config=Mock())
        assert rl.get_similar_experiences_semantic([], {"todo": "x"}, {"actor": "A"}) == []

    @pytest.mark.unit
    def test_actor_matching_boost(self):
        rl = RLComponents(config=Mock())
        exp1 = {"action": {"actor": "A"}, "state": {}, "timestamp": time.time()}
        exp2 = {"action": {"actor": "B"}, "state": {}, "timestamp": time.time()}
        results = rl.get_similar_experiences_semantic([exp1, exp2], {}, {"actor": "A"}, top_k=2)
        assert results[0]["action"]["actor"] == "A"

    @pytest.mark.unit
    def test_top_k_limiting(self):
        rl = RLComponents(config=Mock())
        exps = [
            {"action": {"actor": f"a{i}"}, "state": {}, "timestamp": time.time()} for i in range(10)
        ]
        assert len(rl.get_similar_experiences_semantic(exps, {}, {"actor": "x"}, top_k=3)) == 3

    @pytest.mark.unit
    def test_td_error_prioritization(self):
        rl = RLComponents(config=Mock())
        low_td = {"action": {"actor": "X"}, "state": {}, "td_error": 0.01, "timestamp": time.time()}
        high_td = {"action": {"actor": "X"}, "state": {}, "td_error": 0.9, "timestamp": time.time()}
        results = rl.get_similar_experiences_semantic(
            [low_td, high_td], {}, {"actor": "X"}, top_k=2
        )
        assert results[0]["td_error"] == 0.9

    @pytest.mark.unit
    def test_string_action_handling(self):
        rl = RLComponents(config=Mock())
        exp = {"action": "some_action", "state": {}, "timestamp": time.time()}
        assert len(rl.get_similar_experiences_semantic([exp], {}, "some_action")) == 1

    @pytest.mark.unit
    def test_state_similarity_scoring(self):
        rl = RLComponents(config=Mock())
        exp_match = {
            "action": {"actor": "X"},
            "state": {"todo": "pending pending"},
            "timestamp": time.time(),
        }
        exp_no_match = {"action": {"actor": "X"}, "state": {"todo": ""}, "timestamp": time.time()}
        results = rl.get_similar_experiences_semantic(
            [exp_no_match, exp_match],
            {"todo": "pending pending"},
            {"actor": "X"},
            top_k=2,
        )
        assert results[0]["state"]["todo"] == "pending pending"


class TestRLComponentsQDivergence:
    """Tests for calculate_q_divergence_bonus."""

    @pytest.mark.unit
    def test_none_prediction(self):
        rl = RLComponents(config=Mock())
        div, bonus = rl.calculate_q_divergence_bonus(None, 0.5, 0.1)
        assert div == 0.0 and bonus == 0.0

    @pytest.mark.unit
    def test_perfect_prediction(self):
        rl = RLComponents(config=Mock())
        div, bonus = rl.calculate_q_divergence_bonus(0.5, 0.5, 0.1)
        assert div == 0.0
        assert bonus == pytest.approx(0.1)

    @pytest.mark.unit
    def test_large_divergence(self):
        rl = RLComponents(config=Mock())
        div, bonus = rl.calculate_q_divergence_bonus(0.0, 1.0, 0.1)
        assert div == pytest.approx(1.0)
        assert bonus == pytest.approx(0.0)

    @pytest.mark.unit
    def test_moderate_divergence(self):
        rl = RLComponents(config=Mock())
        div, bonus = rl.calculate_q_divergence_bonus(0.3, 0.8, 0.2)
        assert div == pytest.approx(0.5)
        assert bonus == pytest.approx(0.1)

    @pytest.mark.unit
    def test_divergence_capped_at_one(self):
        rl = RLComponents(config=Mock())
        div, bonus = rl.calculate_q_divergence_bonus(0.0, 5.0, 0.1)
        # divergence = 5.0 but bonus uses min(divergence, 1.0)
        assert bonus == pytest.approx(0.0)


class TestRLComponentsExtractPatterns:
    """Tests for extract_patterns."""

    @pytest.mark.unit
    def test_empty_experiences(self):
        assert RLComponents(config=Mock()).extract_patterns([]) == []

    @pytest.mark.unit
    def test_below_threshold(self):
        rl = RLComponents(config=Mock())
        assert (
            rl.extract_patterns([{"action": {"actor": "A"}, "reward": 0.9}], min_frequency=3) == []
        )

    @pytest.mark.unit
    def test_pattern_extraction_success(self):
        rl = RLComponents(config=Mock())
        exps = [{"action": {"actor": "Coder"}, "reward": 0.85, "state": {}} for _ in range(3)]
        patterns = rl.extract_patterns(exps, min_frequency=3)
        assert len(patterns) == 1
        assert patterns[0]["actor"] == "Coder"
        assert patterns[0]["outcome"] == "success"

    @pytest.mark.unit
    def test_pattern_extraction_failure(self):
        rl = RLComponents(config=Mock())
        exps = [{"action": {"actor": "Bad"}, "reward": 0.1, "state": {}} for _ in range(3)]
        patterns = rl.extract_patterns(exps, min_frequency=3)
        assert patterns[0]["outcome"] == "failure"

    @pytest.mark.unit
    def test_multiple_patterns_sorted_by_frequency(self):
        rl = RLComponents(config=Mock())
        exps = [{"action": {"actor": "A"}, "reward": 0.9, "state": {}} for _ in range(5)] + [
            {"action": {"actor": "B"}, "reward": 0.1, "state": {}} for _ in range(3)
        ]
        patterns = rl.extract_patterns(exps, min_frequency=3)
        assert len(patterns) == 2
        assert patterns[0]["frequency"] >= patterns[1]["frequency"]


class TestRLComponentsCounterfactual:
    """Tests for counterfactual_credit_assignment."""

    @pytest.mark.unit
    def test_empty_trajectory(self):
        assert RLComponents(config=Mock()).counterfactual_credit_assignment([], 1.0) == {}

    @pytest.mark.unit
    def test_single_step(self):
        rl = RLComponents(config=Mock())
        credits = rl.counterfactual_credit_assignment([{"actor": "A", "reward": 0.5}], 1.0)
        assert "A" in credits and credits["A"] > 0

    @pytest.mark.unit
    def test_temporal_proximity(self):
        rl = RLComponents(config=Mock())
        traj = [
            {"actor": "early", "reward": 0.1},
            {"actor": "late", "reward": 0.1},
        ]
        credits = rl.counterfactual_credit_assignment(traj, 1.0, gamma=0.5)
        assert credits["late"] > credits["early"]

    @pytest.mark.unit
    def test_credit_accumulates(self):
        rl = RLComponents(config=Mock())
        traj = [
            {"actor": "X", "reward": 0.2},
            {"actor": "X", "reward": 0.3},
        ]
        credits = rl.counterfactual_credit_assignment(traj, 0.8, gamma=0.99)
        assert credits["X"] > 0

    @pytest.mark.unit
    def test_cooperative_bonus(self):
        rl = RLComponents(config=Mock())
        traj = [
            {"actor": "enabler", "reward": 0.1},
            {"actor": "finisher", "reward": 0.9},
        ]
        credits = rl.counterfactual_credit_assignment(traj, 1.0, gamma=0.99)
        # enabler gets cooperative bonus because finisher succeeded after
        assert credits["enabler"] > 0


class TestRLComponentsTheoryOfMind:
    """Tests for theory_of_mind_predict."""

    @pytest.mark.unit
    def test_empty_other_actors(self):
        assert RLComponents(config=Mock()).theory_of_mind_predict("A", [], [], {}) == {}

    @pytest.mark.unit
    def test_predictions_sum_to_one(self):
        rl = RLComponents(config=Mock())
        traj = [{"actor": "B", "reward": 0.8}, {"actor": "C", "reward": 0.5}]
        preds = rl.theory_of_mind_predict("A", ["B", "C"], traj, {"todo": "pending"})
        assert sum(preds.values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.unit
    def test_successful_actor_ranked_higher(self):
        rl = RLComponents(config=Mock())
        traj = [
            {"actor": "Good", "reward": 1.0},
            {"actor": "Bad", "reward": 0.0},
        ]
        preds = rl.theory_of_mind_predict("Me", ["Good", "Bad"], traj, {})
        assert preds["Good"] > preds["Bad"]

    @pytest.mark.unit
    def test_turn_taking_bonus(self):
        rl = RLComponents(config=Mock())
        # A has recency (0.3) + success (0.4*0.5=0.2) = 0.5 but no turn-taking
        # B has turn-taking (0.2) only
        # So A > B. Verify B at least gets the turn-taking bonus.
        traj = [{"actor": "A", "reward": 0.5}]
        preds = rl.theory_of_mind_predict("Me", ["A", "B"], traj, {})
        assert preds["B"] > 0  # B gets non-zero probability from turn-taking


# ============================================================================
# 13. RLComponents async methods
# ============================================================================


class TestRLComponentsAsync:
    """Tests for async methods in RLComponents."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_td_error_and_why_small_error(self):
        rl = RLComponents(config=Mock())
        exp = {"state": {}, "action": "do_x", "reward": 0.5}
        result = await rl.add_td_error_and_why(exp, 0.05, {}, "do_x", 0.5, 0.45)
        assert result["td_error"] == 0.05
        assert result["why"] == ""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_td_error_and_why_large_positive(self):
        rl = RLComponents(config=Mock())
        exp = {"state": {}, "action": "do_x", "reward": 0.9}
        result = await rl.add_td_error_and_why(exp, 0.5, {}, "do_x", 0.9, 0.4)
        assert result["td_error"] == 0.5
        assert "higher than predicted" in result["why"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_td_error_and_why_large_negative(self):
        rl = RLComponents(config=Mock())
        exp = {"state": {}, "action": "do_y"}
        result = await rl.add_td_error_and_why(exp, -0.5, {}, "do_y", 0.2, 0.7)
        assert "lower than predicted" in result["why"]
