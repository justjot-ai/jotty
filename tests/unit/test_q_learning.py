"""
Tests for Q-Learning Module
============================
Covers LLMQPredictor: init, Q-value estimation, state/action handling,
experience management, tiered memory, and persistence.
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


def _make_config(**overrides):
    """Create a minimal mock config for LLMQPredictor."""
    config = MagicMock()
    config.tier1_max_size = overrides.get("tier1_max_size", 50)
    config.tier2_max_clusters = overrides.get("tier2_max_clusters", 10)
    config.tier3_max_size = overrides.get("tier3_max_size", 500)
    config.max_experience_buffer = overrides.get("max_experience_buffer", 1000)
    config.alpha = overrides.get("alpha", 0.1)
    config.gamma = overrides.get("gamma", 0.99)
    config.epsilon = overrides.get("epsilon", 0.1)
    config.max_q_table_size = overrides.get("max_q_table_size", 10000)
    config.q_prune_percentage = overrides.get("q_prune_percentage", 0.2)
    config.q_value_mode = overrides.get("q_value_mode", "simple")
    return config


def _make_predictor(**config_overrides):
    """Create an LLMQPredictor with mocked dspy internals."""
    with patch("dspy.ChainOfThought") as mock_cot:
        mock_cot.return_value = MagicMock()
        from Jotty.core.intelligence.learning.q_learning import LLMQPredictor

        config = _make_config(**config_overrides)
        predictor = LLMQPredictor(config)
    return predictor


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestLLMQPredictorInit:
    """Tests for LLMQPredictor initialization and basic configuration."""

    def test_init_default_parameters(self):
        """Predictor initializes with correct default learning parameters."""
        predictor = _make_predictor()
        assert predictor.alpha == 0.1
        assert predictor.gamma == 0.99
        assert predictor.epsilon == 0.1
        assert predictor.Q == {}
        assert predictor.experience_buffer == []

    def test_init_custom_parameters(self):
        """Predictor respects custom config values."""
        predictor = _make_predictor(alpha=0.2, gamma=0.9, epsilon=0.3)
        assert predictor.alpha == 0.2
        assert predictor.gamma == 0.9
        assert predictor.epsilon == 0.3

    def test_init_tiered_memory_empty(self):
        """Tiered memory structures are empty on init."""
        predictor = _make_predictor()
        assert predictor.tier1_working == []
        assert predictor.tier2_clusters == {}
        assert predictor.tier3_archive == []
        assert predictor.tier1_threshold == 0.8


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestStateToNaturalLanguage:
    """Tests for _state_to_natural_language conversion."""

    def test_empty_state_returns_initial(self):
        """Empty or None state returns 'Initial state' string."""
        predictor = _make_predictor()
        assert predictor._state_to_natural_language(None) == "Initial state (no history)"
        assert predictor._state_to_natural_language({}) == "Initial state (no history)"

    def test_state_with_query_extracts_intent(self):
        """State with query field extracts QUERY and INTENT."""
        predictor = _make_predictor()
        state = {"query": "count total transactions yesterday"}
        result = predictor._state_to_natural_language(state)
        assert "QUERY:" in result
        assert "COUNT" in result

    def test_state_with_tables_and_errors(self):
        """State with tables and error info is captured in description."""
        predictor = _make_predictor()
        state = {
            "tables": ["schema.fact_transactions"],
            "errors": [{"type": "COLUMN_NOT_FOUND", "column": "txn_date"}],
            "working_column": "dl_last_updated",
        }
        result = predictor._state_to_natural_language(state)
        assert "TABLES:" in result
        assert "fact_transactions" in result
        assert "ERRORS:" in result
        assert "WORKING_COL:" in result

    def test_state_fallback_shows_keys(self):
        """State with only skip_keys falls back to STATE_KEYS."""
        predictor = _make_predictor()
        # 'todo' is in the skip_keys set, and its value is not simple
        # but 'todo' as a dict will go through the todo section (section 10)
        # Use a key that IS in skip_keys but with a value that wont produce
        # output in the dedicated sections (e.g., empty list)
        state = {"todo": {}, "attempts": None, "success": None}
        result = predictor._state_to_natural_language(state)
        # At minimum it should produce some output, not crash
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestActionToNaturalLanguage:
    """Tests for _action_to_natural_language conversion."""

    def test_empty_action(self):
        """Empty or None action returns 'No action'."""
        predictor = _make_predictor()
        assert predictor._action_to_natural_language(None) == "No action"
        assert predictor._action_to_natural_language({}) == "No action"

    def test_action_with_actor_and_task(self):
        """Action with actor and task fields produces rich description."""
        predictor = _make_predictor()
        action = {"actor": "SQLGenerator", "task": "Generate SQL", "tool": "execute_query"}
        result = predictor._action_to_natural_language(action)
        assert "ACTOR: SQLGenerator" in result
        assert "TASK: Generate SQL" in result
        assert "TOOL: execute_query" in result


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestQValueUpdates:
    """Tests for Q-table updates and experience management."""

    def test_add_experience_creates_q_entry(self):
        """Adding experience creates entry in Q-table."""
        predictor = _make_predictor()
        state = {"query": "count users"}
        action = {"actor": "SQL", "task": "generate"}
        predictor.add_experience(state, action, 0.8)

        assert len(predictor.Q) == 1
        assert len(predictor.experience_buffer) == 1
        entry = list(predictor.Q.values())[0]
        assert entry["visit_count"] == 1
        assert 0.0 <= entry["value"] <= 1.0

    def test_q_value_clamped_to_unit_interval(self):
        """Q-values are always clamped to [0, 1]."""
        predictor = _make_predictor()
        # Add a very high reward to try pushing Q above 1.0
        state = {"query": "test clamping"}
        action = {"actor": "tester"}
        for _ in range(20):
            predictor.add_experience(state, action, 1.0)

        state_desc = predictor._state_to_natural_language(state)
        action_desc = predictor._action_to_natural_language(action)
        q_val = predictor._get_q_value(state_desc, action_desc)
        assert 0.0 <= q_val <= 1.0

    def test_experience_buffer_bounded(self):
        """Experience buffer respects max size by evicting low-priority items."""
        predictor = _make_predictor(max_experience_buffer=5)
        state = {"query": "buffer test"}
        action = {"actor": "bot"}

        for i in range(10):
            predictor.add_experience(state, action, float(i) / 10)

        assert len(predictor.experience_buffer) <= 5

    def test_record_outcome_wraps_non_dict_action(self):
        """record_outcome converts non-dict action to dict."""
        predictor = _make_predictor()
        predictor.record_outcome({"query": "test"}, "some_actor", 0.5)
        assert len(predictor.experience_buffer) == 1
        exp = predictor.experience_buffer[0]
        assert exp["action"] == {"actor": "some_actor"}

    def test_visit_count_increments(self):
        """Repeated experiences for same state-action increment visit count."""
        predictor = _make_predictor()
        state = {"query": "repeated"}
        action = {"actor": "bot"}

        predictor.add_experience(state, action, 0.5)
        predictor.add_experience(state, action, 0.7)
        predictor.add_experience(state, action, 0.9)

        entry = list(predictor.Q.values())[0]
        assert entry["visit_count"] == 3


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestQTableLookup:
    """Tests for Q-value lookup including semantic similarity fallback."""

    def test_exact_match_returns_stored_value(self):
        """Exact key match returns stored Q-value."""
        predictor = _make_predictor()
        predictor.Q[("state_a", "action_a")] = {
            "value": 0.75,
            "visit_count": 3,
            "context": [],
            "learned_lessons": [],
            "td_errors": [],
            "avg_reward": 0.7,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        result = predictor._get_q_value_from_table("state_a", "action_a")
        assert result == 0.75

    def test_novel_state_returns_default(self):
        """Unknown state-action pair returns neutral default 0.5."""
        predictor = _make_predictor()
        result = predictor._get_q_value_from_table("never_seen", "never_done")
        assert result == 0.5

    def test_similar_state_returns_discounted_value(self):
        """Semantically similar state returns discounted Q-value (0.9x)."""
        predictor = _make_predictor()
        # Use structured descriptions that will match via _are_similar
        predictor.Q[("QUERY: count all transactions | TABLES: fact_txn", "ACTOR: SQLGen")] = {
            "value": 0.8,
            "visit_count": 5,
            "context": [],
            "learned_lessons": [],
            "td_errors": [],
            "avg_reward": 0.7,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        # Similar structured description with shared fields
        result = predictor._get_q_value_from_table(
            "QUERY: count all transactions | TABLES: fact_txn | FILTERS: date", "ACTOR: SQLGen"
        )
        # Should be 0.8 * 0.9 = 0.72 (from similar match discount)
        assert abs(result - 0.72) < 0.01


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestSimilarityAndParsing:
    """Tests for _are_similar and _parse_structured_fields."""

    def test_identical_strings_are_similar(self):
        """Identical strings return True."""
        predictor = _make_predictor()
        assert predictor._are_similar("hello world", "hello world") is True

    def test_structured_field_parsing(self):
        """Structured KEY: value descriptions are parsed correctly."""
        predictor = _make_predictor()
        fields = predictor._parse_structured_fields(
            "QUERY: count users | TABLES: users_table | ACTOR: SQL"
        )
        assert fields == {"QUERY": "count users", "TABLES": "users_table", "ACTOR": "SQL"}

    def test_dissimilar_strings(self):
        """Completely different strings are not similar."""
        predictor = _make_predictor()
        assert (
            predictor._are_similar(
                "QUERY: count stock prices | TABLES: stocks",
                "ACTOR: email_sender | TOOL: smtp_client",
            )
            is False
        )


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestExtractLesson:
    """Tests for _extract_lesson logic."""

    def test_insignificant_td_error_returns_none(self):
        """Small TD error (< 0.1) produces no lesson."""
        predictor = _make_predictor()
        result = predictor._extract_lesson("state", "action", 0.5, 0.05)
        assert result is None

    def test_high_reward_produces_success_lesson(self):
        """High reward (> 0.7) with significant TD error produces SUCCESS lesson."""
        predictor = _make_predictor()
        result = predictor._extract_lesson(
            "QUERY: count txn | TABLES: fact_txn", "ACTOR: SQLGen | TOOL: execute", 0.9, 0.3
        )
        assert result is not None
        assert "SUCCESS" in result

    def test_low_reward_produces_failure_lesson(self):
        """Low reward (< 0.3) with significant TD error produces FAILED lesson."""
        predictor = _make_predictor()
        result = predictor._extract_lesson(
            "QUERY: bad query | COLS_TRIED: date,dt", "ACTOR: SQLGen", 0.1, -0.4
        )
        assert result is not None
        assert "AVOID" in result or "FAILED" in result


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestPredictQValue:
    """Tests for predict_q_value in simple and fallback modes."""

    def test_simple_mode_returns_average_reward(self):
        """Simple mode returns average reward for actor from experience buffer."""
        predictor = _make_predictor(q_value_mode="simple")
        # Populate experience buffer with known actor rewards
        predictor.experience_buffer = [
            {"action": {"actor": "SQL"}, "reward": 0.6, "state_desc": "", "action_desc": ""},
            {"action": {"actor": "SQL"}, "reward": 0.8, "state_desc": "", "action_desc": ""},
        ]
        q_val, conf, alt = predictor.predict_q_value({"query": "test"}, {"actor": "SQL"})
        assert abs(q_val - 0.7) < 0.01
        assert conf == 0.9
        assert alt is None

    def test_simple_mode_no_experience_falls_through(self):
        """Simple mode with no matching experiences falls back to LLM/table."""
        predictor = _make_predictor(q_value_mode="simple")
        # Mock the predictor to raise so we hit the fallback path
        predictor.predictor = MagicMock(side_effect=Exception("no LLM"))
        q_val, conf, alt = predictor.predict_q_value(
            {"query": "unknown"}, {"actor": "unknown_actor"}
        )
        # Should return a value (from fallback), not raise
        assert isinstance(q_val, float)
        assert 0.0 <= q_val <= 1.0


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestBestAction:
    """Tests for get_best_action epsilon-greedy selection."""

    def test_no_actions_returns_none(self):
        """No available actions returns None with neutral Q-value."""
        predictor = _make_predictor()
        action, q_val, reasoning = predictor.get_best_action({"query": "x"}, [])
        assert action is None
        assert q_val == 0.5

    def test_exploit_selects_highest_q(self):
        """With epsilon=0 (exploit only), best Q-value action is chosen."""
        predictor = _make_predictor(epsilon=0.0)
        predictor.epsilon = 0.0  # Ensure no exploration

        # Pre-populate Q-table so one action has higher value
        s_desc = predictor._state_to_natural_language({"query": "test"})
        a1_desc = predictor._action_to_natural_language({"actor": "A"})
        a2_desc = predictor._action_to_natural_language({"actor": "B"})

        predictor.Q[(s_desc, a1_desc)] = {
            "value": 0.9,
            "visit_count": 5,
            "context": [],
            "learned_lessons": [],
            "td_errors": [],
            "avg_reward": 0.9,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        predictor.Q[(s_desc, a2_desc)] = {
            "value": 0.3,
            "visit_count": 2,
            "context": [],
            "learned_lessons": [],
            "td_errors": [],
            "avg_reward": 0.3,
            "created_at": time.time(),
            "last_updated": time.time(),
        }

        action, q_val, reasoning = predictor.get_best_action(
            {"query": "test"}, [{"actor": "A"}, {"actor": "B"}]
        )
        assert action == {"actor": "A"}
        assert q_val == 0.9
        assert "Exploiting" in reasoning


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestExperienceReplay:
    """Tests for experience_replay batch learning."""

    def test_replay_requires_minimum_buffer(self):
        """Replay returns 0 when buffer is smaller than batch size."""
        predictor = _make_predictor()
        predictor.experience_buffer = [
            {"state_desc": "s", "action_desc": "a", "reward": 0.5, "done": False, "priority": 1.0}
        ]
        result = predictor.experience_replay(batch_size=32)
        assert result == 0

    def test_replay_performs_updates(self):
        """Replay with sufficient buffer performs requested updates."""
        predictor = _make_predictor()
        # Fill buffer with enough experiences
        for i in range(50):
            predictor.experience_buffer.append(
                {
                    "state_desc": f"state_{i}",
                    "action_desc": f"action_{i}",
                    "reward": 0.5,
                    "next_state_desc": None,
                    "done": False,
                    "priority": 1.0,
                    "timestamp": time.time(),
                }
            )

        updates = predictor.experience_replay(batch_size=10)
        assert updates == 10
        # Q-table should have entries now
        assert len(predictor.Q) > 0


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestQTableStats:
    """Tests for Q-table summary and stats methods."""

    def test_empty_q_table_stats(self):
        """Empty Q-table returns zero stats."""
        predictor = _make_predictor()
        stats = predictor.get_q_table_summary()
        assert stats["size"] == 0
        assert stats["total_visits"] == 0

    def test_populated_q_table_stats(self):
        """Populated Q-table returns correct aggregate stats."""
        predictor = _make_predictor()
        predictor.Q[("s1", "a1")] = {
            "value": 0.8,
            "visit_count": 3,
            "context": [],
            "learned_lessons": ["lesson1"],
            "td_errors": [0.1],
            "avg_reward": 0.7,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        predictor.Q[("s2", "a2")] = {
            "value": 0.4,
            "visit_count": 1,
            "context": [],
            "learned_lessons": [],
            "td_errors": [0.2],
            "avg_reward": 0.4,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        stats = predictor.get_q_table_summary()
        assert stats["size"] == 2
        assert stats["total_visits"] == 4
        assert stats["max_value"] == 0.8
        assert stats["min_value"] == 0.4
        assert stats["total_lessons"] == 1

    def test_get_q_table_stats_utilization(self):
        """get_q_table_stats returns utilization ratio."""
        predictor = _make_predictor(max_q_table_size=100)
        predictor.Q[("s1", "a1")] = {
            "value": 0.5,
            "visit_count": 1,
            "context": [],
            "learned_lessons": [],
            "td_errors": [],
            "avg_reward": 0.5,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        stats = predictor.get_q_table_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert abs(stats["utilization"] - 0.01) < 0.001


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestQTablePruning:
    """Tests for Q-table size enforcement and pruning."""

    def test_no_pruning_below_limit(self):
        """No entries pruned when table is below max size."""
        predictor = _make_predictor(max_q_table_size=100)
        predictor.Q[("s1", "a1")] = {
            "value": 0.5,
            "visit_count": 1,
            "context": [],
            "learned_lessons": [],
            "td_errors": [],
            "avg_reward": 0.5,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        pruned = predictor._enforce_q_table_limits()
        assert pruned == 0

    def test_pruning_removes_low_retention_entries(self):
        """When over limit, entries with lowest retention scores are pruned."""
        predictor = _make_predictor(max_q_table_size=5, q_prune_percentage=0.4)
        now = time.time()

        # Add 10 entries, some high-value and some low-value
        for i in range(10):
            predictor.Q[(f"state_{i}", f"action_{i}")] = {
                "value": 0.1 if i < 5 else 0.9,
                "visit_count": 1 if i < 5 else 10,
                "context": [],
                "learned_lessons": [] if i < 5 else ["important"],
                "td_errors": [0.01] if i < 5 else [0.5],
                "avg_reward": 0.1 if i < 5 else 0.9,
                "created_at": now - 7200 if i < 5 else now,
                "last_updated": now - 7200 if i < 5 else now,
            }

        pruned = predictor._enforce_q_table_limits()
        assert pruned > 0
        assert len(predictor.Q) <= 10  # Some were removed


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestPersistence:
    """Tests for save_state and load_state."""

    def test_save_and_load_roundtrip(self):
        """Q-table and experience buffer survive save/load roundtrip."""
        predictor = _make_predictor()
        state = {"query": "count users"}
        action = {"actor": "SQL"}
        predictor.add_experience(state, action, 0.8)
        predictor.add_experience(state, action, 0.6)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            predictor.save_state(path)

            # Create fresh predictor and load
            predictor2 = _make_predictor()
            assert len(predictor2.Q) == 0

            success = predictor2.load_state(path)
            assert success is True
            assert len(predictor2.Q) == len(predictor.Q)
            assert len(predictor2.experience_buffer) == len(predictor.experience_buffer)

            # Verify Q-values match
            for key in predictor.Q:
                assert key in predictor2.Q
                assert abs(predictor2.Q[key]["value"] - predictor.Q[key]["value"]) < 0.001
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_false(self):
        """Loading from nonexistent path returns False."""
        predictor = _make_predictor()
        result = predictor.load_state("/tmp/nonexistent_q_state_xyz123.json")
        assert result is False


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
@pytest.mark.unit
class TestLearnedContext:
    """Tests for get_learned_context method."""

    def test_no_lessons_returns_empty_string(self):
        """With empty Q-table, learned context is empty string."""
        predictor = _make_predictor()
        result = predictor.get_learned_context({"query": "test"})
        assert result == ""

    def test_lessons_included_in_context(self):
        """Stored lessons appear in learned context output."""
        predictor = _make_predictor()
        state = {"query": "count transactions"}
        action = {"actor": "SQL"}

        state_desc = predictor._state_to_natural_language(state)
        action_desc = predictor._action_to_natural_language(action)

        predictor.Q[(state_desc, action_desc)] = {
            "value": 0.8,
            "visit_count": 3,
            "context": [],
            "learned_lessons": ["Use partition column for date filters"],
            "td_errors": [0.2],
            "avg_reward": 0.8,
            "created_at": time.time(),
            "last_updated": time.time(),
        }

        result = predictor.get_learned_context(state, action)
        assert "Q-Learning Lessons" in result
        assert "Use partition column" in result
        assert "Expected Value" in result
