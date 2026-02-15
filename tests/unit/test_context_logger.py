"""
Tests for context_logger module
================================
Tests for EnhancedLogger, ContextRequirements, TokenBudgetManager, and SemanticFilter.

All tests use mocks -- NO real LLM calls, NO real token counting.
"""

import json
import logging
from dataclasses import fields
from unittest.mock import MagicMock, call, patch

import pytest

# =============================================================================
# EnhancedLogger Tests
# =============================================================================


class TestEnhancedLoggerInit:
    """Tests for EnhancedLogger initialization."""

    @pytest.mark.unit
    def test_creates_logger_with_given_name(self):
        """EnhancedLogger creates a stdlib logger with the provided name."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        logger_instance = EnhancedLogger("test.module")
        assert logger_instance.logger.name == "test.module"

    @pytest.mark.unit
    def test_creates_logger_with_empty_name(self):
        """EnhancedLogger with empty string name returns the root logger."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        logger_instance = EnhancedLogger("")
        # logging.getLogger("") returns root logger whose name is "root"
        assert logger_instance.logger.name == "root"

    @pytest.mark.unit
    def test_logger_is_stdlib_logger(self):
        """EnhancedLogger.logger is a standard logging.Logger instance."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("stdlib_check")
        assert isinstance(el.logger, logging.Logger)


class TestEnhancedLoggerCountTokens:
    """Tests for EnhancedLogger._count_tokens."""

    @pytest.mark.unit
    def test_count_tokens_delegates_to_count_tokens_accurate(self):
        """_count_tokens calls count_tokens_accurate with the text and model."""
        with patch(
            "Jotty.core.utils.context_logger.count_tokens_accurate", return_value=42
        ) as mock_ct:
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ct_test")
            result = el._count_tokens("hello world")
        assert result == 42
        mock_ct.assert_called_with("hello world", model="gpt-4")

    @pytest.mark.unit
    def test_count_tokens_empty_string(self):
        """_count_tokens handles empty string input."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=0):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("empty")
            result = el._count_tokens("")
        assert result == 0


class TestEnhancedLoggerFormat:
    """Tests for EnhancedLogger._format."""

    @pytest.mark.unit
    def test_format_dict(self):
        """_format renders dicts as indented JSON."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("fmt")
        result = el._format({"key": "value"})
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    @pytest.mark.unit
    def test_format_list(self):
        """_format renders lists as indented JSON."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("fmt")
        result = el._format([1, 2, 3])
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    @pytest.mark.unit
    def test_format_string(self):
        """_format returns str() for plain strings."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("fmt")
        assert el._format("hello") == "hello"

    @pytest.mark.unit
    def test_format_integer(self):
        """_format returns str() for integers."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("fmt")
        assert el._format(42) == "42"

    @pytest.mark.unit
    def test_format_none(self):
        """_format returns 'None' for None values."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("fmt")
        assert el._format(None) == "None"

    @pytest.mark.unit
    def test_format_nested_dict_with_non_serializable(self):
        """_format handles non-serializable values via default=str."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger
        el = EnhancedLogger("fmt")
        from datetime import datetime

        dt = datetime(2026, 1, 1, 12, 0, 0)
        result = el._format({"ts": dt})
        assert "2026" in result


class TestEnhancedLoggerLogActorStart:
    """Tests for EnhancedLogger.log_actor_start."""

    @pytest.mark.unit
    def test_log_actor_start_logs_actor_name_and_attempt(self):
        """log_actor_start logs the actor name and attempt number."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=5):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("start_test")
        el.logger = MagicMock()
        el.log_actor_start("MyActor", 2, {"param1": "val1"}, {"ctx1": 100})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "MyActor" in joined
        assert "Attempt 2" in joined

    @pytest.mark.unit
    def test_log_actor_start_logs_input_count(self):
        """log_actor_start logs the number of input parameters."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=3):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("start_test")
        el.logger = MagicMock()
        inputs = {"a": 1, "b": 2, "c": 3}
        el.log_actor_start("Actor", 1, inputs, {})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "3 parameters" in joined

    @pytest.mark.unit
    def test_log_actor_start_logs_context_total_tokens(self):
        """log_actor_start logs the total context tokens."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=1):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ctx")
        el.logger = MagicMock()
        el.log_actor_start("Actor", 1, {}, {"meta": 200, "mem": 300})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "500" in joined  # 200 + 300

    @pytest.mark.unit
    def test_log_actor_start_empty_inputs(self):
        """log_actor_start handles empty inputs dict."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=0):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("empty")
        el.logger = MagicMock()
        el.log_actor_start("Actor", 1, {}, {})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "0 parameters" in joined


class TestEnhancedLoggerLogActorEnd:
    """Tests for EnhancedLogger.log_actor_end."""

    @pytest.mark.unit
    def test_log_actor_end_logs_success_true(self):
        """log_actor_end logs success status when True."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=10):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("end_test")
        el.logger = MagicMock()
        el.log_actor_end("Actor", "some output", True, 1.23)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "True" in joined

    @pytest.mark.unit
    def test_log_actor_end_logs_duration(self):
        """log_actor_end logs the duration formatted to 2 decimals."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=10):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("dur")
        el.logger = MagicMock()
        el.log_actor_end("Actor", "output", True, 3.456)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "3.46" in joined

    @pytest.mark.unit
    def test_log_actor_end_with_dspy_prediction(self):
        """log_actor_end handles DSPy Prediction objects with _store attribute."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=5):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("dspy")
        el.logger = MagicMock()
        mock_output = MagicMock()
        mock_output._store = {"answer": "42", "reasoning": "because"}
        el.log_actor_end("Actor", mock_output, True, 0.5)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "DSPy Prediction" in joined
        assert "2 fields" in joined

    @pytest.mark.unit
    def test_log_actor_end_with_raw_output(self):
        """log_actor_end logs raw output when not a DSPy Prediction."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=7):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("raw")
        el.logger = MagicMock()
        el.log_actor_end("Actor", "plain text", False, 2.0)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "Raw Output" in joined

    @pytest.mark.unit
    def test_log_actor_end_success_false(self):
        """log_actor_end correctly logs failure."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=3):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("fail")
        el.logger = MagicMock()
        el.log_actor_end("Actor", "error", False, 0.1)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "False" in joined


class TestEnhancedLoggerLogValidation:
    """Tests for EnhancedLogger.log_validation."""

    @pytest.mark.unit
    def test_log_validation_valid_decision(self):
        """log_validation logs VALID when decision is True."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=10):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("val")
        el.logger = MagicMock()
        el.log_validation("TestValidator", "ACTOR", {"inp": "data"}, True, 0.95, "Looks good")
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "VALID" in joined
        assert "ACTOR" in joined
        assert "TestValidator" in joined

    @pytest.mark.unit
    def test_log_validation_invalid_decision(self):
        """log_validation logs INVALID when decision is False."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=8):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("val")
        el.logger = MagicMock()
        el.log_validation("SwarmValidator", "SWARM", {}, False, 0.3, "Bad output")
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "INVALID" in joined
        assert "SWARM" in joined

    @pytest.mark.unit
    def test_log_validation_confidence_format(self):
        """log_validation formats confidence to 2 decimal places."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=5):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("conf")
        el.logger = MagicMock()
        el.log_validation("V", "ACTOR", {}, True, 0.8765, "ok")
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "0.88" in joined

    @pytest.mark.unit
    def test_log_validation_logs_full_reasoning(self):
        """log_validation logs the full reasoning string without truncation."""
        long_reasoning = "A" * 500
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=125):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("reason")
        el.logger = MagicMock()
        el.log_validation("V", "ACTOR", {}, True, 0.9, long_reasoning)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert long_reasoning in joined


class TestEnhancedLoggerLogCompression:
    """Tests for EnhancedLogger.log_compression."""

    @pytest.mark.unit
    def test_log_compression_normal_ratio(self):
        """log_compression computes correct compression ratio."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("comp")
        el.logger = MagicMock()
        el.log_compression("metadata", 1000, 500, "summarize", "reduce context")
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "50.0%" in joined
        assert "metadata" in joined
        assert "summarize" in joined
        assert "reduce context" in joined

    @pytest.mark.unit
    def test_log_compression_zero_original_tokens(self):
        """log_compression handles zero original_tokens without division error."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("zero")
        el.logger = MagicMock()
        # Should not raise ZeroDivisionError
        el.log_compression("comp", 0, 0, "none", "test")
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "0.0%" in joined

    @pytest.mark.unit
    def test_log_compression_logs_all_fields(self):
        """log_compression logs original, compressed, ratio, method, and purpose."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("fields")
        el.logger = MagicMock()
        el.log_compression("trajectory", 2000, 400, "truncate", "fit budget")
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "2000" in joined
        assert "400" in joined
        assert "truncate" in joined
        assert "fit budget" in joined


class TestEnhancedLoggerLogMemory:
    """Tests for EnhancedLogger.log_memory."""

    @pytest.mark.unit
    def test_log_memory_uppercases_operation(self):
        """log_memory uppercases the operation name."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=15):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("mem")
        el.logger = MagicMock()
        el.log_memory("store", "some content", "episodic", {"key": "val"})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "STORE" in joined

    @pytest.mark.unit
    def test_log_memory_logs_level(self):
        """log_memory logs the memory level."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=5):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("mem")
        el.logger = MagicMock()
        el.log_memory("retrieve", "data", "semantic", {})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "semantic" in joined

    @pytest.mark.unit
    def test_log_memory_logs_full_content(self):
        """log_memory logs full content without truncation."""
        content = "x" * 1000
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=250):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("mem")
        el.logger = MagicMock()
        el.log_memory("store", content, "procedural", {"src": "test"})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert content in joined

    @pytest.mark.unit
    def test_log_memory_serializes_metadata_as_json(self):
        """log_memory serializes metadata dict as JSON."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate", return_value=5):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("mem")
        el.logger = MagicMock()
        metadata = {"score": 0.95, "source": "test"}
        el.log_memory("store", "c", "episodic", metadata)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "score" in joined
        assert "0.95" in joined


class TestEnhancedLoggerLogRlUpdate:
    """Tests for EnhancedLogger.log_rl_update."""

    @pytest.mark.unit
    def test_log_rl_update_logs_actor_and_action(self):
        """log_rl_update logs the actor and action."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("rl")
        el.logger = MagicMock()
        el.log_rl_update("SqlActor", {"step": 1}, "generate_query", 0.8, 0.2, 0.5, 0.7)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "SqlActor" in joined
        assert "generate_query" in joined

    @pytest.mark.unit
    def test_log_rl_update_logs_q_values_transition(self):
        """log_rl_update logs old and new Q-values with delta."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("rl")
        el.logger = MagicMock()
        el.log_rl_update("Actor", {}, "act", 1.0, 0.1, 0.500, 0.600)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "0.500" in joined
        assert "0.600" in joined
        assert "+0.100" in joined

    @pytest.mark.unit
    def test_log_rl_update_negative_delta(self):
        """log_rl_update shows negative delta when q_new < q_old."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("rl")
        el.logger = MagicMock()
        el.log_rl_update("Actor", {}, "act", -0.5, -0.3, 0.800, 0.500)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "-0.300" in joined

    @pytest.mark.unit
    def test_log_rl_update_logs_state_as_json(self):
        """log_rl_update serializes state dict as JSON."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("rl")
        el.logger = MagicMock()
        state = {"attempt": 3, "last_score": 0.6}
        el.log_rl_update("Actor", state, "retry", 0.5, 0.1, 0.4, 0.5)
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "attempt" in joined
        assert "last_score" in joined


class TestEnhancedLoggerLogContextBuilding:
    """Tests for EnhancedLogger.log_context_building."""

    @pytest.mark.unit
    def test_log_context_building_with_dict_budget(self):
        """log_context_building sums dict budget values for total."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ctx")
        el.logger = MagicMock()
        budget = {"metadata": 5000, "memories": 3000}
        el.log_context_building("Actor", budget, 7500, {"metadata": 4500, "memories": 3000})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "8000" in joined  # 5000 + 3000

    @pytest.mark.unit
    def test_log_context_building_with_scalar_budget(self):
        """log_context_building handles scalar (non-dict) budget."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ctx")
        el.logger = MagicMock()
        el.log_context_building("Actor", 10000, 9000, {"a": 5000, "b": 4000})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "10000" in joined

    @pytest.mark.unit
    def test_log_context_building_empty_components(self):
        """log_context_building handles empty components dict."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ctx")
        el.logger = MagicMock()
        el.log_context_building("Actor", {"x": 1000}, 0, {})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "No component breakdown available" in joined

    @pytest.mark.unit
    def test_log_context_building_component_over_budget(self):
        """log_context_building marks components that exceed total budget as OVER."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ctx")
        el.logger = MagicMock()
        budget = {"meta": 100}
        el.log_context_building("Actor", budget, 500, {"big_component": 200})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "OVER" in joined

    @pytest.mark.unit
    def test_log_context_building_component_within_budget(self):
        """log_context_building marks components within budget as OK."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import EnhancedLogger

            el = EnhancedLogger("ctx")
        el.logger = MagicMock()
        budget = {"meta": 10000}
        el.log_context_building("Actor", budget, 5000, {"small": 50})
        info_calls = [str(c) for c in el.logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "OK" in joined


# =============================================================================
# ContextRequirements Tests
# =============================================================================


class TestContextRequirements:
    """Tests for the ContextRequirements dataclass."""

    @pytest.mark.unit
    def test_default_metadata_fields_is_none(self):
        """Default metadata_fields is None."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        assert cr.metadata_fields is None

    @pytest.mark.unit
    def test_default_metadata_detail(self):
        """Default metadata_detail is 'full'."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        assert cr.metadata_detail == "full"

    @pytest.mark.unit
    def test_default_memories_scope(self):
        """Default memories_scope is 'recent'."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        assert cr.memories_scope == "recent"

    @pytest.mark.unit
    def test_default_memories_limit(self):
        """Default memories_limit is 5."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        assert cr.memories_limit == 5

    @pytest.mark.unit
    def test_default_trajectory_needed(self):
        """Default trajectory_needed is True."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        assert cr.trajectory_needed is True

    @pytest.mark.unit
    def test_default_previous_outputs_needed_is_none(self):
        """Default previous_outputs_needed is None."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        assert cr.previous_outputs_needed is None

    @pytest.mark.unit
    def test_default_budget_proportions(self):
        """Default budget_proportions sums to 1.0."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements()
        total = sum(cr.budget_proportions.values())
        assert abs(total - 1.0) < 1e-9
        assert cr.budget_proportions["metadata"] == 0.4
        assert cr.budget_proportions["memories"] == 0.2
        assert cr.budget_proportions["trajectory"] == 0.1
        assert cr.budget_proportions["previous_outputs"] == 0.3

    @pytest.mark.unit
    def test_custom_metadata_fields(self):
        """ContextRequirements accepts custom metadata_fields."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr = ContextRequirements(metadata_fields=["tables", "columns"])
        assert cr.metadata_fields == ["tables", "columns"]

    @pytest.mark.unit
    def test_custom_budget_proportions_independent_per_instance(self):
        """Each instance gets its own budget_proportions dict (no shared state)."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import ContextRequirements
        cr1 = ContextRequirements()
        cr2 = ContextRequirements()
        cr1.budget_proportions["metadata"] = 0.99
        assert cr2.budget_proportions["metadata"] == 0.4


# =============================================================================
# TokenBudgetManager Tests
# =============================================================================


class TestTokenBudgetManagerInit:
    """Tests for TokenBudgetManager initialization."""

    @pytest.mark.unit
    def test_default_total_budget(self):
        """Default total budget is 30000."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import TokenBudgetManager
        mgr = TokenBudgetManager()
        assert mgr.total == 30000

    @pytest.mark.unit
    def test_default_output_reserve(self):
        """Default output reserve is 8000."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import TokenBudgetManager
        mgr = TokenBudgetManager()
        assert mgr.output_reserve == 8000

    @pytest.mark.unit
    def test_input_budget_calculated(self):
        """Input budget = total - output_reserve."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import TokenBudgetManager
        mgr = TokenBudgetManager(total_budget=50000, output_reserve=10000)
        assert mgr.input_budget == 40000

    @pytest.mark.unit
    def test_custom_total_and_reserve(self):
        """Custom total and reserve are stored correctly."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import TokenBudgetManager
        mgr = TokenBudgetManager(total_budget=100000, output_reserve=20000)
        assert mgr.total == 100000
        assert mgr.output_reserve == 20000
        assert mgr.input_budget == 80000


class TestTokenBudgetManagerAllocate:
    """Tests for TokenBudgetManager.allocate."""

    @pytest.mark.unit
    def test_allocate_essential_components_capped_at_2000(self):
        """Essential components are capped at 2000 tokens each."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager()
        req = ContextRequirements()
        sizes = {"goal": 5000, "task": 3000}
        alloc = mgr.allocate(req, sizes)
        assert alloc["goal"] == 2000
        assert alloc["task"] == 2000

    @pytest.mark.unit
    def test_allocate_essential_below_cap(self):
        """Essential components below 2000 get their actual size."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager()
        req = ContextRequirements()
        sizes = {"goal": 500, "task": 800}
        alloc = mgr.allocate(req, sizes)
        assert alloc["goal"] == 500
        assert alloc["task"] == 800

    @pytest.mark.unit
    def test_allocate_missing_essential_gets_zero(self):
        """Essential components missing from sizes get 0."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager()
        req = ContextRequirements()
        alloc = mgr.allocate(req, {})
        assert alloc.get("goal", 0) == 0
        assert alloc.get("task", 0) == 0

    @pytest.mark.unit
    def test_allocate_flexible_uses_proportions(self):
        """Flexible components get proportional allocation from remaining budget."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager(total_budget=30000, output_reserve=8000)
        # input_budget = 22000, no essentials used = 22000 remaining
        req = ContextRequirements()
        alloc = mgr.allocate(req, {})
        # metadata: 0.4 * 22000 = 8800
        assert alloc["metadata"] == int(22000 * 0.4)
        # memories: 0.2 * 22000 = 4400
        assert alloc["memories"] == int(22000 * 0.2)
        # trajectory: 0.1 * 22000 = 2200
        assert alloc["trajectory"] == int(22000 * 0.1)
        # previous_outputs: 0.3 * 22000 = 6600
        assert alloc["previous_outputs"] == int(22000 * 0.3)

    @pytest.mark.unit
    def test_allocate_remaining_reduced_by_essentials(self):
        """Flexible budget is reduced by essential component allocations."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager(total_budget=30000, output_reserve=8000)
        req = ContextRequirements()
        # 2 essentials at 2000 each = 4000 used
        sizes = {"goal": 2000, "task": 2000}
        alloc = mgr.allocate(req, sizes)
        # remaining = 22000 - 4000 = 18000
        assert alloc["metadata"] == int(18000 * 0.4)

    @pytest.mark.unit
    def test_allocate_with_custom_proportions(self):
        """Allocate respects custom budget_proportions."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager(total_budget=30000, output_reserve=8000)
        req = ContextRequirements(budget_proportions={"metadata": 0.5, "memories": 0.5})
        alloc = mgr.allocate(req, {})
        # remaining = 22000
        assert alloc["metadata"] == int(22000 * 0.5)
        assert alloc["memories"] == int(22000 * 0.5)

    @pytest.mark.unit
    def test_allocate_error_context_essential(self):
        """error_context is treated as an essential component."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager()
        req = ContextRequirements()
        sizes = {"error_context": 1500}
        alloc = mgr.allocate(req, sizes)
        assert alloc["error_context"] == 1500

    @pytest.mark.unit
    def test_allocate_required_params_essential(self):
        """required_params is treated as an essential component."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                TokenBudgetManager,
            )
        mgr = TokenBudgetManager()
        req = ContextRequirements()
        sizes = {"required_params": 1800}
        alloc = mgr.allocate(req, sizes)
        assert alloc["required_params"] == 1800

    @pytest.mark.unit
    def test_allocate_with_none_requirements_uses_fallback_proportions(self):
        """When requirements is None (falsy), fallback proportions are used."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import TokenBudgetManager
        mgr = TokenBudgetManager(total_budget=30000, output_reserve=8000)
        alloc = mgr.allocate(None, {})
        # Fallback proportions: metadata=0.3, memories=0.2, trajectory=0.3, previous_outputs=0.2
        remaining = 22000
        assert alloc["metadata"] == int(remaining * 0.3)
        assert alloc["memories"] == int(remaining * 0.2)
        assert alloc["trajectory"] == int(remaining * 0.3)
        assert alloc["previous_outputs"] == int(remaining * 0.2)


# =============================================================================
# SemanticFilter Tests
# =============================================================================


class TestSemanticFilterFilterByRequirements:
    """Tests for SemanticFilter.filter_by_requirements."""

    @pytest.mark.unit
    def test_returns_content_when_no_requirements(self):
        """filter_by_requirements returns content as-is when requirements is None."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import SemanticFilter
        sf = SemanticFilter()
        content = {"a": 1, "b": 2}
        result = sf.filter_by_requirements(content, None, "any goal")
        assert result == content

    @pytest.mark.unit
    def test_filters_dict_by_metadata_fields(self):
        """filter_by_requirements filters dict to only requested metadata_fields."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                SemanticFilter,
            )
        sf = SemanticFilter()
        content = {"tables": ["t1"], "columns": ["c1"], "indexes": ["i1"]}
        req = ContextRequirements(metadata_fields=["tables", "columns"])
        result = sf.filter_by_requirements(content, req, "generate SQL")
        assert result == {"tables": ["t1"], "columns": ["c1"]}
        assert "indexes" not in result

    @pytest.mark.unit
    def test_filters_dict_missing_fields_skipped(self):
        """filter_by_requirements skips requested fields not present in content."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                SemanticFilter,
            )
        sf = SemanticFilter()
        content = {"tables": ["t1"]}
        req = ContextRequirements(metadata_fields=["tables", "nonexistent"])
        result = sf.filter_by_requirements(content, req, "goal")
        assert result == {"tables": ["t1"]}

    @pytest.mark.unit
    def test_summarizes_list_items_in_summary_mode(self):
        """filter_by_requirements summarizes list items when detail is 'summary'."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                SemanticFilter,
            )
        sf = SemanticFilter()
        content = [
            {"name": "table1", "description": "Main table", "rows": 1000, "extra": "data"},
            {"name": "table2", "title": "Secondary"},
        ]
        req = ContextRequirements(metadata_detail="summary")
        result = sf.filter_by_requirements(content, req, "goal")
        assert len(result) == 2
        assert result[0] == {"name": "table1", "description": "Main table"}
        assert result[1] == {"name": "table2", "title": "Secondary"}

    @pytest.mark.unit
    def test_returns_content_unchanged_for_string_input(self):
        """filter_by_requirements returns string content unchanged."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                SemanticFilter,
            )
        sf = SemanticFilter()
        req = ContextRequirements()
        result = sf.filter_by_requirements("plain text", req, "goal")
        assert result == "plain text"

    @pytest.mark.unit
    def test_dict_without_metadata_fields_returns_unchanged(self):
        """filter_by_requirements returns dict unchanged when metadata_fields is None."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                SemanticFilter,
            )
        sf = SemanticFilter()
        content = {"a": 1, "b": 2, "c": 3}
        req = ContextRequirements(metadata_fields=None)
        result = sf.filter_by_requirements(content, req, "goal")
        assert result == content

    @pytest.mark.unit
    def test_list_without_summary_mode_returns_unchanged(self):
        """filter_by_requirements returns list unchanged when detail is not 'summary'."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import (
                ContextRequirements,
                SemanticFilter,
            )
        sf = SemanticFilter()
        content = [{"name": "x", "extra": "y"}]
        req = ContextRequirements(metadata_detail="full")
        result = sf.filter_by_requirements(content, req, "goal")
        assert result == content


class TestSemanticFilterSummarizeItem:
    """Tests for SemanticFilter._summarize_item."""

    @pytest.mark.unit
    def test_summarize_dict_keeps_key_fields(self):
        """_summarize_item keeps name, id, title, description from dict."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import SemanticFilter
        sf = SemanticFilter()
        item = {
            "name": "test",
            "id": 1,
            "title": "Test Item",
            "description": "A test",
            "extra_field": "removed",
            "other": 42,
        }
        result = sf._summarize_item(item)
        assert result == {
            "name": "test",
            "id": 1,
            "title": "Test Item",
            "description": "A test",
        }

    @pytest.mark.unit
    def test_summarize_dict_no_known_keys_returns_original(self):
        """_summarize_item returns original dict if no known keys present."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import SemanticFilter
        sf = SemanticFilter()
        item = {"foo": "bar", "baz": 123}
        result = sf._summarize_item(item)
        assert result == item

    @pytest.mark.unit
    def test_summarize_non_dict_returns_as_is(self):
        """_summarize_item returns non-dict items unchanged."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import SemanticFilter
        sf = SemanticFilter()
        assert sf._summarize_item("string") == "string"
        assert sf._summarize_item(42) == 42
        assert sf._summarize_item(None) is None

    @pytest.mark.unit
    def test_summarize_empty_dict_returns_original(self):
        """_summarize_item returns empty dict unchanged (no summary keys found)."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import SemanticFilter
        sf = SemanticFilter()
        result = sf._summarize_item({})
        assert result == {}

    @pytest.mark.unit
    def test_summarize_dict_partial_keys(self):
        """_summarize_item keeps only the known keys that exist."""
        with patch("Jotty.core.utils.context_logger.count_tokens_accurate"):
            from Jotty.core.infrastructure.utils.context_logger import SemanticFilter
        sf = SemanticFilter()
        item = {"name": "only_name", "unknown": "dropped"}
        result = sf._summarize_item(item)
        assert result == {"name": "only_name"}
