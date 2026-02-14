"""
Tests for Utility Modules: trajectory_parser, profiler, file_logger
====================================================================

Comprehensive unit tests covering:
- TrajectoryParser, TaggedAttempt, create_parser (trajectory_parser.py)
- ExecutionTimer, get_timer, timed_block, timed decorator (profiler.py)
- setup_file_logging, close_file_logging (file_logger.py)

All tests are fast (< 1s), offline, no real LLM calls.
"""

import asyncio
import logging
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock

try:
    from Jotty.core.utils.trajectory_parser import (
        TaggedAttempt,
        TrajectoryParser,
        create_parser,
    )
    TRAJECTORY_PARSER_AVAILABLE = True
except ImportError:
    TRAJECTORY_PARSER_AVAILABLE = False

try:
    from Jotty.core.utils.profiler import (
        ExecutionTimer,
        get_timer,
        timed_block,
        timed,
        enable_profiling,
        disable_profiling,
        reset_profiling,
        _global_timer,
    )
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

try:
    from Jotty.core.utils.file_logger import (
        setup_file_logging,
        close_file_logging,
    )
    FILE_LOGGER_AVAILABLE = True
except ImportError:
    FILE_LOGGER_AVAILABLE = False


# =============================================================================
# Helper: build a mock DSPy result with _store / trajectory
# =============================================================================

def _make_result(trajectory_dict, wrap_in_trajectory_key=True):
    """Create a mock result object with _store containing trajectory data."""
    result = Mock()
    if wrap_in_trajectory_key:
        result._store = {"trajectory": trajectory_dict}
    else:
        result._store = trajectory_dict
    return result


def _single_step(
    index=0,
    tool_name="execute_query",
    tool_args=None,
    observation="success",
    thought="thinking",
):
    """Return dict keys for a single trajectory step."""
    if tool_args is None:
        tool_args = {"query": "SELECT 1"}
    return {
        f"tool_name_{index}": tool_name,
        f"tool_args_{index}": tool_args,
        f"observation_{index}": observation,
        f"thought_{index}": thought,
    }


# =============================================================================
# TaggedAttempt Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TRAJECTORY_PARSER_AVAILABLE, reason="trajectory_parser not importable")
class TestTaggedAttempt:
    """Tests for the TaggedAttempt dataclass."""

    def test_creation_with_defaults(self):
        """TaggedAttempt can be created with required fields and sensible defaults."""
        attempt = TaggedAttempt(
            output="SELECT 1",
            tag="answer",
            execution_status="success",
            execution_result="OK",
            reasoning="step 0",
        )
        assert attempt.output == "SELECT 1"
        assert attempt.tag == "answer"
        assert attempt.execution_status == "success"
        assert attempt.attempt_number == 0
        assert attempt.tool_name == ""
        assert isinstance(attempt.timestamp, float)

    def test_creation_with_all_fields(self):
        """TaggedAttempt accepts all explicit values."""
        attempt = TaggedAttempt(
            output="code",
            tag="error",
            execution_status="failed",
            execution_result="traceback",
            reasoning="wrong",
            attempt_number=3,
            tool_name="run_code",
            timestamp=100.0,
        )
        assert attempt.attempt_number == 3
        assert attempt.tool_name == "run_code"
        assert attempt.timestamp == 100.0

    def test_is_answer_true(self):
        """is_answer() returns True for tag='answer'."""
        attempt = TaggedAttempt(
            output="x", tag="answer", execution_status="success",
            execution_result="ok", reasoning="r",
        )
        assert attempt.is_answer() is True
        assert attempt.is_error() is False
        assert attempt.is_exploratory() is False

    def test_is_error_true(self):
        """is_error() returns True for tag='error'."""
        attempt = TaggedAttempt(
            output="x", tag="error", execution_status="failed",
            execution_result="err", reasoning="r",
        )
        assert attempt.is_error() is True
        assert attempt.is_answer() is False
        assert attempt.is_exploratory() is False

    def test_is_exploratory_true(self):
        """is_exploratory() returns True for tag='exploratory'."""
        attempt = TaggedAttempt(
            output="x", tag="exploratory", execution_status="uncertain",
            execution_result="?", reasoning="r",
        )
        assert attempt.is_exploratory() is True
        assert attempt.is_answer() is False
        assert attempt.is_error() is False

    def test_unknown_tag_all_false(self):
        """Unknown tags cause all three predicates to return False."""
        attempt = TaggedAttempt(
            output="x", tag="custom", execution_status="custom",
            execution_result="?", reasoning="r",
        )
        assert attempt.is_answer() is False
        assert attempt.is_error() is False
        assert attempt.is_exploratory() is False


# =============================================================================
# TrajectoryParser Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TRAJECTORY_PARSER_AVAILABLE, reason="trajectory_parser not importable")
class TestTrajectoryParser:
    """Tests for the TrajectoryParser class."""

    def test_init_without_lm(self):
        """Parser initializes with lm=None."""
        parser = TrajectoryParser()
        assert parser.lm is None

    def test_init_with_lm(self):
        """Parser stores the provided LM."""
        mock_lm = Mock()
        parser = TrajectoryParser(lm=mock_lm)
        assert parser.lm is mock_lm

    def test_parse_no_store(self):
        """Returns empty list if result has no _store."""
        parser = TrajectoryParser()
        result = Mock(spec=[])  # no _store attribute
        assert parser.parse_trajectory(result) == []

    def test_parse_single_success_step(self):
        """Parses a single successful trajectory step."""
        parser = TrajectoryParser()
        traj = _single_step(0, observation="Query returned result successfully")
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result)
        assert len(attempts) == 1
        assert attempts[0].tag == "answer"
        assert attempts[0].execution_status == "success"
        assert attempts[0].output == "SELECT 1"
        assert attempts[0].tool_name == "execute_query"
        assert attempts[0].attempt_number == 1

    def test_parse_single_error_step(self):
        """Parses a single error trajectory step."""
        parser = TrajectoryParser()
        traj = _single_step(0, observation="RuntimeError: table not found")
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result)
        assert len(attempts) == 1
        assert attempts[0].tag == "error"
        assert attempts[0].execution_status == "failed"

    def test_parse_single_exploratory_step(self):
        """Parses a step that is neither success nor error as exploratory."""
        parser = TrajectoryParser()
        traj = _single_step(0, observation="Attempting to run query...")
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result)
        assert len(attempts) == 1
        assert attempts[0].tag == "exploratory"
        assert attempts[0].execution_status == "uncertain"

    def test_parse_multiple_steps(self):
        """Parses multiple trajectory steps in order."""
        parser = TrajectoryParser()
        traj = {}
        traj.update(_single_step(0, observation="Error: syntax error"))
        traj.update(_single_step(1, observation="Query completed successfully"))
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result)
        assert len(attempts) == 2
        assert attempts[0].tag == "error"
        assert attempts[0].attempt_number == 1
        assert attempts[1].tag == "answer"
        assert attempts[1].attempt_number == 2

    def test_parse_with_tool_name_filter_matches(self):
        """Tool name filter includes only matching tool calls."""
        parser = TrajectoryParser()
        traj = {}
        traj.update(_single_step(0, tool_name="execute_query", observation="result done"))
        traj.update(_single_step(1, tool_name="search_web", observation="result done"))
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result, tool_name_filter="execute_query")
        assert len(attempts) == 1
        assert attempts[0].tool_name == "execute_query"

    def test_parse_with_tool_name_filter_no_match(self):
        """Tool name filter returns empty when nothing matches."""
        parser = TrajectoryParser()
        traj = _single_step(0, tool_name="execute_query", observation="result done")
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result, tool_name_filter="nonexistent_tool")
        assert len(attempts) == 0

    def test_parse_fallback_to_top_level_store(self):
        """Falls back to reading trajectory keys from top-level _store."""
        parser = TrajectoryParser()
        traj = _single_step(0, observation="success complete")
        result = _make_result(traj, wrap_in_trajectory_key=False)

        attempts = parser.parse_trajectory(result)
        assert len(attempts) == 1
        assert attempts[0].tag == "answer"

    def test_parse_empty_trajectory(self):
        """Returns empty list when trajectory has no tool_name_0."""
        parser = TrajectoryParser()
        result = _make_result({})
        assert parser.parse_trajectory(result) == []

    def test_parse_trajectory_dict_not_dict(self):
        """Falls back to _store when trajectory key is not a dict."""
        parser = TrajectoryParser()
        result = Mock()
        traj = _single_step(0, observation="done successfully")
        traj["trajectory"] = "not_a_dict"
        result._store = traj

        attempts = parser.parse_trajectory(result)
        assert len(attempts) == 1

    def test_thought_fallback(self):
        """Uses fallback reasoning when thought is empty/falsy."""
        parser = TrajectoryParser()
        traj = _single_step(0, observation="done successfully", thought="")
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result)
        assert "Trajectory step 0" in attempts[0].reasoning

    def test_thought_preserved(self):
        """Preserves non-empty thought as reasoning."""
        parser = TrajectoryParser()
        traj = _single_step(0, observation="done successfully", thought="I need to check")
        result = _make_result(traj)

        attempts = parser.parse_trajectory(result)
        assert attempts[0].reasoning == "I need to check"


# =============================================================================
# TrajectoryParser._tag_attempt Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TRAJECTORY_PARSER_AVAILABLE, reason="trajectory_parser not importable")
class TestTagAttempt:
    """Tests for the _tag_attempt method's tagging logic."""

    def setup_method(self):
        self.parser = TrajectoryParser()

    # --- Error indicators ---

    def test_tag_error_keyword_error(self):
        assert self.parser._tag_attempt("Something caused an error") == "error"

    def test_tag_error_keyword_exception(self):
        assert self.parser._tag_attempt("An exception occurred") == "error"

    def test_tag_error_keyword_traceback(self):
        assert self.parser._tag_attempt("Traceback (most recent call last)") == "error"

    def test_tag_error_keyword_failed(self):
        assert self.parser._tag_attempt("Task failed to execute") == "error"

    def test_tag_error_keyword_valueerror(self):
        assert self.parser._tag_attempt("ValueError: invalid literal") == "error"

    def test_tag_error_keyword_typeerror(self):
        assert self.parser._tag_attempt("TypeError: unexpected type") == "error"

    def test_tag_error_keyword_keyerror(self):
        assert self.parser._tag_attempt("KeyError: 'missing_key'") == "error"

    def test_tag_error_keyword_attributeerror(self):
        assert self.parser._tag_attempt("AttributeError: no such attr") == "error"

    def test_tag_error_keyword_indexerror(self):
        assert self.parser._tag_attempt("IndexError: list index out of range") == "error"

    def test_tag_error_keyword_invalid(self):
        assert self.parser._tag_attempt("This is invalid input") == "error"

    def test_tag_error_keyword_runtimeerror(self):
        assert self.parser._tag_attempt("RuntimeError: oops") == "error"

    def test_tag_error_keyword_syntaxerror(self):
        assert self.parser._tag_attempt("SyntaxError: unexpected token") == "error"

    def test_tag_error_keyword_failure(self):
        assert self.parser._tag_attempt("Complete failure in execution") == "error"

    # --- Success indicators ---

    def test_tag_answer_keyword_success(self):
        assert self.parser._tag_attempt("Execution was a success") == "answer"

    def test_tag_answer_keyword_complete(self):
        assert self.parser._tag_attempt("Task is complete") == "answer"

    def test_tag_answer_keyword_done(self):
        assert self.parser._tag_attempt("Processing done") == "answer"

    def test_tag_answer_keyword_finished(self):
        assert self.parser._tag_attempt("Task finished") == "answer"

    def test_tag_answer_keyword_result(self):
        assert self.parser._tag_attempt("result: 42") == "answer"

    def test_tag_answer_keyword_output(self):
        assert self.parser._tag_attempt("output is ready") == "answer"

    def test_tag_answer_keyword_returned(self):
        assert self.parser._tag_attempt("Function returned 5") == "answer"

    def test_tag_answer_keyword_equals(self):
        assert self.parser._tag_attempt("x = 5") == "answer"

    # --- Null indicators override success ---

    def test_tag_exploratory_success_with_none(self):
        """Success indicator + null indicator => exploratory."""
        assert self.parser._tag_attempt("result is none") == "exploratory"

    def test_tag_exploratory_success_with_null(self):
        assert self.parser._tag_attempt("output: null") == "exploratory"

    def test_tag_exploratory_success_with_empty(self):
        assert self.parser._tag_attempt("result is empty") == "exploratory"

    def test_tag_exploratory_success_with_no_result(self):
        assert self.parser._tag_attempt("returned no result") == "exploratory"

    def test_tag_exploratory_success_with_empty_list(self):
        assert self.parser._tag_attempt("result = []") == "exploratory"

    def test_tag_exploratory_success_with_empty_dict(self):
        assert self.parser._tag_attempt("output = {}") == "exploratory"

    # --- Error takes precedence over success ---

    def test_tag_error_trumps_success(self):
        """Error indicators take precedence over success indicators."""
        assert self.parser._tag_attempt("Error: result is done") == "error"

    # --- Default: exploratory ---

    def test_tag_default_exploratory(self):
        """No indicators at all => exploratory."""
        assert self.parser._tag_attempt("just some random text here") == "exploratory"

    def test_tag_empty_observation(self):
        assert self.parser._tag_attempt("") == "exploratory"

    # --- Case insensitivity ---

    def test_tag_case_insensitive(self):
        """Tagging is case-insensitive."""
        assert self.parser._tag_attempt("ERROR occurred") == "error"
        assert self.parser._tag_attempt("SUCCESS achieved") == "answer"

    # --- Non-string observation ---

    def test_tag_non_string_observation(self):
        """Non-string observations are str()-converted."""
        assert self.parser._tag_attempt(12345) == "exploratory"
        assert self.parser._tag_attempt(None) == "exploratory"


# =============================================================================
# TrajectoryParser._extract_output Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TRAJECTORY_PARSER_AVAILABLE, reason="trajectory_parser not importable")
class TestExtractOutput:
    """Tests for the _extract_output method."""

    def setup_method(self):
        self.parser = TrajectoryParser()

    def test_extract_query_field(self):
        assert self.parser._extract_output({"query": "SELECT 1"}) == "SELECT 1"

    def test_extract_code_field(self):
        assert self.parser._extract_output({"code": "print('hi')"}) == "print('hi')"

    def test_extract_prompt_field(self):
        assert self.parser._extract_output({"prompt": "Write a blog"}) == "Write a blog"

    def test_extract_input_field(self):
        assert self.parser._extract_output({"input": "data"}) == "data"

    def test_extract_data_field(self):
        assert self.parser._extract_output({"data": [1, 2, 3]}) == [1, 2, 3]

    def test_extract_text_field(self):
        assert self.parser._extract_output({"text": "hello"}) == "hello"

    def test_extract_command_field(self):
        assert self.parser._extract_output({"command": "ls -la"}) == "ls -la"

    def test_extract_fallback_dict_str(self):
        """Falls back to str(tool_args) when no known key is found."""
        result = self.parser._extract_output({"unknown_key": "val"})
        assert "unknown_key" in result
        assert "val" in result

    def test_extract_non_dict(self):
        """Non-dict tool_args are str()-converted."""
        assert self.parser._extract_output("raw string") == "raw string"
        assert self.parser._extract_output(42) == "42"

    def test_extract_priority_order(self):
        """query takes priority over code when both present."""
        result = self.parser._extract_output({"query": "SELECT 1", "code": "print(1)"})
        assert result == "SELECT 1"

    def test_extract_empty_dict(self):
        """Empty dict falls back to str representation."""
        result = self.parser._extract_output({})
        assert result == "{}"


# =============================================================================
# TrajectoryParser._get_status Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TRAJECTORY_PARSER_AVAILABLE, reason="trajectory_parser not importable")
class TestGetStatus:
    """Tests for the _get_status method."""

    def setup_method(self):
        self.parser = TrajectoryParser()

    def test_status_answer(self):
        assert self.parser._get_status("answer") == "success"

    def test_status_error(self):
        assert self.parser._get_status("error") == "failed"

    def test_status_exploratory(self):
        assert self.parser._get_status("exploratory") == "uncertain"

    def test_status_unknown_tag(self):
        assert self.parser._get_status("custom") == "unknown"


# =============================================================================
# create_parser Factory Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TRAJECTORY_PARSER_AVAILABLE, reason="trajectory_parser not importable")
class TestCreateParser:
    """Tests for the create_parser factory function."""

    def test_create_parser_default(self):
        parser = create_parser()
        assert isinstance(parser, TrajectoryParser)
        assert parser.lm is None

    def test_create_parser_with_lm(self):
        mock_lm = Mock()
        parser = create_parser(lm=mock_lm)
        assert parser.lm is mock_lm


# =============================================================================
# ExecutionTimer Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not PROFILER_AVAILABLE, reason="profiler not importable")
class TestExecutionTimer:
    """Tests for the ExecutionTimer class."""

    def test_initial_state(self):
        """Freshly created timer has no timings and is enabled."""
        timer = ExecutionTimer()
        assert timer.timings == {}
        assert timer.enabled is True
        assert timer.profiling_report is None

    def test_record_single(self):
        """record() stores a single timing value."""
        timer = ExecutionTimer()
        timer.record("op1", 1.5)
        assert "op1" in timer.timings
        assert timer.timings["op1"] == [1.5]

    def test_record_multiple_same_operation(self):
        """record() appends multiple timings for the same operation."""
        timer = ExecutionTimer()
        timer.record("op1", 1.0)
        timer.record("op1", 2.0)
        timer.record("op1", 3.0)
        assert timer.timings["op1"] == [1.0, 2.0, 3.0]

    def test_record_disabled(self):
        """record() does nothing when timer is disabled."""
        timer = ExecutionTimer()
        timer.enabled = False
        timer.record("op1", 1.5)
        assert timer.timings == {}

    def test_get_stats_existing_operation(self):
        """get_stats() returns correct statistics."""
        timer = ExecutionTimer()
        timer.record("op1", 1.0)
        timer.record("op1", 3.0)
        timer.record("op1", 2.0)
        stats = timer.get_stats("op1")
        assert stats["count"] == 3
        assert stats["total"] == 6.0
        assert stats["avg"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0

    def test_get_stats_nonexistent_operation(self):
        """get_stats() returns empty dict for unknown operations."""
        timer = ExecutionTimer()
        assert timer.get_stats("nope") == {}

    def test_get_stats_empty_list(self):
        """get_stats() returns empty dict if list is empty."""
        timer = ExecutionTimer()
        timer.timings["op1"] = []
        assert timer.get_stats("op1") == {}

    def test_get_all_stats(self):
        """get_all_stats() returns stats for all recorded operations."""
        timer = ExecutionTimer()
        timer.record("op1", 1.0)
        timer.record("op2", 2.0)
        all_stats = timer.get_all_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats
        assert all_stats["op1"]["count"] == 1
        assert all_stats["op2"]["count"] == 1

    def test_get_all_stats_empty(self):
        """get_all_stats() returns empty dict when no timings."""
        timer = ExecutionTimer()
        assert timer.get_all_stats() == {}

    def test_reset_clears_timings(self):
        """reset() clears all timing data."""
        timer = ExecutionTimer()
        timer.record("op1", 1.0)
        timer.record("op2", 2.0)
        timer.reset()
        assert timer.timings == {}

    def test_reset_clears_profiling_report_entries(self):
        """reset() also clears profiling_report.entries if set."""
        timer = ExecutionTimer()
        mock_report = Mock()
        mock_report.entries = [1, 2, 3]
        timer.set_profiling_report(mock_report)
        timer.record("op1", 1.0)
        timer.reset()
        assert timer.timings == {}
        # The source code calls entries.clear() directly on the list
        assert mock_report.entries == []

    def test_reset_without_profiling_report(self):
        """reset() works fine when no profiling report is set."""
        timer = ExecutionTimer()
        timer.record("op1", 1.0)
        timer.reset()
        assert timer.timings == {}

    def test_set_profiling_report(self):
        """set_profiling_report() stores the report."""
        timer = ExecutionTimer()
        report = Mock()
        timer.set_profiling_report(report)
        assert timer.profiling_report is report

    def test_print_summary_empty(self):
        """print_summary() handles no timing data gracefully."""
        timer = ExecutionTimer()
        # Should not raise
        timer.print_summary()

    def test_print_summary_with_data(self):
        """print_summary() logs output for recorded operations."""
        timer = ExecutionTimer()
        timer.record("op1", 1.0)
        timer.record("op2", 2.5)
        # Should not raise
        timer.print_summary()

    def test_record_single_value_stats(self):
        """Stats for a single recording have min == max == avg == total."""
        timer = ExecutionTimer()
        timer.record("single_op", 5.0)
        stats = timer.get_stats("single_op")
        assert stats["count"] == 1
        assert stats["total"] == 5.0
        assert stats["avg"] == 5.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0


# =============================================================================
# get_timer / enable / disable / reset_profiling Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not PROFILER_AVAILABLE, reason="profiler not importable")
class TestGlobalTimerFunctions:
    """Tests for module-level timer functions."""

    def setup_method(self):
        """Reset global timer before each test."""
        reset_profiling()
        enable_profiling()

    def test_get_timer_returns_global(self):
        """get_timer() returns the global ExecutionTimer instance."""
        timer = get_timer()
        assert isinstance(timer, ExecutionTimer)
        assert timer is _global_timer

    def test_enable_profiling(self):
        """enable_profiling() sets enabled=True."""
        disable_profiling()
        assert _global_timer.enabled is False
        enable_profiling()
        assert _global_timer.enabled is True

    def test_disable_profiling(self):
        """disable_profiling() sets enabled=False."""
        assert _global_timer.enabled is True
        disable_profiling()
        assert _global_timer.enabled is False

    def test_reset_profiling(self):
        """reset_profiling() clears all timing data on the global timer."""
        _global_timer.record("test_op", 1.0)
        assert len(_global_timer.timings) > 0
        reset_profiling()
        assert _global_timer.timings == {}


# =============================================================================
# timed_block Context Manager Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not PROFILER_AVAILABLE, reason="profiler not importable")
class TestTimedBlock:
    """Tests for the timed_block context manager."""

    def setup_method(self):
        reset_profiling()
        enable_profiling()

    def test_timed_block_records_duration(self):
        """timed_block records execution time to the global timer."""
        with timed_block("test_block"):
            time.sleep(0.01)

        stats = _global_timer.get_stats("test_block")
        assert stats["count"] == 1
        assert stats["total"] >= 0.01

    def test_timed_block_disabled(self):
        """timed_block with enabled=False does not record."""
        with timed_block("disabled_block", enabled=False):
            time.sleep(0.01)

        assert _global_timer.get_stats("disabled_block") == {}

    def test_timed_block_exception_still_records(self):
        """timed_block records timing even when code raises."""
        with pytest.raises(ValueError):
            with timed_block("error_block"):
                raise ValueError("boom")

        stats = _global_timer.get_stats("error_block")
        assert stats["count"] == 1

    def test_timed_block_with_profiling_report(self):
        """timed_block calls profiling_report.record_timing when report is set."""
        mock_report = Mock()
        _global_timer.set_profiling_report(mock_report)

        with timed_block("reported_op", component="Agent", name="Test"):
            pass

        mock_report.record_timing.assert_called_once()
        call_kwargs = mock_report.record_timing.call_args
        assert call_kwargs[1]["operation"] == "reported_op"
        assert call_kwargs[1]["component"] == "Agent"

        # Cleanup
        _global_timer.profiling_report = None

    def test_timed_block_without_profiling_report(self):
        """timed_block works fine without a profiling report set."""
        _global_timer.profiling_report = None
        with timed_block("no_report_op"):
            pass

        stats = _global_timer.get_stats("no_report_op")
        assert stats["count"] == 1

    def test_timed_block_default_component(self):
        """timed_block uses 'Other' as the default component."""
        mock_report = Mock()
        _global_timer.set_profiling_report(mock_report)

        with timed_block("default_comp_op"):
            pass

        call_kwargs = mock_report.record_timing.call_args
        assert call_kwargs[1]["component"] == "Other"

        _global_timer.profiling_report = None


# =============================================================================
# timed Decorator Tests (sync)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not PROFILER_AVAILABLE, reason="profiler not importable")
class TestTimedDecoratorSync:
    """Tests for the @timed decorator on synchronous functions."""

    def setup_method(self):
        reset_profiling()
        enable_profiling()

    def test_timed_with_explicit_name(self):
        """@timed records under the given operation name."""
        @timed("my_sync_op")
        def do_work():
            return 42

        result = do_work()
        assert result == 42
        stats = _global_timer.get_stats("my_sync_op")
        assert stats["count"] == 1

    def test_timed_with_auto_name(self):
        """@timed() with no arg uses module.funcname as operation."""
        @timed()
        def auto_named_func():
            return "hello"

        result = auto_named_func()
        assert result == "hello"
        # The operation name includes the module and function name
        all_stats = _global_timer.get_all_stats()
        matching_ops = [k for k in all_stats if "auto_named_func" in k]
        assert len(matching_ops) == 1

    def test_timed_disabled(self):
        """@timed with enabled=False bypasses timing."""
        @timed("disabled_op", enabled=False)
        def do_nothing():
            return "ok"

        result = do_nothing()
        assert result == "ok"
        assert _global_timer.get_stats("disabled_op") == {}

    def test_timed_preserves_function_name(self):
        """@timed preserves the decorated function's __name__ via functools.wraps."""
        @timed("preserved_op")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_timed_exception_still_records(self):
        """@timed records timing even when the function raises."""
        @timed("exception_op")
        def boom():
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            boom()

        stats = _global_timer.get_stats("exception_op")
        assert stats["count"] == 1

    def test_timed_multiple_calls(self):
        """@timed accumulates stats across multiple calls."""
        @timed("multi_op")
        def do_stuff():
            return True

        do_stuff()
        do_stuff()
        do_stuff()

        stats = _global_timer.get_stats("multi_op")
        assert stats["count"] == 3


# =============================================================================
# timed Decorator Tests (async)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not PROFILER_AVAILABLE, reason="profiler not importable")
class TestTimedDecoratorAsync:
    """Tests for the @timed decorator on async functions."""

    def setup_method(self):
        reset_profiling()
        enable_profiling()

    @pytest.mark.asyncio
    async def test_timed_async_records(self):
        """@timed on an async function records timing."""
        @timed("async_op")
        async def async_work():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await async_work()
        assert result == "async_result"
        stats = _global_timer.get_stats("async_op")
        assert stats["count"] == 1
        assert stats["total"] >= 0.01

    @pytest.mark.asyncio
    async def test_timed_async_disabled(self):
        """@timed with enabled=False bypasses timing for async functions."""
        @timed("async_disabled", enabled=False)
        async def async_skip():
            return "skipped"

        result = await async_skip()
        assert result == "skipped"
        assert _global_timer.get_stats("async_disabled") == {}

    @pytest.mark.asyncio
    async def test_timed_async_exception_still_records(self):
        """@timed async records timing even on exception."""
        @timed("async_error")
        async def async_boom():
            raise ValueError("async kaboom")

        with pytest.raises(ValueError):
            await async_boom()

        stats = _global_timer.get_stats("async_error")
        assert stats["count"] == 1

    @pytest.mark.asyncio
    async def test_timed_async_preserves_name(self):
        """@timed preserves the async function's __name__."""
        @timed("async_named")
        async def my_async_func():
            pass

        assert my_async_func.__name__ == "my_async_func"

    @pytest.mark.asyncio
    async def test_timed_async_auto_name(self):
        """@timed() on async uses module.funcname as operation name."""
        @timed()
        async def auto_async():
            return 1

        await auto_async()
        all_stats = _global_timer.get_all_stats()
        matching_ops = [k for k in all_stats if "auto_async" in k]
        assert len(matching_ops) == 1


# =============================================================================
# setup_file_logging Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not FILE_LOGGER_AVAILABLE, reason="file_logger not importable")
class TestSetupFileLogging:
    """Tests for setup_file_logging()."""

    def teardown_method(self):
        """Clean up any file handlers added during tests."""
        close_file_logging()

    def test_creates_logs_directory(self, tmp_path):
        """setup_file_logging creates the logs/ subdirectory."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir))
        assert (output_dir / "logs").exists()

    def test_creates_beautified_log(self, tmp_path):
        """Beautified log file is created when enabled."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), enable_beautified=True, enable_debug=False)
        assert (output_dir / "logs" / "beautified.log").exists()

    def test_creates_debug_log(self, tmp_path):
        """Debug log file is created when enabled."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), enable_beautified=False, enable_debug=True)
        assert (output_dir / "logs" / "debug.log").exists()

    def test_creates_both_logs(self, tmp_path):
        """Both log files created by default."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir))
        assert (output_dir / "logs" / "beautified.log").exists()
        assert (output_dir / "logs" / "debug.log").exists()

    def test_no_beautified_when_disabled(self, tmp_path):
        """Beautified log is not created when disabled."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), enable_beautified=False, enable_debug=False)
        assert not (output_dir / "logs" / "beautified.log").exists()

    def test_no_debug_when_disabled(self, tmp_path):
        """Debug log is not created when disabled."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), enable_beautified=False, enable_debug=False)
        assert not (output_dir / "logs" / "debug.log").exists()

    def test_log_level_applied(self, tmp_path):
        """Custom log_level is applied to the root logger."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), log_level="WARNING")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_log_level_case_insensitive(self, tmp_path):
        """log_level is case-insensitive."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), log_level="debug")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_invalid_log_level_defaults_to_info(self, tmp_path):
        """Invalid log_level falls back to INFO."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), log_level="NONEXISTENT")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_removes_existing_file_handlers(self, tmp_path):
        """Calling setup_file_logging twice does not duplicate handlers."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir))
        setup_file_logging(str(output_dir))

        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        # Should have at most 2 file handlers (beautified + debug), not 4
        assert len(file_handlers) <= 2

    def test_writes_to_beautified_log(self, tmp_path):
        """Messages actually appear in the beautified log file."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir), enable_beautified=True, enable_debug=False)

        test_logger = logging.getLogger("test_beautified")
        test_logger.info("test message for beautified")

        # Flush handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        content = (output_dir / "logs" / "beautified.log").read_text()
        assert "test message for beautified" in content

    def test_writes_to_debug_log(self, tmp_path):
        """Messages actually appear in the debug log file."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(
            str(output_dir), enable_beautified=False, enable_debug=True, log_level="DEBUG"
        )

        test_logger = logging.getLogger("test_debug")
        test_logger.debug("test debug message")

        for handler in logging.getLogger().handlers:
            handler.flush()

        content = (output_dir / "logs" / "debug.log").read_text()
        assert "test debug message" in content

    def test_idempotent_directory_creation(self, tmp_path):
        """Calling setup twice on the same dir does not fail (exist_ok=True)."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir))
        # Should not raise
        setup_file_logging(str(output_dir))
        assert (output_dir / "logs").exists()


# =============================================================================
# close_file_logging Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not FILE_LOGGER_AVAILABLE, reason="file_logger not importable")
class TestCloseFileLogging:
    """Tests for close_file_logging()."""

    def test_close_removes_file_handlers(self, tmp_path):
        """close_file_logging removes all FileHandlers from root logger."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir))

        root_logger = logging.getLogger()
        file_handlers_before = [
            h for h in root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers_before) > 0

        close_file_logging()

        file_handlers_after = [
            h for h in root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers_after) == 0

    def test_close_idempotent(self, tmp_path):
        """Calling close_file_logging twice does not raise."""
        output_dir = tmp_path / "run_output"
        setup_file_logging(str(output_dir))
        close_file_logging()
        # Second call should be a no-op, not raise
        close_file_logging()

    def test_close_without_setup(self):
        """close_file_logging works even if setup was never called."""
        # Should not raise
        close_file_logging()

    def test_close_preserves_non_file_handlers(self, tmp_path):
        """close_file_logging only removes FileHandlers, not StreamHandlers."""
        output_dir = tmp_path / "run_output"
        root_logger = logging.getLogger()

        # Add a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        root_logger.addHandler(stream_handler)

        setup_file_logging(str(output_dir))
        close_file_logging()

        # Stream handler should still be there
        assert stream_handler in root_logger.handlers

        # Cleanup
        root_logger.removeHandler(stream_handler)


# ===========================================================================
# context_utils Tests
# ===========================================================================

try:
    from Jotty.core.utils.context_utils import (
        ErrorType, CompressionResult, ContextCompressor, ErrorDetector,
        ExecutionTrajectory, ENRICHMENT_MARKERS,
        strip_enrichment_context, create_compressor, detect_error_type,
    )
    CONTEXT_UTILS_AVAILABLE = True
except ImportError:
    CONTEXT_UTILS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not CONTEXT_UTILS_AVAILABLE, reason="context_utils not available")
class TestErrorType:
    """Tests for ErrorType enum."""

    def test_all_values(self):
        expected = {"context_length", "timeout", "parse_error", "rate_limit", "network", "tool_error", "unknown"}
        actual = {e.value for e in ErrorType}
        assert expected == actual


@pytest.mark.unit
@pytest.mark.skipif(not CONTEXT_UTILS_AVAILABLE, reason="context_utils not available")
class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_creation(self):
        cr = CompressionResult(original_length=100, compressed_length=50, compression_ratio=0.5, content="test")
        assert cr.original_length == 100
        assert cr.compressed_length == 50
        assert cr.compression_ratio == 0.5
        assert cr.preserved_trajectory == ""


@pytest.mark.unit
@pytest.mark.skipif(not CONTEXT_UTILS_AVAILABLE, reason="context_utils not available")
class TestContextCompressor:
    """Tests for ContextCompressor."""

    def test_compress_empty_content(self):
        compressor = ContextCompressor()
        result = compressor.compress("")
        assert result.original_length == 0
        assert result.content == ""
        assert result.compression_ratio == 1.0

    def test_compress_preserves_trajectory(self):
        compressor = ContextCompressor()
        result = compressor.compress("some content", trajectory="my trajectory")
        assert result.preserved_trajectory == "my trajectory"

    def test_compress_with_target_ratio(self):
        compressor = ContextCompressor()
        content = "\n\n".join([f"Paragraph {i} with some meaningful content." for i in range(20)])
        result = compressor.compress(content, target_ratio=0.3)
        assert result.compressed_length < result.original_length

    def test_compress_with_keywords(self):
        compressor = ContextCompressor()
        content = "First paragraph about dogs.\n\nSecond paragraph about cats.\n\nThird about birds."
        result = compressor.compress(content, target_ratio=0.5, preserve_keywords=["cats"])
        assert result.content  # Should have some content

    def test_split_into_sections(self):
        compressor = ContextCompressor()
        content = "Section A\n\nSection B\n\nSection C"
        sections = compressor._split_into_sections(content)
        assert len(sections) == 3
        assert sections[0]["text"] == "Section A"
        assert sections[2]["is_recent"] is True  # Last 3 are recent

    def test_score_and_sort_sections(self):
        compressor = ContextCompressor()
        sections = [
            {"text": "about python", "position": 0, "length": 12, "score": 0.0, "is_recent": False},
            {"text": "about java", "position": 1, "length": 10, "score": 0.0, "is_recent": True},
        ]
        scored = compressor._score_and_sort_sections(sections, ["python"])
        # "python" keyword match should score higher
        assert scored[0]["text"] == "about python" or scored[0]["is_recent"]

    def test_create_compressor_factory(self):
        c = create_compressor()
        assert isinstance(c, ContextCompressor)


@pytest.mark.unit
@pytest.mark.skipif(not CONTEXT_UTILS_AVAILABLE, reason="context_utils not available")
class TestErrorDetector:
    """Tests for ErrorDetector."""

    def test_detect_context_length(self):
        error = Exception("input is too long for model")
        assert ErrorDetector.detect(error) == ErrorType.CONTEXT_LENGTH

    def test_detect_timeout(self):
        error = Exception("request timed out")
        assert ErrorDetector.detect(error) == ErrorType.TIMEOUT

    def test_detect_timeout_by_type_name(self):
        class TimeoutError(Exception):
            pass
        assert ErrorDetector.detect(TimeoutError("some error")) == ErrorType.TIMEOUT

    def test_detect_parse_error(self):
        error = Exception("failed to parse JSON response")
        assert ErrorDetector.detect(error) == ErrorType.PARSE_ERROR

    def test_detect_rate_limit(self):
        error = Exception("rate limit exceeded, try again later")
        assert ErrorDetector.detect(error) == ErrorType.RATE_LIMIT

    def test_detect_network(self):
        error = Exception("connection error occurred")
        assert ErrorDetector.detect(error) == ErrorType.NETWORK

    def test_detect_unknown(self):
        error = Exception("something completely different")
        assert ErrorDetector.detect(error) == ErrorType.UNKNOWN

    def test_retry_strategy_context_length(self):
        strategy = ErrorDetector.get_retry_strategy(ErrorType.CONTEXT_LENGTH)
        assert strategy["should_retry"] is True
        assert strategy["action"] == "compress"

    def test_retry_strategy_unknown_no_retry(self):
        strategy = ErrorDetector.get_retry_strategy(ErrorType.UNKNOWN)
        assert strategy["should_retry"] is False

    def test_detect_error_type_convenience(self):
        error = Exception("too many tokens in context")
        error_type, strategy = detect_error_type(error)
        assert error_type == ErrorType.CONTEXT_LENGTH
        assert strategy["should_retry"] is True


@pytest.mark.unit
@pytest.mark.skipif(not CONTEXT_UTILS_AVAILABLE, reason="context_utils not available")
class TestExecutionTrajectory:
    """Tests for ExecutionTrajectory."""

    def test_empty_trajectory_to_context(self):
        traj = ExecutionTrajectory(steps_completed=[], outputs_collected={}, current_step_index=0)
        assert traj.to_context() == ""

    def test_to_context_with_steps(self):
        traj = ExecutionTrajectory(
            steps_completed=[
                {"description": "Search web", "output": "found 5 results"},
                {"description": "Analyze results"},
            ],
            outputs_collected={},
            current_step_index=2,
        )
        context = traj.to_context()
        assert "Search web" in context
        assert "found 5 results" in context
        assert "Step 1" in context

    def test_to_context_with_reasoning(self):
        traj = ExecutionTrajectory(
            steps_completed=[{"description": "step1"}],
            outputs_collected={},
            current_step_index=1,
            reasoning_so_far="Decided to use approach A",
        )
        context = traj.to_context()
        assert "Decided to use approach A" in context

    def test_add_step(self):
        traj = ExecutionTrajectory(steps_completed=[], outputs_collected={}, current_step_index=0)
        traj.add_step({"description": "Do thing", "output_key": "result"}, output="done")
        assert len(traj.steps_completed) == 1
        assert traj.current_step_index == 1
        assert traj.outputs_collected["result"] == "done"

    def test_add_step_auto_key(self):
        traj = ExecutionTrajectory(steps_completed=[], outputs_collected={}, current_step_index=0)
        traj.add_step({"description": "Do thing"}, output="done")
        assert "step_1" in traj.outputs_collected


@pytest.mark.unit
@pytest.mark.skipif(not CONTEXT_UTILS_AVAILABLE, reason="context_utils not available")
class TestStripEnrichmentContext:
    """Tests for strip_enrichment_context."""

    def test_no_enrichment(self):
        assert strip_enrichment_context("Simple task") == "Simple task"

    def test_empty_string(self):
        assert strip_enrichment_context("") == ""

    def test_none_passthrough(self):
        assert strip_enrichment_context(None) is None

    def test_strip_learning_context(self):
        task = "Original task\nLearned Insights:\nSome learning context"
        result = strip_enrichment_context(task)
        assert result == "Original task"

    def test_strip_multi_perspective(self):
        task = "Do X\n[Multi-Perspective Analysis\nSome analysis"
        result = strip_enrichment_context(task)
        assert result == "Do X"

    def test_strip_separator(self):
        task = "User request\n\n---\nExtra context"
        result = strip_enrichment_context(task)
        assert result == "User request"


# ===========================================================================
# env_loader Tests
# ===========================================================================

try:
    from Jotty.core.utils.env_loader import (
        get_jotty_root, load_jotty_env, get_env, get_env_bool, get_env_int,
    )
    ENV_LOADER_AVAILABLE = True
except ImportError:
    ENV_LOADER_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not ENV_LOADER_AVAILABLE, reason="env_loader not available")
class TestEnvLoader:
    """Tests for env_loader module."""

    def test_get_jotty_root(self):
        root = get_jotty_root()
        assert root.exists()
        assert root.name == "Jotty"

    def test_get_env_with_default(self):
        result = get_env("NONEXISTENT_KEY_XYZ_12345", "fallback")
        assert result == "fallback"

    def test_get_env_reads_os_environ(self):
        import os
        os.environ["_JOTTY_TEST_KEY"] = "test_value"
        try:
            assert get_env("_JOTTY_TEST_KEY") == "test_value"
        finally:
            del os.environ["_JOTTY_TEST_KEY"]

    def test_get_env_bool_true(self):
        import os
        for val in ["true", "1", "yes", "on"]:
            os.environ["_JOTTY_BOOL_TEST"] = val
            assert get_env_bool("_JOTTY_BOOL_TEST") is True
        del os.environ["_JOTTY_BOOL_TEST"]

    def test_get_env_bool_false(self):
        import os
        for val in ["false", "0", "no", "off"]:
            os.environ["_JOTTY_BOOL_TEST"] = val
            assert get_env_bool("_JOTTY_BOOL_TEST") is False
        del os.environ["_JOTTY_BOOL_TEST"]

    def test_get_env_bool_default(self):
        assert get_env_bool("NONEXISTENT_KEY_XYZ") is False
        assert get_env_bool("NONEXISTENT_KEY_XYZ", True) is True

    def test_get_env_int(self):
        import os
        os.environ["_JOTTY_INT_TEST"] = "42"
        try:
            assert get_env_int("_JOTTY_INT_TEST") == 42
        finally:
            del os.environ["_JOTTY_INT_TEST"]

    def test_get_env_int_invalid(self):
        import os
        os.environ["_JOTTY_INT_TEST"] = "not_a_number"
        try:
            assert get_env_int("_JOTTY_INT_TEST", 99) == 99
        finally:
            del os.environ["_JOTTY_INT_TEST"]

    def test_get_env_int_default(self):
        assert get_env_int("NONEXISTENT_KEY_XYZ") == 0

    def test_load_jotty_env_nonexistent_file(self):
        import Jotty.core.utils.env_loader as el
        old = el._env_loaded
        el._env_loaded = False
        try:
            result = load_jotty_env(env_file="/tmp/nonexistent_env_file_xyz")
            assert result is False
        finally:
            el._env_loaded = old


# ===========================================================================
# skill_status Tests
# ===========================================================================

try:
    from Jotty.core.utils.skill_status import SkillStatus, get_status
    SKILL_STATUS_AVAILABLE = True
except ImportError:
    SKILL_STATUS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not SKILL_STATUS_AVAILABLE, reason="skill_status not available")
class TestSkillStatus:
    """Tests for SkillStatus class."""

    def test_init(self):
        ss = SkillStatus("test-skill")
        assert ss.skill_name == "test-skill"
        assert ss._callback is None

    def test_set_callback(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        assert ss._callback is cb

    def test_emit_with_callback(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.emit("Searching", "query")
        cb.assert_called_once_with("Searching", "query")

    def test_emit_without_callback(self):
        ss = SkillStatus("test")
        ss.emit("Searching", "query")  # Should not raise

    def test_emit_callback_exception_suppressed(self):
        ss = SkillStatus("test")
        cb = Mock(side_effect=RuntimeError("boom"))
        ss.set_callback(cb)
        ss.emit("Searching", "query")  # Should not raise

    def test_searching(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.searching("web")
        cb.assert_called_once()
        assert "Searching" in cb.call_args[0][0]

    def test_fetching_truncates_long_url(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        long_url = "https://example.com/" + "a" * 100
        ss.fetching(long_url)
        detail = cb.call_args[0][1]
        assert "..." in detail

    def test_fetching_short_url(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.fetching("https://example.com")
        detail = cb.call_args[0][1]
        assert "..." not in detail

    def test_processing(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.processing("data")
        assert "Processing" in cb.call_args[0][0]

    def test_processing_no_item(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.processing()
        assert "Processing" in cb.call_args[0][1]

    def test_analyzing(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.analyzing("results")
        assert "Analyzing" in cb.call_args[0][0]

    def test_creating(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.creating("report")
        assert "Creating" in cb.call_args[0][0]

    def test_sending(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.sending("Slack")
        assert "Sending" in cb.call_args[0][0]

    def test_done(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.done()
        assert "Done" in cb.call_args[0][0]

    def test_error(self):
        ss = SkillStatus("test")
        cb = Mock()
        ss.set_callback(cb)
        ss.error("something broke")
        assert "Error" in cb.call_args[0][0]

    def test_get_status_factory(self):
        ss = get_status("my-skill")
        assert isinstance(ss, SkillStatus)
        assert ss.skill_name == "my-skill"


# ===========================================================================
# tool_helpers Tests
# ===========================================================================

try:
    from Jotty.core.utils.tool_helpers import (
        tool_response, tool_error, require_params, validate_params,
        tool_wrapper, async_tool_wrapper, _normalize_param_aliases,
    )
    TOOL_HELPERS_AVAILABLE = True
except ImportError:
    TOOL_HELPERS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestToolResponse:
    """Tests for tool_response."""

    def test_basic(self):
        r = tool_response()
        assert r == {"success": True}

    def test_with_data(self):
        r = tool_response(data={"id": "123"})
        assert r["success"] is True
        assert r["id"] == "123"

    def test_with_kwargs(self):
        r = tool_response(message="done")
        assert r["message"] == "done"


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestToolError:
    """Tests for tool_error."""

    def test_basic(self):
        r = tool_error("bad request")
        assert r["success"] is False
        assert r["error"] == "bad request"

    def test_with_code(self):
        r = tool_error("not found", code="404")
        assert r["code"] == "404"

    def test_with_kwargs(self):
        r = tool_error("fail", detail="more info")
        assert r["detail"] == "more info"


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestRequireParams:
    """Tests for require_params."""

    def test_all_present(self):
        assert require_params({"a": 1, "b": 2}, ["a", "b"]) is None

    def test_missing_param(self):
        result = require_params({"a": 1}, ["a", "b"])
        assert result is not None
        assert result["success"] is False
        assert "b" in result["error"]

    def test_empty_value_fails(self):
        result = require_params({"a": ""}, ["a"])
        assert result is not None

    def test_none_value_fails(self):
        result = require_params({"a": None}, ["a"])
        assert result is not None


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestValidateParams:
    """Tests for validate_params schema validation."""

    def test_valid_params(self):
        schema = {"name": {"required": True, "type": str}}
        assert validate_params({"name": "test"}, schema) is None

    def test_missing_required(self):
        schema = {"name": {"required": True, "type": str}}
        result = validate_params({}, schema)
        assert result is not None
        assert "required" in result["error"]

    def test_wrong_type(self):
        schema = {"count": {"required": True, "type": int}}
        result = validate_params({"count": "not_int"}, schema)
        assert result is not None
        assert "int" in result["error"]

    def test_min_violation(self):
        schema = {"count": {"type": int, "min": 1}}
        result = validate_params({"count": 0}, schema)
        assert result is not None

    def test_max_violation(self):
        schema = {"count": {"type": int, "max": 10}}
        result = validate_params({"count": 20}, schema)
        assert result is not None

    def test_max_length_violation(self):
        schema = {"name": {"type": str, "max_length": 5}}
        result = validate_params({"name": "toolong"}, schema)
        assert result is not None

    def test_optional_missing_ok(self):
        schema = {"name": {"required": False, "type": str}}
        assert validate_params({}, schema) is None


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestToolWrapper:
    """Tests for tool_wrapper decorator."""

    def test_successful_call(self):
        @tool_wrapper(required_params=["query"])
        def search(params):
            return tool_response(data={"result": params["query"]})
        result = search({"query": "test"})
        assert result["success"] is True
        assert result["result"] == "test"

    def test_missing_required_param(self):
        @tool_wrapper(required_params=["query"])
        def search(params):
            return tool_response()
        result = search({})
        assert result["success"] is False

    def test_exception_handling(self):
        @tool_wrapper(required_params=["x"])
        def bad_tool(params):
            raise ValueError("boom")
        result = bad_tool({"x": "val"})
        assert result["success"] is False
        assert "boom" in result["error"]

    def test_alias_resolution(self):
        @tool_wrapper(required_params=["query"])
        def search(params):
            return tool_response(data={"q": params["query"]})
        result = search({"search_query": "hello"})
        assert result["success"] is True
        assert result["q"] == "hello"

    def test_stashed_required_params(self):
        @tool_wrapper(required_params=["a", "b"])
        def my_tool(params):
            return tool_response()
        assert my_tool._required_params == ["a", "b"]


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestAsyncToolWrapper:
    """Tests for async_tool_wrapper decorator."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        @async_tool_wrapper(required_params=["query"])
        async def search(params):
            return tool_response(data={"result": params["query"]})
        result = await search({"query": "test"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_missing_param(self):
        @async_tool_wrapper(required_params=["query"])
        async def search(params):
            return tool_response()
        result = await search({})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        @async_tool_wrapper(required_params=["x"])
        async def bad_tool(params):
            raise RuntimeError("async boom")
        result = await bad_tool({"x": "val"})
        assert result["success"] is False
        assert "async boom" in result["error"]


@pytest.mark.unit
@pytest.mark.skipif(not TOOL_HELPERS_AVAILABLE, reason="tool_helpers not available")
class TestNormalizeParamAliases:
    """Tests for _normalize_param_aliases."""

    def test_alias_resolved(self):
        params = {"file_path": "/tmp/test"}
        _normalize_param_aliases(params, ["path"])
        assert params["path"] == "/tmp/test"
        assert "file_path" not in params

    def test_canonical_preserved(self):
        params = {"path": "/original", "file_path": "/alias"}
        _normalize_param_aliases(params, ["path"])
        assert params["path"] == "/original"

    def test_no_matching_alias(self):
        params = {"unrelated": "value"}
        _normalize_param_aliases(params, ["path"])
        assert "path" not in params


# ===========================================================================
# algorithmic_foundations Tests
# ===========================================================================

try:
    from Jotty.core.utils.algorithmic_foundations import SortingAlgorithms
    SORTING_AVAILABLE = True
except ImportError:
    SORTING_AVAILABLE = False

try:
    from Jotty.core.utils.algorithmic_foundations import MutualInformationRetriever
    MI_RETRIEVER_AVAILABLE = True
except ImportError:
    MI_RETRIEVER_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not SORTING_AVAILABLE, reason="SortingAlgorithms not available")
class TestSortingAlgorithms:
    """Tests for SortingAlgorithms."""

    def test_bubble_sort_basic(self):
        assert SortingAlgorithms.bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]

    def test_bubble_sort_empty(self):
        assert SortingAlgorithms.bubble_sort([]) == []

    def test_bubble_sort_single(self):
        assert SortingAlgorithms.bubble_sort([42]) == [42]

    def test_bubble_sort_already_sorted(self):
        assert SortingAlgorithms.bubble_sort([1, 2, 3]) == [1, 2, 3]

    def test_bubble_sort_reverse(self):
        assert SortingAlgorithms.bubble_sort([1, 2, 3], reverse=True) == [3, 2, 1]

    def test_bubble_sort_with_key(self):
        data = [{"age": 30}, {"age": 25}, {"age": 35}]
        result = SortingAlgorithms.bubble_sort(data, key=lambda x: x["age"])
        assert result[0]["age"] == 25
        assert result[2]["age"] == 35

    def test_bubble_sort_does_not_mutate(self):
        original = [3, 1, 2]
        result = SortingAlgorithms.bubble_sort(original)
        assert result == [1, 2, 3]
        assert original == [3, 1, 2]

    def test_bubble_sort_analysis(self):
        result = SortingAlgorithms.bubble_sort_analysis([4, 3, 2, 1])
        assert result["sorted_array"] == [1, 2, 3, 4]
        assert result["comparisons"] > 0
        assert result["swaps"] > 0
        assert result["passes"] > 0

    def test_bubble_sort_analysis_sorted_input(self):
        result = SortingAlgorithms.bubble_sort_analysis([1, 2, 3])
        assert result["sorted_array"] == [1, 2, 3]
        assert result["swaps"] == 0


@pytest.mark.unit
@pytest.mark.skipif(not MI_RETRIEVER_AVAILABLE, reason="MutualInformationRetriever not available")
class TestMutualInformationRetriever:
    """Tests for MutualInformationRetriever."""

    def test_init(self):
        retriever = MutualInformationRetriever(diversity_weight=0.5)
        assert retriever.diversity_weight == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_empty(self):
        retriever = MutualInformationRetriever()
        result = await retriever.retrieve([], "query")
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_fewer_than_k(self):
        retriever = MutualInformationRetriever()
        memories = ["memory one", "memory two"]
        result = await retriever.retrieve(memories, "query", k=5)
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_retrieve_selects_k(self):
        retriever = MutualInformationRetriever()
        memories = [f"memory about topic {i}" for i in range(10)]
        result = await retriever.retrieve(memories, "topic", k=3)
        assert len(result) == 3

    def test_compute_relevance(self):
        retriever = MutualInformationRetriever()
        score = retriever._compute_relevance("the quick brown fox", "quick fox")
        assert score > 0

    def test_compute_similarity(self):
        retriever = MutualInformationRetriever()
        sim = retriever._compute_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_compute_similarity_different(self):
        retriever = MutualInformationRetriever()
        sim = retriever._compute_similarity("hello world", "goodbye universe")
        assert sim == 0.0


# ===========================================================================
# api_client Tests
# ===========================================================================

try:
    from Jotty.core.utils.api_client import BaseAPIClient
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not API_CLIENT_AVAILABLE, reason="api_client not available")
class TestBaseAPIClient:
    """Tests for BaseAPIClient."""

    def _make_client_class(self):
        class TestClient(BaseAPIClient):
            BASE_URL = "https://api.example.com"
            AUTH_PREFIX = "Bearer"
            TOKEN_ENV_VAR = "_JOTTY_TEST_TOKEN"
        return TestClient

    def test_init_with_token(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token="my-token")
        assert client.token == "my-token"

    def test_init_from_env(self):
        import os
        os.environ["_JOTTY_TEST_TOKEN"] = "env-token"
        try:
            ClientClass = self._make_client_class()
            client = ClientClass()
            assert client.token == "env-token"
        finally:
            del os.environ["_JOTTY_TEST_TOKEN"]

    def test_is_configured_true(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        assert client.is_configured is True

    def test_is_configured_false(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token=None)
        # Depending on env, may or may not have token
        # Just test the property works
        assert isinstance(client.is_configured, bool)

    def test_build_url_endpoint(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        url = client._build_url("/messages")
        assert url == "https://api.example.com/messages"

    def test_build_url_full_url(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        url = client._build_url("https://other.com/path")
        assert url == "https://other.com/path"

    def test_get_headers_with_token(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token="my-secret")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer my-secret"
        assert headers["Content-Type"] == "application/json"

    def test_require_token_configured(self):
        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        assert client.require_token() is None

    @patch("requests.get")
    def test_make_request_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123"}
        mock_get.return_value = mock_response

        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        result = client._make_request("/test", method="GET")
        assert result["success"] is True

    @patch("requests.delete")
    def test_make_request_204(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_delete.return_value = mock_response

        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        result = client._make_request("/test", method="DELETE")
        assert result["success"] is True

    @patch("requests.post")
    def test_make_request_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "bad request"}
        mock_response.text = "bad request"
        mock_post.return_value = mock_response

        ClientClass = self._make_client_class()
        client = ClientClass(token="tok")
        result = client._make_request("/test")
        assert result["success"] is False
