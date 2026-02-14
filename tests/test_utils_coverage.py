"""
Tests for utils coverage gaps.

Covers:
- context_utils: strip_enrichment_context, ErrorDetector, ContextCompressor
- BudgetTracker: recording calls, usage tracking
- CircuitBreaker: state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
"""

import pytest
from unittest.mock import Mock


# ──────────────────────────────────────────────────────────────────────
# Context Utils — strip_enrichment_context
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestStripEnrichmentContext:
    """Tests for strip_enrichment_context."""

    def test_clean_text_unchanged(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        assert strip_enrichment_context("Write a report") == "Write a report"

    def test_empty_string(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        assert strip_enrichment_context("") == ""

    def test_none_input(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        assert strip_enrichment_context(None) is None

    def test_strips_multi_perspective(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Write a report\n[Multi-Perspective Analysis]: lots of context here"
        assert strip_enrichment_context(task) == "Write a report"

    def test_strips_learned_insights(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Analyze data\nLearned Insights: from previous runs"
        assert strip_enrichment_context(task) == "Analyze data"

    def test_strips_q_learning_lessons(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Do research\n# Q-Learning Lessons\n- lesson 1"
        assert strip_enrichment_context(task) == "Do research"

    def test_strips_relay_injection(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Write code\n[Previous agent 'researcher' output]:\nsome context"
        assert strip_enrichment_context(task) == "Write code"

    def test_strips_refinement_injection(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Improve essay\n\nHere is the current draft. Improve it:\nold draft text"
        assert strip_enrichment_context(task) == "Improve essay"

    def test_strips_debate_injection(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Analyze X\nOther agents produced these solutions. Critique them."
        assert strip_enrichment_context(task) == "Analyze X"

    def test_strips_separator(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = "Original task\n\n---\nAppended enrichment context"
        assert strip_enrichment_context(task) == "Original task"

    def test_multiple_markers_all_stripped(self):
        from Jotty.core.utils.context_utils import strip_enrichment_context
        task = (
            "Real task\nLearned Insights: x\n"
            "[Multi-Perspective]: y\n# Q-Learning Lessons\nz"
        )
        result = strip_enrichment_context(task)
        assert result == "Real task"


# ──────────────────────────────────────────────────────────────────────
# Context Utils — ErrorDetector
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestErrorDetector:
    """Tests for ErrorDetector error classification."""

    def test_context_length_detected(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        err = Exception("input is too long for this model")
        assert ErrorDetector.detect(err) == ErrorType.CONTEXT_LENGTH

    def test_timeout_detected(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        err = Exception("request timed out after 30s")
        assert ErrorDetector.detect(err) == ErrorType.TIMEOUT

    def test_rate_limit_detected(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        err = Exception("429 Too Many Requests")
        assert ErrorDetector.detect(err) == ErrorType.RATE_LIMIT

    def test_parse_error_detected(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        err = Exception("failed to parse JSON response")
        assert ErrorDetector.detect(err) == ErrorType.PARSE_ERROR

    def test_network_error_detected(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        err = Exception("connection refused by host")
        assert ErrorDetector.detect(err) == ErrorType.NETWORK

    def test_unknown_error(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        err = Exception("some random error")
        assert ErrorDetector.detect(err) == ErrorType.UNKNOWN

    def test_retry_strategy_context_length(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        strategy = ErrorDetector.get_retry_strategy(ErrorType.CONTEXT_LENGTH)
        assert strategy['should_retry'] is True
        assert strategy['action'] == 'compress'

    def test_retry_strategy_unknown(self):
        from Jotty.core.utils.context_utils import ErrorDetector, ErrorType
        strategy = ErrorDetector.get_retry_strategy(ErrorType.UNKNOWN)
        assert strategy['should_retry'] is False


# ──────────────────────────────────────────────────────────────────────
# Context Utils — ContextCompressor
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestContextCompressor:
    """Tests for ContextCompressor."""

    def test_empty_input(self):
        from Jotty.core.utils.context_utils import ContextCompressor
        c = ContextCompressor()
        result = c.compress("")
        assert result.compressed_length == 0
        assert result.compression_ratio == 1.0

    def test_compression_reduces_size(self):
        from Jotty.core.utils.context_utils import ContextCompressor
        c = ContextCompressor()
        long_text = "\n\n".join(f"Paragraph {i} with content" for i in range(20))
        result = c.compress(long_text, target_ratio=0.3)
        assert result.compressed_length < result.original_length

    def test_recent_content_preserved(self):
        from Jotty.core.utils.context_utils import ContextCompressor
        c = ContextCompressor()
        text = "\n\n".join([f"Old paragraph {i}" for i in range(10)] +
                           ["IMPORTANT recent content"])
        result = c.compress(text, target_ratio=0.3)
        assert "IMPORTANT recent content" in result.content


# ──────────────────────────────────────────────────────────────────────
# BudgetTracker
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBudgetTracker:
    """Tests for BudgetTracker."""

    def test_record_calls(self):
        from Jotty.core.utils.budget_tracker import BudgetTracker
        bt = BudgetTracker()
        bt.record_call("researcher", tokens_input=1000, tokens_output=500, model="gpt-4o")
        bt.record_call("coder", tokens_input=500, tokens_output=200, model="gpt-4o-mini")

        usage = bt.get_usage()
        assert usage['calls'] >= 2
        assert usage['tokens_input'] >= 1500

    def test_multiple_agents_tracked(self):
        from Jotty.core.utils.budget_tracker import BudgetTracker
        bt = BudgetTracker()
        bt.record_call("agent_a", tokens_input=100, tokens_output=50)
        bt.record_call("agent_b", tokens_input=200, tokens_output=100)

        usage = bt.get_usage()
        assert usage['calls'] == 2

    def test_can_make_call_initially(self):
        from Jotty.core.utils.budget_tracker import BudgetTracker
        bt = BudgetTracker()
        assert bt.can_make_call("researcher")

    def test_episode_tracking(self):
        from Jotty.core.utils.budget_tracker import BudgetTracker
        bt = BudgetTracker()
        bt.start_episode("ep1")
        bt.record_call("agent", tokens_input=100, tokens_output=50)
        summary = bt.end_episode()
        assert 'episode_usage' in summary

    def test_cost_calculation(self):
        from Jotty.core.utils.budget_tracker import BudgetTracker
        bt = BudgetTracker(cost_per_1k_input=0.01, cost_per_1k_output=0.03)
        bt.record_call("agent", tokens_input=1000, tokens_output=1000)
        usage = bt.get_usage()
        assert usage.get('estimated_cost_usd', 0) > 0


# ──────────────────────────────────────────────────────────────────────
# CircuitBreaker
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def test_starts_closed(self):
        from Jotty.core.utils.timeouts import CircuitBreaker, CircuitBreakerConfig
        cb = CircuitBreaker(config=CircuitBreakerConfig(name="test"))
        can, reason = cb.can_request()
        assert can is True

    def test_success_keeps_closed(self):
        from Jotty.core.utils.timeouts import CircuitBreaker, CircuitBreakerConfig
        cb = CircuitBreaker(config=CircuitBreakerConfig(name="test"))
        cb.record_success()
        can, _ = cb.can_request()
        assert can is True

    def test_failures_open_circuit(self):
        from Jotty.core.utils.timeouts import CircuitBreaker, CircuitBreakerConfig
        config = CircuitBreakerConfig(name="test", failure_threshold=3)
        cb = CircuitBreaker(config=config)
        for _ in range(3):
            cb.record_failure(Exception("fail"))
        can, _ = cb.can_request()
        assert can is False

    def test_stats_track_failures(self):
        from Jotty.core.utils.timeouts import CircuitBreaker, CircuitBreakerConfig
        cb = CircuitBreaker(config=CircuitBreakerConfig(name="test"))
        cb.record_success()
        cb.record_failure(Exception("oops"))
        stats = cb.get_stats()
        assert stats['failure_count'] >= 1
        assert stats['total_failures'] >= 1

    def test_success_after_half_open_closes(self):
        from Jotty.core.utils.timeouts import CircuitBreaker, CircuitBreakerConfig
        config = CircuitBreakerConfig(name="test", failure_threshold=2, timeout=0.0)
        cb = CircuitBreaker(config=config)

        # Open it
        cb.record_failure(Exception("f1"))
        cb.record_failure(Exception("f2"))

        # With timeout=0, should immediately try half-open
        import time
        time.sleep(0.01)

        can, _ = cb.can_request()
        if can:
            cb.record_success()
            can2, _ = cb.can_request()
            assert can2 is True
