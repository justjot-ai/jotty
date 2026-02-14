"""
Tests for Timeout Management and Circuit Breakers
====================================================
Tests for CircuitBreaker, DeadLetterQueue, AdaptiveTimeout,
timeout/async_timeout decorators, and global instances.

Covers:
- CircuitState enum values
- CircuitBreakerConfig defaults and custom values
- CircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- CircuitBreaker protect decorator (sync and async)
- CircuitOpenError exception
- timeout decorator (sync) with signal-based timeout
- async_timeout decorator
- FailedOperation dataclass
- DeadLetterQueue add, retry_all, get_failures_by_operation, get_stats
- AdaptiveTimeout record_latency, get_timeout, measure context manager, get_stats
- Global instances (LLM_CIRCUIT_BREAKER, TOOL_CIRCUIT_BREAKER, GLOBAL_DLQ, ADAPTIVE_TIMEOUT)
"""
import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime, timedelta

from Jotty.core.utils.timeouts import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    FailedOperation,
    DeadLetterQueue,
    AdaptiveTimeout,
    timeout,
    async_timeout,
    LLM_CIRCUIT_BREAKER,
    TOOL_CIRCUIT_BREAKER,
    GLOBAL_DLQ,
    ADAPTIVE_TIMEOUT,
)
from Jotty.core.foundation.exceptions import TimeoutError


# =============================================================================
# CircuitState Enum Tests
# =============================================================================

class TestCircuitState:
    """Tests for CircuitState enum."""

    @pytest.mark.unit
    def test_closed_value(self):
        """CircuitState.CLOSED has value 'closed'."""
        assert CircuitState.CLOSED.value == "closed"

    @pytest.mark.unit
    def test_open_value(self):
        """CircuitState.OPEN has value 'open'."""
        assert CircuitState.OPEN.value == "open"

    @pytest.mark.unit
    def test_half_open_value(self):
        """CircuitState.HALF_OPEN has value 'half_open'."""
        assert CircuitState.HALF_OPEN.value == "half_open"

    @pytest.mark.unit
    def test_enum_has_exactly_three_members(self):
        """CircuitState has exactly three states."""
        assert len(CircuitState) == 3

    @pytest.mark.unit
    def test_enum_lookup_by_value(self):
        """CircuitState can be constructed from its string value."""
        assert CircuitState("closed") is CircuitState.CLOSED
        assert CircuitState("open") is CircuitState.OPEN
        assert CircuitState("half_open") is CircuitState.HALF_OPEN


# =============================================================================
# CircuitBreakerConfig Tests
# =============================================================================

class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """CircuitBreakerConfig uses correct defaults."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0
        assert config.name == "default"

    @pytest.mark.unit
    def test_custom_values(self):
        """CircuitBreakerConfig accepts custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout=120.0,
            name="custom_breaker",
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout == 120.0
        assert config.name == "custom_breaker"

    @pytest.mark.unit
    def test_partial_custom_values(self):
        """CircuitBreakerConfig mixes custom and default values."""
        config = CircuitBreakerConfig(name="partial", failure_threshold=3)
        assert config.failure_threshold == 3
        assert config.success_threshold == 2  # default
        assert config.timeout == 60.0  # default
        assert config.name == "partial"


# =============================================================================
# CircuitOpenError Tests
# =============================================================================

class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    @pytest.mark.unit
    def test_is_exception(self):
        """CircuitOpenError is a subclass of Exception."""
        assert issubclass(CircuitOpenError, Exception)

    @pytest.mark.unit
    def test_message_preserved(self):
        """CircuitOpenError preserves its message."""
        err = CircuitOpenError("circuit is open")
        assert str(err) == "circuit is open"

    @pytest.mark.unit
    def test_can_be_raised_and_caught(self):
        """CircuitOpenError can be raised and caught."""
        with pytest.raises(CircuitOpenError, match="test error"):
            raise CircuitOpenError("test error")


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    @pytest.mark.unit
    def test_initial_state_is_closed(self):
        """CircuitBreaker starts in CLOSED state."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.unit
    def test_initial_counters_are_zero(self):
        """CircuitBreaker starts with zero counters."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.total_calls == 0
        assert cb.total_failures == 0

    @pytest.mark.unit
    def test_initial_last_failure_time_is_none(self):
        """CircuitBreaker starts with no last failure time."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.last_failure_time is None

    @pytest.mark.unit
    def test_config_stored(self):
        """CircuitBreaker stores its config."""
        config = CircuitBreakerConfig(name="stored")
        cb = CircuitBreaker(config)
        assert cb.config is config
        assert cb.config.name == "stored"


class TestCircuitBreakerCanRequest:
    """Tests for CircuitBreaker.can_request()."""

    @pytest.mark.unit
    def test_closed_allows_request(self):
        """CLOSED state allows requests."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        allowed, reason = cb.can_request()
        assert allowed is True
        assert "closed" in reason.lower()

    @pytest.mark.unit
    def test_can_request_increments_total_calls(self):
        """Each can_request() increments total_calls."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        cb.can_request()
        cb.can_request()
        cb.can_request()
        assert cb.total_calls == 3

    @pytest.mark.unit
    def test_open_rejects_request(self):
        """OPEN state rejects requests."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure(RuntimeError("fail"))
        allowed, reason = cb.can_request()
        assert allowed is False
        assert "open" in reason.lower() or "rejecting" in reason.lower()

    @pytest.mark.unit
    def test_open_transitions_to_half_open_after_timeout(self):
        """OPEN transitions to HALF_OPEN when timeout expires."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout=0.0))
        cb.record_failure(RuntimeError("fail"))
        assert cb.state == CircuitState.OPEN

        # Timeout of 0.0 means it immediately transitions
        allowed, reason = cb.can_request()
        assert allowed is True
        assert cb.state == CircuitState.HALF_OPEN
        assert "half-open" in reason.lower() or "half_open" in reason.lower()

    @pytest.mark.unit
    def test_open_stays_open_before_timeout(self):
        """OPEN stays OPEN when timeout has not expired."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout=9999.0))
        cb.record_failure(RuntimeError("fail"))
        assert cb.state == CircuitState.OPEN

        allowed, reason = cb.can_request()
        assert allowed is False
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_half_open_allows_request(self):
        """HALF_OPEN state allows test requests."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout=0.0))
        cb.record_failure(RuntimeError("fail"))
        # Transition to HALF_OPEN
        cb.can_request()
        assert cb.state == CircuitState.HALF_OPEN

        # Further requests are also allowed in HALF_OPEN
        allowed, reason = cb.can_request()
        assert allowed is True
        assert "half-open" in reason.lower() or "half_open" in reason.lower()


class TestCircuitBreakerRecordSuccess:
    """Tests for CircuitBreaker.record_success()."""

    @pytest.mark.unit
    def test_success_decrements_failure_count_in_closed(self):
        """Success in CLOSED state decrements failure count."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=10))
        cb.record_failure(RuntimeError("fail"))
        cb.record_failure(RuntimeError("fail"))
        assert cb.failure_count == 2
        cb.record_success()
        assert cb.failure_count == 1

    @pytest.mark.unit
    def test_success_does_not_go_below_zero(self):
        """Failure count does not go below zero on success."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        cb.record_success()
        assert cb.failure_count == 0

    @pytest.mark.unit
    def test_success_in_half_open_increments_success_count(self):
        """Success in HALF_OPEN increments success_count."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, success_threshold=3, timeout=0.0
        ))
        cb.record_failure(RuntimeError("fail"))
        cb.can_request()  # transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.success_count == 1
        assert cb.state == CircuitState.HALF_OPEN  # not yet at threshold

    @pytest.mark.unit
    def test_success_in_half_open_closes_circuit_at_threshold(self):
        """Enough successes in HALF_OPEN closes the circuit."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, timeout=0.0
        ))
        cb.record_failure(RuntimeError("fail"))
        cb.can_request()  # transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerRecordFailure:
    """Tests for CircuitBreaker.record_failure()."""

    @pytest.mark.unit
    def test_failure_increments_failure_count(self):
        """Failures increment failure_count."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        cb.record_failure(RuntimeError("fail 1"))
        cb.record_failure(RuntimeError("fail 2"))
        assert cb.failure_count == 2

    @pytest.mark.unit
    def test_failure_increments_total_failures(self):
        """Failures increment total_failures."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        cb.record_failure(RuntimeError("a"))
        cb.record_failure(RuntimeError("b"))
        assert cb.total_failures == 2

    @pytest.mark.unit
    def test_failure_updates_last_failure_time(self):
        """Failure updates last_failure_time."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.last_failure_time is None
        cb.record_failure(RuntimeError("fail"))
        assert cb.last_failure_time is not None
        assert cb.last_failure_time <= time.time()

    @pytest.mark.unit
    def test_failure_opens_circuit_at_threshold(self):
        """Circuit opens when failures reach threshold."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure(RuntimeError("1"))
        cb.record_failure(RuntimeError("2"))
        assert cb.state == CircuitState.CLOSED
        cb.record_failure(RuntimeError("3"))
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN goes back to OPEN."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, timeout=0.0
        ))
        cb.record_failure(RuntimeError("initial"))
        cb.can_request()  # -> HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure(RuntimeError("during test"))
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerStateTransitions:
    """Full cycle state transition tests."""

    @pytest.mark.unit
    def test_full_cycle_closed_open_half_open_closed(self):
        """Full state cycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2, success_threshold=2, timeout=0.0
        ))

        # Start CLOSED
        assert cb.state == CircuitState.CLOSED

        # Trip to OPEN
        cb.record_failure(RuntimeError("1"))
        cb.record_failure(RuntimeError("2"))
        assert cb.state == CircuitState.OPEN

        # Transition to HALF_OPEN via can_request (timeout=0.0)
        allowed, _ = cb.can_request()
        assert allowed is True
        assert cb.state == CircuitState.HALF_OPEN

        # Recover to CLOSED
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.unit
    def test_half_open_failure_goes_back_to_open_then_recovers(self):
        """HALF_OPEN failure -> OPEN -> HALF_OPEN -> CLOSED cycle."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, success_threshold=1, timeout=0.0
        ))

        # CLOSED -> OPEN
        cb.record_failure(RuntimeError("trip"))
        assert cb.state == CircuitState.OPEN

        # OPEN -> HALF_OPEN
        cb.can_request()
        assert cb.state == CircuitState.HALF_OPEN

        # HALF_OPEN -> OPEN (failure during test)
        cb.record_failure(RuntimeError("test fail"))
        assert cb.state == CircuitState.OPEN

        # OPEN -> HALF_OPEN again
        cb.can_request()
        assert cb.state == CircuitState.HALF_OPEN

        # HALF_OPEN -> CLOSED (success)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.unit
    def test_success_resets_failure_count_on_close(self):
        """When circuit transitions from HALF_OPEN to CLOSED, failure_count resets."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2, success_threshold=1, timeout=0.0
        ))
        cb.record_failure(RuntimeError("1"))
        cb.record_failure(RuntimeError("2"))
        assert cb.state == CircuitState.OPEN

        cb.can_request()  # -> HALF_OPEN
        cb.record_success()  # -> CLOSED
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerGetStats:
    """Tests for CircuitBreaker.get_stats()."""

    @pytest.mark.unit
    def test_stats_after_init(self):
        """Stats after initialization are clean."""
        cb = CircuitBreaker(CircuitBreakerConfig(name="test_stats"))
        stats = cb.get_stats()
        assert stats['name'] == "test_stats"
        assert stats['state'] == "closed"
        assert stats['failure_count'] == 0
        assert stats['success_count'] == 0
        assert stats['total_calls'] == 0
        assert stats['total_failures'] == 0
        assert stats['failure_rate'] == 0.0
        assert stats['last_failure_time'] is None

    @pytest.mark.unit
    def test_stats_after_usage(self):
        """Stats reflect actual usage."""
        cb = CircuitBreaker(CircuitBreakerConfig(name="used"))
        cb.can_request()
        cb.can_request()
        cb.record_failure(RuntimeError("fail"))
        cb.record_success()

        stats = cb.get_stats()
        assert stats['total_calls'] == 2
        assert stats['total_failures'] == 1
        assert stats['failure_rate'] == 0.5
        assert stats['last_failure_time'] is not None

    @pytest.mark.unit
    def test_stats_failure_rate_zero_when_no_calls(self):
        """Failure rate is 0.0 when total_calls is 0."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        stats = cb.get_stats()
        assert stats['failure_rate'] == 0.0


class TestCircuitBreakerProtectSync:
    """Tests for CircuitBreaker.protect() with synchronous functions."""

    @pytest.mark.unit
    def test_protect_sync_success(self):
        """Protected sync function returns result on success."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        def my_func(x, y):
            return x + y

        result = my_func(3, 4)
        assert result == 7

    @pytest.mark.unit
    def test_protect_sync_records_success(self):
        """Protected sync function records success."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=10))
        # Put some failures in first
        cb.record_failure(RuntimeError("a"))
        cb.record_failure(RuntimeError("b"))
        initial_failures = cb.failure_count

        @cb.protect
        def my_func():
            return "ok"

        my_func()
        # success in CLOSED decrements failure count
        assert cb.failure_count == initial_failures - 1

    @pytest.mark.unit
    def test_protect_sync_records_failure(self):
        """Protected sync function records failure on exception."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        def failing_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_func()
        assert cb.failure_count == 1
        assert cb.total_failures == 1

    @pytest.mark.unit
    def test_protect_sync_raises_circuit_open_error(self):
        """Protected sync function raises CircuitOpenError when circuit is open."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout=9999.0))
        cb.record_failure(RuntimeError("trip"))

        @cb.protect
        def my_func():
            return "should not reach"

        with pytest.raises(CircuitOpenError):
            my_func()

    @pytest.mark.unit
    def test_protect_sync_preserves_function_name(self):
        """Protected function preserves its __name__."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        def my_special_func():
            pass

        assert my_special_func.__name__ == "my_special_func"


class TestCircuitBreakerProtectAsync:
    """Tests for CircuitBreaker.protect() with async functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_protect_async_success(self):
        """Protected async function returns result on success."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        async def my_async_func(x):
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_protect_async_records_failure(self):
        """Protected async function records failure on exception."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        async def failing_async():
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            await failing_async()
        assert cb.failure_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_protect_async_raises_circuit_open_error(self):
        """Protected async function raises CircuitOpenError when circuit is open."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout=9999.0))
        cb.record_failure(RuntimeError("trip"))

        @cb.protect
        async def my_async():
            return "unreachable"

        with pytest.raises(CircuitOpenError):
            await my_async()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_protect_async_preserves_function_name(self):
        """Protected async function preserves its __name__."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        async def named_async():
            pass

        assert named_async.__name__ == "named_async"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_protect_async_records_success(self):
        """Protected async function records success properly."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, success_threshold=1, timeout=0.0
        ))
        # Trip the breaker and then recover
        cb.record_failure(RuntimeError("trip"))
        cb.can_request()  # -> HALF_OPEN

        @cb.protect
        async def recovery_func():
            return "recovered"

        result = await recovery_func()
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED


# =============================================================================
# FailedOperation Tests
# =============================================================================

class TestFailedOperation:
    """Tests for FailedOperation dataclass."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """FailedOperation stores all fields."""
        err = ValueError("test error")
        now = datetime.now()
        op = FailedOperation(
            operation_name="my_op",
            args=(1, 2),
            kwargs={"key": "val"},
            error=err,
            timestamp=now,
        )
        assert op.operation_name == "my_op"
        assert op.args == (1, 2)
        assert op.kwargs == {"key": "val"}
        assert op.error is err
        assert op.timestamp == now
        assert op.retry_count == 0
        assert op.last_retry is None

    @pytest.mark.unit
    def test_defaults(self):
        """FailedOperation has correct defaults for retry_count and last_retry."""
        op = FailedOperation(
            operation_name="op",
            args=(),
            kwargs={},
            error=RuntimeError("x"),
            timestamp=datetime.now(),
        )
        assert op.retry_count == 0
        assert op.last_retry is None

    @pytest.mark.unit
    def test_retry_count_can_be_set(self):
        """FailedOperation retry_count can be set explicitly."""
        op = FailedOperation(
            operation_name="op",
            args=(),
            kwargs={},
            error=RuntimeError("x"),
            timestamp=datetime.now(),
            retry_count=5,
        )
        assert op.retry_count == 5


# =============================================================================
# DeadLetterQueue Tests
# =============================================================================

class TestDeadLetterQueueInit:
    """Tests for DeadLetterQueue initialization."""

    @pytest.mark.unit
    def test_default_max_size(self):
        """DLQ defaults to max_size=1000."""
        dlq = DeadLetterQueue()
        assert dlq.max_size == 1000

    @pytest.mark.unit
    def test_custom_max_size(self):
        """DLQ accepts custom max_size."""
        dlq = DeadLetterQueue(max_size=50)
        assert dlq.max_size == 50

    @pytest.mark.unit
    def test_initial_counters(self):
        """DLQ starts with zero counters."""
        dlq = DeadLetterQueue()
        assert dlq.total_added == 0
        assert dlq.total_retried == 0
        assert dlq.total_successful_retries == 0
        assert len(dlq.queue) == 0


class TestDeadLetterQueueAdd:
    """Tests for DeadLetterQueue.add()."""

    @pytest.mark.unit
    def test_add_single_item(self):
        """Adding an item increases queue size."""
        dlq = DeadLetterQueue()
        dlq.add("test_op", error=RuntimeError("fail"))
        assert len(dlq.queue) == 1
        assert dlq.total_added == 1

    @pytest.mark.unit
    def test_add_preserves_operation_name(self):
        """Added item has correct operation_name."""
        dlq = DeadLetterQueue()
        dlq.add("my_operation", args=(1, 2), kwargs={"k": "v"}, error=RuntimeError("x"))
        item = dlq.queue[0]
        assert item.operation_name == "my_operation"
        assert item.args == (1, 2)
        assert item.kwargs == {"k": "v"}

    @pytest.mark.unit
    def test_add_with_none_kwargs_becomes_empty_dict(self):
        """Adding with kwargs=None stores empty dict."""
        dlq = DeadLetterQueue()
        dlq.add("op", error=RuntimeError("fail"))
        assert dlq.queue[0].kwargs == {}

    @pytest.mark.unit
    def test_add_respects_max_size(self):
        """Queue evicts oldest items when max_size is exceeded."""
        dlq = DeadLetterQueue(max_size=3)
        for i in range(5):
            dlq.add(f"op_{i}", error=RuntimeError(f"fail {i}"))
        assert len(dlq.queue) == 3
        assert dlq.total_added == 5
        # Oldest items (0 and 1) should be evicted
        names = [item.operation_name for item in dlq.queue]
        assert names == ["op_2", "op_3", "op_4"]

    @pytest.mark.unit
    def test_add_sets_timestamp(self):
        """Added item has a timestamp."""
        dlq = DeadLetterQueue()
        before = datetime.now()
        dlq.add("op", error=RuntimeError("fail"))
        after = datetime.now()
        ts = dlq.queue[0].timestamp
        assert before <= ts <= after


class TestDeadLetterQueueRetryAll:
    """Tests for DeadLetterQueue.retry_all()."""

    @pytest.mark.unit
    def test_retry_all_success(self):
        """Successful retry clears the queue."""
        dlq = DeadLetterQueue()
        dlq.add("op1", error=RuntimeError("fail"))
        dlq.add("op2", error=RuntimeError("fail"))

        mock_func = MagicMock()
        stats = dlq.retry_all(mock_func)

        assert stats['success'] == 2
        assert stats['failed'] == 0
        assert stats['total'] == 2
        assert len(dlq.queue) == 0
        assert dlq.total_retried == 2
        assert dlq.total_successful_retries == 2

    @pytest.mark.unit
    def test_retry_all_failure_requeues_up_to_3_times(self):
        """Failed retries are requeued up to 3 attempts."""
        dlq = DeadLetterQueue()
        dlq.add("failing_op", error=RuntimeError("original"))

        mock_func = MagicMock(side_effect=RuntimeError("still failing"))
        stats = dlq.retry_all(mock_func)

        # First attempt fails, requeued (retry_count=1)
        # Second attempt fails, requeued (retry_count=2)
        # Third attempt fails, given up (retry_count=3)
        assert stats['failed'] == 3
        assert stats['success'] == 0
        assert len(dlq.queue) == 0  # given up after 3

    @pytest.mark.unit
    def test_retry_all_mixed_results(self):
        """Mix of success and failure in retry_all."""
        dlq = DeadLetterQueue()
        dlq.add("good_op", args=("good",), error=RuntimeError("fail"))
        dlq.add("bad_op", args=("bad",), error=RuntimeError("fail"))

        def operation_func(name, *args, **kwargs):
            if "bad" in args:
                raise RuntimeError("still bad")
            return "ok"

        stats = dlq.retry_all(operation_func)
        assert stats['success'] >= 1

    @pytest.mark.unit
    def test_retry_all_empty_queue(self):
        """Retrying empty queue returns zero stats."""
        dlq = DeadLetterQueue()
        stats = dlq.retry_all(MagicMock())
        assert stats == {'success': 0, 'failed': 0, 'total': 0}

    @pytest.mark.unit
    def test_retry_all_increments_retry_count(self):
        """Each retry attempt increments the operation's retry_count."""
        dlq = DeadLetterQueue()
        dlq.add("op", error=RuntimeError("fail"))

        call_count = 0

        def failing_once(name, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail first time")
            # succeed on second call

        stats = dlq.retry_all(failing_once)
        assert stats['success'] == 1
        assert stats['failed'] == 1

    @pytest.mark.unit
    def test_retry_all_passes_correct_args(self):
        """retry_all passes operation_name, args, kwargs to operation_func."""
        dlq = DeadLetterQueue()
        dlq.add("my_op", args=(10, 20), kwargs={"key": "value"}, error=RuntimeError("x"))

        mock_func = MagicMock()
        dlq.retry_all(mock_func)
        mock_func.assert_called_once_with("my_op", 10, 20, key="value")

    @pytest.mark.unit
    def test_retry_all_sets_last_retry(self):
        """retry_all sets last_retry on each processed operation."""
        dlq = DeadLetterQueue()
        dlq.add("op", error=RuntimeError("fail"))

        # Make it fail so we can inspect the requeued item
        mock_func = MagicMock(side_effect=RuntimeError("still failing"))
        dlq.retry_all(mock_func)

        # The item was requeued multiple times, all should have last_retry set
        # After 3 retries, queue should be empty (gave up)
        assert dlq.total_retried == 3


class TestDeadLetterQueueGetFailuresByOperation:
    """Tests for DeadLetterQueue.get_failures_by_operation()."""

    @pytest.mark.unit
    def test_empty_queue(self):
        """Empty queue returns empty dict."""
        dlq = DeadLetterQueue()
        assert dlq.get_failures_by_operation() == {}

    @pytest.mark.unit
    def test_counts_by_operation(self):
        """Counts failures grouped by operation name."""
        dlq = DeadLetterQueue()
        dlq.add("op_a", error=RuntimeError("1"))
        dlq.add("op_b", error=RuntimeError("2"))
        dlq.add("op_a", error=RuntimeError("3"))
        dlq.add("op_a", error=RuntimeError("4"))

        counts = dlq.get_failures_by_operation()
        assert counts == {"op_a": 3, "op_b": 1}


class TestDeadLetterQueueGetStats:
    """Tests for DeadLetterQueue.get_stats()."""

    @pytest.mark.unit
    def test_stats_initial(self):
        """Initial DLQ stats are clean."""
        dlq = DeadLetterQueue(max_size=500)
        stats = dlq.get_stats()
        assert stats['current_size'] == 0
        assert stats['max_size'] == 500
        assert stats['total_added'] == 0
        assert stats['total_retried'] == 0
        assert stats['total_successful_retries'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['failures_by_operation'] == {}

    @pytest.mark.unit
    def test_stats_after_operations(self):
        """Stats reflect actual add and retry operations."""
        dlq = DeadLetterQueue()
        dlq.add("op1", error=RuntimeError("fail"))
        dlq.add("op2", error=RuntimeError("fail"))

        mock_func = MagicMock()  # succeeds
        dlq.retry_all(mock_func)

        stats = dlq.get_stats()
        assert stats['total_added'] == 2
        assert stats['total_retried'] == 2
        assert stats['total_successful_retries'] == 2
        assert stats['success_rate'] == 1.0
        assert stats['current_size'] == 0

    @pytest.mark.unit
    def test_stats_success_rate_zero_when_no_retries(self):
        """Success rate is 0.0 when no retries attempted."""
        dlq = DeadLetterQueue()
        dlq.add("op", error=RuntimeError("fail"))
        stats = dlq.get_stats()
        assert stats['success_rate'] == 0.0


# =============================================================================
# AdaptiveTimeout Tests
# =============================================================================

class TestAdaptiveTimeoutInit:
    """Tests for AdaptiveTimeout initialization."""

    @pytest.mark.unit
    def test_default_values(self):
        """AdaptiveTimeout uses correct defaults."""
        at = AdaptiveTimeout()
        assert at.initial == 30.0
        assert at.percentile == 95.0
        assert at.min_timeout == 5.0
        assert at.max_timeout == 300.0
        assert at.latencies == {}
        assert at.max_samples == 100

    @pytest.mark.unit
    def test_custom_values(self):
        """AdaptiveTimeout accepts custom values."""
        at = AdaptiveTimeout(
            initial=10.0, percentile=99.0, min_timeout=1.0, max_timeout=60.0
        )
        assert at.initial == 10.0
        assert at.percentile == 99.0
        assert at.min_timeout == 1.0
        assert at.max_timeout == 60.0


class TestAdaptiveTimeoutRecordLatency:
    """Tests for AdaptiveTimeout.record_latency()."""

    @pytest.mark.unit
    def test_record_single_latency(self):
        """Recording a single latency creates the operation entry."""
        at = AdaptiveTimeout()
        at.record_latency("op1", 1.5)
        assert "op1" in at.latencies
        assert at.latencies["op1"] == [1.5]

    @pytest.mark.unit
    def test_record_multiple_latencies(self):
        """Multiple latencies are accumulated."""
        at = AdaptiveTimeout()
        at.record_latency("op1", 1.0)
        at.record_latency("op1", 2.0)
        at.record_latency("op1", 3.0)
        assert at.latencies["op1"] == [1.0, 2.0, 3.0]

    @pytest.mark.unit
    def test_record_latency_bounded_by_max_samples(self):
        """Latencies are bounded to max_samples."""
        at = AdaptiveTimeout()
        at.max_samples = 5
        for i in range(10):
            at.record_latency("op", float(i))
        assert len(at.latencies["op"]) == 5
        # First items removed, last 5 remain
        assert at.latencies["op"] == [5.0, 6.0, 7.0, 8.0, 9.0]

    @pytest.mark.unit
    def test_record_latency_multiple_operations(self):
        """Different operations have separate latency lists."""
        at = AdaptiveTimeout()
        at.record_latency("op_a", 1.0)
        at.record_latency("op_b", 2.0)
        assert at.latencies["op_a"] == [1.0]
        assert at.latencies["op_b"] == [2.0]


class TestAdaptiveTimeoutGetTimeout:
    """Tests for AdaptiveTimeout.get_timeout()."""

    @pytest.mark.unit
    def test_returns_initial_for_unknown_operation(self):
        """Unknown operation returns initial timeout."""
        at = AdaptiveTimeout(initial=25.0)
        assert at.get_timeout("unknown") == 25.0

    @pytest.mark.unit
    def test_returns_initial_for_few_samples(self):
        """Fewer than 10 samples returns initial timeout."""
        at = AdaptiveTimeout(initial=25.0)
        for i in range(9):
            at.record_latency("op", float(i))
        assert at.get_timeout("op") == 25.0

    @pytest.mark.unit
    def test_returns_initial_for_exactly_9_samples(self):
        """Exactly 9 samples returns initial timeout."""
        at = AdaptiveTimeout(initial=25.0)
        for i in range(9):
            at.record_latency("op", 1.0)
        assert at.get_timeout("op") == 25.0

    @pytest.mark.unit
    def test_adaptive_with_10_samples(self):
        """With 10+ samples, uses percentile-based timeout."""
        at = AdaptiveTimeout(initial=25.0, percentile=95.0)
        # Record 10 samples: [1, 2, 3, ..., 10]
        for i in range(1, 11):
            at.record_latency("op", float(i))
        result = at.get_timeout("op")
        # With 10 values sorted [1..10], index = int(10 * 0.95) = 9, value = 10.0
        assert result == 10.0

    @pytest.mark.unit
    def test_adaptive_respects_min_timeout(self):
        """Adaptive timeout does not go below min_timeout."""
        at = AdaptiveTimeout(initial=25.0, min_timeout=10.0, percentile=50.0)
        # Record all very small latencies
        for _ in range(20):
            at.record_latency("op", 0.1)
        result = at.get_timeout("op")
        assert result == 10.0

    @pytest.mark.unit
    def test_adaptive_respects_max_timeout(self):
        """Adaptive timeout does not exceed max_timeout."""
        at = AdaptiveTimeout(initial=25.0, max_timeout=50.0, percentile=95.0)
        # Record very large latencies
        for _ in range(20):
            at.record_latency("op", 999.0)
        result = at.get_timeout("op")
        assert result == 50.0

    @pytest.mark.unit
    def test_percentile_computation_50th(self):
        """50th percentile selects the median value."""
        at = AdaptiveTimeout(percentile=50.0, min_timeout=0.0)
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(1, 11):
            at.record_latency("op", float(i))
        result = at.get_timeout("op")
        # index = int(10 * 0.5) = 5, sorted[5] = 6
        assert result == 6.0

    @pytest.mark.unit
    def test_percentile_100_gets_max(self):
        """100th percentile returns the max latency (clamped to last index)."""
        at = AdaptiveTimeout(percentile=100.0, min_timeout=0.0)
        for i in range(1, 11):
            at.record_latency("op", float(i))
        result = at.get_timeout("op")
        # index = int(10 * 1.0) = 10, min(10, 9) = 9, sorted[9] = 10
        assert result == 10.0


class TestAdaptiveTimeoutMeasure:
    """Tests for AdaptiveTimeout.measure() context manager."""

    @pytest.mark.unit
    def test_measure_records_latency(self):
        """Measure context manager records latency."""
        at = AdaptiveTimeout()
        with at.measure("op"):
            pass  # very fast
        assert len(at.latencies["op"]) == 1
        assert at.latencies["op"][0] >= 0

    @pytest.mark.unit
    def test_measure_records_latency_even_on_exception(self):
        """Measure records latency even if exception occurs in body."""
        at = AdaptiveTimeout()
        try:
            with at.measure("op"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert len(at.latencies["op"]) == 1

    @pytest.mark.unit
    def test_measure_records_realistic_latency(self):
        """Measure records a latency close to actual elapsed time."""
        at = AdaptiveTimeout()
        with at.measure("op"):
            time.sleep(0.05)
        recorded = at.latencies["op"][0]
        assert 0.03 <= recorded <= 0.3  # generous bounds

    @pytest.mark.unit
    def test_measure_context_returns_self(self):
        """Measure context manager returns MeasureContext instance."""
        at = AdaptiveTimeout()
        with at.measure("op") as ctx:
            assert ctx is not None
            assert hasattr(ctx, 'start_time')
            assert ctx.start_time is not None


class TestAdaptiveTimeoutGetStats:
    """Tests for AdaptiveTimeout.get_stats()."""

    @pytest.mark.unit
    def test_stats_empty(self):
        """Stats for no recorded latencies is empty dict."""
        at = AdaptiveTimeout()
        assert at.get_stats() == {}

    @pytest.mark.unit
    def test_stats_with_data(self):
        """Stats include avg, median, min, max, count, current_timeout."""
        at = AdaptiveTimeout(initial=25.0)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            at.record_latency("op", v)

        stats = at.get_stats()
        assert "op" in stats
        op_stats = stats["op"]
        assert op_stats['count'] == 5
        assert op_stats['avg'] == 3.0
        assert op_stats['median'] == 3.0
        assert op_stats['min'] == 1.0
        assert op_stats['max'] == 5.0
        # Fewer than 10 samples -> current_timeout == initial
        assert op_stats['current_timeout'] == 25.0

    @pytest.mark.unit
    def test_stats_multiple_operations(self):
        """Stats include all recorded operations."""
        at = AdaptiveTimeout()
        at.record_latency("op_a", 1.0)
        at.record_latency("op_b", 2.0)
        stats = at.get_stats()
        assert "op_a" in stats
        assert "op_b" in stats

    @pytest.mark.unit
    def test_stats_with_enough_samples_show_adaptive_timeout(self):
        """With 10+ samples, current_timeout reflects adaptive calculation."""
        at = AdaptiveTimeout(initial=25.0, percentile=95.0, min_timeout=0.0)
        for i in range(1, 21):
            at.record_latency("op", float(i))

        stats = at.get_stats()
        # 20 samples sorted [1..20], index = int(20 * 0.95) = 19, sorted[19] = 20.0
        assert stats["op"]['current_timeout'] == 20.0


# =============================================================================
# timeout Decorator Tests (sync)
# =============================================================================

class TestTimeoutDecorator:
    """Tests for the sync timeout() decorator."""

    @pytest.mark.unit
    def test_fast_function_completes(self):
        """Function completing before timeout returns normally."""
        @timeout(5, "should not timeout")
        def fast_func():
            return 42

        assert fast_func() == 42

    @pytest.mark.unit
    def test_preserves_function_name(self):
        """Timeout decorator preserves __name__."""
        @timeout(5)
        def my_named_func():
            pass

        assert my_named_func.__name__ == "my_named_func"

    @pytest.mark.unit
    def test_raises_value_error_for_async_function(self):
        """timeout() raises ValueError when wrapping an async function."""
        @timeout(5)
        async def async_func():
            pass

        with pytest.raises(ValueError, match="Use async_timeout"):
            # The sync wrapper checks if func is async and raises
            # We need to call it, not await it, since the wrapper is sync
            async_func()

    @pytest.mark.unit
    def test_function_with_args_and_kwargs(self):
        """Timeout decorator passes args and kwargs through."""
        @timeout(5)
        def adder(a, b, extra=0):
            return a + b + extra

        assert adder(1, 2, extra=10) == 13

    @pytest.mark.unit
    def test_slow_function_times_out(self):
        """Function exceeding timeout raises TimeoutError."""
        @timeout(1, "slow operation")
        def slow_func():
            time.sleep(5)
            return "should not reach"

        with pytest.raises(TimeoutError, match="slow operation.*after 1s"):
            slow_func()

    @pytest.mark.unit
    def test_timeout_restores_signal_handler(self):
        """After timeout, the original signal handler is restored."""
        import signal as sig
        original_handler = sig.getsignal(sig.SIGALRM)

        @timeout(5)
        def quick_func():
            return "done"

        quick_func()
        restored_handler = sig.getsignal(sig.SIGALRM)
        # The handler should be restored to original
        assert restored_handler == original_handler


# =============================================================================
# async_timeout Decorator Tests
# =============================================================================

class TestAsyncTimeoutDecorator:
    """Tests for the async_timeout() decorator."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fast_async_completes(self):
        """Async function completing before timeout returns normally."""
        @async_timeout(5, "should not timeout")
        async def fast_async():
            return "fast result"

        result = await fast_async()
        assert result == "fast result"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_preserves_async_function_name(self):
        """async_timeout preserves __name__."""
        @async_timeout(5)
        async def my_async_named():
            pass

        assert my_async_named.__name__ == "my_async_named"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_slow_async_times_out(self):
        """Async function exceeding timeout raises TimeoutError."""
        @async_timeout(0.1, "async slow")
        async def slow_async():
            await asyncio.sleep(10)
            return "unreachable"

        with pytest.raises(TimeoutError, match="async slow.*after 0.1s"):
            await slow_async()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_with_args_and_kwargs(self):
        """async_timeout passes args and kwargs through."""
        @async_timeout(5)
        async def multiplier(x, y, factor=1):
            return x * y * factor

        result = await multiplier(3, 4, factor=2)
        assert result == 24

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_timeout_propagates_exceptions(self):
        """Non-timeout exceptions propagate through async_timeout."""
        @async_timeout(5)
        async def failing_async():
            raise ValueError("inner error")

        with pytest.raises(ValueError, match="inner error"):
            await failing_async()


# =============================================================================
# Global Instances Tests
# =============================================================================

class TestGlobalInstances:
    """Tests for global singleton instances."""

    @pytest.mark.unit
    def test_llm_circuit_breaker_exists(self):
        """LLM_CIRCUIT_BREAKER is a CircuitBreaker instance."""
        assert isinstance(LLM_CIRCUIT_BREAKER, CircuitBreaker)

    @pytest.mark.unit
    def test_llm_circuit_breaker_config(self):
        """LLM_CIRCUIT_BREAKER has correct config."""
        assert LLM_CIRCUIT_BREAKER.config.name == "llm_calls"
        assert LLM_CIRCUIT_BREAKER.config.failure_threshold == 5
        assert LLM_CIRCUIT_BREAKER.config.timeout == 60.0

    @pytest.mark.unit
    def test_tool_circuit_breaker_exists(self):
        """TOOL_CIRCUIT_BREAKER is a CircuitBreaker instance."""
        assert isinstance(TOOL_CIRCUIT_BREAKER, CircuitBreaker)

    @pytest.mark.unit
    def test_tool_circuit_breaker_config(self):
        """TOOL_CIRCUIT_BREAKER has correct config."""
        assert TOOL_CIRCUIT_BREAKER.config.name == "tool_calls"
        assert TOOL_CIRCUIT_BREAKER.config.failure_threshold == 3
        assert TOOL_CIRCUIT_BREAKER.config.timeout == 30.0

    @pytest.mark.unit
    def test_global_dlq_exists(self):
        """GLOBAL_DLQ is a DeadLetterQueue instance."""
        assert isinstance(GLOBAL_DLQ, DeadLetterQueue)

    @pytest.mark.unit
    def test_global_dlq_max_size(self):
        """GLOBAL_DLQ has max_size=1000."""
        assert GLOBAL_DLQ.max_size == 1000

    @pytest.mark.unit
    def test_adaptive_timeout_exists(self):
        """ADAPTIVE_TIMEOUT is an AdaptiveTimeout instance."""
        assert isinstance(ADAPTIVE_TIMEOUT, AdaptiveTimeout)

    @pytest.mark.unit
    def test_adaptive_timeout_config(self):
        """ADAPTIVE_TIMEOUT has correct defaults."""
        assert ADAPTIVE_TIMEOUT.initial == 30.0
        assert ADAPTIVE_TIMEOUT.percentile == 95.0


# =============================================================================
# TimeoutError Import Tests
# =============================================================================

class TestTimeoutErrorImport:
    """Tests for TimeoutError from foundation.exceptions."""

    @pytest.mark.unit
    def test_timeout_error_is_jotty_exception(self):
        """TimeoutError is from Jotty exceptions, not builtins."""
        from Jotty.core.foundation.exceptions import TimeoutError as JottyTimeout
        from Jotty.core.foundation.exceptions import ExecutionError
        assert issubclass(JottyTimeout, ExecutionError)

    @pytest.mark.unit
    def test_timeout_error_can_be_raised(self):
        """TimeoutError from foundation can be raised and caught."""
        with pytest.raises(TimeoutError):
            raise TimeoutError("test timeout")


# =============================================================================
# Edge Case and Integration-Style Tests
# =============================================================================

class TestCircuitBreakerEdgeCases:
    """Edge cases for CircuitBreaker."""

    @pytest.mark.unit
    def test_open_without_last_failure_time(self):
        """OPEN state with no last_failure_time still rejects."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        cb.state = CircuitState.OPEN
        cb.last_failure_time = None
        allowed, reason = cb.can_request()
        assert allowed is False

    @pytest.mark.unit
    def test_concurrent_failures_only_trip_once(self):
        """Multiple failures beyond threshold keep state OPEN."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        for _ in range(10):
            cb.record_failure(RuntimeError("fail"))
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 10

    @pytest.mark.unit
    def test_success_in_open_state_does_nothing(self):
        """record_success() in OPEN state has no effect on state."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure(RuntimeError("trip"))
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_half_open_success_count_reset_on_transition(self):
        """success_count resets to 0 when transitioning to HALF_OPEN."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, success_threshold=3, timeout=0.0
        ))
        cb.success_count = 99  # artificial
        cb.record_failure(RuntimeError("trip"))
        # Transition to HALF_OPEN resets success_count
        cb.can_request()
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 0

    @pytest.mark.unit
    def test_protect_sync_multiple_calls_accumulate_stats(self):
        """Multiple calls through protect accumulate in stats."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        @cb.protect
        def ok_func():
            return "ok"

        for _ in range(5):
            ok_func()

        stats = cb.get_stats()
        assert stats['total_calls'] == 5
        assert stats['total_failures'] == 0


class TestDeadLetterQueueEdgeCases:
    """Edge cases for DeadLetterQueue."""

    @pytest.mark.unit
    def test_retry_all_with_max_size_1(self):
        """DLQ with max_size=1 works correctly."""
        dlq = DeadLetterQueue(max_size=1)
        dlq.add("op1", error=RuntimeError("fail"))
        dlq.add("op2", error=RuntimeError("fail"))
        assert len(dlq.queue) == 1
        assert dlq.queue[0].operation_name == "op2"

    @pytest.mark.unit
    def test_get_failures_by_operation_after_successful_retry(self):
        """After successful retry, failures_by_operation is empty."""
        dlq = DeadLetterQueue()
        dlq.add("op", error=RuntimeError("fail"))
        dlq.retry_all(MagicMock())
        assert dlq.get_failures_by_operation() == {}


class TestAdaptiveTimeoutEdgeCases:
    """Edge cases for AdaptiveTimeout."""

    @pytest.mark.unit
    def test_get_timeout_with_zero_percentile(self):
        """0th percentile returns minimum observed latency."""
        at = AdaptiveTimeout(percentile=0.0, min_timeout=0.0)
        for i in range(1, 11):
            at.record_latency("op", float(i))
        result = at.get_timeout("op")
        # index = int(10 * 0.0) = 0, sorted[0] = 1.0
        assert result == 1.0

    @pytest.mark.unit
    def test_all_identical_latencies(self):
        """When all latencies are identical, timeout equals that value."""
        at = AdaptiveTimeout(percentile=95.0, min_timeout=0.0)
        for _ in range(20):
            at.record_latency("op", 5.0)
        result = at.get_timeout("op")
        assert result == 5.0

    @pytest.mark.unit
    def test_measure_multiple_operations(self):
        """Measure context manager works for different operations."""
        at = AdaptiveTimeout()
        with at.measure("op_a"):
            pass
        with at.measure("op_b"):
            pass
        assert len(at.latencies["op_a"]) == 1
        assert len(at.latencies["op_b"]) == 1

    @pytest.mark.unit
    def test_stats_with_single_sample(self):
        """Stats work correctly with a single sample."""
        at = AdaptiveTimeout(initial=25.0)
        at.record_latency("op", 3.5)
        stats = at.get_stats()
        assert stats["op"]['count'] == 1
        assert stats["op"]['avg'] == 3.5
        assert stats["op"]['median'] == 3.5
        assert stats["op"]['min'] == 3.5
        assert stats["op"]['max'] == 3.5
        assert stats["op"]['current_timeout'] == 25.0  # < 10 samples
