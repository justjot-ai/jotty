"""
Tests for Jotty.core.utils.async_utils
=======================================

Comprehensive unit tests covering:
- safe_status(): callback invocation, None handling, exception suppression
- StatusReporter: construction, __call__, prefix chaining, logging
- run_sync(): coroutine execution from sync context
- sync_wrapper(): decorator for making async functions sync-callable
- ensure_async(): wrapping sync functions as async
- gather_with_limit(): concurrency-limited async gather
- AGENT_EVENT_TYPES: frozenset membership
- AgentEvent: dataclass construction, __post_init__ validation
- AgentEventBroadcaster: singleton, subscribe, emit, unsubscribe, emit_async, reset

All tests are fast (< 1s), offline, no real LLM calls.
"""

import asyncio
import logging
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from Jotty.core.infrastructure.utils.async_utils import (
    AGENT_EVENT_TYPES,
    AgentEvent,
    AgentEventBroadcaster,
    StatusCallback,
    StatusReporter,
    StreamingCallback,
    ensure_async,
    gather_with_limit,
    run_sync,
    safe_status,
    sync_wrapper,
)

# =============================================================================
# Tests for safe_status
# =============================================================================


@pytest.mark.unit
class TestSafeStatus:
    """Tests for the safe_status helper function."""

    def test_calls_callback_with_stage_and_detail(self):
        """safe_status should invoke callback(stage, detail) when callback is not None."""
        cb = Mock()
        safe_status(cb, "Planning", "step 1")
        cb.assert_called_once_with("Planning", "step 1")

    def test_none_callback_does_nothing(self):
        """safe_status should silently return when callback is None."""
        # Should not raise
        safe_status(None, "Planning", "step 1")

    def test_suppresses_callback_exception(self):
        """safe_status should catch and suppress exceptions from the callback."""
        cb = Mock(side_effect=RuntimeError("callback exploded"))
        # Should not raise
        safe_status(cb, "Error", "details")
        cb.assert_called_once()

    def test_default_detail_is_empty_string(self):
        """safe_status detail parameter should default to empty string."""
        cb = Mock()
        safe_status(cb, "Loading")
        cb.assert_called_once_with("Loading", "")

    def test_logs_debug_on_exception(self):
        """safe_status should log a debug message when callback raises."""
        cb = Mock(side_effect=ValueError("bad value"))
        with patch("Jotty.core.utils.async_utils._logger") as mock_logger:
            safe_status(cb, "X", "Y")
            mock_logger.debug.assert_called_once()

    def test_does_not_log_on_success(self):
        """safe_status should not produce debug log on successful callback."""
        cb = Mock()
        with patch("Jotty.core.utils.async_utils._logger") as mock_logger:
            safe_status(cb, "OK", "fine")
            mock_logger.debug.assert_not_called()

    def test_callback_receives_exact_strings(self):
        """safe_status should pass stage and detail without modification."""
        cb = Mock()
        safe_status(cb, "  spaces  ", "detail\nnewline")
        cb.assert_called_once_with("  spaces  ", "detail\nnewline")


# =============================================================================
# Tests for StatusReporter
# =============================================================================


@pytest.mark.unit
class TestStatusReporter:
    """Tests for the StatusReporter class."""

    def test_call_invokes_callback(self):
        """StatusReporter.__call__ should forward to the callback."""
        cb = Mock()
        reporter = StatusReporter(callback=cb)
        reporter("Planning", "step 1")
        cb.assert_called_once_with("Planning", "step 1")

    def test_call_with_no_callback(self):
        """StatusReporter.__call__ should not fail when callback is None."""
        reporter = StatusReporter(callback=None)
        # Should not raise
        reporter("Planning", "step 1")

    def test_call_with_prefix(self):
        """StatusReporter should prepend prefix to stage."""
        cb = Mock()
        reporter = StatusReporter(callback=cb, prefix="[agent1]")
        reporter("Planning", "detail")
        cb.assert_called_once_with("[agent1] Planning", "detail")

    def test_call_without_prefix_no_extra_space(self):
        """When prefix is empty, stage should not have leading/trailing spaces."""
        cb = Mock()
        reporter = StatusReporter(callback=cb, prefix="")
        reporter("Planning", "x")
        cb.assert_called_once_with("Planning", "x")

    def test_logs_info_when_logger_provided(self):
        """StatusReporter should call logger.info when a logger is provided."""
        mock_logger = Mock(spec=logging.Logger)
        reporter = StatusReporter(callback=None, logger=mock_logger, emoji="")
        reporter("Loading", "data")
        mock_logger.info.assert_called_once()
        logged_msg = mock_logger.info.call_args[0][0]
        assert "Loading" in logged_msg
        assert "data" in logged_msg

    def test_logs_emoji_prefix(self):
        """StatusReporter should include emoji in log message."""
        mock_logger = Mock(spec=logging.Logger)
        reporter = StatusReporter(callback=None, logger=mock_logger, emoji=">>")
        reporter("Step", "info")
        logged_msg = mock_logger.info.call_args[0][0]
        assert logged_msg.startswith(">>")

    def test_log_message_without_detail(self):
        """When detail is empty, log message should not contain a colon separator."""
        mock_logger = Mock(spec=logging.Logger)
        reporter = StatusReporter(callback=None, logger=mock_logger)
        reporter("Step", "")
        logged_msg = mock_logger.info.call_args[0][0]
        assert ":" not in logged_msg

    def test_no_logging_when_logger_is_none(self):
        """StatusReporter should not attempt logging when logger is None."""
        # If logger is None, no AttributeError should occur
        reporter = StatusReporter(callback=None, logger=None)
        reporter("Step", "detail")  # Should not raise

    def test_with_prefix_returns_new_reporter(self):
        """with_prefix should return a new StatusReporter instance."""
        cb = Mock()
        parent = StatusReporter(callback=cb, emoji="!")
        child = parent.with_prefix("[sub]")
        assert isinstance(child, StatusReporter)
        assert child is not parent

    def test_with_prefix_child_uses_parent_callback(self):
        """Child reporter from with_prefix should use the same callback."""
        cb = Mock()
        parent = StatusReporter(callback=cb)
        child = parent.with_prefix("[sub]")
        child("Running", "go")
        cb.assert_called_once_with("[sub] Running", "go")

    def test_with_prefix_preserves_emoji(self):
        """Child reporter should preserve the parent's emoji setting."""
        mock_logger = Mock(spec=logging.Logger)
        parent = StatusReporter(callback=None, logger=mock_logger, emoji="ZZ")
        child = parent.with_prefix("[sub]")
        child("Test", "detail")
        logged_msg = mock_logger.info.call_args[0][0]
        assert "ZZ" in logged_msg

    def test_with_prefix_preserves_logger(self):
        """Child reporter should inherit the parent's logger."""
        mock_logger = Mock(spec=logging.Logger)
        parent = StatusReporter(callback=None, logger=mock_logger)
        child = parent.with_prefix("[sub]")
        child("X", "Y")
        mock_logger.info.assert_called_once()

    def test_callback_exception_suppressed(self):
        """StatusReporter should suppress callback exceptions via safe_status."""
        cb = Mock(side_effect=TypeError("oops"))
        reporter = StatusReporter(callback=cb)
        # Should not raise
        reporter("Error", "bad")

    def test_slots_defined(self):
        """StatusReporter should use __slots__ for memory efficiency."""
        assert hasattr(StatusReporter, "__slots__")
        reporter = StatusReporter()
        with pytest.raises(AttributeError):
            reporter.arbitrary_attr = "nope"


# =============================================================================
# Tests for run_sync
# =============================================================================


@pytest.mark.unit
class TestRunSync:
    """Tests for the run_sync function."""

    def test_runs_simple_coroutine(self):
        """run_sync should execute a coroutine and return its result."""

        async def coro():
            return 42

        result = run_sync(coro())
        assert result == 42

    def test_runs_coroutine_with_await(self):
        """run_sync should handle coroutines that await other coroutines."""

        async def inner():
            return "hello"

        async def outer():
            return await inner()

        result = run_sync(outer())
        assert result == "hello"

    def test_propagates_exception(self):
        """run_sync should propagate exceptions from the coroutine."""

        async def failing():
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            run_sync(failing())

    def test_returns_none(self):
        """run_sync should handle coroutines that return None."""

        async def noop():
            pass

        result = run_sync(noop())
        assert result is None


# =============================================================================
# Tests for sync_wrapper
# =============================================================================


@pytest.mark.unit
class TestSyncWrapper:
    """Tests for the sync_wrapper decorator."""

    def test_wraps_async_function(self):
        """sync_wrapper should make an async function callable synchronously."""

        @sync_wrapper
        async def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_preserves_function_name(self):
        """sync_wrapper should preserve the wrapped function's __name__."""

        @sync_wrapper
        async def my_func():
            pass

        assert my_func.__name__ == "my_func"

    def test_passes_kwargs(self):
        """sync_wrapper should forward keyword arguments correctly."""

        @sync_wrapper
        async def greet(name="world"):
            return f"hello {name}"

        assert greet(name="test") == "hello test"

    def test_propagates_exception(self):
        """sync_wrapper should let exceptions propagate."""

        @sync_wrapper
        async def boom():
            raise RuntimeError("sync boom")

        with pytest.raises(RuntimeError, match="sync boom"):
            boom()


# =============================================================================
# Tests for ensure_async
# =============================================================================


@pytest.mark.unit
class TestEnsureAsync:
    """Tests for the ensure_async function."""

    @pytest.mark.asyncio
    async def test_wraps_sync_function(self):
        """ensure_async should make a sync function awaitable."""

        def add(a, b):
            return a + b

        async_add = ensure_async(add)
        result = await async_add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_returns_async_function_unchanged(self):
        """ensure_async should return an already-async function as-is."""

        async def original():
            return "async"

        wrapped = ensure_async(original)
        assert wrapped is original

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """ensure_async should preserve the wrapped function's __name__."""

        def my_sync_func():
            pass

        wrapped = ensure_async(my_sync_func)
        assert wrapped.__name__ == "my_sync_func"

    @pytest.mark.asyncio
    async def test_wrapped_sync_is_coroutine_function(self):
        """ensure_async result should be detected as a coroutine function."""

        def sync_fn():
            return 1

        wrapped = ensure_async(sync_fn)
        assert asyncio.iscoroutinefunction(wrapped)

    @pytest.mark.asyncio
    async def test_passes_kwargs(self):
        """ensure_async should forward keyword arguments."""

        def greet(name="world"):
            return f"hi {name}"

        wrapped = ensure_async(greet)
        result = await wrapped(name="async")
        assert result == "hi async"

    @pytest.mark.asyncio
    async def test_propagates_exception(self):
        """ensure_async should propagate exceptions from sync function."""

        def failing():
            raise ValueError("sync fail")

        wrapped = ensure_async(failing)
        with pytest.raises(ValueError, match="sync fail"):
            await wrapped()


# =============================================================================
# Tests for gather_with_limit
# =============================================================================


@pytest.mark.unit
class TestGatherWithLimit:
    """Tests for the gather_with_limit async function."""

    @pytest.mark.asyncio
    async def test_runs_all_coroutines(self):
        """gather_with_limit should run all coroutines and return results."""

        async def make(val):
            return val

        results = await gather_with_limit([make(1), make(2), make(3)])
        assert results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_input(self):
        """gather_with_limit should return [] when given no coroutines."""
        results = await gather_with_limit([])
        assert results == []

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        """gather_with_limit should not exceed the concurrency limit."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked(val):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.01)
            async with lock:
                current -= 1
            return val

        coros = [tracked(i) for i in range(10)]
        results = await gather_with_limit(coros, limit=3)
        assert len(results) == 10
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_default_limit_is_ten(self):
        """gather_with_limit default limit should allow up to 10 concurrent."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked(val):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.01)
            async with lock:
                current -= 1
            return val

        coros = [tracked(i) for i in range(20)]
        results = await gather_with_limit(coros)
        assert len(results) == 20
        assert max_concurrent <= 10

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """gather_with_limit should return results in the same order as input coros."""

        async def make(val):
            await asyncio.sleep(0.01 * (5 - val))  # reverse delay
            return val

        results = await gather_with_limit([make(i) for i in range(5)], limit=5)
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_limit_of_one_serializes(self):
        """gather_with_limit with limit=1 should run coroutines one at a time."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked(val):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.005)
            async with lock:
                current -= 1
            return val

        coros = [tracked(i) for i in range(5)]
        results = await gather_with_limit(coros, limit=1)
        assert len(results) == 5
        assert max_concurrent == 1


# =============================================================================
# Tests for AGENT_EVENT_TYPES
# =============================================================================


@pytest.mark.unit
class TestAgentEventTypes:
    """Tests for the AGENT_EVENT_TYPES frozenset."""

    def test_is_frozenset(self):
        """AGENT_EVENT_TYPES should be a frozenset."""
        assert isinstance(AGENT_EVENT_TYPES, frozenset)

    def test_contains_expected_types(self):
        """AGENT_EVENT_TYPES should contain all documented event types."""
        expected = {
            "tool_start",
            "tool_end",
            "step_start",
            "step_end",
            "status",
            "error",
            "streaming",
        }
        assert expected == AGENT_EVENT_TYPES

    def test_immutable(self):
        """AGENT_EVENT_TYPES should not be mutable."""
        with pytest.raises(AttributeError):
            AGENT_EVENT_TYPES.add("new_type")


# =============================================================================
# Tests for AgentEvent
# =============================================================================


@pytest.mark.unit
class TestAgentEvent:
    """Tests for the AgentEvent dataclass."""

    def test_creation_with_valid_type(self):
        """AgentEvent should be creatable with a valid event type."""
        event = AgentEvent(type="tool_start")
        assert event.type == "tool_start"

    def test_default_data_is_empty_dict(self):
        """AgentEvent data should default to an empty dict."""
        event = AgentEvent(type="status")
        assert event.data == {}

    def test_default_agent_id_is_none(self):
        """AgentEvent agent_id should default to None."""
        event = AgentEvent(type="status")
        assert event.agent_id is None

    def test_timestamp_auto_set(self):
        """AgentEvent timestamp should be auto-populated close to current time."""
        before = time.time()
        event = AgentEvent(type="status")
        after = time.time()
        assert before <= event.timestamp <= after

    def test_custom_data(self):
        """AgentEvent should store custom data dict."""
        event = AgentEvent(type="tool_start", data={"skill": "web-search"})
        assert event.data == {"skill": "web-search"}

    def test_custom_agent_id(self):
        """AgentEvent should store a custom agent_id."""
        event = AgentEvent(type="error", agent_id="agent-007")
        assert event.agent_id == "agent-007"

    def test_unknown_type_logs_debug(self):
        """AgentEvent with unknown type should trigger a debug log."""
        with patch("Jotty.core.utils.async_utils._logger") as mock_logger:
            AgentEvent(type="unknown_event_type")
            mock_logger.debug.assert_called_once()
            assert "unknown_event_type" in mock_logger.debug.call_args[0][1]

    def test_known_type_no_debug_log(self):
        """AgentEvent with a known type should not trigger a debug log."""
        with patch("Jotty.core.utils.async_utils._logger") as mock_logger:
            AgentEvent(type="tool_end")
            mock_logger.debug.assert_not_called()

    def test_all_valid_types_accepted(self):
        """All AGENT_EVENT_TYPES should be accepted without debug logging."""
        with patch("Jotty.core.utils.async_utils._logger") as mock_logger:
            for etype in AGENT_EVENT_TYPES:
                AgentEvent(type=etype)
            mock_logger.debug.assert_not_called()


# =============================================================================
# Tests for AgentEventBroadcaster
# =============================================================================


@pytest.mark.unit
class TestAgentEventBroadcaster:
    """Tests for the AgentEventBroadcaster singleton."""

    def setup_method(self):
        """Reset singleton before each test to ensure isolation."""
        AgentEventBroadcaster.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        AgentEventBroadcaster.reset_instance()

    def test_get_instance_returns_broadcaster(self):
        """get_instance should return an AgentEventBroadcaster."""
        bus = AgentEventBroadcaster.get_instance()
        assert isinstance(bus, AgentEventBroadcaster)

    def test_singleton_identity(self):
        """get_instance should always return the same object."""
        bus1 = AgentEventBroadcaster.get_instance()
        bus2 = AgentEventBroadcaster.get_instance()
        assert bus1 is bus2

    def test_reset_instance_clears_singleton(self):
        """reset_instance should cause next get_instance to return a new object."""
        bus1 = AgentEventBroadcaster.get_instance()
        AgentEventBroadcaster.reset_instance()
        bus2 = AgentEventBroadcaster.get_instance()
        assert bus1 is not bus2

    def test_subscribe_and_emit(self):
        """Subscribed callback should be called on emit."""
        bus = AgentEventBroadcaster.get_instance()
        handler = Mock()
        bus.subscribe("tool_start", handler)
        event = AgentEvent(type="tool_start", data={"skill": "calc"})
        bus.emit(event)
        handler.assert_called_once_with(event)

    def test_emit_only_notifies_matching_type(self):
        """emit should only invoke callbacks subscribed to the event's type."""
        bus = AgentEventBroadcaster.get_instance()
        tool_handler = Mock()
        status_handler = Mock()
        bus.subscribe("tool_start", tool_handler)
        bus.subscribe("status", status_handler)
        event = AgentEvent(type="tool_start")
        bus.emit(event)
        tool_handler.assert_called_once()
        status_handler.assert_not_called()

    def test_multiple_subscribers_same_type(self):
        """Multiple callbacks for the same event type should all be called."""
        bus = AgentEventBroadcaster.get_instance()
        h1 = Mock()
        h2 = Mock()
        bus.subscribe("error", h1)
        bus.subscribe("error", h2)
        event = AgentEvent(type="error")
        bus.emit(event)
        h1.assert_called_once_with(event)
        h2.assert_called_once_with(event)

    def test_unsubscribe_removes_callback(self):
        """After unsubscribe, callback should no longer be invoked."""
        bus = AgentEventBroadcaster.get_instance()
        handler = Mock()
        bus.subscribe("status", handler)
        bus.unsubscribe("status", handler)
        bus.emit(AgentEvent(type="status"))
        handler.assert_not_called()

    def test_unsubscribe_nonexistent_callback_no_error(self):
        """Unsubscribing a callback that was never subscribed should not raise."""
        bus = AgentEventBroadcaster.get_instance()
        handler = Mock()
        # Should not raise
        bus.unsubscribe("status", handler)

    def test_unsubscribe_nonexistent_type_no_error(self):
        """Unsubscribing from a type with no subscribers should not raise."""
        bus = AgentEventBroadcaster.get_instance()
        handler = Mock()
        bus.unsubscribe("nonexistent_type", handler)

    def test_emit_suppresses_listener_exception(self):
        """emit should catch and suppress exceptions from listeners."""
        bus = AgentEventBroadcaster.get_instance()
        bad_handler = Mock(side_effect=RuntimeError("handler crash"))
        good_handler = Mock()
        bus.subscribe("error", bad_handler)
        bus.subscribe("error", good_handler)
        event = AgentEvent(type="error")
        bus.emit(event)
        bad_handler.assert_called_once()
        good_handler.assert_called_once()

    def test_emit_logs_debug_on_listener_exception(self):
        """emit should log debug when a listener raises."""
        bus = AgentEventBroadcaster.get_instance()
        bad_handler = Mock(side_effect=ValueError("oops"))
        bus.subscribe("status", bad_handler)
        with patch("Jotty.core.utils.async_utils._logger") as mock_logger:
            bus.emit(AgentEvent(type="status"))
            mock_logger.debug.assert_called_once()

    def test_emit_no_subscribers_no_error(self):
        """emit with no subscribers for the event type should not raise."""
        bus = AgentEventBroadcaster.get_instance()
        bus.emit(AgentEvent(type="tool_end"))

    def test_emit_async_falls_back_to_emit_outside_loop(self):
        """emit_async should fall back to sync emit when no event loop is running."""
        bus = AgentEventBroadcaster.get_instance()
        handler = Mock()
        bus.subscribe("status", handler)
        event = AgentEvent(type="status", data={"msg": "hello"})
        bus.emit_async(event)
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_emit_async_uses_call_soon_threadsafe(self):
        """emit_async inside an event loop should use call_soon_threadsafe."""
        bus = AgentEventBroadcaster.get_instance()
        handler = Mock()
        bus.subscribe("step_start", handler)
        event = AgentEvent(type="step_start")

        # Patch asyncio.get_running_loop to return a mock loop
        mock_loop = Mock()
        with patch("Jotty.core.utils.async_utils.asyncio.get_running_loop", return_value=mock_loop):
            bus.emit_async(event)
            mock_loop.call_soon_threadsafe.assert_called_once_with(bus.emit, event)

    def test_singleton_thread_safety(self):
        """get_instance should be thread-safe and always return the same instance."""
        AgentEventBroadcaster.reset_instance()
        instances = []

        def get():
            instances.append(AgentEventBroadcaster.get_instance())

        threads = [threading.Thread(target=get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 20
        assert all(inst is instances[0] for inst in instances)


# =============================================================================
# Tests for type aliases
# =============================================================================


@pytest.mark.unit
class TestTypeAliases:
    """Tests for StatusCallback and StreamingCallback type aliases."""

    def test_status_callback_accepts_none(self):
        """StatusCallback typed as Optional should accept None values."""
        cb: StatusCallback = None
        assert cb is None

    def test_status_callback_accepts_callable(self):
        """StatusCallback should accept a callable."""
        cb: StatusCallback = lambda stage, detail: None
        assert callable(cb)

    def test_streaming_callback_accepts_none(self):
        """StreamingCallback typed as Optional should accept None values."""
        cb: StreamingCallback = None
        assert cb is None

    def test_streaming_callback_accepts_callable(self):
        """StreamingCallback should accept a callable taking AgentEvent."""
        cb: StreamingCallback = lambda event: None
        assert callable(cb)
