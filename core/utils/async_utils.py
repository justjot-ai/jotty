"""
Async Utilities
===============

Helpers for async/sync interop and safe callback invocation.
Eliminates duplicate async wrapper patterns across skills
and duplicate status_callback try/except patterns across orchestration.

Usage:
    from Jotty.core.utils.async_utils import run_sync, safe_status, StatusReporter

    # Convert async function to sync
    result = run_sync(async_function(args))

    # Safe status callback (replaces 50+ inline try/except blocks)
    safe_status(callback, "Planning", "step 1 of 3")

    # StatusReporter: reusable reporter with optional logging
    report = StatusReporter(callback, logger, prefix="[agent1]")
    report("Planning", "step 1 of 3")
"""

import asyncio
import functools
import logging
import threading
import time as _time
from dataclasses import dataclass, field
from typing import TypeVar, Callable, Any, Coroutine, Dict, List, Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status callback helpers — centralises the try/except-guarded callback
# pattern that was duplicated 50+ times across orchestration & agent code.
# ---------------------------------------------------------------------------

# Type alias for status callbacks used throughout Jotty
StatusCallback = Optional[Callable[[str, str], Any]]


def safe_status(
    callback: StatusCallback,
    stage: str,
    detail: str = "",
) -> None:
    """
    Invoke a status callback safely, suppressing any exception.

    Replaces the ubiquitous pattern::

        if status_callback:
            try:
                status_callback(stage, detail)
            except Exception:
                pass

    Args:
        callback: Optional status callback ``(stage, detail) -> None``.
        stage: Short label for the current phase (e.g. ``"Planning"``).
        detail: Human-readable progress detail.
    """
    if callback is None:
        return
    try:
        callback(stage, detail)
    except Exception as exc:
        _logger.debug("Status callback error: %s", exc)


class StatusReporter:
    """
    Reusable status reporter that wraps a callback with optional logging.

    Replaces the per-method ``_status()`` closures that were re-defined
    in every orchestration method::

        def _status(stage, detail=""):
            if status_callback:
                try: status_callback(stage, detail)
                except Exception: pass
            logger.info(f"stage: {detail}")

    Usage::

        report = StatusReporter(callback, logger, emoji="")
        report("Planning", "discovering skills")
        # → calls callback("Planning", "discovering skills")
        # → logs " Planning: discovering skills"

        # With prefix (for multi-agent):
        sub = report.with_prefix("[agent1]")
        sub("Executing", "step 2")
        # → calls callback("[agent1] Executing", "step 2")
    """

    __slots__ = ("_callback", "_logger", "_emoji", "_prefix")

    def __init__(
        self,
        callback: StatusCallback = None,
        logger: Optional[logging.Logger] = None,
        emoji: str = "",
        prefix: str = "",
    ) -> None:
        self._callback = callback
        self._logger = logger
        self._emoji = emoji
        self._prefix = prefix

    def __call__(self, stage: str, detail: str = "") -> None:
        full_stage = f"{self._prefix} {stage}".strip() if self._prefix else stage
        safe_status(self._callback, full_stage, detail)
        if self._logger:
            msg = f"{self._emoji} {full_stage}".strip()
            if detail:
                msg = f"{msg}: {detail}"
            self._logger.info(msg)

    def with_prefix(self, prefix: str) -> "StatusReporter":
        """Return a child reporter that prepends *prefix* to every stage."""
        return StatusReporter(
            callback=self._callback,
            logger=self._logger,
            emoji=self._emoji,
            prefix=prefix,
        )

T = TypeVar('T')


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine synchronously with event-loop detection.

    Detects whether a loop is already running (e.g. Web gateway, Jupyter)
    and adapts:
    - Running loop: schedules via ``run_coroutine_threadsafe`` in a worker thread
    - No loop: creates a new loop with ``asyncio.run``

    This prevents ``RuntimeError: This event loop is already running`` which
    occurs when calling ``run_sync`` from inside an async context.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an async context — schedule on the running loop
        # from a background thread to avoid deadlock
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    # No running loop — safe to create one
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def sync_wrapper(async_func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Decorator to create sync wrapper for async function.

    Usage:
        @sync_wrapper
        async def my_async_function(x, y):
            await some_async_operation()
            return result

        # Now callable as sync:
        result = my_async_function(1, 2)
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs) -> T:
        return run_sync(async_func(*args, **kwargs))
    return wrapper


def ensure_async(func: Callable) -> Callable[..., Coroutine]:
    """
    Ensure a function is async (wrap sync functions).

    Args:
        func: Function to wrap

    Returns:
        Async version of the function
    """
    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


async def gather_with_limit(coros: list, limit: int = 10) -> list:
    """
    Run coroutines with concurrency limit.

    Args:
        coros: List of coroutines
        limit: Max concurrent coroutines

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited(c) for c in coros])


# ---------------------------------------------------------------------------
# Agent Event System — lightweight event envelope + singleton broadcaster
# for tool → agent → UI communication.
# ---------------------------------------------------------------------------

AGENT_EVENT_TYPES = frozenset({
    "tool_start", "tool_end",
    "step_start", "step_end",
    "status", "error", "streaming",
})


@dataclass
class AgentEvent:
    """Lightweight event envelope used by all agent broadcasting."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_time.time)
    agent_id: Optional[str] = None

    def __post_init__(self):
        if self.type not in AGENT_EVENT_TYPES:
            _logger.debug("AgentEvent created with unknown type: %s", self.type)


class AgentEventBroadcaster:
    """
    Singleton event bus for tool / agent / UI communication.

    Usage::

        bus = AgentEventBroadcaster.get_instance()
        bus.subscribe("tool_start", my_handler)
        bus.emit(AgentEvent(type="tool_start", data={"skill": "shell-exec"}))
    """

    _instance: Optional["AgentEventBroadcaster"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    @classmethod
    def get_instance(cls) -> "AgentEventBroadcaster":
        """Get or create the singleton broadcaster."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AgentEventBroadcaster()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register a listener for an event type."""
        self._listeners.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a listener for an event type."""
        listeners = self._listeners.get(event_type, [])
        try:
            listeners.remove(callback)
        except ValueError:
            pass

    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all subscribed listeners (suppresses exceptions)."""
        for cb in self._listeners.get(event.type, []):
            try:
                cb(event)
            except Exception as exc:
                _logger.debug("Event listener error for %s: %s", event.type, exc)

    def emit_async(self, event: AgentEvent) -> None:
        """Thread-safe emit for tools running in executors."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self.emit, event)
        except RuntimeError:
            self.emit(event)


# Streaming callback type alias — carries richer AgentEvent payloads
StreamingCallback = Optional[Callable[[AgentEvent], Any]]
