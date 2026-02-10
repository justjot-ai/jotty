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
from typing import TypeVar, Callable, Any, Coroutine, Optional

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

        report = StatusReporter(callback, logger, emoji="📍")
        report("Planning", "discovering skills")
        # → calls callback("Planning", "discovering skills")
        # → logs  "📍 Planning: discovering skills"

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
    Run an async coroutine synchronously.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
