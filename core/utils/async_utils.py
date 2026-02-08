"""
Async Utilities
===============

Helpers for async/sync interop.
Eliminates duplicate async wrapper patterns across skills.

Usage:
    from Jotty.core.utils.async_utils import run_sync

    # Convert async function to sync
    result = run_sync(async_function(args))

    # Or use decorator
    @sync_wrapper
    async def my_async_func():
        ...

    my_async_func()  # Now callable synchronously
"""

import asyncio
import functools
from typing import TypeVar, Callable, Any, Coroutine

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
