"""
Tests for _SessionLockManager — per-session lane queue serialization.

Covers:
    1. Same session_id returns the same lock object
    2. Different session_ids return different lock objects
    3. Concurrent requests for the same session are serialized
    4. Concurrent requests for different sessions run in parallel
    5. Expired locks are cleaned up after TTL
    6. Locked (in-use) sessions are NOT cleaned up even if expired
"""

import asyncio
import time

import pytest

from Jotty.core.intelligence.orchestration.core.swarm_manager import _SessionLockManager


@pytest.mark.unit
@pytest.mark.asyncio
class TestSessionLockManagerIdentity:
    """Lock identity: same session -> same lock, different session -> different lock."""

    async def test_same_session_returns_same_lock(self):
        mgr = _SessionLockManager()
        lock_a = await mgr.get_lock("session-1")
        lock_b = await mgr.get_lock("session-1")
        assert lock_a is lock_b

    async def test_different_sessions_return_different_locks(self):
        mgr = _SessionLockManager()
        lock_a = await mgr.get_lock("session-1")
        lock_b = await mgr.get_lock("session-2")
        assert lock_a is not lock_b

    async def test_lock_is_asyncio_lock(self):
        mgr = _SessionLockManager()
        lock = await mgr.get_lock("session-1")
        assert isinstance(lock, asyncio.Lock)


@pytest.mark.unit
@pytest.mark.asyncio
class TestSessionLockManagerSerialization:
    """Concurrent requests for the same session must be serialized."""

    async def test_same_session_serialized(self):
        """Two tasks sharing a session execute one after the other."""
        mgr = _SessionLockManager()
        order: list = []

        async def worker(name: str, delay: float):
            lock = await mgr.get_lock("session-1")
            async with lock:
                order.append(f"{name}-start")
                await asyncio.sleep(delay)
                order.append(f"{name}-end")

        t1 = asyncio.create_task(worker("A", 0.02))
        # Small yield to let task A grab the lock first
        await asyncio.sleep(0.001)
        t2 = asyncio.create_task(worker("B", 0.01))
        await asyncio.gather(t1, t2)

        # A must fully complete before B starts
        assert order == ["A-start", "A-end", "B-start", "B-end"]

    async def test_different_sessions_parallel(self):
        """Two tasks on different sessions overlap in time."""
        mgr = _SessionLockManager()
        timestamps: dict = {}

        async def worker(session_id: str, delay: float):
            lock = await mgr.get_lock(session_id)
            async with lock:
                timestamps[f"{session_id}-start"] = time.monotonic()
                await asyncio.sleep(delay)
                timestamps[f"{session_id}-end"] = time.monotonic()

        t1 = asyncio.create_task(worker("session-1", 0.03))
        t2 = asyncio.create_task(worker("session-2", 0.03))
        await asyncio.gather(t1, t2)

        # session-2 must have started before session-1 ended (overlap)
        assert timestamps["session-2-start"] < timestamps["session-1-end"], (
            "Different sessions should run in parallel, but session-2 started "
            "after session-1 ended"
        )

    async def test_three_requests_same_session_queued(self):
        """Three concurrent requests on one session execute in FIFO order."""
        mgr = _SessionLockManager()
        order: list = []

        async def worker(name: str):
            lock = await mgr.get_lock("shared")
            async with lock:
                order.append(name)
                await asyncio.sleep(0.01)

        t1 = asyncio.create_task(worker("1"))
        await asyncio.sleep(0.001)
        t2 = asyncio.create_task(worker("2"))
        await asyncio.sleep(0.001)
        t3 = asyncio.create_task(worker("3"))
        await asyncio.gather(t1, t2, t3)

        assert order == ["1", "2", "3"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestSessionLockManagerCleanup:
    """TTL-based cleanup of idle session locks."""

    async def test_expired_lock_cleaned_up(self):
        """An idle lock older than TTL is removed on next get_lock call."""
        mgr = _SessionLockManager()
        # Set TTL to 0 so everything expires immediately
        original_ttl = _SessionLockManager._TTL_SECONDS
        try:
            _SessionLockManager._TTL_SECONDS = 0

            lock_old = await mgr.get_lock("old-session")
            assert "old-session" in mgr._locks

            # Brief sleep so the timestamp is in the past relative to TTL=0
            await asyncio.sleep(0.01)

            # Getting a lock for a *different* session triggers cleanup
            await mgr.get_lock("new-session")

            # old-session should have been cleaned up
            assert "old-session" not in mgr._locks
            # new-session should exist
            assert "new-session" in mgr._locks
        finally:
            _SessionLockManager._TTL_SECONDS = original_ttl

    async def test_locked_session_not_cleaned_up(self):
        """A lock that is currently held must NOT be cleaned up, even if expired."""
        mgr = _SessionLockManager()
        original_ttl = _SessionLockManager._TTL_SECONDS
        try:
            _SessionLockManager._TTL_SECONDS = 0

            lock = await mgr.get_lock("busy-session")

            # Hold the lock while triggering cleanup
            async with lock:
                await asyncio.sleep(0.01)
                # This call triggers cleanup, but busy-session is locked
                await mgr.get_lock("other-session")

                # busy-session must still exist because it is locked
                assert "busy-session" in mgr._locks

        finally:
            _SessionLockManager._TTL_SECONDS = original_ttl

    async def test_cleanup_only_removes_expired(self):
        """Cleanup should leave non-expired locks untouched."""
        mgr = _SessionLockManager()
        original_ttl = _SessionLockManager._TTL_SECONDS
        try:
            # Use a large TTL so nothing expires
            _SessionLockManager._TTL_SECONDS = 9999

            await mgr.get_lock("session-a")
            await mgr.get_lock("session-b")
            await asyncio.sleep(0.01)
            await mgr.get_lock("session-c")

            # All three should still be present
            assert "session-a" in mgr._locks
            assert "session-b" in mgr._locks
            assert "session-c" in mgr._locks
        finally:
            _SessionLockManager._TTL_SECONDS = original_ttl

    async def test_last_used_updated_on_access(self):
        """Accessing a session updates its last-used timestamp."""
        mgr = _SessionLockManager()

        await mgr.get_lock("session-1")
        ts1 = mgr._last_used["session-1"]

        await asyncio.sleep(0.01)

        await mgr.get_lock("session-1")
        ts2 = mgr._last_used["session-1"]

        assert ts2 > ts1, "last_used should be updated on every access"
