"""
Execution Memory Backend Tests
==============================

Unit tests for JSONMemory and NoOpMemory backends.
All tests use tmp_path to avoid filesystem side effects.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from Jotty.core.modes.execution.memory import JSONMemory, NoOpMemory

# =============================================================================
# JSONMemory Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestJSONMemoryStoreRetrieve:
    """JSONMemory store then retrieve returns the stored entry."""

    async def test_store_then_retrieve_returns_entry(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(
            goal="summarize report", result="Report summary here", success=True, confidence=0.95
        )

        entries = await mem.retrieve(goal="summarize report")

        assert len(entries) == 1
        assert entries[0]["goal"] == "summarize report"
        assert entries[0]["result"] == "Report summary here"
        assert entries[0]["success"] is True
        assert entries[0]["confidence"] == 0.95


@pytest.mark.unit
@pytest.mark.asyncio
class TestJSONMemoryRetrieveEmpty:
    """JSONMemory retrieve on a fresh store returns an empty list."""

    async def test_retrieve_empty_returns_empty_list(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)

        entries = await mem.retrieve(goal="nonexistent goal")

        assert entries == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestJSONMemoryClear:
    """JSONMemory clear removes all stored entries."""

    async def test_clear_removes_all_entries(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(goal="task alpha", result="result alpha", success=True, confidence=0.8)
        await mem.store(goal="task beta", result="result beta", success=True, confidence=0.9)

        await mem.clear()

        entries_alpha = await mem.retrieve(goal="task alpha")
        entries_beta = await mem.retrieve(goal="task beta")
        assert entries_alpha == []
        assert entries_beta == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestJSONMemoryExpiry:
    """JSONMemory filters out expired entries on retrieval."""

    async def test_expired_entries_are_filtered_out(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)

        # Store an entry with a very short TTL
        await mem.store(
            goal="expiring task", result="old result", success=True, confidence=0.7, ttl_hours=1
        )

        # Patch datetime.now to simulate time passing beyond TTL
        future_time = datetime.now() + timedelta(hours=2)
        with patch("Jotty.core.execution.memory.json_memory.datetime") as mock_dt:
            mock_dt.now.return_value = future_time
            mock_dt.fromisoformat = datetime.fromisoformat

            entries = await mem.retrieve(goal="expiring task")

        assert entries == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestJSONMemoryCapacity:
    """JSONMemory caps entries at 10 per file, keeping the most recent."""

    async def test_capacity_capped_at_ten_entries(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)

        # Store 15 entries under the same goal pattern
        for i in range(15):
            await mem.store(
                goal="repeated goal",
                result=f"result {i}",
                success=True,
                confidence=0.5,
            )

        # Retrieve with limit=10 to see all stored entries
        entries = await mem.retrieve(goal="repeated goal", limit=10)

        # Only the last 10 entries should be kept (indices 5-14)
        assert len(entries) == 10
        # The oldest stored entries should have been dropped; newest retained
        results = [e["result"] for e in entries]
        assert "result 14" in results
        assert "result 5" in results
        assert "result 4" not in results


# =============================================================================
# NoOpMemory Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestNoOpMemoryStore:
    """NoOpMemory store does nothing and raises no errors."""

    async def test_store_does_nothing(self):
        mem = NoOpMemory()
        # Should complete without error and return None
        result = await mem.store(goal="test", result="data", success=True, confidence=1.0)
        assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestNoOpMemoryRetrieve:
    """NoOpMemory retrieve always returns an empty list."""

    async def test_retrieve_returns_empty_list(self):
        mem = NoOpMemory()
        entries = await mem.retrieve(goal="anything", limit=10)
        assert entries == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestNoOpMemoryClear:
    """NoOpMemory clear does nothing and raises no errors."""

    async def test_clear_does_nothing(self):
        mem = NoOpMemory()
        result = await mem.clear()
        assert result is None
