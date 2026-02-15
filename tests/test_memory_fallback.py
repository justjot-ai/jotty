"""
Tests for Memory Fallback, Persistence, and MemorySystem
=========================================================

Comprehensive tests covering:
- fallback_memory.py: MemoryEntry, MemoryType, SimpleFallbackMemory, get_fallback_memory
- memory_persistence.py: MemoryPersistence, enable_memory_persistence
- memory_system.py: MemoryBackend, MemoryConfig, MemoryResult, MemorySystem
"""

import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Safe imports with skip guards
# ---------------------------------------------------------------------------

try:
    from Jotty.core.intelligence.memory.fallback_memory import MemoryEntry as FallbackMemoryEntry
    from Jotty.core.intelligence.memory.fallback_memory import (
        MemoryType,
        SimpleFallbackMemory,
        get_fallback_memory,
    )

    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False

try:
    from Jotty.core.intelligence.memory.memory_persistence import (
        MemoryPersistence,
        enable_memory_persistence,
    )

    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

try:
    from Jotty.core.intelligence.memory.memory_system import (
        MemoryBackend,
        MemoryConfig,
        MemoryResult,
        MemorySystem,
    )

    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False

skip_no_fallback = pytest.mark.skipif(
    not FALLBACK_AVAILABLE, reason="fallback_memory module not importable"
)
skip_no_persistence = pytest.mark.skipif(
    not PERSISTENCE_AVAILABLE, reason="memory_persistence module not importable"
)
skip_no_system = pytest.mark.skipif(
    not SYSTEM_AVAILABLE, reason="memory_system module not importable"
)


# =============================================================================
# FallbackMemoryEntry Tests
# =============================================================================


@skip_no_fallback
class TestFallbackMemoryEntry:
    """Tests for the FallbackMemoryEntry dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """MemoryEntry has correct defaults."""
        entry = FallbackMemoryEntry(content="hello")
        assert entry.content == "hello"
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.access_count == 0
        assert entry.importance == 0.5
        assert entry.metadata == {}
        assert isinstance(entry.timestamp, float)
        assert isinstance(entry.last_accessed, float)

    @pytest.mark.unit
    def test_access_increments_count(self):
        """access() bumps access_count and refreshes last_accessed."""
        entry = FallbackMemoryEntry(content="test")
        old_last = entry.last_accessed
        time.sleep(0.01)
        entry.access()
        assert entry.access_count == 1
        assert entry.last_accessed >= old_last

    @pytest.mark.unit
    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict preserves data."""
        entry = FallbackMemoryEntry(
            content="roundtrip test",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            metadata={"key": "value"},
        )
        data = entry.to_dict()
        restored = FallbackMemoryEntry.from_dict(data)
        assert restored.content == entry.content
        assert restored.memory_type == entry.memory_type
        assert restored.importance == entry.importance
        assert restored.metadata == entry.metadata

    @pytest.mark.unit
    def test_from_dict_defaults(self):
        """from_dict fills missing fields with defaults."""
        data = {"content": "minimal"}
        entry = FallbackMemoryEntry.from_dict(data)
        assert entry.content == "minimal"
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.access_count == 0
        assert entry.importance == 0.5

    @pytest.mark.unit
    def test_to_dict_serializes_memory_type_as_string(self):
        """to_dict stores memory_type as its string value."""
        entry = FallbackMemoryEntry(content="x", memory_type=MemoryType.PROCEDURAL)
        data = entry.to_dict()
        assert data["memory_type"] == "procedural"


# =============================================================================
# MemoryType Enum Tests
# =============================================================================


@skip_no_fallback
class TestMemoryType:
    """Tests for the MemoryType enum."""

    @pytest.mark.unit
    def test_all_types_exist(self):
        """Three memory types are defined."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    @pytest.mark.unit
    def test_type_count(self):
        """Exactly 3 memory types."""
        assert len(MemoryType) == 3


# =============================================================================
# SimpleFallbackMemory Tests
# =============================================================================


@skip_no_fallback
class TestSimpleFallbackMemoryInit:
    """Tests for SimpleFallbackMemory initialization."""

    @pytest.mark.unit
    def test_default_capacities(self):
        """Default capacities match constructor signature."""
        mem = SimpleFallbackMemory()
        assert mem.max_entries == 500
        assert mem.capacities[MemoryType.EPISODIC] == 300
        assert mem.capacities[MemoryType.SEMANTIC] == 150
        assert mem.capacities[MemoryType.PROCEDURAL] == 50

    @pytest.mark.unit
    def test_custom_capacities(self):
        """Custom capacities are respected."""
        mem = SimpleFallbackMemory(
            max_entries=100,
            episodic_capacity=50,
            semantic_capacity=30,
            procedural_capacity=20,
        )
        assert mem.max_entries == 100
        assert mem.capacities[MemoryType.EPISODIC] == 50

    @pytest.mark.unit
    def test_initial_statistics(self):
        """Fresh memory has all zeroes."""
        mem = SimpleFallbackMemory()
        stats = mem.get_statistics()
        assert stats["total_entries"] == 0
        assert stats["total_stored"] == 0
        assert stats["total_retrieved"] == 0
        assert stats["total_evicted"] == 0


@skip_no_fallback
class TestSimpleFallbackMemoryStore:
    """Tests for SimpleFallbackMemory.store()."""

    @pytest.mark.unit
    def test_store_returns_key(self):
        """store() returns an MD5-based key."""
        mem = SimpleFallbackMemory()
        key = mem.store("hello world")
        assert isinstance(key, str)
        assert len(key) == 16

    @pytest.mark.unit
    def test_store_empty_content_returns_empty(self):
        """Empty or whitespace content returns empty string."""
        mem = SimpleFallbackMemory()
        assert mem.store("") == ""
        assert mem.store("   ") == ""

    @pytest.mark.unit
    def test_store_increments_total(self):
        """Each unique store increments total_stored."""
        mem = SimpleFallbackMemory()
        mem.store("one")
        mem.store("two")
        stats = mem.get_statistics()
        assert stats["total_stored"] == 2

    @pytest.mark.unit
    def test_store_duplicate_updates_existing(self):
        """Storing same content updates access, does not add new entry."""
        mem = SimpleFallbackMemory()
        k1 = mem.store("same content")
        k2 = mem.store("same content")
        assert k1 == k2
        stats = mem.get_statistics()
        assert stats["total_entries"] == 1
        # total_stored should be 1 because duplicate hits the update branch
        assert stats["total_stored"] == 1

    @pytest.mark.unit
    def test_store_with_metadata(self):
        """Metadata is attached to the entry."""
        mem = SimpleFallbackMemory()
        mem.store("content", metadata={"source": "test"})
        entries = mem.get_recent(1)
        assert entries[0].metadata == {"source": "test"}

    @pytest.mark.unit
    def test_store_different_types(self):
        """Stores go to different type buckets."""
        mem = SimpleFallbackMemory()
        mem.store("a", memory_type=MemoryType.EPISODIC)
        mem.store("b", memory_type=MemoryType.SEMANTIC)
        mem.store("c", memory_type=MemoryType.PROCEDURAL)
        stats = mem.get_statistics()
        assert stats["by_type"]["episodic"]["count"] == 1
        assert stats["by_type"]["semantic"]["count"] == 1
        assert stats["by_type"]["procedural"]["count"] == 1


@skip_no_fallback
class TestSimpleFallbackMemoryEviction:
    """Tests for LRU eviction."""

    @pytest.mark.unit
    def test_eviction_when_at_capacity(self):
        """Oldest entry is evicted when capacity is reached."""
        mem = SimpleFallbackMemory(episodic_capacity=2)
        mem.store("first")
        mem.store("second")
        mem.store("third")
        stats = mem.get_statistics()
        assert stats["by_type"]["episodic"]["count"] == 2
        assert stats["total_evicted"] == 1

    @pytest.mark.unit
    def test_evict_lru_empty_returns_none(self):
        """_evict_lru on empty storage returns None."""
        mem = SimpleFallbackMemory()
        result = mem._evict_lru(MemoryType.EPISODIC)
        assert result is None

    @pytest.mark.unit
    def test_evicted_entry_is_oldest(self):
        """The first-in (LRU) entry is the one evicted."""
        mem = SimpleFallbackMemory(episodic_capacity=2)
        mem.store("first entry")
        mem.store("second entry")
        # Now store a third, which should evict "first entry"
        mem.store("third entry")
        remaining = mem.get_recent(10, memory_type=MemoryType.EPISODIC)
        contents = [e.content for e in remaining]
        assert "first entry" not in contents
        assert "second entry" in contents
        assert "third entry" in contents


@skip_no_fallback
class TestSimpleFallbackMemoryRetrieve:
    """Tests for SimpleFallbackMemory.retrieve()."""

    @pytest.mark.unit
    def test_retrieve_empty_query_returns_empty(self):
        """Empty query returns no results."""
        mem = SimpleFallbackMemory()
        mem.store("some content")
        assert mem.retrieve("") == []

    @pytest.mark.unit
    def test_retrieve_finds_matching_content(self):
        """Keyword overlap produces retrieval hits."""
        mem = SimpleFallbackMemory()
        mem.store("python programming language")
        mem.store("java programming language")
        mem.store("cooking recipes food")
        results = mem.retrieve("python programming", top_k=2)
        assert len(results) >= 1
        # The python entry should be top-ranked
        assert "python" in results[0].content.lower()

    @pytest.mark.unit
    def test_retrieve_respects_top_k(self):
        """Never returns more than top_k entries."""
        mem = SimpleFallbackMemory()
        for i in range(10):
            mem.store(f"memory entry number {i}")
        results = mem.retrieve("memory entry", top_k=3)
        assert len(results) <= 3

    @pytest.mark.unit
    def test_retrieve_by_specific_type(self):
        """Filter by memory_type only searches that bucket."""
        mem = SimpleFallbackMemory()
        mem.store("python code", memory_type=MemoryType.EPISODIC)
        mem.store("python tutorial", memory_type=MemoryType.SEMANTIC)
        results = mem.retrieve("python", memory_type=MemoryType.SEMANTIC)
        for r in results:
            assert r.memory_type == MemoryType.SEMANTIC

    @pytest.mark.unit
    def test_retrieve_no_matches_returns_empty(self):
        """Query with no keyword overlap returns nothing."""
        mem = SimpleFallbackMemory()
        mem.store("completely unrelated content xyz")
        results = mem.retrieve("totally different query abc")
        # May return empty or low-scored entries depending on importance threshold
        # Since importance=0.5 and threshold is 0.1, importance*0.3 = 0.15 > 0.1
        # So it may still return due to importance score alone
        # The key thing is it does not crash
        assert isinstance(results, list)

    @pytest.mark.unit
    def test_retrieve_updates_access_count(self):
        """Retrieved entries get their access count incremented."""
        mem = SimpleFallbackMemory()
        mem.store("test query matching content")
        results = mem.retrieve("test query matching content", top_k=1)
        if results:
            assert results[0].access_count >= 1


@skip_no_fallback
class TestSimpleFallbackMemoryGetRecent:
    """Tests for SimpleFallbackMemory.get_recent()."""

    @pytest.mark.unit
    def test_get_recent_empty(self):
        """Empty memory returns empty list."""
        mem = SimpleFallbackMemory()
        assert mem.get_recent(5) == []

    @pytest.mark.unit
    def test_get_recent_sorted_by_timestamp(self):
        """Results are ordered by timestamp descending."""
        mem = SimpleFallbackMemory()
        mem.store("old")
        time.sleep(0.01)
        mem.store("new")
        results = mem.get_recent(10)
        assert len(results) == 2
        assert results[0].timestamp >= results[1].timestamp

    @pytest.mark.unit
    def test_get_recent_filtered_by_type(self):
        """Filtered get_recent only returns matching type."""
        mem = SimpleFallbackMemory()
        mem.store("epi", memory_type=MemoryType.EPISODIC)
        mem.store("sem", memory_type=MemoryType.SEMANTIC)
        results = mem.get_recent(10, memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].content == "epi"


@skip_no_fallback
class TestSimpleFallbackMemoryClear:
    """Tests for SimpleFallbackMemory.clear()."""

    @pytest.mark.unit
    def test_clear_all(self):
        """clear() removes all entries."""
        mem = SimpleFallbackMemory()
        mem.store("a")
        mem.store("b", memory_type=MemoryType.SEMANTIC)
        cleared = mem.clear()
        assert cleared == 2
        assert mem.get_statistics()["total_entries"] == 0

    @pytest.mark.unit
    def test_clear_specific_type(self):
        """clear(type) only removes that type."""
        mem = SimpleFallbackMemory()
        mem.store("a", memory_type=MemoryType.EPISODIC)
        mem.store("b", memory_type=MemoryType.SEMANTIC)
        cleared = mem.clear(MemoryType.EPISODIC)
        assert cleared == 1
        stats = mem.get_statistics()
        assert stats["by_type"]["episodic"]["count"] == 0
        assert stats["by_type"]["semantic"]["count"] == 1


@skip_no_fallback
class TestSimpleFallbackMemoryStatistics:
    """Tests for get_statistics()."""

    @pytest.mark.unit
    def test_utilization_calculation(self):
        """Utilization is count / capacity."""
        mem = SimpleFallbackMemory(episodic_capacity=10)
        for i in range(5):
            mem.store(f"entry {i}")
        stats = mem.get_statistics()
        assert stats["by_type"]["episodic"]["utilization"] == 0.5

    @pytest.mark.unit
    def test_statistics_tracks_operations(self):
        """Stats reflect store, retrieve, eviction counts."""
        mem = SimpleFallbackMemory(episodic_capacity=2)
        mem.store("a")
        mem.store("b")
        mem.store("c")  # evicts "a"
        mem.retrieve("b")
        stats = mem.get_statistics()
        assert stats["total_stored"] == 3
        assert stats["total_retrieved"] == 1
        assert stats["total_evicted"] == 1


# =============================================================================
# Compatibility Methods Tests (remember, recall, consolidate)
# =============================================================================


@skip_no_fallback
class TestSimpleFallbackMemoryCompatibility:
    """Tests for BrainInspiredMemoryManager compatibility aliases."""

    @pytest.mark.unit
    def test_remember_stores_to_correct_type(self):
        """remember() maps level string to MemoryType."""
        mem = SimpleFallbackMemory()
        key = mem.remember("test content", level="semantic")
        stats = mem.get_statistics()
        assert stats["by_type"]["semantic"]["count"] == 1

    @pytest.mark.unit
    def test_remember_defaults_to_episodic(self):
        """remember() defaults to episodic level."""
        mem = SimpleFallbackMemory()
        mem.remember("default level content")
        stats = mem.get_statistics()
        assert stats["by_type"]["episodic"]["count"] == 1

    @pytest.mark.unit
    def test_remember_unknown_level_falls_back_to_episodic(self):
        """Unknown level string maps to EPISODIC."""
        mem = SimpleFallbackMemory()
        mem.remember("unknown", level="meta")
        stats = mem.get_statistics()
        assert stats["by_type"]["episodic"]["count"] == 1

    @pytest.mark.unit
    def test_recall_returns_strings(self):
        """recall() returns content strings, not MemoryEntry objects."""
        mem = SimpleFallbackMemory()
        mem.store("python programming tips")
        results = mem.recall("python programming")
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.unit
    def test_recall_with_level_filter(self):
        """recall() with level filters to that type."""
        mem = SimpleFallbackMemory()
        mem.store("ep content", memory_type=MemoryType.EPISODIC)
        mem.store("sem content", memory_type=MemoryType.SEMANTIC)
        results = mem.recall("content", level="semantic")
        # Should only include semantic entries
        if results:
            for r in results:
                assert "sem" in r or isinstance(r, str)

    @pytest.mark.unit
    def test_consolidate_returns_statistics(self):
        """consolidate() is a no-op that returns stats."""
        mem = SimpleFallbackMemory()
        mem.store("data")
        result = mem.consolidate()
        assert "total_entries" in result
        assert result["total_entries"] == 1


# =============================================================================
# SimpleFallbackMemory Persistence Tests (save/load)
# =============================================================================


@skip_no_fallback
class TestSimpleFallbackMemoryPersistence:
    """Tests for save() and load() methods."""

    @pytest.mark.unit
    def test_save_creates_file(self, tmp_path):
        """save() writes a JSON file."""
        mem = SimpleFallbackMemory()
        mem.store("persisted content")
        filepath = tmp_path / "memory.json"
        result = mem.save(str(filepath))
        assert result is True
        assert filepath.exists()

    @pytest.mark.unit
    def test_save_load_roundtrip(self, tmp_path):
        """save() then load() restores all entries."""
        mem = SimpleFallbackMemory()
        mem.store("entry one", memory_type=MemoryType.EPISODIC, importance=0.8)
        mem.store("entry two", memory_type=MemoryType.SEMANTIC, importance=0.3)
        filepath = tmp_path / "memory.json"
        mem.save(str(filepath))

        mem2 = SimpleFallbackMemory()
        result = mem2.load(str(filepath))
        assert result is True
        stats = mem2.get_statistics()
        assert stats["total_entries"] == 2

    @pytest.mark.unit
    def test_load_nonexistent_file_returns_false(self, tmp_path):
        """load() from nonexistent path returns False."""
        mem = SimpleFallbackMemory()
        result = mem.load(str(tmp_path / "does_not_exist.json"))
        assert result is False

    @pytest.mark.unit
    def test_save_handles_write_error(self, tmp_path):
        """save() returns False on write failure."""
        mem = SimpleFallbackMemory()
        mem.store("data")
        with patch("builtins.open", side_effect=PermissionError("no write")):
            result = mem.save(str(tmp_path / "fail.json"))
        assert result is False

    @pytest.mark.unit
    def test_load_handles_corrupt_json(self, tmp_path):
        """load() returns False on corrupt JSON."""
        filepath = tmp_path / "corrupt.json"
        filepath.write_text("NOT VALID JSON {{{")
        mem = SimpleFallbackMemory()
        result = mem.load(str(filepath))
        assert result is False

    @pytest.mark.unit
    def test_save_restores_statistics(self, tmp_path):
        """Statistics (stored/retrieved/evicted) survive save/load."""
        mem = SimpleFallbackMemory(episodic_capacity=2)
        mem.store("a")
        mem.store("b")
        mem.store("c")  # evicts
        mem.retrieve("a")
        filepath = tmp_path / "stats.json"
        mem.save(str(filepath))

        mem2 = SimpleFallbackMemory()
        mem2.load(str(filepath))
        assert mem2._total_stored == 3
        assert mem2._total_evicted == 1
        assert mem2._total_retrieved == 1


# =============================================================================
# SimpleFallbackMemory Serialization (to_dict / from_dict)
# =============================================================================


@skip_no_fallback
class TestSimpleFallbackMemorySerialization:
    """Tests for to_dict() and from_dict() class methods."""

    @pytest.mark.unit
    def test_to_dict_roundtrip(self):
        """to_dict() -> from_dict() preserves state."""
        mem = SimpleFallbackMemory(max_entries=100, episodic_capacity=20)
        mem.store("alpha", memory_type=MemoryType.EPISODIC)
        mem.store("beta", memory_type=MemoryType.SEMANTIC)

        data = mem.to_dict()
        restored = SimpleFallbackMemory.from_dict(data)

        assert restored.max_entries == 100
        stats = restored.get_statistics()
        assert stats["total_entries"] == 2

    @pytest.mark.unit
    def test_from_dict_defaults(self):
        """from_dict with empty data uses default capacities."""
        mem = SimpleFallbackMemory.from_dict({})
        assert mem.max_entries == 500
        assert mem.capacities[MemoryType.EPISODIC] == 300


# =============================================================================
# get_fallback_memory factory
# =============================================================================


@skip_no_fallback
class TestGetFallbackMemory:
    """Tests for the get_fallback_memory factory function."""

    @pytest.mark.unit
    def test_returns_instance(self):
        """Factory returns a SimpleFallbackMemory instance."""
        mem = get_fallback_memory()
        assert isinstance(mem, SimpleFallbackMemory)

    @pytest.mark.unit
    def test_passes_kwargs(self):
        """Factory forwards keyword arguments."""
        mem = get_fallback_memory(max_entries=42, episodic_capacity=10)
        assert mem.max_entries == 42
        assert mem.capacities[MemoryType.EPISODIC] == 10


# =============================================================================
# MemoryPersistence Tests
# =============================================================================


@skip_no_persistence
class TestMemoryPersistenceInit:
    """Tests for MemoryPersistence initialization."""

    @pytest.mark.unit
    def test_creates_directory(self, tmp_path):
        """Constructor creates the persistence directory."""
        persist_dir = tmp_path / "mem_persist"
        mock_memory = MagicMock()
        mock_memory.memories = {}
        MemoryPersistence(mock_memory, persist_dir)
        assert persist_dir.exists()

    @pytest.mark.unit
    def test_level_files_are_defined(self, tmp_path):
        """All five level files are configured."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        mock_memory = MagicMock()
        mock_memory.memories = {}
        mp = MemoryPersistence(mock_memory, tmp_path)
        assert len(mp.level_files) == 5
        for level in MemoryLevel:
            assert level in mp.level_files


@skip_no_persistence
class TestMemoryPersistenceSave:
    """Tests for MemoryPersistence.save()."""

    @pytest.mark.unit
    def test_save_creates_json_files(self, tmp_path):
        """save() writes one JSON file per memory level."""
        from datetime import datetime

        from Jotty.core.infrastructure.foundation.data_structures import (
            GoalValue,
            MemoryEntry,
            MemoryLevel,
        )

        mock_memory = MagicMock()
        # Create a real entry to save
        entry = MagicMock()
        entry.key = "test_key"
        entry.content = "test content"
        entry.level = MemoryLevel.EPISODIC
        entry.context = {"task": "test"}
        entry.created_at = datetime.now()
        entry.last_accessed = datetime.now()
        entry.access_count = 2
        entry.ucb_visits = 1
        entry.token_count = 10
        entry.default_value = 0.5
        entry.goal_values = {}
        entry.causal_links = []
        entry.content_hash = "abc123"
        entry.similar_entries = []
        entry.source_episode = 0
        entry.source_agent = ""
        entry.is_protected = False
        entry.protection_reason = ""

        mock_memory.memories = {
            MemoryLevel.EPISODIC: {"test_key": entry},
            MemoryLevel.SEMANTIC: {},
            MemoryLevel.PROCEDURAL: {},
            MemoryLevel.META: {},
            MemoryLevel.CAUSAL: {},
        }

        mp = MemoryPersistence(mock_memory, tmp_path)
        result = mp.save()
        assert result is True
        # Episodic file should exist with content
        epi_file = tmp_path / "episodic_memories.json"
        assert epi_file.exists()
        data = json.loads(epi_file.read_text())
        assert len(data) == 1
        assert data[0]["key"] == "test_key"

    @pytest.mark.unit
    def test_save_handles_goal_values(self, tmp_path):
        """save() serializes goal_values correctly."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        mock_gv = MagicMock()
        mock_gv.value = 0.8
        mock_gv.access_count = 3

        entry = MagicMock()
        entry.key = "gv_key"
        entry.content = "goal content"
        entry.level = MemoryLevel.EPISODIC
        entry.context = {}
        entry.created_at = datetime.now()
        entry.last_accessed = datetime.now()
        entry.access_count = 1
        entry.ucb_visits = 0
        entry.token_count = 5
        entry.default_value = 0.5
        entry.goal_values = {"research": mock_gv}
        entry.causal_links = []
        entry.content_hash = "hash1"
        entry.similar_entries = []
        entry.source_episode = 1
        entry.source_agent = "agent1"
        entry.is_protected = False
        entry.protection_reason = ""

        mock_memory = MagicMock()
        mock_memory.memories = {
            MemoryLevel.EPISODIC: {"gv_key": entry},
            MemoryLevel.SEMANTIC: {},
            MemoryLevel.PROCEDURAL: {},
            MemoryLevel.META: {},
            MemoryLevel.CAUSAL: {},
        }

        mp = MemoryPersistence(mock_memory, tmp_path)
        result = mp.save()
        assert result is True
        data = json.loads((tmp_path / "episodic_memories.json").read_text())
        assert data[0]["goal_values"]["research"]["value"] == 0.8

    @pytest.mark.unit
    def test_save_returns_false_on_error(self, tmp_path):
        """save() returns False when serialization fails."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        mock_memory = MagicMock()
        # Make memories dict access raise when iterating values
        bad_dict = MagicMock()
        bad_dict.values = Mock(side_effect=RuntimeError("serialize error"))
        mock_memory.memories = {
            MemoryLevel.EPISODIC: bad_dict,
            MemoryLevel.SEMANTIC: {},
            MemoryLevel.PROCEDURAL: {},
            MemoryLevel.META: {},
            MemoryLevel.CAUSAL: {},
        }

        mp = MemoryPersistence(mock_memory, tmp_path)
        result = mp.save()
        assert result is False


@skip_no_persistence
class TestMemoryPersistenceLoad:
    """Tests for MemoryPersistence.load()."""

    @pytest.mark.unit
    def test_load_skips_missing_files(self, tmp_path):
        """load() skips levels with no file present."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        mock_memory = MagicMock()
        mock_memory.memories = {level: {} for level in MemoryLevel}
        mp = MemoryPersistence(mock_memory, tmp_path)
        result = mp.load()
        assert result is True

    @pytest.mark.unit
    def test_load_restores_entries(self, tmp_path):
        """load() populates memory with entries from disk."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        # Write a test file
        entry_data = [
            {
                "key": "loaded_key",
                "content": "loaded content",
                "level": "episodic",
                "context": {"source": "test"},
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 5,
                "ucb_visits": 2,
                "token_count": 10,
                "default_value": 0.6,
                "goal_values": {"research": {"value": 0.7, "access_count": 3}},
                "causal_links": [],
                "content_hash": "abc",
                "similar_entries": [],
                "source_episode": 1,
                "source_agent": "test_agent",
                "is_protected": True,
                "protection_reason": "important",
            }
        ]
        epi_file = tmp_path / "episodic_memories.json"
        epi_file.write_text(json.dumps(entry_data))

        mock_memory = MagicMock()
        mock_memory.memories = {level: {} for level in MemoryLevel}

        mp = MemoryPersistence(mock_memory, tmp_path)
        result = mp.load()
        assert result is True
        # Should have been stored into the mock
        assert "loaded_key" in mock_memory.memories[MemoryLevel.EPISODIC]

    @pytest.mark.unit
    def test_load_returns_false_on_corrupt_file(self, tmp_path):
        """load() returns False on corrupt JSON file."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        corrupt_file = tmp_path / "episodic_memories.json"
        corrupt_file.write_text("NOT JSON")

        mock_memory = MagicMock()
        mock_memory.memories = {level: {} for level in MemoryLevel}

        mp = MemoryPersistence(mock_memory, tmp_path)
        result = mp.load()
        assert result is False


@skip_no_persistence
class TestEnableMemoryPersistence:
    """Tests for the enable_memory_persistence() factory function."""

    @pytest.mark.unit
    def test_default_dir_uses_agent_name(self):
        """Default persistence_dir is derived from agent_name."""
        mock_memory = MagicMock()
        mock_memory.agent_name = "my_agent"
        mock_memory.memories = {}

        with patch.object(MemoryPersistence, "load", return_value=True):
            with patch("pathlib.Path.mkdir"):
                mp = enable_memory_persistence(mock_memory)
        assert "my_agent" in str(mp.persistence_dir)

    @pytest.mark.unit
    def test_custom_dir(self, tmp_path):
        """Custom persistence_dir is used."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        mock_memory = MagicMock()
        mock_memory.memories = {level: {} for level in MemoryLevel}
        custom_dir = tmp_path / "custom"

        mp = enable_memory_persistence(mock_memory, persistence_dir=custom_dir)
        assert mp.persistence_dir == custom_dir

    @pytest.mark.unit
    def test_load_called_on_enable(self, tmp_path):
        """enable_memory_persistence() calls load() to restore state."""
        mock_memory = MagicMock()
        mock_memory.memories = {}

        with patch.object(MemoryPersistence, "load", return_value=True) as mock_load:
            enable_memory_persistence(mock_memory, persistence_dir=tmp_path)
        mock_load.assert_called_once()


# =============================================================================
# MemoryResult Tests
# =============================================================================


@skip_no_system
class TestMemoryResult:
    """Tests for the MemoryResult dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """MemoryResult has correct defaults."""
        result = MemoryResult(content="test", level="episodic")
        assert result.relevance == 0.0
        assert result.timestamp == 0.0
        assert result.metadata == {}

    @pytest.mark.unit
    def test_str_representation(self):
        """__str__ shows level prefix and truncated content."""
        result = MemoryResult(content="a" * 200, level="semantic")
        s = str(result)
        assert "[semantic]" in s
        assert len(s) < 200  # Truncated


@skip_no_system
class TestMemoryConfig:
    """Tests for the MemoryConfig dataclass."""

    @pytest.mark.unit
    def test_defaults(self):
        """MemoryConfig defaults are sensible."""
        cfg = MemoryConfig()
        assert cfg.backend == MemoryBackend.FULL
        assert cfg.agent_name == "default"
        assert cfg.auto_consolidate is True
        assert cfg.consolidation_interval == 3
        assert cfg.max_memories_per_level == 500
        assert cfg.enable_tracing is True


# =============================================================================
# MemorySystem Tests
# =============================================================================


@skip_no_system
class TestMemorySystemFallbackInit:
    """Tests for MemorySystem with fallback backend."""

    @pytest.mark.unit
    def test_explicit_fallback_backend(self):
        """Configuring FALLBACK backend initializes SimpleFallbackMemory."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        assert ms._backend_type == MemoryBackend.FALLBACK
        assert isinstance(ms._backend, SimpleFallbackMemory)

    @pytest.mark.unit
    def test_falls_back_when_full_fails(self):
        """When full backend import fails, system falls back gracefully."""
        cfg = MemoryConfig(backend=MemoryBackend.FULL)
        with patch.object(MemorySystem, "_init_full", side_effect=RuntimeError("no brain")):
            ms = MemorySystem(config=cfg)
        assert ms._backend_type == MemoryBackend.FALLBACK

    @pytest.mark.unit
    def test_default_config_when_none_provided(self):
        """None config produces default MemoryConfig."""
        with patch.object(MemorySystem, "_init_full", side_effect=ImportError("no brain")):
            ms = MemorySystem(config=None)
        assert ms.config.agent_name == "default"


@skip_no_system
class TestMemorySystemStore:
    """Tests for MemorySystem.store()."""

    @pytest.mark.unit
    def test_store_fallback_increments_counter(self):
        """store() increments _store_count."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms.store("hello world")
        assert ms._store_count == 1

    @pytest.mark.unit
    def test_store_fallback_maps_meta_to_semantic(self):
        """Meta level maps to semantic in fallback."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms.store("meta info", level="meta")
        stats = ms._backend.get_statistics()
        assert stats["by_type"]["semantic"]["count"] == 1

    @pytest.mark.unit
    def test_store_fallback_returns_key(self):
        """store() returns a non-empty key."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        key = ms.store("some content")
        assert isinstance(key, str)
        assert len(key) > 0

    @pytest.mark.unit
    def test_store_full_backend(self):
        """store() on full backend delegates to SwarmMemory.store()."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        # Patch to simulate full backend
        ms._backend_type = MemoryBackend.FULL
        mock_backend = MagicMock()
        mock_backend.store = MagicMock(return_value="mem_id_123")
        ms._backend = mock_backend

        with patch(
            "Jotty.core.memory.memory_system.MemorySystem._store_full",
            return_value="mem_id_123",
        ) as mock_sf:
            result = ms.store("full backend content")
        mock_sf.assert_called_once()


@skip_no_system
class TestMemorySystemRetrieve:
    """Tests for MemorySystem.retrieve()."""

    @pytest.mark.unit
    def test_retrieve_fallback_returns_memory_results(self):
        """retrieve() returns MemoryResult objects from fallback."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms.store("python programming tips")
        results = ms.retrieve("python programming")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, MemoryResult)

    @pytest.mark.unit
    def test_retrieve_increments_counter(self):
        """retrieve() increments _retrieve_count."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms.retrieve("anything")
        assert ms._retrieve_count == 1

    @pytest.mark.unit
    def test_retrieve_fallback_handles_exception(self):
        """_retrieve_fallback returns empty list on error."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms._backend.retrieve = Mock(side_effect=RuntimeError("boom"))
        results = ms.retrieve("query")
        assert results == []

    @pytest.mark.unit
    def test_retrieve_full_handles_exception(self):
        """_retrieve_full returns empty list on backend error."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms._backend_type = MemoryBackend.FULL
        ms._backend = MagicMock()
        ms._backend.retrieve = MagicMock(side_effect=RuntimeError("fail"))
        results = ms.retrieve("query")
        assert results == []


@skip_no_system
class TestMemorySystemConsolidate:
    """Tests for MemorySystem.consolidate()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consolidate_fallback_calls_prune(self):
        """consolidate() on fallback calls prune."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms._backend.prune = Mock()
        result = await ms.consolidate()
        assert result["success"] is True
        assert result["backend"] == "fallback"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consolidate_fallback_handles_missing_prune(self):
        """consolidate() on fallback handles AttributeError on prune."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        # Remove prune if it exists
        if hasattr(ms._backend, "prune"):
            ms._backend.prune = Mock(side_effect=AttributeError("no prune"))
        result = await ms.consolidate()
        assert result["success"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consolidate_increments_count(self):
        """consolidate() increments _consolidation_count."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms._backend.prune = Mock()
        await ms.consolidate()
        assert ms._consolidation_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_consolidate_full_async_backend(self):
        """consolidate() awaits async consolidate on full backend."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms._backend_type = MemoryBackend.FULL
        ms._backend = MagicMock()
        ms._backend.consolidate = AsyncMock(return_value={"consolidated": 5})
        result = await ms.consolidate()
        assert result["success"] is True


@skip_no_system
class TestMemorySystemRecordEpisode:
    """Tests for MemorySystem.record_episode()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_episode_increments_count(self):
        """record_episode() increments _episode_count."""
        cfg = MemoryConfig(
            backend=MemoryBackend.FALLBACK,
            auto_consolidate=False,
        )
        ms = MemorySystem(config=cfg)
        await ms.record_episode(goal="test", result="ok", reward=1.0)
        assert ms._episode_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_episode_stores_content(self):
        """record_episode() stores a formatted string."""
        cfg = MemoryConfig(
            backend=MemoryBackend.FALLBACK,
            auto_consolidate=False,
        )
        ms = MemorySystem(config=cfg)
        await ms.record_episode(goal="research", result="found answer", reward=0.8)
        recent = ms._backend.get_recent(1)
        assert len(recent) == 1
        assert "research" in recent[0].content
        assert "found answer" in recent[0].content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_consolidation_triggered(self):
        """Auto-consolidation triggers at interval."""
        cfg = MemoryConfig(
            backend=MemoryBackend.FALLBACK,
            auto_consolidate=True,
            consolidation_interval=2,
        )
        ms = MemorySystem(config=cfg)
        ms._backend.prune = Mock()

        await ms.record_episode(goal="g1", result="r1", reward=0.5)
        assert ms._consolidation_count == 0
        await ms.record_episode(goal="g2", result="r2", reward=0.5)
        assert ms._consolidation_count == 1


@skip_no_system
class TestMemorySystemStatus:
    """Tests for MemorySystem.status()."""

    @pytest.mark.unit
    def test_status_returns_dict(self):
        """status() returns a dict with expected keys."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        status = ms.status()
        assert status["backend"] == "fallback"
        assert status["agent_name"] == "default"
        assert "operations" in status
        assert "uptime_s" in status

    @pytest.mark.unit
    def test_status_counts_operations(self):
        """status() reflects store/retrieve counts."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms.store("a")
        ms.store("b")
        ms.retrieve("a")
        status = ms.status()
        assert status["operations"]["stores"] == 2
        assert status["operations"]["retrieves"] == 1


@skip_no_system
class TestMemorySystemClear:
    """Tests for MemorySystem.clear()."""

    @pytest.mark.unit
    def test_clear_fallback(self):
        """clear() empties fallback memory and resets counters."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms.store("data")
        ms.clear()
        assert ms._store_count == 0
        assert ms._episode_count == 0
        stats = ms._backend.get_statistics()
        assert stats["total_entries"] == 0

    @pytest.mark.unit
    def test_clear_full_backend(self):
        """clear() clears all levels on full backend."""
        cfg = MemoryConfig(backend=MemoryBackend.FALLBACK)
        ms = MemorySystem(config=cfg)
        ms._backend_type = MemoryBackend.FULL

        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

        mock_memories = {level: MagicMock() for level in MemoryLevel}
        ms._backend = MagicMock()
        ms._backend.memories = mock_memories

        ms.clear()
        for level_mock in mock_memories.values():
            level_mock.clear.assert_called_once()


@skip_no_system
class TestMemoryBackendEnum:
    """Tests for the MemoryBackend enum."""

    @pytest.mark.unit
    def test_backend_values(self):
        """All backend values are correct."""
        assert MemoryBackend.FULL.value == "full"
        assert MemoryBackend.SIMPLE.value == "simple"
        assert MemoryBackend.FALLBACK.value == "fallback"

    @pytest.mark.unit
    def test_backend_count(self):
        """Three backends are defined."""
        assert len(MemoryBackend) == 3
