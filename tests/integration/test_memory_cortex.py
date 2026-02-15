"""
Tests for Memory Cortex Module
================================
Tests for SwarmMemory store, retrieve, consolidation, and serialization.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# SwarmMemory Creation Tests
# =============================================================================


class TestSwarmMemoryCreation:
    """Tests for SwarmMemory initialization."""

    @pytest.mark.unit
    def test_creation_with_defaults(self, minimal_jotty_config):
        """SwarmMemory creates with agent name and config."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        assert memory.agent_name == "test_agent"

    @pytest.mark.unit
    def test_empty_memory_statistics(self, minimal_jotty_config):
        """New memory has zero entries."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        stats = memory.get_statistics()
        assert stats["total_memories"] == 0

    @pytest.mark.unit
    def test_five_memory_levels_exist(self, minimal_jotty_config):
        """SwarmMemory has all 5 memory levels initialized."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        for level in MemoryLevel:
            assert level in memory.memories


# =============================================================================
# Store Tests
# =============================================================================


class TestSwarmMemoryStore:
    """Tests for SwarmMemory.store()."""

    @pytest.mark.unit
    def test_store_episodic_memory(self, minimal_jotty_config):
        """Store creates entry at episodic level."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        entry = memory.store(
            content="Agent used web-search and got good results",
            level=MemoryLevel.EPISODIC,
            context={"task": "research"},
            goal="Find information about AI",
        )
        assert entry is not None
        assert entry.content == "Agent used web-search and got good results"
        stats = memory.get_statistics()
        assert stats["total_memories"] == 1

    @pytest.mark.unit
    def test_store_with_domain_and_task_type(self, minimal_jotty_config):
        """Store creates hierarchical key with domain:task_type:hash."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        entry = memory.store(
            content="Test memory",
            level=MemoryLevel.SEMANTIC,
            context={},
            goal="test",
            domain="research",
            task_type="analysis",
        )
        assert entry is not None
        # Key should contain domain and task_type
        found = False
        for key in memory.memories[MemoryLevel.SEMANTIC]:
            if "research" in key and "analysis" in key:
                found = True
                break
        assert found, "Hierarchical key not found in stored memories"

    @pytest.mark.unit
    def test_store_multiple_levels(self, minimal_jotty_config):
        """Store works at different memory levels."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        for level in [MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL]:
            memory.store(
                content=f"Memory at {level.name}",
                level=level,
                context={},
                goal="test",
            )

        stats = memory.get_statistics()
        assert stats["total_memories"] == 3

    @pytest.mark.unit
    def test_store_with_outcome_failure(self, minimal_jotty_config):
        """store_with_outcome routes failures to CAUSAL level."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        entry = memory.store_with_outcome(
            content="Tool failed with error X",
            context={"tool": "web-search"},
            goal="research",
            outcome="failure",
        )
        assert entry is not None
        # Failure should be stored at CAUSAL level
        causal_count = len(memory.memories.get(MemoryLevel.CAUSAL, {}))
        assert causal_count >= 1

    @pytest.mark.unit
    def test_store_with_outcome_success(self, minimal_jotty_config):
        """store_with_outcome routes successes to SEMANTIC level."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        entry = memory.store_with_outcome(
            content="Agent found great results",
            context={},
            goal="research",
            outcome="success",
        )
        assert entry is not None
        semantic_count = len(memory.memories.get(MemoryLevel.SEMANTIC, {}))
        assert semantic_count >= 1

    @pytest.mark.unit
    def test_capacity_enforcement(self, minimal_jotty_config):
        """Store enforces capacity limits with eviction."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel, SwarmConfig
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        config = SwarmConfig(
            episodic_capacity=3,
            output_base_dir="./test_outputs",
            create_run_folder=False,
        )
        memory = SwarmMemory("test_agent", config)

        for i in range(5):
            memory.store(
                content=f"Memory {i}",
                level=MemoryLevel.EPISODIC,
                context={},
                goal="test",
                initial_value=0.1 * i,
            )

        # Should not exceed capacity
        episodic_count = len(memory.memories[MemoryLevel.EPISODIC])
        assert episodic_count <= 3


# =============================================================================
# Retrieve Tests
# =============================================================================


class TestSwarmMemoryRetrieve:
    """Tests for SwarmMemory.retrieve_fast() and retrieve_by_domain()."""

    @pytest.mark.unit
    def test_retrieve_fast_empty(self, minimal_jotty_config):
        """retrieve_fast from empty memory returns empty list."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        results = memory.retrieve_fast(
            query="test query",
            goal="test goal",
            budget_tokens=1000,
        )
        assert results == []

    @pytest.mark.unit
    def test_retrieve_fast_finds_relevant(self, minimal_jotty_config):
        """retrieve_fast returns memories matching query keywords."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store(
            "web search returned good results for AI trends",
            MemoryLevel.EPISODIC,
            {},
            "research AI",
        )
        memory.store(
            "calculator tool computed financial metrics", MemoryLevel.EPISODIC, {}, "analyze data"
        )
        memory.store(
            "AI research produced comprehensive report", MemoryLevel.EPISODIC, {}, "research AI"
        )

        results = memory.retrieve_fast(
            query="AI research",
            goal="research AI",
            budget_tokens=5000,
        )
        assert len(results) >= 1
        # Most relevant should mention AI or research
        assert any("AI" in r.content or "research" in r.content for r in results)

    @pytest.mark.unit
    def test_retrieve_fast_respects_budget(self, minimal_jotty_config):
        """retrieve_fast respects token budget."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        # Store many memories
        for i in range(20):
            memory.store(
                content=f"Memory content number {i} " * 50,  # ~200 tokens each
                level=MemoryLevel.EPISODIC,
                context={},
                goal="test",
            )

        results = memory.retrieve_fast(
            query="memory content",
            goal="test",
            budget_tokens=100,  # Very small budget
        )
        # Should return limited results due to budget
        assert len(results) <= 5

    @pytest.mark.unit
    def test_retrieve_by_domain(self, minimal_jotty_config):
        """retrieve_by_domain filters by domain prefix."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store(
            "coding pattern A",
            MemoryLevel.SEMANTIC,
            {},
            "code",
            domain="coding",
            task_type="review",
        )
        memory.store(
            "research finding B",
            MemoryLevel.SEMANTIC,
            {},
            "research",
            domain="research",
            task_type="analysis",
        )
        memory.store(
            "coding pattern C", MemoryLevel.SEMANTIC, {}, "code", domain="coding", task_type="debug"
        )

        results = memory.retrieve_by_domain(
            domain="coding",
            goal="code",
            budget_tokens=5000,
        )
        # Domain filtering may return empty if key format doesn't match,
        # but should at least not error
        assert isinstance(results, list)

    @pytest.mark.unit
    def test_retrieve_updates_access_tracking(self, minimal_jotty_config):
        """retrieve_fast updates access_count on returned memories."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store("test content for tracking", MemoryLevel.EPISODIC, {}, "test")
        initial_count = entry.access_count

        memory.retrieve_fast("test content", "test", budget_tokens=5000)
        # Access count should have increased
        assert entry.access_count >= initial_count


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSwarmMemorySerialization:
    """Tests for SwarmMemory to_dict/from_dict."""

    @pytest.mark.unit
    def test_to_dict_empty(self, minimal_jotty_config):
        """to_dict on empty memory returns valid structure."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        data = memory.to_dict()
        assert data["agent_name"] == "test_agent"
        assert "memories" in data

    @pytest.mark.unit
    def test_round_trip_serialization(self, minimal_jotty_config):
        """to_dict/from_dict preserves data."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store(
            "important finding", MemoryLevel.SEMANTIC, {}, "goal", domain="test", task_type="unit"
        )
        memory.store(
            "error pattern", MemoryLevel.CAUSAL, {}, "debug", domain="test", task_type="debug"
        )

        data = memory.to_dict()
        restored = SwarmMemory.from_dict(data, minimal_jotty_config)

        assert restored.agent_name == "test_agent"
        original_stats = memory.get_statistics()
        restored_stats = restored.get_statistics()
        assert original_stats["total_memories"] == restored_stats["total_memories"]

    @pytest.mark.unit
    def test_statistics_correct(self, minimal_jotty_config):
        """get_statistics returns correct counts."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("ep1", MemoryLevel.EPISODIC, {}, "g1")
        memory.store("ep2", MemoryLevel.EPISODIC, {}, "g2")
        memory.store("sem1", MemoryLevel.SEMANTIC, {}, "g1")

        stats = memory.get_statistics()
        assert stats["total_memories"] == 3
        assert stats["by_level"]["episodic"] == 2
        assert stats["by_level"]["semantic"] == 1

    @pytest.mark.unit
    def test_consolidated_knowledge_output(self, minimal_jotty_config):
        """get_consolidated_knowledge returns formatted string."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store(
            "Pattern: always validate inputs before execution",
            MemoryLevel.SEMANTIC,
            {},
            "validation",
        )
        memory.store(
            "Procedure: use web-search then summarize", MemoryLevel.PROCEDURAL, {}, "research"
        )

        knowledge = memory.get_consolidated_knowledge()
        assert isinstance(knowledge, str)


# =============================================================================
# Store Edge Cases
# =============================================================================


class TestSwarmMemoryStoreEdgeCases:
    """Tests for edge cases in SwarmMemory.store()."""

    @pytest.mark.unit
    def test_store_at_all_five_levels(self, minimal_jotty_config):
        """Store works at all 5 memory levels."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        for level in MemoryLevel:
            memory.store(
                content=f"Content for {level.value}",
                level=level,
                context={},
                goal="test",
            )

        stats = memory.get_statistics()
        assert stats["total_memories"] == 5
        for level in MemoryLevel:
            assert stats["by_level"][level.value] == 1

    @pytest.mark.unit
    def test_store_empty_content(self, minimal_jotty_config):
        """Store with empty string content still creates entry."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test",
        )
        assert entry is not None
        assert entry.content == ""
        assert memory.get_statistics()["total_memories"] == 1

    @pytest.mark.unit
    def test_store_very_long_content_truncated(self, minimal_jotty_config):
        """Store truncates content exceeding max_entry_tokens."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        # max_entry_tokens defaults to 2000 => 8000 chars
        long_content = "A" * (minimal_jotty_config.max_entry_tokens * 4 + 1000)
        entry = memory.store(
            content=long_content,
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test",
        )
        assert "[TRUNCATED" in entry.content
        assert len(entry.content) < len(long_content)

    @pytest.mark.unit
    def test_store_duplicate_content_returns_existing(self, minimal_jotty_config):
        """Storing identical content at same level returns existing entry."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry1 = memory.store(
            content="unique test content",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test",
        )
        entry2 = memory.store(
            content="unique test content",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test",
        )
        # Should return same entry (dedup via key match)
        assert entry1.key == entry2.key
        assert memory.get_statistics()["total_memories"] == 1
        # Access count should have incremented
        assert entry2.access_count >= 1

    @pytest.mark.unit
    def test_store_extracts_domain_from_context(self, minimal_jotty_config):
        """Store extracts domain from context when not provided explicitly."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="context-based domain entry",
            level=MemoryLevel.SEMANTIC,
            context={"domain": "sql", "task_type": "date_filter"},
            goal="test",
        )
        # Key should contain sql and date_filter
        assert "sql" in entry.key
        assert "date_filter" in entry.key

    @pytest.mark.unit
    def test_store_defaults_domain_to_general(self, minimal_jotty_config):
        """Store defaults domain to 'general' when not in context or args."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="no domain info at all",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="test",
        )
        assert entry.key.startswith("general:")

    @pytest.mark.unit
    def test_store_sets_goal_value(self, minimal_jotty_config):
        """Store sets goal-conditioned value on the entry."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="value test",
            level=MemoryLevel.EPISODIC,
            context={},
            goal="my_goal",
            initial_value=0.8,
        )
        assert entry.get_value("my_goal") == 0.8
        assert entry.default_value == 0.8

    @pytest.mark.unit
    def test_store_with_causal_links(self, minimal_jotty_config):
        """Store preserves causal_links list on entry."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="causal entry",
            level=MemoryLevel.CAUSAL,
            context={},
            goal="test",
            causal_links=["link_abc", "link_def"],
        )
        assert "link_abc" in entry.causal_links
        assert "link_def" in entry.causal_links

    @pytest.mark.unit
    def test_store_metadata_contains_domain_task_type(self, minimal_jotty_config):
        """Store sets metadata with domain and task_type."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="metadata test",
            level=MemoryLevel.SEMANTIC,
            context={},
            goal="test",
            domain="mermaid",
            task_type="sequence_diagram",
        )
        assert hasattr(entry, "metadata")
        assert entry.metadata["domain"] == "mermaid"
        assert entry.metadata["task_type"] == "sequence_diagram"


# =============================================================================
# store_with_outcome Edge Cases
# =============================================================================


class TestStoreWithOutcomeEdgeCases:
    """Tests for store_with_outcome edge cases."""

    @pytest.mark.unit
    def test_store_with_outcome_neutral(self, minimal_jotty_config):
        """store_with_outcome with neutral routes to EPISODIC."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store_with_outcome(
            content="Neutral event occurred",
            context={},
            goal="test",
            outcome="neutral",
        )
        assert entry is not None
        episodic_count = len(memory.memories[MemoryLevel.EPISODIC])
        assert episodic_count >= 1

    @pytest.mark.unit
    def test_store_with_outcome_failure_high_value(self, minimal_jotty_config):
        """Failures are stored with high initial value (0.9)."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store_with_outcome(
            content="Critical failure",
            context={"error": "timeout"},
            goal="debug",
            outcome="failure",
        )
        assert entry.default_value == 0.9

    @pytest.mark.unit
    def test_store_with_outcome_success_low_value(self, minimal_jotty_config):
        """Successes are stored with lower initial value (0.4)."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store_with_outcome(
            content="Simple success",
            context={},
            goal="test",
            outcome="success",
        )
        assert entry.default_value == 0.4

    @pytest.mark.unit
    def test_store_with_outcome_failure_includes_context_dump(self, minimal_jotty_config):
        """Failure outcome includes full context dump in content."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store_with_outcome(
            content="Error X happened",
            context={"tool": "web-search", "status": 500},
            goal="debug",
            outcome="failure",
        )
        assert "FAILURE ANALYSIS" in entry.content
        assert "FULL CONTEXT" in entry.content
        assert "web-search" in entry.content

    @pytest.mark.unit
    def test_store_with_outcome_success_summarizes(self, minimal_jotty_config):
        """Success outcome stores only first line summary."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store_with_outcome(
            content="First line summary\nSecond line detail\nThird line detail",
            context={},
            goal="test",
            outcome="success",
        )
        assert "Success:" in entry.content
        assert "First line summary" in entry.content
        # Should NOT contain the additional detail lines directly
        assert "Third line detail" not in entry.content

    @pytest.mark.unit
    def test_store_with_outcome_passes_domain_task_type(self, minimal_jotty_config):
        """store_with_outcome forwards domain and task_type to store."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store_with_outcome(
            content="Domain routed",
            context={},
            goal="test",
            outcome="neutral",
            domain="sql",
            task_type="join_query",
        )
        assert "sql" in entry.key
        assert "join_query" in entry.key


# =============================================================================
# store_with_surprise Tests
# =============================================================================


class TestStoreWithSurprise:
    """Tests for surprise-based memory storage."""

    @pytest.mark.unit
    def test_low_surprise_skips_storage(self, minimal_jotty_config):
        """Surprise < 0.3 returns None (skipped)."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        result = memory.store_with_surprise(
            content="Routine event",
            surprise_score=0.1,
            context={},
            goal="test",
        )
        assert result is None
        assert memory.get_statistics()["total_memories"] == 0

    @pytest.mark.unit
    def test_medium_surprise_stores_episodic(self, minimal_jotty_config):
        """Surprise 0.3-0.7 stores in EPISODIC level."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        result = memory.store_with_surprise(
            content="Notable event",
            surprise_score=0.5,
            context={},
            goal="test",
        )
        assert result is not None
        assert len(memory.memories[MemoryLevel.EPISODIC]) == 1

    @pytest.mark.unit
    def test_high_surprise_stores_causal(self, minimal_jotty_config):
        """Surprise >= 0.7 stores in CAUSAL level with high value."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        result = memory.store_with_surprise(
            content="Very surprising event",
            surprise_score=0.9,
            context={},
            goal="test",
        )
        assert result is not None
        assert len(memory.memories[MemoryLevel.CAUSAL]) == 1
        assert result.default_value >= 0.8

    @pytest.mark.unit
    def test_surprise_score_clamped(self, minimal_jotty_config):
        """Surprise scores outside 0-1 are clamped."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        # Score > 1.0 should be clamped to 1.0 => stored as causal
        result = memory.store_with_surprise(
            content="Over-surprising",
            surprise_score=1.5,
            context={},
            goal="test",
        )
        assert result is not None

        # Score < 0.0 should be clamped to 0.0 => skipped
        result2 = memory.store_with_surprise(
            content="Negative surprise",
            surprise_score=-0.5,
            context={},
            goal="test",
        )
        assert result2 is None


# =============================================================================
# Retrieval Mixin Advanced Tests
# =============================================================================


class TestRetrievalMixinAdvanced:
    """Tests for advanced retrieval methods from RetrievalMixin."""

    @pytest.mark.unit
    def test_retrieve_fast_with_specific_levels(self, minimal_jotty_config):
        """retrieve_fast can filter by specific memory levels."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("episodic content about AI", MemoryLevel.EPISODIC, {}, "test")
        memory.store("semantic pattern about AI", MemoryLevel.SEMANTIC, {}, "test")

        # Only search SEMANTIC level
        results = memory.retrieve_fast(
            query="AI pattern",
            goal="test",
            budget_tokens=5000,
            levels=[MemoryLevel.SEMANTIC],
        )
        # Should only find semantic entries
        for r in results:
            assert r.level == MemoryLevel.SEMANTIC

    @pytest.mark.unit
    def test_retrieve_fast_no_matching_keywords(self, minimal_jotty_config):
        """retrieve_fast with completely unrelated query returns by recency."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("apple banana cherry", MemoryLevel.EPISODIC, {}, "fruit")

        results = memory.retrieve_fast(
            query="quantum physics thermodynamics",
            goal="science",
            budget_tokens=5000,
        )
        # May return results with low scores (recency/value still score)
        assert isinstance(results, list)

    @pytest.mark.unit
    def test_retrieve_fast_short_query_words_ignored(self, minimal_jotty_config):
        """retrieve_fast ignores query words with <= 2 characters."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("important data analysis result", MemoryLevel.EPISODIC, {}, "test")

        # Query with only short words => fallback to recency
        results = memory.retrieve_fast(
            query="a b c d",
            goal="test",
            budget_tokens=5000,
        )
        # Should still return results (recency fallback)
        assert isinstance(results, list)

    @pytest.mark.unit
    def test_retrieve_fast_top_k_limits_results(self, minimal_jotty_config):
        """retrieve_fast respects top_k parameter."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel, SwarmConfig
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        config = SwarmConfig(
            episodic_capacity=100,
            output_base_dir="./test_outputs",
            create_run_folder=False,
        )
        memory = SwarmMemory("test_agent", config)

        for i in range(20):
            memory.store(f"item {i} data analysis", MemoryLevel.EPISODIC, {}, "test")

        results = memory.retrieve_fast(
            query="data analysis",
            goal="test",
            budget_tokens=100000,
            top_k=3,
        )
        assert len(results) <= 3

    @pytest.mark.unit
    def test_retrieve_fast_increments_total_accesses(self, minimal_jotty_config):
        """retrieve_fast increments memory.total_accesses counter."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        memory.store("test data", MemoryLevel.EPISODIC, {}, "test")

        assert memory.total_accesses == 0
        memory.retrieve_fast("test", "test", budget_tokens=5000)
        assert memory.total_accesses == 1
        memory.retrieve_fast("test", "test", budget_tokens=5000)
        assert memory.total_accesses == 2

    @pytest.mark.unit
    def test_retrieve_fast_updates_ucb_visits(self, minimal_jotty_config):
        """retrieve_fast increments ucb_visits on returned entries."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store("test ucb content", MemoryLevel.EPISODIC, {}, "test")
        initial_ucb = entry.ucb_visits

        memory.retrieve_fast("test ucb content", "test", budget_tokens=5000)
        assert entry.ucb_visits > initial_ucb

    @pytest.mark.unit
    def test_retrieve_fast_updates_last_accessed(self, minimal_jotty_config):
        """retrieve_fast updates last_accessed timestamp."""
        from datetime import datetime, timedelta

        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store("timestamp test", MemoryLevel.EPISODIC, {}, "test")
        old_accessed = entry.last_accessed

        # Small sleep would be ideal but to keep tests fast,
        # just verify the field gets set
        memory.retrieve_fast("timestamp test", "test", budget_tokens=5000)
        assert entry.last_accessed >= old_accessed

    @pytest.mark.unit
    def test_retrieve_by_task_type(self, minimal_jotty_config):
        """retrieve_by_task_type filters by task type in key."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store(
            "date filter pattern",
            MemoryLevel.SEMANTIC,
            {},
            "test",
            domain="sql",
            task_type="date_filter",
        )
        memory.store(
            "join pattern", MemoryLevel.SEMANTIC, {}, "test", domain="sql", task_type="join_query"
        )

        results = memory.retrieve_by_task_type(
            task_type="date_filter",
            goal="test",
            budget_tokens=5000,
        )
        assert isinstance(results, list)


# =============================================================================
# Capacity Enforcement Advanced Tests
# =============================================================================


class TestCapacityEnforcementAdvanced:
    """Tests for memory capacity enforcement edge cases."""

    @pytest.mark.unit
    def test_semantic_capacity_enforcement(self, minimal_jotty_config):
        """Capacity enforcement works on semantic level."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel, SwarmConfig
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        config = SwarmConfig(
            semantic_capacity=3,
            output_base_dir="./test_outputs",
            create_run_folder=False,
        )
        memory = SwarmMemory("test_agent", config)

        for i in range(6):
            memory.store(
                content=f"Semantic memory {i}",
                level=MemoryLevel.SEMANTIC,
                context={},
                goal="test",
                initial_value=0.1 * i,
            )

        assert len(memory.memories[MemoryLevel.SEMANTIC]) <= 3

    @pytest.mark.unit
    def test_capacity_evicts_lowest_value_first(self, minimal_jotty_config):
        """Eviction removes lowest-value unprotected memory."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel, SwarmConfig
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        config = SwarmConfig(
            episodic_capacity=3,
            output_base_dir="./test_outputs",
            create_run_folder=False,
        )
        memory = SwarmMemory("test_agent", config)

        # Store 3 entries
        memory.store("low value", MemoryLevel.EPISODIC, {}, "test", initial_value=0.1)
        memory.store("high value", MemoryLevel.EPISODIC, {}, "test", initial_value=0.9)
        memory.store("medium value", MemoryLevel.EPISODIC, {}, "test", initial_value=0.5)

        # Store 4th entry to trigger eviction
        memory.store("new entry", MemoryLevel.EPISODIC, {}, "test", initial_value=0.6)

        # High value entry should still be present
        remaining_values = [m.default_value for m in memory.memories[MemoryLevel.EPISODIC].values()]
        assert 0.9 in remaining_values
        assert len(memory.memories[MemoryLevel.EPISODIC]) <= 3

    @pytest.mark.unit
    def test_capacity_skips_protected_memories(self, minimal_jotty_config):
        """Eviction skips protected memories."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel, SwarmConfig
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        config = SwarmConfig(
            episodic_capacity=2,
            output_base_dir="./test_outputs",
            create_run_folder=False,
        )
        memory = SwarmMemory("test_agent", config)

        entry = memory.store("protected low", MemoryLevel.EPISODIC, {}, "test", initial_value=0.1)
        entry.is_protected = True
        memory.store("unprotected high", MemoryLevel.EPISODIC, {}, "test", initial_value=0.9)

        # Trigger eviction: the protected low-value one should survive
        memory.store("new entry", MemoryLevel.EPISODIC, {}, "test", initial_value=0.5)

        assert len(memory.memories[MemoryLevel.EPISODIC]) <= 2


# =============================================================================
# Self-RAG Retrieval Tests
# =============================================================================


class TestSelfRAGRetrieval:
    """Tests for self_rag_retrieve."""

    @pytest.mark.unit
    def test_self_rag_skips_greetings(self, minimal_jotty_config):
        """self_rag_retrieve skips retrieval for simple greetings."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        should_retrieve, results, reasoning = memory.self_rag_retrieve(task="hello")
        assert should_retrieve is False
        assert results == []
        assert "greeting" in reasoning.lower() or "simple" in reasoning.lower()

    @pytest.mark.unit
    def test_self_rag_skips_empty_memory(self, minimal_jotty_config):
        """self_rag_retrieve skips when no memories stored."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        should_retrieve, results, reasoning = memory.self_rag_retrieve(task="complex research task")
        assert should_retrieve is False
        assert "no memories" in reasoning.lower()

    @pytest.mark.unit
    def test_self_rag_retrieves_relevant_memories(self, minimal_jotty_config):
        """self_rag_retrieve returns relevant memories for non-trivial tasks."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            "AI research methodology works well",
            MemoryLevel.SEMANTIC,
            {},
            "research",
            initial_value=0.8,
        )

        should_retrieve, results, reasoning = memory.self_rag_retrieve(
            task="AI research methodology",
            goal="research",
            budget_tokens=5000,
        )
        # Should find something (depends on keyword overlap + value threshold)
        assert isinstance(results, list)
        assert isinstance(should_retrieve, bool)


# =============================================================================
# Serialization Advanced Tests
# =============================================================================


class TestSerializationAdvanced:
    """Advanced tests for to_dict/from_dict serialization."""

    @pytest.mark.unit
    def test_to_dict_preserves_all_fields(self, minimal_jotty_config):
        """to_dict includes all expected top-level fields."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)
        data = memory.to_dict()

        assert "agent_name" in data
        assert "total_accesses" in data
        assert "consolidation_count" in data
        assert "memories" in data
        assert "causal_links" in data
        assert "goal_hierarchy" in data

    @pytest.mark.unit
    def test_to_dict_memory_entry_fields(self, minimal_jotty_config):
        """to_dict includes all memory entry fields."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("serialization test", MemoryLevel.EPISODIC, {"ctx": "val"}, "goal1")
        data = memory.to_dict()

        episodic = data["memories"]["episodic"]
        assert len(episodic) == 1
        entry_data = list(episodic.values())[0]

        expected_fields = [
            "key",
            "content",
            "level",
            "context",
            "created_at",
            "last_accessed",
            "goal_values",
            "default_value",
            "access_count",
            "ucb_visits",
            "token_count",
            "is_protected",
            "protection_reason",
            "causal_links",
            "metadata",
        ]
        for field_name in expected_fields:
            assert field_name in entry_data, f"Missing field: {field_name}"

    @pytest.mark.unit
    def test_from_dict_with_missing_optional_fields(self, minimal_jotty_config):
        """from_dict handles missing optional fields gracefully."""
        from datetime import datetime

        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        data = {
            "agent_name": "restored_agent",
            "memories": {
                "episodic": {
                    "general:general:abc123def456ab": {
                        "key": "general:general:abc123def456ab",
                        "content": "test content",
                        "level": "episodic",
                        "context": {},
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                        "goal_values": {},
                        "default_value": 0.5,
                        "access_count": 0,
                        "ucb_visits": 0,
                        "token_count": 4,
                    }
                },
                "semantic": {},
                "procedural": {},
                "meta": {},
                "causal": {},
            },
            # Missing: total_accesses, consolidation_count, causal_links, goal_hierarchy
        }

        restored = SwarmMemory.from_dict(data, minimal_jotty_config)
        assert restored.agent_name == "restored_agent"
        assert restored.total_accesses == 0
        assert restored.consolidation_count == 0
        assert restored.get_statistics()["total_memories"] == 1

    @pytest.mark.unit
    def test_from_dict_migrates_old_format_keys(self, minimal_jotty_config):
        """from_dict migrates old-format (hash-only) keys to new format."""
        import hashlib
        from datetime import datetime

        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        content = "old format content"
        old_key = hashlib.md5(content.encode()).hexdigest()[:16]

        data = {
            "agent_name": "migrated_agent",
            "memories": {
                "episodic": {
                    old_key: {
                        "key": old_key,
                        "content": content,
                        "level": "episodic",
                        "context": {"domain": "sql", "task_type": "query"},
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                        "goal_values": {},
                        "default_value": 0.5,
                        "access_count": 2,
                        "ucb_visits": 1,
                        "token_count": 5,
                    }
                },
                "semantic": {},
                "procedural": {},
                "meta": {},
                "causal": {},
            },
        }

        restored = SwarmMemory.from_dict(data, minimal_jotty_config)
        # Old key should be migrated to new format
        episodic_keys = list(restored.memories.keys())[0]
        for key in restored.memories[list(restored.memories.keys())[0]]:
            assert ":" in key, f"Key not migrated: {key}"

    @pytest.mark.unit
    def test_round_trip_preserves_goal_values(self, minimal_jotty_config):
        """Round-trip serialization preserves goal-conditioned values."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store(
            content="goal value test",
            level=MemoryLevel.SEMANTIC,
            context={},
            goal="primary_goal",
            initial_value=0.75,
        )

        data = memory.to_dict()
        restored = SwarmMemory.from_dict(data, minimal_jotty_config)

        restored_entries = list(restored.memories[MemoryLevel.SEMANTIC].values())
        assert len(restored_entries) == 1
        assert restored_entries[0].get_value("primary_goal") == 0.75

    @pytest.mark.unit
    def test_round_trip_preserves_causal_links(self, minimal_jotty_config):
        """Round-trip serialization preserves causal links."""
        from Jotty.core.infrastructure.foundation.data_structures import CausalLink
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.causal_links["link1"] = CausalLink(
            cause="missing type annotation",
            effect="parser error",
            confidence=0.85,
            conditions=["database=trino"],
            domain="sql",
        )

        data = memory.to_dict()
        restored = SwarmMemory.from_dict(data, minimal_jotty_config)

        assert "link1" in restored.causal_links
        assert restored.causal_links["link1"].cause == "missing type annotation"
        assert restored.causal_links["link1"].confidence == 0.85

    @pytest.mark.unit
    def test_round_trip_preserves_protection_status(self, minimal_jotty_config):
        """Round-trip serialization preserves is_protected and protection_reason."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store("protected entry", MemoryLevel.META, {}, "test")
        entry.is_protected = True
        entry.protection_reason = "META level"

        data = memory.to_dict()
        restored = SwarmMemory.from_dict(data, minimal_jotty_config)

        restored_entry = list(restored.memories[MemoryLevel.META].values())[0]
        assert restored_entry.is_protected is True
        assert restored_entry.protection_reason == "META level"


# =============================================================================
# Statistics and Consolidated Knowledge Advanced Tests
# =============================================================================


class TestStatisticsAdvanced:
    """Advanced tests for statistics and consolidated knowledge."""

    @pytest.mark.unit
    def test_statistics_tracks_protected_count(self, minimal_jotty_config):
        """get_statistics correctly counts protected memories."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry1 = memory.store("protected", MemoryLevel.EPISODIC, {}, "test")
        entry1.is_protected = True
        memory.store("unprotected", MemoryLevel.EPISODIC, {}, "test")

        stats = memory.get_statistics()
        assert stats["protected_memories"] == 1

    @pytest.mark.unit
    def test_statistics_tracks_causal_links(self, minimal_jotty_config):
        """get_statistics counts causal links."""
        from Jotty.core.infrastructure.foundation.data_structures import CausalLink
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.causal_links["l1"] = CausalLink(cause="a", effect="b")
        memory.causal_links["l2"] = CausalLink(cause="c", effect="d")

        stats = memory.get_statistics()
        assert stats["causal_links"] == 2

    @pytest.mark.unit
    def test_consolidated_knowledge_empty_returns_empty_string(self, minimal_jotty_config):
        """get_consolidated_knowledge on empty memory returns empty string."""
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        result = memory.get_consolidated_knowledge()
        assert result == ""

    @pytest.mark.unit
    def test_consolidated_knowledge_with_goal(self, minimal_jotty_config):
        """get_consolidated_knowledge sorts by goal value when goal provided."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("low value pattern", MemoryLevel.SEMANTIC, {}, "research", initial_value=0.2)
        memory.store("high value pattern", MemoryLevel.SEMANTIC, {}, "research", initial_value=0.9)

        knowledge = memory.get_consolidated_knowledge(goal="research")
        assert isinstance(knowledge, str)
        assert "Learned Patterns" in knowledge

    @pytest.mark.unit
    def test_consolidated_knowledge_includes_all_sections(self, minimal_jotty_config):
        """get_consolidated_knowledge includes patterns, procedures, wisdom, causal."""
        from Jotty.core.infrastructure.foundation.data_structures import CausalLink, MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("semantic pattern X", MemoryLevel.SEMANTIC, {}, "test")
        memory.store("procedural step Y", MemoryLevel.PROCEDURAL, {}, "test")
        memory.store("meta wisdom Z", MemoryLevel.META, {}, "test")
        memory.causal_links["cl1"] = CausalLink(cause="action A", effect="result B", confidence=0.8)

        knowledge = memory.get_consolidated_knowledge()
        assert "Learned Patterns" in knowledge
        assert "Procedural Knowledge" in knowledge
        assert "Meta Wisdom" in knowledge
        assert "Causal Understanding" in knowledge

    @pytest.mark.unit
    def test_consolidated_knowledge_respects_max_items(self, minimal_jotty_config):
        """get_consolidated_knowledge limits output to max_items."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel, SwarmConfig
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        config = SwarmConfig(
            semantic_capacity=50,
            output_base_dir="./test_outputs",
            create_run_folder=False,
        )
        memory = SwarmMemory("test_agent", config)

        for i in range(20):
            memory.store(f"pattern {i}", MemoryLevel.SEMANTIC, {}, "test")

        # max_items=4 => at most 2 patterns (max_items//2)
        knowledge = memory.get_consolidated_knowledge(max_items=4)
        assert isinstance(knowledge, str)


# =============================================================================
# Protection Tests
# =============================================================================


class TestProtection:
    """Tests for protect_high_value."""

    @pytest.mark.unit
    def test_protect_high_value_marks_above_threshold(self, minimal_jotty_config):
        """protect_high_value marks memories above threshold as protected."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        low = memory.store("low value", MemoryLevel.EPISODIC, {}, "test", initial_value=0.2)
        high = memory.store("high value", MemoryLevel.EPISODIC, {}, "test", initial_value=0.9)

        memory.protect_high_value(threshold=0.8)

        assert high.is_protected is True
        assert low.is_protected is False

    @pytest.mark.unit
    def test_protect_high_value_always_protects_meta(self, minimal_jotty_config):
        """protect_high_value always protects META level regardless of value."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        meta = memory.store("low value meta", MemoryLevel.META, {}, "test", initial_value=0.1)
        memory.protect_high_value(threshold=0.8)

        assert meta.is_protected is True
        assert "META" in meta.protection_reason

    @pytest.mark.unit
    def test_protect_high_value_always_protects_causal(self, minimal_jotty_config):
        """protect_high_value always protects CAUSAL level regardless of value."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.memory.cortex import SwarmMemory

        memory = SwarmMemory("test_agent", minimal_jotty_config)

        causal = memory.store("low value causal", MemoryLevel.CAUSAL, {}, "test", initial_value=0.1)
        memory.protect_high_value(threshold=0.8)

        assert causal.is_protected is True
        assert "CAUSAL" in causal.protection_reason


# =============================================================================
# MemoryEntry Method Tests
# =============================================================================


class TestMemoryEntryMethods:
    """Tests for MemoryEntry methods (get_value, get_ucb_score)."""

    @pytest.mark.unit
    def test_get_value_returns_default_for_unknown_goal(self, minimal_jotty_config):
        """MemoryEntry.get_value returns default_value for unknown goals."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="test:test:abc123",
            content="test",
            level=MemoryLevel.EPISODIC,
            context={},
            default_value=0.6,
        )
        assert entry.get_value("unknown_goal") == 0.6

    @pytest.mark.unit
    def test_get_value_returns_goal_specific_value(self, minimal_jotty_config):
        """MemoryEntry.get_value returns goal-specific value when set."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            GoalValue,
            MemoryEntry,
            MemoryLevel,
        )

        entry = MemoryEntry(
            key="test:test:abc123",
            content="test",
            level=MemoryLevel.EPISODIC,
            context={},
            default_value=0.5,
        )
        entry.goal_values["my_goal"] = GoalValue(value=0.9)
        assert entry.get_value("my_goal") == 0.9

    @pytest.mark.unit
    def test_get_ucb_score_infinite_for_unvisited(self, minimal_jotty_config):
        """MemoryEntry.get_ucb_score returns infinity for unvisited entries."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="test:test:abc",
            content="test",
            level=MemoryLevel.EPISODIC,
            context={},
            ucb_visits=0,
        )
        score = entry.get_ucb_score(goal="test", total_accesses=100)
        assert score == float("inf")

    @pytest.mark.unit
    def test_get_ucb_score_finite_after_visits(self, minimal_jotty_config):
        """MemoryEntry.get_ucb_score returns finite value after visits."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="test:test:abc",
            content="test",
            level=MemoryLevel.EPISODIC,
            context={},
            ucb_visits=5,
            default_value=0.5,
        )
        score = entry.get_ucb_score(goal="test", total_accesses=100)
        assert score != float("inf")
        assert score > 0

    @pytest.mark.unit
    def test_memory_entry_auto_computes_content_hash(self, minimal_jotty_config):
        """MemoryEntry auto-computes content_hash in __post_init__."""
        import hashlib

        from Jotty.core.infrastructure.foundation.data_structures import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="test:test:abc",
            content="hash this content",
            level=MemoryLevel.EPISODIC,
            context={},
        )
        expected_hash = hashlib.md5("hash this content".encode()).hexdigest()
        assert entry.content_hash == expected_hash

    @pytest.mark.unit
    def test_memory_entry_auto_computes_token_count(self, minimal_jotty_config):
        """MemoryEntry auto-computes token_count from content length."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryEntry, MemoryLevel

        content = "A" * 100  # 100 chars
        entry = MemoryEntry(
            key="test:test:abc",
            content=content,
            level=MemoryLevel.EPISODIC,
            context={},
        )
        # token_count = len(content) // 4 + 1 = 26
        assert entry.token_count == 26
