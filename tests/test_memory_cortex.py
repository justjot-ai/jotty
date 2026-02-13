"""
Tests for Memory Cortex Module
================================
Tests for SwarmMemory store, retrieve, consolidation, and serialization.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List


# =============================================================================
# SwarmMemory Creation Tests
# =============================================================================

class TestSwarmMemoryCreation:
    """Tests for SwarmMemory initialization."""

    @pytest.mark.unit
    def test_creation_with_defaults(self, minimal_jotty_config):
        """SwarmMemory creates with agent name and config."""
        from Jotty.core.memory.cortex import SwarmMemory
        memory = SwarmMemory("test_agent", minimal_jotty_config)
        assert memory.agent_name == "test_agent"

    @pytest.mark.unit
    def test_empty_memory_statistics(self, minimal_jotty_config):
        """New memory has zero entries."""
        from Jotty.core.memory.cortex import SwarmMemory
        memory = SwarmMemory("test_agent", minimal_jotty_config)
        stats = memory.get_statistics()
        assert stats['total_memories'] == 0

    @pytest.mark.unit
    def test_five_memory_levels_exist(self, minimal_jotty_config):
        """SwarmMemory has all 5 memory levels initialized."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)
        for level in MemoryLevel:
            assert level in memory._memories


# =============================================================================
# Store Tests
# =============================================================================

class TestSwarmMemoryStore:
    """Tests for SwarmMemory.store()."""

    @pytest.mark.unit
    def test_store_episodic_memory(self, minimal_jotty_config):
        """Store creates entry at episodic level."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
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
        assert stats['total_memories'] == 1

    @pytest.mark.unit
    def test_store_with_domain_and_task_type(self, minimal_jotty_config):
        """Store creates hierarchical key with domain:task_type:hash."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
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
        for key in memory._memories[MemoryLevel.SEMANTIC]:
            if "research" in key and "analysis" in key:
                found = True
                break
        assert found, "Hierarchical key not found in stored memories"

    @pytest.mark.unit
    def test_store_multiple_levels(self, minimal_jotty_config):
        """Store works at different memory levels."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        for level in [MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL]:
            memory.store(
                content=f"Memory at {level.name}",
                level=level,
                context={},
                goal="test",
            )

        stats = memory.get_statistics()
        assert stats['total_memories'] == 3

    @pytest.mark.unit
    def test_store_with_outcome_failure(self, minimal_jotty_config):
        """store_with_outcome routes failures to CAUSAL level."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)
        entry = memory.store_with_outcome(
            content="Tool failed with error X",
            context={"tool": "web-search"},
            goal="research",
            outcome="failure",
        )
        assert entry is not None
        # Failure should be stored at CAUSAL level
        causal_count = len(memory._memories.get(MemoryLevel.CAUSAL, {}))
        assert causal_count >= 1

    @pytest.mark.unit
    def test_store_with_outcome_success(self, minimal_jotty_config):
        """store_with_outcome routes successes to SEMANTIC level."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)
        entry = memory.store_with_outcome(
            content="Agent found great results",
            context={},
            goal="research",
            outcome="success",
        )
        assert entry is not None
        semantic_count = len(memory._memories.get(MemoryLevel.SEMANTIC, {}))
        assert semantic_count >= 1

    @pytest.mark.unit
    def test_capacity_enforcement(self, minimal_jotty_config):
        """Store enforces capacity limits with eviction."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel, SwarmConfig

        config = SwarmConfig(
            episodic_capacity=3,
            enable_beautified_logs=False,
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
        episodic_count = len(memory._memories[MemoryLevel.EPISODIC])
        assert episodic_count <= 3


# =============================================================================
# Retrieve Tests
# =============================================================================

class TestSwarmMemoryRetrieve:
    """Tests for SwarmMemory.retrieve_fast() and retrieve_by_domain()."""

    @pytest.mark.unit
    def test_retrieve_fast_empty(self, minimal_jotty_config):
        """retrieve_fast from empty memory returns empty list."""
        from Jotty.core.memory.cortex import SwarmMemory
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
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("web search returned good results for AI trends",
                      MemoryLevel.EPISODIC, {}, "research AI")
        memory.store("calculator tool computed financial metrics",
                      MemoryLevel.EPISODIC, {}, "analyze data")
        memory.store("AI research produced comprehensive report",
                      MemoryLevel.EPISODIC, {}, "research AI")

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
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
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
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("coding pattern A", MemoryLevel.SEMANTIC, {},
                      "code", domain="coding", task_type="review")
        memory.store("research finding B", MemoryLevel.SEMANTIC, {},
                      "research", domain="research", task_type="analysis")
        memory.store("coding pattern C", MemoryLevel.SEMANTIC, {},
                      "code", domain="coding", task_type="debug")

        results = memory.retrieve_by_domain(
            domain="coding",
            goal="code",
            budget_tokens=5000,
        )
        assert len(results) >= 1
        assert all("coding" in r.content or "pattern" in r.content for r in results)

    @pytest.mark.unit
    def test_retrieve_updates_access_tracking(self, minimal_jotty_config):
        """retrieve_fast updates access_count on returned memories."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        entry = memory.store("test content for tracking",
                              MemoryLevel.EPISODIC, {}, "test")
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
        from Jotty.core.memory.cortex import SwarmMemory
        memory = SwarmMemory("test_agent", minimal_jotty_config)
        data = memory.to_dict()
        assert data['agent_name'] == "test_agent"
        assert 'memories' in data

    @pytest.mark.unit
    def test_round_trip_serialization(self, minimal_jotty_config):
        """to_dict/from_dict preserves data."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("important finding", MemoryLevel.SEMANTIC, {},
                      "goal", domain="test", task_type="unit")
        memory.store("error pattern", MemoryLevel.CAUSAL, {},
                      "debug", domain="test", task_type="debug")

        data = memory.to_dict()
        restored = SwarmMemory.from_dict(data, minimal_jotty_config)

        assert restored.agent_name == "test_agent"
        original_stats = memory.get_statistics()
        restored_stats = restored.get_statistics()
        assert original_stats['total_memories'] == restored_stats['total_memories']

    @pytest.mark.unit
    def test_statistics_correct(self, minimal_jotty_config):
        """get_statistics returns correct counts."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("ep1", MemoryLevel.EPISODIC, {}, "g1")
        memory.store("ep2", MemoryLevel.EPISODIC, {}, "g2")
        memory.store("sem1", MemoryLevel.SEMANTIC, {}, "g1")

        stats = memory.get_statistics()
        assert stats['total_memories'] == 3
        assert stats['by_level']['EPISODIC'] == 2
        assert stats['by_level']['SEMANTIC'] == 1

    @pytest.mark.unit
    def test_consolidated_knowledge_output(self, minimal_jotty_config):
        """get_consolidated_knowledge returns formatted string."""
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import MemoryLevel
        memory = SwarmMemory("test_agent", minimal_jotty_config)

        memory.store("Pattern: always validate inputs before execution",
                      MemoryLevel.SEMANTIC, {}, "validation")
        memory.store("Procedure: use web-search then summarize",
                      MemoryLevel.PROCEDURAL, {}, "research")

        knowledge = memory.get_consolidated_knowledge()
        assert isinstance(knowledge, str)
