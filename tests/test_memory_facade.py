"""
Tests for the Memory Subsystem Facade (Phase 2b).

Verifies each memory accessor returns the correct type.
All tests use mocks where needed and run offline.
"""

import pytest


@pytest.mark.unit
class TestMemoryFacade:
    """Tests for memory facade accessor functions."""

    def test_get_memory_system_returns_memory_system(self):
        from Jotty.core.intelligence.memory.facade import get_memory_system
        from Jotty.core.intelligence.memory.memory_system import MemorySystem
        ms = get_memory_system()
        assert isinstance(ms, MemorySystem)

    def test_get_brain_manager_returns_manager(self):
        from Jotty.core.intelligence.memory.facade import get_brain_manager
        from Jotty.core.intelligence.memory.memory_orchestrator import BrainInspiredMemoryManager
        mgr = get_brain_manager()
        assert isinstance(mgr, BrainInspiredMemoryManager)

    def test_get_consolidator_returns_consolidator(self):
        from Jotty.core.intelligence.memory.facade import get_consolidator
        from Jotty.core.intelligence.memory.consolidation_engine import SharpWaveRippleConsolidator
        con = get_consolidator()
        assert isinstance(con, SharpWaveRippleConsolidator)

    def test_get_rag_retriever_returns_retriever(self):
        from Jotty.core.intelligence.memory.facade import get_rag_retriever
        from Jotty.core.intelligence.memory.llm_rag import LLMRAGRetriever
        retriever = get_rag_retriever()
        assert isinstance(retriever, LLMRAGRetriever)

    def test_list_components_returns_dict(self):
        from Jotty.core.intelligence.memory.facade import list_components
        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.intelligence.memory.facade import list_components
        components = list_components()
        expected = [
            "MemorySystem",
            "BrainInspiredMemoryManager",
            "SharpWaveRippleConsolidator",
            "LLMRAGRetriever",
            "SwarmMemory",
        ]
        for name in expected:
            assert name in components, f"Missing component: {name}"

    def test_list_components_values_are_strings(self):
        from Jotty.core.intelligence.memory.facade import list_components
        for name, desc in list_components().items():
            assert isinstance(desc, str)
            assert len(desc) > 0


@pytest.mark.unit
class TestMemoryFacadeFromInit:
    """Test facade functions are accessible from __init__."""

    def test_import_get_memory_system(self):
        from Jotty.core.intelligence.memory import get_memory_system
        assert callable(get_memory_system)

    def test_import_get_brain_manager(self):
        from Jotty.core.intelligence.memory import get_brain_manager
        assert callable(get_brain_manager)

    def test_import_get_consolidator(self):
        from Jotty.core.intelligence.memory import get_consolidator
        assert callable(get_consolidator)

    def test_import_get_rag_retriever(self):
        from Jotty.core.intelligence.memory import get_rag_retriever
        assert callable(get_rag_retriever)
