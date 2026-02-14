"""
Memory Subsystem Facade
========================

Clean, discoverable API for all memory components.
No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.memory.facade import get_memory_system, list_components

    # Zero-config memory (recommended)
    memory = get_memory_system()
    memory.store("Experience X succeeded", level="episodic")
    results = memory.retrieve("How to handle X?", top_k=5)
"""

from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.foundation.data_structures import SwarmConfig


def get_memory_system():
    """
    Return a MemorySystem instance (recommended entry point).

    Returns:
        MemorySystem instance with auto-detected backend.
    """
    from Jotty.core.memory.memory_system import MemorySystem
    return MemorySystem()


def get_brain_manager():
    """
    Return a BrainInspiredMemoryManager for hierarchical memory management.

    Returns:
        BrainInspiredMemoryManager instance.
    """
    from Jotty.core.memory.memory_orchestrator import BrainInspiredMemoryManager
    return BrainInspiredMemoryManager()


def get_consolidator(config=None):
    """
    Return a SharpWaveRippleConsolidator for brain-inspired memory consolidation.

    Args:
        config: Optional BrainModeConfig. If None, uses defaults.

    Returns:
        SharpWaveRippleConsolidator instance.
    """
    from Jotty.core.memory.consolidation_engine import SharpWaveRippleConsolidator, BrainModeConfig
    if config is None:
        config = BrainModeConfig()
    return SharpWaveRippleConsolidator(config=config)


def get_rag_retriever(config: Optional['SwarmConfig'] = None):
    """
    Return an LLMRAGRetriever for LLM-powered retrieval-augmented generation.

    Args:
        config: Optional SwarmConfig. If None, uses defaults.

    Returns:
        LLMRAGRetriever instance.
    """
    from Jotty.core.memory.llm_rag import LLMRAGRetriever
    if config is None:
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig()
    return LLMRAGRetriever(config=config)


def list_components() -> Dict[str, str]:
    """
    List all memory subsystem components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "MemorySystem": "Unified facade: store/retrieve/consolidate with auto-backend selection",
        "BrainInspiredMemoryManager": "5-level memory hierarchy with brain-inspired consolidation",
        "SharpWaveRippleConsolidator": "Sleep-like consolidation (episodic → semantic → procedural)",
        "LLMRAGRetriever": "LLM-powered retrieval with deduplication and causal extraction",
        "SwarmMemory": "Low-level 5-level storage backend (cortex)",
        "SimpleBrain": "User-friendly memory API with consolidation triggers",
        "HippocampalExtractor": "Extracts key facts from episodic memories",
        "BrainStateMachine": "Manages brain mode transitions (wake/consolidation/sleep)",
        "MongoDBMemoryBackend": "MongoDB-backed persistent memory storage",
        "SlidingWindowChunker": "Chunks large text into overlapping windows for retrieval",
        "RecencyValueRanker": "Ranks memories by recency and value score",
        "DeduplicationEngine": "Removes duplicate or near-duplicate memories",
    }
