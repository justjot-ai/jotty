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

import threading
from typing import Optional, Union, Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Jotty.core.foundation.data_structures import SwarmLearningConfig
    from Jotty.core.foundation.configs import MemoryConfig
    from Jotty.core.memory.memory_system import MemorySystem
    from Jotty.core.memory.memory_orchestrator import BrainInspiredMemoryManager
    from Jotty.core.memory.consolidation_engine import SharpWaveRippleConsolidator
    from Jotty.core.memory.llm_rag import LLMRAGRetriever

_lock = threading.Lock()
_singletons: Dict[str, object] = {}


def _resolve_memory_config(config: Any) -> 'SwarmConfig':
    """Convert MemoryConfig or SwarmConfig to SwarmConfig for internal use.

    Accepts:
        - None → default SwarmConfig
        - MemoryConfig → SwarmConfig.from_configs(memory=config)
        - SwarmConfig → pass through
    """
    if config is None:
        from Jotty.core.foundation.data_structures import SwarmLearningConfig
        return SwarmConfig()

    from Jotty.core.foundation.configs.memory import MemoryConfig
    if isinstance(config, MemoryConfig):
        from Jotty.core.foundation.data_structures import SwarmLearningConfig
        return SwarmConfig.from_configs(memory=config)

    # Assume SwarmConfig
    return config


def get_memory_system() -> 'MemorySystem':
    """
    Return a MemorySystem singleton (recommended entry point).

    Thread-safe with double-checked locking.

    Returns:
        MemorySystem instance with auto-detected backend.
    """
    key = 'memory_system'
    if key not in _singletons:
        with _lock:
            if key not in _singletons:
                from Jotty.core.memory.memory_system import MemorySystem
                _singletons[key] = MemorySystem()
    return _singletons[key]


def get_brain_manager() -> 'BrainInspiredMemoryManager':
    """
    Return a BrainInspiredMemoryManager singleton for hierarchical memory management.

    Thread-safe with double-checked locking.

    Returns:
        BrainInspiredMemoryManager instance.
    """
    key = 'brain_manager'
    if key not in _singletons:
        with _lock:
            if key not in _singletons:
                from Jotty.core.memory.memory_orchestrator import BrainInspiredMemoryManager
                _singletons[key] = BrainInspiredMemoryManager()
    return _singletons[key]


def get_consolidator(config: Any = None) -> 'SharpWaveRippleConsolidator':
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


def get_rag_retriever(config: Optional[Union['MemoryConfig', 'SwarmConfig']] = None) -> 'LLMRAGRetriever':
    """
    Return an LLMRAGRetriever for LLM-powered retrieval-augmented generation.

    Args:
        config: Optional MemoryConfig or SwarmConfig. If None, uses defaults.

    Returns:
        LLMRAGRetriever instance.
    """
    from Jotty.core.memory.llm_rag import LLMRAGRetriever
    resolved = _resolve_memory_config(config)
    return LLMRAGRetriever(config=resolved)


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
