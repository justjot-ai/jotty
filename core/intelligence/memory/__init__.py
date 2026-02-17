from typing import Any

"""
Memory Layer - Hierarchical Memory Systems
==========================================

Brain-inspired memory with 5 levels and consolidation.

RECOMMENDED ENTRY POINT:
    from Jotty.core.intelligence.memory import MemorySystem
    memory = MemorySystem()  # Zero-config, auto-detects best backend
"""

import importlib as _importlib

from .consolidation_engine import (
    AgentAbstractor,
    AgentRole,
    BrainMode,
    BrainModeConfig,
    BrainStateMachine,
    ConsolidationResult,
    HippocampalExtractor,
    MemoryCandidate,
    SharpWaveRippleConsolidator,
)
from .llm_rag import (
    CausalExtractor,
    DeduplicationEngine,
    LLMRAGRetriever,
    LLMRelevanceScorer,
    RecencyValueRanker,
    SlidingWindowChunker,
)

# ── Eager imports (lightweight, no DSPy) ────────────────────────────
from .memory_orchestrator import (
    BrainInspiredMemoryManager,
    BrainPreset,
    ConsolidationTrigger,
    EpisodicMemory,
    Experience,
    SemanticPattern,
    SimpleBrain,
    calculate_chunk_size,
    get_model_context,
    load_brain_config,
)
from .memory_system import MemoryBackend, MemoryConfig, MemoryResult, MemorySystem
from .mongodb_backend import MongoDBMemoryBackend, enable_mongodb_memory

# ── Lazy imports (DSPy-dependent, loaded on first access) ───────────
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # cortex (SwarmMemory itself is lightweight once consolidation is deferred)
    "SwarmMemory": (".cortex", "SwarmMemory"),
    # Consolidation signatures/classes (heavy DSPy, lazy-loaded from consolidation_engine)
    "MemoryCluster": (".consolidation_engine", "MemoryCluster"),
    "MemoryLevelClassificationSignature": (
        ".consolidation_engine",
        "MemoryLevelClassificationSignature",
    ),
    "MemoryLevelClassifier": (".consolidation_engine", "MemoryLevelClassifier"),
    "MetaWisdomSignature": (".consolidation_engine", "MetaWisdomSignature"),
    "PatternExtractionSignature": (".consolidation_engine", "PatternExtractionSignature"),
    "ProceduralExtractionSignature": (".consolidation_engine", "ProceduralExtractionSignature"),
    "ConsolidationValidationSignature": (
        ".consolidation_engine",
        "ConsolidationValidationSignature",
    ),
    "ConsolidationValidator": (".consolidation_engine", "ConsolidationValidator"),
    # JustJot sync adapter (bidirectional Jotty<->JustJot memory sync)
    "JustJotMemorySyncAdapter": (".justjot_sync_adapter", "JustJotMemorySyncAdapter"),
    "JustJotMemory": (".justjot_sync_adapter", "JustJotMemory"),
}


_FACADE_IMPORTS = {
    "get_memory_system",
    "get_brain_manager",
    "get_consolidator",
    "get_rag_retriever",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _FACADE_IMPORTS:
        from . import memory_system as _ms

        value = getattr(_ms, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # memory_system (PREFERRED entry point)
    "MemorySystem",
    "MemoryConfig",
    "MemoryBackend",
    "MemoryResult",
    # memory_orchestrator
    "SimpleBrain",
    "BrainPreset",
    "ConsolidationTrigger",
    "Experience",
    "calculate_chunk_size",
    "get_model_context",
    "load_brain_config",
    "BrainInspiredMemoryManager",
    "EpisodicMemory",
    "SemanticPattern",
    # consolidation_engine
    "AgentAbstractor",
    "AgentRole",
    "BrainMode",
    "BrainModeConfig",
    "BrainStateMachine",
    "ConsolidationResult",
    "HippocampalExtractor",
    "MemoryCandidate",
    "SharpWaveRippleConsolidator",
    # cortex (SwarmMemory)
    "SwarmMemory",
    # consolidation_engine (DSPy, lazy-loaded)
    "MemoryCluster",
    "MemoryLevelClassificationSignature",
    "MemoryLevelClassifier",
    "MetaWisdomSignature",
    "PatternExtractionSignature",
    "ProceduralExtractionSignature",
    "ConsolidationValidationSignature",
    "ConsolidationValidator",
    # llm_rag
    "CausalExtractor",
    "DeduplicationEngine",
    "LLMRAGRetriever",
    "LLMRelevanceScorer",
    "RecencyValueRanker",
    "SlidingWindowChunker",
    # mongodb_backend
    "MongoDBMemoryBackend",
    "enable_mongodb_memory",
    # facade
    "get_memory_system",
    "get_brain_manager",
    "get_consolidator",
    "get_rag_retriever",
    # JustJot sync
    "JustJotMemorySyncAdapter",
    "JustJotMemory",
]
