"""
Memory Layer - Hierarchical Memory Systems
==========================================

Brain-inspired memory with 5 levels and consolidation.

Modules:
--------
- cortex: 5-level hierarchical memory (EPISODIC → SEMANTIC → PROCEDURAL → META → CAUSAL)
- consolidation_engine: Hippocampal extraction, sharp-wave ripple consolidation
- memory_orchestrator: Unified memory API (SimpleBrain + BrainInspiredMemoryManager)
- llm_rag: LLM-based retrieval, deduplication, causal extraction
"""

from .memory_orchestrator import (
    # SimpleBrain API
    BrainPreset,
    ConsolidationTrigger,
    Experience,
    SimpleBrain,
    calculate_chunk_size,
    get_model_context,
    load_brain_config,
    # BrainInspiredMemoryManager
    BrainInspiredMemoryManager,
    EpisodicMemory,
    SemanticPattern,
)
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
from .cortex import (
    HierarchicalMemory,
    MemoryCluster,
    MemoryLevelClassificationSignature,
    MemoryLevelClassifier,
    MetaWisdomSignature,
    PatternExtractionSignature,
    ProceduralExtractionSignature,
)
from .llm_rag import (
    CausalExtractor,
    DeduplicationEngine,
    LLMRAGRetriever,
    LLMRelevanceScorer,
    RecencyValueRanker,
    SlidingWindowChunker,
)

__all__ = [
    # memory_orchestrator
    'SimpleBrain',
    'BrainPreset',
    'ConsolidationTrigger',
    'Experience',
    'calculate_chunk_size',
    'get_model_context',
    'load_brain_config',
    'BrainInspiredMemoryManager',
    'EpisodicMemory',
    'SemanticPattern',
    # consolidation_engine
    'AgentAbstractor',
    'AgentRole',
    'BrainMode',
    'BrainModeConfig',
    'BrainStateMachine',
    'ConsolidationResult',
    'HippocampalExtractor',
    'MemoryCandidate',
    'SharpWaveRippleConsolidator',
    # cortex
    'HierarchicalMemory',
    'MemoryCluster',
    'MemoryLevelClassificationSignature',
    'MemoryLevelClassifier',
    'MetaWisdomSignature',
    'PatternExtractionSignature',
    'ProceduralExtractionSignature',
    # llm_rag
    'CausalExtractor',
    'DeduplicationEngine',
    'LLMRAGRetriever',
    'LLMRelevanceScorer',
    'RecencyValueRanker',
    'SlidingWindowChunker',
]
