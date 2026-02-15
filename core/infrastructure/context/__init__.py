"""
Context Layer - Context Management & Protection
===============================================

Token management, auto-chunking, compression, and context overflow prevention.

Unified Architecture:
--------------------
- models: Unified data structures (ContextChunk, ContextPriority, configs)
- utils: Shared utilities (token estimation, compression, chunking)
- context_manager: Smart context coordination
- global_context_guard: Global context protection
- content_gate: Content filtering & relevance
- chunker: LLM-based semantic chunking
- compressor: LLM-based compression with Shapley credits
- context_gradient: Context-as-gradient learning
"""

# Import unified models (single source of truth)
from .models import (
    ContextChunk,
    ContextPriority,
    ProcessedContent,
    ContextOverflowInfo,
    CompressionConfig,
    ChunkingConfig,
)

# Import shared utilities
from . import utils as context_utils

# Import specific components
from .chunker import (
    ContextChunker,
    ChunkingSignature,
    CombiningSignature,
)
from .compressor import (
    AgenticCompressor,
    CompressionSignature,
)
from .content_gate import (
    ContentGate,
    RelevanceEstimator,
    RelevanceSignature,
    with_content_gate,
)
from .context_gradient import (
    ContextApplier,
    ContextGradient,
    ContextUpdate,
    CooperationGradientSignature,
    MemoryGradientSignature,
)
from .context_manager import (
    SmartContextManager,
    OverflowDetector,
    with_smart_context,
    patch_dspy_with_guard,
    unpatch_dspy,
)

from .facade import (
    get_context_manager,
    get_context_guard,
    get_content_gate,
)

__all__ = [
    # Unified models (no duplicates!)
    'ContextChunk',
    'ContextPriority',
    'ProcessedContent',
    'ContextOverflowInfo',
    'CompressionConfig',
    'ChunkingConfig',
    # Shared utilities
    'context_utils',
    # Facades
    'get_context_manager',
    'get_context_guard',
    'get_content_gate',
    # Chunker
    'ContextChunker',
    'ChunkingSignature',
    'CombiningSignature',
    # Compressor
    'AgenticCompressor',
    'CompressionSignature',
    # Content gate
    'ContentGate',
    'RelevanceEstimator',
    'RelevanceSignature',
    'with_content_gate',
    # Context gradient
    'ContextApplier',
    'ContextGradient',
    'ContextUpdate',
    'CooperationGradientSignature',
    'MemoryGradientSignature',
    # Unified context manager (includes all guard features)
    'SmartContextManager',
    'OverflowDetector',
    'with_smart_context',
    'patch_dspy_with_guard',
    'unpatch_dspy',
]
