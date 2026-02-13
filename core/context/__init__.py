"""
Context Layer - Context Management & Protection
===============================================

Token management, auto-chunking, compression, and context overflow prevention.

Modules:
--------
- context_manager: Auto-chunking, compression coordinator
- global_context_guard: Global context protection
- context_guard: Context overflow prevention
- context_gradient: Context-as-gradient learning
- content_gate: Content filtering
- chunker: LLM-based chunking
- compressor: LLM-based compression
"""

# Import all context management components
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
    ContentChunk,
    ContentGate,
    ProcessedContent,
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
from .context_guard import (
    LLMContextManager,
)
from .global_context_guard import (
    GlobalContextGuard,
    patch_dspy_with_guard,
    unpatch_dspy,
)
from .context_manager import (
    ContextChunk,
    ContextPriority,
    SmartContextManager,
    with_smart_context,
)

__all__ = [
    # chunker
    'ContextChunker',
    'ChunkingSignature',
    'CombiningSignature',
    # compressor
    'AgenticCompressor',
    'CompressionSignature',
    # content_gate
    'ContentChunk',
    'ContentGate',
    'ProcessedContent',
    'RelevanceEstimator',
    'RelevanceSignature',
    'with_content_gate',
    # context_gradient
    'ContextApplier',
    'ContextGradient',
    'ContextUpdate',
    'CooperationGradientSignature',
    'MemoryGradientSignature',
    # context_guard
    'LLMContextManager',
    # global_context_guard
    'GlobalContextGuard',
    'patch_dspy_with_guard',
    'unpatch_dspy',
    # context_manager
    'ContextChunk',
    'ContextPriority',
    'SmartContextManager',
    'with_smart_context',
]
