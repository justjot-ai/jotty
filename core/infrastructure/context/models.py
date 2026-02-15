"""
Unified Context Models - Best of All Implementations
=====================================================

Consolidates data structures from:
- context_manager.py (ContextChunk, ContextPriority)
- content_gate.py (ContentChunk, ProcessedContent)
- global_context_guard.py (ContextOverflowInfo)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..utils.tokenizer import SmartTokenizer

# =============================================================================
# PRIORITY LEVELS (Unified from context_manager + global_context_guard)
# =============================================================================


class ContextPriority(Enum):
    """
    Priority levels for context preservation during compression.

    Unified from multiple implementations with consistent values (0-3).
    """

    CRITICAL = 0  # NEVER compress: current task, goal, critical memories
    HIGH = 1  # Compress last: recent memories, errors, tool results
    MEDIUM = 2  # Compress when needed: trajectory, history (default)
    LOW = 3  # Compress first: verbose logs, old memories


# =============================================================================
# CONTEXT CHUNKS (Unified from context_manager + content_gate)
# =============================================================================


@dataclass
class ContextChunk:
    """
    A chunk of context with metadata for smart compression and relevance scoring.

    Merges best features from:
    - context_manager.ContextChunk (priority, compression tracking)
    - content_gate.ContentChunk (relevance scoring, chunk indexing)
    """

    content: str
    priority: ContextPriority
    category: str  # "task", "todo", "memory", "trajectory", "tool_result"

    # Token tracking
    tokens: int = 0
    is_compressed: bool = False
    original_tokens: int = 0

    # Relevance tracking (from content_gate)
    relevance_score: float = 0.0
    extracted_info: str = ""

    # Chunk indexing (from content_gate)
    index: int = 0
    total_chunks: int = 1

    def __post_init__(self) -> None:
        """Auto-calculate tokens if not provided."""
        if not self.tokens:
            self.tokens = SmartTokenizer.get_instance().count_tokens(self.content)
        if not self.original_tokens:
            self.original_tokens = self.tokens


@dataclass
class ProcessedContent:
    """
    Result of content processing (from content_gate).

    Tracks chunking and compression results.
    """

    content: str
    was_chunked: bool
    original_tokens: int
    final_tokens: int
    chunks_used: int
    chunks_total: int

    # Compression tracking (NEW)
    was_compressed: bool = False
    compression_ratio: float = 1.0


# =============================================================================
# OVERFLOW DETECTION (from global_context_guard)
# =============================================================================


@dataclass
class ContextOverflowInfo:
    """
    Information about a detected context overflow error.

    From global_context_guard - best structural detection implementation.
    """

    is_overflow: bool
    detected_tokens: Optional[int] = None
    max_allowed: Optional[int] = None
    provider_hint: Optional[str] = None
    detection_method: str = "unknown"


# =============================================================================
# COMPRESSION CONFIGURATION (NEW - unified config)
# =============================================================================


@dataclass
class CompressionConfig:
    """Configuration for compression strategies."""

    # Compression methods
    use_llm_compression: bool = True  # Try LLM-based compression for CRITICAL/HIGH
    use_shapley_credits: bool = False  # Use Shapley impact scores if available

    # Quality thresholds
    min_quality_score: float = 5.0  # Minimum acceptable quality (0-10)
    warn_on_low_quality: bool = True

    # Fallback behavior
    chars_per_token: int = 4  # Character estimation for simple compression
    safety_margin: float = 0.85  # Use only this fraction of max_tokens

    # Retry configuration
    max_retries: int = 3
    compression_per_retry: float = 0.7  # Reduce by 30% each retry


@dataclass
class ChunkingConfig:
    """Configuration for content chunking."""

    # Chunking strategy
    use_semantic_chunking: bool = True  # LLM-powered semantic chunking
    use_relevance_scoring: bool = True  # Score chunks by relevance

    # Chunk size
    max_chunk_tokens: int = 4000
    overlap_tokens: int = 200  # Overlap between chunks

    # Relevance thresholds
    min_relevance_score: float = 0.3  # Skip chunks below this score


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ContextPriority",
    "ContextChunk",
    "ProcessedContent",
    "ContextOverflowInfo",
    "CompressionConfig",
    "ChunkingConfig",
]
