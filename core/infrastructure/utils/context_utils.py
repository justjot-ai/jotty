"""
Context Utilities — Re-export Shim
===================================

All context utilities now live in ``core.infrastructure.context``.
This module re-exports them for backward compatibility.

Canonical locations:
- Error handling:    ``core.infrastructure.context.error_handling``
- Models:           ``core.infrastructure.context.models``
- Enrichment strip: ``core.infrastructure.context.utils``
- Compression:      ``core.infrastructure.context.utils``
"""

from typing import List, Optional

# ── Error handling ──────────────────────────────────────────────────
from Jotty.core.infrastructure.context.error_handling import (
    CompressionResult,
    ErrorDetector,
    ErrorType,
    detect_error_type,
)

# ── Models ──────────────────────────────────────────────────────────
from Jotty.core.infrastructure.context.models import ExecutionTrajectory

# ── Enrichment stripping ────────────────────────────────────────────
from Jotty.core.infrastructure.context.utils import (
    ENRICHMENT_MARKERS,
    prefix_suffix_compress,
    strip_enrichment_context,
)


# ── ContextCompressor thin wrapper (backward compat) ────────────────
class ContextCompressor:
    """
    Backward-compatible wrapper that delegates to ``prefix_suffix_compress``.

    New code should use ``SmartContextManager.compress_parts()`` or
    ``core.infrastructure.context.utils.prefix_suffix_compress()`` directly.
    """

    def __init__(self, max_compression_retries: int = 3) -> None:
        self.max_compression_retries = max_compression_retries
        self._compression_ratio = 0.7

    def compress(
        self,
        content: str,
        target_ratio: float = 0.5,
        preserve_keywords: Optional[List[str]] = None,
        trajectory: str = "",
    ) -> CompressionResult:
        if not content or not content.strip():
            return CompressionResult(
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                content="",
                preserved_trajectory=trajectory,
            )

        original_length = len(content)
        target_length = int(original_length * target_ratio)
        # Rough char-to-token: 4 chars per token
        target_tokens = target_length // 4

        compressed = prefix_suffix_compress(content, target_tokens)

        if len(compressed) < original_length:
            compressed = "...[earlier context compressed]...\n\n" + compressed

        return CompressionResult(
            original_length=original_length,
            compressed_length=len(compressed),
            compression_ratio=len(compressed) / original_length if original_length > 0 else 1.0,
            content=compressed,
            preserved_trajectory=trajectory,
        )


def create_compressor() -> ContextCompressor:
    """Factory function to create a context compressor."""
    return ContextCompressor()


__all__ = [
    # Error handling
    "ErrorType",
    "CompressionResult",
    "ErrorDetector",
    "detect_error_type",
    # Models
    "ExecutionTrajectory",
    # Enrichment stripping
    "ENRICHMENT_MARKERS",
    "strip_enrichment_context",
    # Backward compat
    "ContextCompressor",
    "create_compressor",
]
