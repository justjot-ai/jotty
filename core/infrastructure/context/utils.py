"""
Shared Context Utilities - DRY Consolidation
=============================================

Extracts duplicated functions from:
- context_manager.py (5 copies of estimate_tokens)
- context_guard.py
- global_context_guard.py
- compressor.py
- content_gate.py

Provides unified token estimation, compression, and chunking utilities.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ..utils.tokenizer import SmartTokenizer

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN ESTIMATION (Unified - Single Source of Truth)
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using SmartTokenizer.

    Unified from 5 duplicate implementations across the context module.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count (0 if text is None or empty)
    """
    if not text:
        return 0
    return SmartTokenizer.get_instance().count_tokens(text)


# =============================================================================
# SIMPLE COMPRESSION STRATEGIES
# =============================================================================


def simple_truncate(text: str, target_tokens: int, chars_per_token: int = 4) -> str:
    """
    Simple truncation with ellipsis.

    Fast fallback when LLM compression isn't available.

    Args:
        text: Text to truncate
        target_tokens: Target token count
        chars_per_token: Character estimation ratio

    Returns:
        Truncated text with ellipsis marker
    """
    target_chars = target_tokens * chars_per_token

    if len(text) <= target_chars:
        return text

    # Truncate with ellipsis
    return text[: target_chars - 50] + "\n... (content truncated to fit context) ..."


def prefix_suffix_compress(text: str, target_tokens: int, chars_per_token: int = 4) -> str:
    """
    Keep start + end, compress middle.

    Better than simple truncation - preserves context from both ends.
    From context_manager._simple_compress.

    Args:
        text: Text to compress
        target_tokens: Target token count
        chars_per_token: Character estimation ratio

    Returns:
        Compressed text preserving start and end
    """
    target_chars = target_tokens * chars_per_token

    if len(text) <= target_chars:
        return text

    keep_start = target_chars // 2
    keep_end = target_chars // 2 - 50

    return text[:keep_start] + "\n[...COMPRESSED...]\n" + text[-keep_end:]


def structured_extract(
    text: str,
    target_tokens: int,
    preserve_keywords: Optional[List[str]] = None,
    chars_per_token: int = 4,
) -> str:
    """
    Heuristic line-based compression preserving important lines.

    From context_guard.compress_structured - keeps lines with critical keywords.

    Args:
        text: Text to compress
        target_tokens: Target token count
        preserve_keywords: Keywords that mark important lines
        chars_per_token: Character estimation ratio

    Returns:
        Structured extraction preserving critical lines
    """
    if not preserve_keywords:
        preserve_keywords = ["CRITICAL", "IMPORTANT", "ERROR", "MUST", "NEVER"]

    lines = text.split("\n")
    critical_lines = []
    other_lines = []

    # Separate critical from non-critical lines
    for line in lines:
        is_critical = any(kw.upper() in line.upper() for kw in preserve_keywords)
        if is_critical:
            critical_lines.append(line)
        else:
            other_lines.append(line)

    # Always include critical lines
    critical_text = "\n".join(critical_lines)
    critical_tokens = estimate_tokens(critical_text)

    # Calculate remaining budget for other lines
    remaining_tokens = target_tokens - critical_tokens

    if remaining_tokens <= 0:
        # Critical lines already exceed budget - truncate them
        return simple_truncate(critical_text, target_tokens, chars_per_token)

    # Add other lines until budget exhausted
    other_text = "\n".join(other_lines)
    other_tokens = estimate_tokens(other_text)

    if other_tokens <= remaining_tokens:
        # Everything fits
        return critical_text + "\n" + other_text

    # Truncate other lines to fit
    truncated_other = simple_truncate(other_text, remaining_tokens, chars_per_token)
    return critical_text + "\n" + truncated_other


# =============================================================================
# INTELLIGENT COMPRESSION (LLM-based)
# =============================================================================


async def intelligent_compress(
    text: str,
    target_tokens: int,
    task_context: Optional[Dict[str, Any]] = None,
    preserve_keywords: Optional[List[str]] = None,
    shapley_credits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    LLM-powered intelligent compression.

    Best compression quality - understands content and preserves important parts.
    From compressor.AgenticCompressor with Shapley impact prioritization.

    Args:
        text: Text to compress
        target_tokens: Target token count
        task_context: Optional context about the task
        preserve_keywords: Keywords to preserve
        shapley_credits: Optional Shapley impact scores for prioritization

    Returns:
        Dict with:
            - compressed_content: Compressed text
            - compression_ratio: Ratio achieved
            - quality_score: Self-assessed quality (0-10)
            - what_was_removed: Summary of removed content
    """
    if not DSPY_AVAILABLE:
        logger.debug("DSPy not available, falling back to structured extraction")
        compressed = structured_extract(text, target_tokens, preserve_keywords)
        return {
            "compressed_content": compressed,
            "compression_ratio": len(compressed) / max(len(text), 1),
            "quality_score": 7.0,  # Estimated
            "what_was_removed": "Used structured extraction (DSPy not available)",
        }

    # Build DSPy signature for compression
    class CompressionSignature(dspy.Signature):
        """Intelligently compress content while preserving critical information."""

        full_content = dspy.InputField(desc="Full content to compress")
        task_description = dspy.InputField(desc="What the agent needs to do")
        target_tokens = dspy.InputField(desc="Target token count")
        priority_keywords = dspy.InputField(desc="Keywords that MUST be preserved")
        high_impact_items = dspy.InputField(desc="High Shapley credit items (preserve first)")
        low_impact_items = dspy.InputField(desc="Low Shapley credit items (can remove)")

        compressed_content = dspy.OutputField(desc="Compressed content")
        compression_ratio = dspy.OutputField(desc="Percentage retained")
        what_was_removed = dspy.OutputField(desc="What was removed")
        quality_score = dspy.OutputField(desc="Quality score 0-10")

    # Build inputs
    if not task_context:
        task_context = {}

    priority_keywords = ", ".join(preserve_keywords or [])
    if not priority_keywords:
        priority_keywords = "No specific keywords - preserve important information"

    # Extract high/low impact items from Shapley credits
    high_impact_items = "None specified"
    low_impact_items = "None specified"

    if shapley_credits:
        sorted_credits = sorted(shapley_credits.items(), key=lambda x: x[1], reverse=True)
        top_n = max(1, len(sorted_credits) // 5)
        high_impact = [item for item, credit in sorted_credits[:top_n]]
        low_impact = [item for item, credit in sorted_credits[-top_n:] if credit < 0.3]

        high_impact_items = ", ".join(high_impact) if high_impact else "None specified"
        low_impact_items = ", ".join(low_impact) if low_impact else "None specified"

    # Call LLM compressor
    try:
        compressor = dspy.ChainOfThought(CompressionSignature)
        result = compressor(
            full_content=text,
            task_description=task_context.get("goal", "Process this content"),
            target_tokens=str(target_tokens),
            priority_keywords=priority_keywords,
            high_impact_items=high_impact_items,
            low_impact_items=low_impact_items,
        )

        return {
            "compressed_content": result.compressed_content,
            "compression_ratio": result.compression_ratio,
            "quality_score": result.quality_score,
            "what_was_removed": result.what_was_removed,
        }

    except Exception as e:
        logger.warning(f"LLM compression failed: {e}, falling back to structured extraction")
        compressed = structured_extract(text, target_tokens, preserve_keywords)
        return {
            "compressed_content": compressed,
            "compression_ratio": len(compressed) / max(len(text), 1),
            "quality_score": 6.0,
            "what_was_removed": f"Fallback extraction (LLM failed: {e})",
        }


# =============================================================================
# CHUNKING UTILITIES
# =============================================================================


def create_chunks(
    content: str,
    max_chunk_tokens: int = 4000,
    overlap_tokens: int = 200,
    preserve_sentences: bool = True,
) -> List[str]:
    """
    Split content into chunks with optional sentence boundary preservation.

    Unified from chunker.py, content_gate.py, context_guard.py.

    Args:
        content: Content to chunk
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks for context
        preserve_sentences: Try to split at sentence boundaries

    Returns:
        List of content chunks
    """
    # Estimate characters per chunk
    chars_per_token = 4
    max_chunk_chars = max_chunk_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token

    if len(content) <= max_chunk_chars:
        return [content]

    chunks = []
    start = 0

    while start < len(content):
        end = start + max_chunk_chars

        # Try to end at sentence boundary if requested
        if preserve_sentences and end < len(content):
            # Look for sentence endings in the last 20% of the chunk
            search_start = end - int(max_chunk_chars * 0.2)
            search_region = content[search_start:end]

            # Find last sentence ending
            sentence_ends = [".", "!", "?", "\n\n"]
            last_end = -1

            for ending in sentence_ends:
                pos = search_region.rfind(ending)
                if pos > last_end:
                    last_end = pos

            if last_end > 0:
                # Adjust end to sentence boundary
                end = search_start + last_end + 1

        chunk = content[start:end]
        chunks.append(chunk)

        # Move start forward with overlap
        start = end - overlap_chars
        if start >= len(content):
            break

    logger.debug(f"Created {len(chunks)} chunks from {len(content)} chars")
    return chunks


# =============================================================================
# OVERFLOW ERROR DETECTION
# =============================================================================


def detect_overflow_error(error: Exception, max_tokens: int = 28000) -> bool:
    """
    Detect if an exception is a context overflow error.

    Simplified from global_context_guard.OverflowDetector.
    Uses structural detection, not hardcoded string matching.

    Args:
        error: Exception to check
        max_tokens: Maximum token limit

    Returns:
        True if this is a context overflow error
    """
    # Structural indicators in class/type names
    overflow_indicators = {
        "overflow",
        "length",
        "size",
        "limit",
        "exhausted",
        "context",
        "token",
        "sequence",
        "long",
        "exceed",
        "capacity",
        "window",
        "memory",
        "oom",
    }

    # Method 1: Check error type hierarchy
    error_types = [cls.__name__.lower() for cls in type(error).__mro__]
    for type_name in error_types:
        if any(indicator in type_name for indicator in overflow_indicators):
            return True

    # Method 2: Numeric extraction
    error_str = str(error)
    numbers = re.findall(r"\d+", error_str)
    for num_str in numbers:
        try:
            num = int(num_str)
            if num > max_tokens and num < 1000000:
                return True
        except ValueError:
            continue

    # Method 3: Check error attributes
    for attr in ["code", "error_code", "status_code", "message", "type"]:
        if hasattr(error, attr):
            value = str(getattr(error, attr, "")).lower()
            if any(indicator in value for indicator in overflow_indicators):
                return True

    return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "estimate_tokens",
    "simple_truncate",
    "prefix_suffix_compress",
    "structured_extract",
    "intelligent_compress",
    "create_chunks",
    "detect_overflow_error",
]
