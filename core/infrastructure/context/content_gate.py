"""
Jotty v7.6 - Content Gate
==========================

A-Team Approved: Transparent chunking layer.

Every piece of content passes through ContentGate:
1. Estimates tokens
2. Chunks if needed (automatic)
3. Extracts relevant information (conditional intelligence)
4. Considers future task requirements

Key Insight: Chunking should be TRANSPARENT and UNIVERSAL.
Not a separate tool called manually.
"""

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.tokenizer import SmartTokenizer
from . import utils as ctx_utils
from .models import ContextChunk, ContextPriority, ProcessedContent

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# RELEVANCE ESTIMATOR
# =============================================================================


class RelevanceSignature(dspy.Signature):
    """
    Estimate how relevant a text chunk is to a query and future tasks.

    Consider:
    - Direct relevance to current query
    - Relevance to upcoming tasks
    - Unique information not found elsewhere
    """

    chunk_content = dspy.InputField(desc="Content of the chunk")
    query = dspy.InputField(desc="Current query/task")
    future_tasks = dspy.InputField(desc="Upcoming tasks to consider")
    chunk_index = dspy.InputField(desc="Position in document (start, middle, end)")

    relevance_score = dspy.OutputField(desc="Relevance 0.0-1.0")
    key_information = dspy.OutputField(desc="Most important info from this chunk")
    reasoning = dspy.OutputField(desc="Why this is/isn't relevant")


class RelevanceEstimator:
    """
    Estimate chunk relevance using LLM.

    Considers:
    - Current query
    - Future tasks (if provided)
    - Chunk position (start/end often more important)
    """

    def __init__(self) -> None:
        if DSPY_AVAILABLE:
            self.estimator = dspy.ChainOfThought(RelevanceSignature)
        else:
            self.estimator = None

    async def estimate_relevance(
        self, chunk: ContextChunk, query: str, future_tasks: List[str] = None
    ) -> Tuple[float, str]:
        """
        Estimate relevance and extract key info.

        Returns: (relevance_score, extracted_info)
        """
        # Position-based baseline
        position_weight = self._position_weight(chunk.index, chunk.total_chunks)

        if not self.estimator:
            # Fallback: keyword overlap
            relevance = self._keyword_overlap(chunk.content, query)
            return (relevance * position_weight, "")

        try:
            position_desc = self._describe_position(chunk.index, chunk.total_chunks)

            result = self.estimator(
                chunk_content=chunk.content,  # Limit for estimation
                query=query,
                future_tasks=json.dumps(future_tasks or []),
                chunk_index=position_desc,
            )

            score = self._parse_score(result.relevance_score)
            key_info = result.key_information or ""

            return (score * position_weight, key_info)

        except Exception as e:
            logger.debug(f"Relevance estimation failed: {e}")
            relevance = self._keyword_overlap(chunk.content, query)
            return (relevance * position_weight, "")

    def _position_weight(self, index: int, total: int) -> float:
        """Weight based on position (start/end are often more important)."""
        if total <= 1:
            return 1.0

        # U-shaped curve: start and end are more important
        relative_pos = index / (total - 1)  # 0 to 1

        # Peak at 0 and 1, minimum at 0.5
        u_curve = 1.0 - 0.3 * math.sin(math.pi * relative_pos)

        return u_curve

    def _describe_position(self, index: int, total: int) -> str:
        """Describe chunk position."""
        if total == 1:
            return "only_chunk"
        if index == 0:
            return "start"
        if index == total - 1:
            return "end"
        if index < total / 3:
            return "early"
        if index > 2 * total / 3:
            return "late"
        return "middle"

    def _keyword_overlap(self, chunk: str, query: str) -> float:
        """Simple keyword overlap for fallback."""
        chunk_words = set(chunk.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5

        overlap = len(chunk_words & query_words)
        return min(1.0, overlap / len(query_words))

    def _parse_score(self, score_str: str) -> float:
        """Parse score from LLM output."""
        if isinstance(score_str, (int, float)):
            return float(score_str)

        try:
            return float(score_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Score parsing failed: {e}")
            numbers = re.findall(r"[\d.]+", str(score_str))
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
            return 0.5


# =============================================================================
# CONTENT GATE
# =============================================================================


class ContentGate:
    """
    Transparent layer for content processing.

    ALL content should pass through here:
    - Estimates tokens
    - Auto-chunks if needed
    - Extracts relevant info
    - Considers future tasks

    Usage:
        gate = ContentGate(max_tokens=28000)
        processed = await gate.process(large_document, query, future_tasks)
    """

    def __init__(
        self, max_tokens: int = 28000, chunk_overlap: int = 200, relevance_threshold: float = 0.3
    ) -> None:
        self.max_tokens = max_tokens
        # Reserve 40% of tokens for this content (rest for system, output, etc.)
        self.usable_tokens = int(max_tokens * 0.4)
        self.chunk_size = self.usable_tokens // 4  # Each chunk is 10% of total
        self.chunk_overlap = chunk_overlap
        self.relevance_threshold = relevance_threshold

        self.relevance_estimator = RelevanceEstimator()
        self._tokenizer = SmartTokenizer.get_instance()

        # Statistics
        self.total_processed = 0
        self.chunked_count = 0

        logger.info(
            f" ContentGate initialized (max_tokens={max_tokens}, " f"chunk_size={self.chunk_size})"
        )

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count using shared utility."""
        return ctx_utils.estimate_tokens(content)

    async def process(
        self, content: str, query: str, future_tasks: List[str] = None
    ) -> ProcessedContent:
        """
        Process content through the gate.

        Automatically chunks and extracts if content is too large.
        """
        self.total_processed += 1
        original_tokens = self.estimate_tokens(content)

        # If fits easily, return as-is
        if original_tokens <= self.usable_tokens:
            return ProcessedContent(
                content=content,
                was_chunked=False,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                chunks_used=1,
                chunks_total=1,
            )

        # Need to chunk and extract
        self.chunked_count += 1
        chunks = self._create_chunks(content)

        # Score and extract from each chunk
        extracted_parts = []
        for chunk in chunks:
            score, key_info = await self.relevance_estimator.estimate_relevance(
                chunk, query, future_tasks
            )
            chunk.relevance_score = score
            chunk.extracted_info = key_info

            if score >= self.relevance_threshold:
                # Include this chunk's key info
                if key_info:
                    extracted_parts.append(
                        f"[From chunk {chunk.index + 1}/{chunk.total_chunks}] {key_info}"
                    )
                else:
                    # No extraction available, include summary
                    extracted_parts.append(
                        f"[Chunk {chunk.index + 1}/{chunk.total_chunks}] " f"{chunk.content}..."
                    )

        # Combine extracted parts
        if extracted_parts:
            final_content = "\n\n".join(extracted_parts)
        else:
            # No relevant chunks found - keep first and last
            final_content = (
                f"[Start] {chunks[0].content}\n\n"
                f"...[{len(chunks) - 2} chunks omitted]...\n\n"
                f"[End] {chunks[-1].content}"
            )

        final_tokens = self.estimate_tokens(final_content)

        return ProcessedContent(
            content=final_content,
            was_chunked=True,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            chunks_used=len(extracted_parts),
            chunks_total=len(chunks),
        )

    def _create_chunks(self, content: str) -> List[ContextChunk]:
        """Split content into overlapping chunks."""
        # Use ~4 chars per token as approximation for chunking boundaries
        chars_per_token = 4
        chunk_chars = self.chunk_size * chars_per_token
        overlap_chars = self.chunk_overlap * chars_per_token

        chunks = []
        pos = 0

        while pos < len(content):
            end = min(pos + chunk_chars, len(content))
            chunk_content = content[pos:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_sentence = chunk_content.rfind(". ")
                if last_sentence > chunk_chars * 0.7:  # At least 70% of chunk
                    chunk_content = chunk_content[: last_sentence + 1]
                    end = pos + last_sentence + 1

            chunks.append(chunk_content)
            pos = end - overlap_chars

        # Convert to ContextChunk objects
        return [
            ContextChunk(
                content=c,
                priority=ContextPriority.MEDIUM,  # Default priority for content chunks
                category="content",  # Category for content gate chunks
                index=i,
                total_chunks=len(chunks),
            )
            for i, c in enumerate(chunks)
        ]

    def process_sync(
        self, content: str, query: str, future_tasks: List[str] = None
    ) -> ProcessedContent:
        """Synchronous version of process."""
        try:
            asyncio.get_running_loop()
            # Already in async context — run in a new thread to avoid
            # "this event loop is already running" errors
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.process(content, query, future_tasks)).result()
        except RuntimeError:
            return asyncio.run(self.process(content, query, future_tasks))

    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            "total_processed": self.total_processed,
            "chunked_count": self.chunked_count,
            "chunk_rate": self.chunked_count / max(self.total_processed, 1),
            "max_tokens": self.max_tokens,
            "usable_tokens": self.usable_tokens,
            "chunk_size": self.chunk_size,
        }


# =============================================================================
# DECORATOR FOR AUTOMATIC GATING
# =============================================================================


def with_content_gate(max_tokens: int = 28000) -> Any:
    """
    Decorator to automatically process large content arguments.

    Usage:
        @with_content_gate(max_tokens=28000)
        async def my_agent(query: str, context: str):
            # context is automatically chunked if too large
            ...
    """
    gate = ContentGate(max_tokens=max_tokens)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find and process large string arguments
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str) and gate.estimate_tokens(value) > gate.usable_tokens:
                    # Get query from kwargs or first positional arg
                    query = kwargs.get("query", kwargs.get("task", str(args[0]) if args else ""))
                    processed = await gate.process(value, query)
                    new_kwargs[key] = processed.content
                else:
                    new_kwargs[key] = value

            return await func(*args, **new_kwargs)

        return wrapper

    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

import functools

__all__ = [
    "ContextChunk",
    "ProcessedContent",
    "RelevanceEstimator",
    "ContentGate",
    "with_content_gate",
]
