"""
Context Compressor

Compresses context between steps to reduce size and improve performance.
Based on OAgents context management principles.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CompressionResult:
    """Result of context compression."""

    original_context: str
    compressed_context: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_context": self.original_context,
            "compressed_context": self.compressed_context,
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "compression_ratio": self.compression_ratio,
            "method": self.method,
        }


class ContextCompressor:
    """
    Compresses context to reduce size while preserving important information.

    Strategies:
    - Summarization (LLM-based)
    - Truncation (keep most recent)
    - Key point extraction
    - Semantic compression
    """

    def __init__(self, max_length: int = 1000, strategy: str = "truncate") -> None:
        """
        Initialize context compressor.

        Args:
            max_length: Maximum context length
            strategy: Compression strategy ("truncate", "summarize", "key_points")
        """
        self.max_length = max_length
        self.strategy = strategy

    def compress(self, context: str, llm: Optional[Any] = None) -> CompressionResult:
        """
        Compress context.

        Args:
            context: Context to compress
            llm: Optional LLM for summarization

        Returns:
            CompressionResult
        """
        original_length = len(context)

        if original_length <= self.max_length:
            # No compression needed
            return CompressionResult(
                original_context=context,
                compressed_context=context,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                method="none",
            )

        if self.strategy == "truncate":
            compressed = self._truncate(context)
            method = "truncate"
        elif self.strategy == "summarize" and llm:
            compressed = self._summarize(context, llm)
            method = "summarize"
        elif self.strategy == "key_points":
            compressed = self._extract_key_points(context)
            method = "key_points"
        else:
            # Fallback to truncate
            compressed = self._truncate(context)
            method = "truncate"

        compressed_length = len(compressed)
        compression_ratio = compressed_length / original_length if original_length > 0 else 1.0

        return CompressionResult(
            original_context=context,
            compressed_context=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            method=method,
        )

    def _truncate(self, context: str) -> str:
        """Truncate context, keeping most recent AND important parts."""
        # Split by lines or sentences
        lines = context.split("\n")

        # Strategy: Keep both beginning (important context) and end (most recent)
        # If context is too long, keep first 30% and last 70%

        if len(context) <= self.max_length:
            return context

        # Try to keep last N lines that fit
        compressed_lines = []
        current_length = 0

        for line in reversed(lines):
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length <= self.max_length:
                compressed_lines.insert(0, line)
                current_length += line_length
            else:
                break

        compressed = "\n".join(compressed_lines)

        # If still too long, use smart truncation
        if len(compressed) > self.max_length:
            # Keep the END (most recent) - this preserves the latest step output
            # The end is more important for multi-step tasks
            compressed = compressed[-self.max_length :]

            # Add indicator that beginning was truncated
            if not compressed.startswith("..."):
                compressed = "..." + compressed

        return compressed

    def _extract_key_points(self, context: str) -> str:
        """Extract key points from context."""
        # Simple heuristic: extract sentences with keywords
        sentences = re.split(r"[.!?]\s+", context)

        # Keywords that indicate important information
        important_keywords = [
            "important",
            "key",
            "main",
            "critical",
            "essential",
            "result",
            "conclusion",
            "summary",
            "note",
            "remember",
        ]

        key_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in important_keywords):
                key_sentences.append(sentence)

        # If no key sentences found, use first and last sentences
        if not key_sentences and sentences:
            key_sentences = [sentences[0]]
            if len(sentences) > 1:
                key_sentences.append(sentences[-1])

        compressed = ". ".join(key_sentences)

        # Truncate if still too long
        if len(compressed) > self.max_length:
            compressed = compressed[: self.max_length - 3] + "..."

        return compressed

    async def _summarize(self, context: str, llm: Any) -> str:
        """Summarize context using LLM."""
        import asyncio

        prompt = f"""Summarize this context in {self.max_length // 4} characters or less, preserving key information:

{context}

Summary:"""

        try:
            response = await asyncio.to_thread(llm.generate, prompt, max_tokens=200)
            summary = response.text if hasattr(response, "text") else str(response)

            # Ensure it fits
            if len(summary) > self.max_length:
                summary = summary[: self.max_length - 3] + "..."

            return summary
        except Exception:
            # Fallback to truncate
            return self._truncate(context)


class ContextManager:
    """
    Manages context across multiple steps with compression.

    Features:
    - Automatic compression when context grows
    - Configurable compression strategy
    - Context summarization
    """

    def __init__(
        self,
        max_length: int = 2000,
        compression_strategy: str = "truncate",
        compress_threshold: float = 0.8,
    ) -> None:
        """
        Initialize context manager.

        Args:
            max_length: Maximum context length
            compression_strategy: Strategy to use
            compress_threshold: When to compress (0.0-1.0)
        """
        self.max_length = max_length
        self.compressor = ContextCompressor(max_length, compression_strategy)
        self.compress_threshold = compress_threshold
        self.context_history: List[str] = []

    def add_step(self, step_output: str, llm: Optional[Any] = None) -> str:
        """
        Add a step output to context.

        Args:
            step_output: Output from this step
            llm: Optional LLM for summarization

        Returns:
            Updated context (may be compressed)
        """
        # Add to history
        self.context_history.append(step_output)

        # Build current context
        current_context = "\n\n".join(self.context_history)

        # Check if compression needed
        if len(current_context) > self.max_length * self.compress_threshold:
            # Compress
            compression_result = self.compressor.compress(current_context, llm)
            current_context = compression_result.compressed_context

            # Update history with compressed version
            # Keep recent steps uncompressed, compress older ones
            recent_steps = self.context_history[-2:]  # Keep last 2 steps
            compressed_older = compression_result.compressed_context

            # Rebuild history
            if len(recent_steps) < len(self.context_history):
                # Some steps were compressed
                self.context_history = [compressed_older] + recent_steps

        return current_context

    def get_context(self) -> str:
        """Get current context."""
        return "\n\n".join(self.context_history)

    def clear(self) -> None:
        """Clear context."""
        self.context_history = []
