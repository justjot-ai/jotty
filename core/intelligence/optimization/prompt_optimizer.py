"""
Prompt Optimizer

Optimizes prompts to reduce LLM call time and costs.
Based on OAgents efficiency principles.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original_prompt: str
    optimized_prompt: str
    original_length: int
    optimized_length: int
    reduction_percent: float
    optimizations_applied: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "original_length": self.original_length,
            "optimized_length": self.optimized_length,
            "reduction_percent": self.reduction_percent,
            "optimizations_applied": self.optimizations_applied,
        }


class PromptOptimizer:
    """
    Optimizes prompts to reduce length while maintaining quality.

    Strategies:
    - Remove redundant words
    - Use abbreviations where clear
    - Remove unnecessary context
    - Simplify instructions
    - Use bullet points instead of paragraphs
    """

    def __init__(self, aggressive: bool = False) -> None:
        """
        Initialize prompt optimizer.

        Args:
            aggressive: Use more aggressive optimizations (may reduce clarity)
        """
        self.aggressive = aggressive

    def optimize(self, prompt: str, max_length: Optional[int] = None) -> OptimizationResult:
        """
        Optimize a prompt.

        Args:
            prompt: Original prompt
            max_length: Optional maximum length target

        Returns:
            OptimizationResult
        """
        original_length = len(prompt)
        optimized = prompt
        optimizations_applied = []

        # 1. Remove extra whitespace
        optimized = re.sub(r"\s+", " ", optimized).strip()
        if len(optimized) < original_length:
            optimizations_applied.append("whitespace_removal")

        # 2. Remove redundant phrases
        redundant_patterns = [
            (r"\bplease\s+note\s+that\b", "", "redundant_phrases"),
            (r"\bit\s+is\s+important\s+to\s+note\s+that\b", "Note:", "redundant_phrases"),
            (r"\bI\s+would\s+like\s+you\s+to\b", "", "redundant_phrases"),
            (r"\bcan\s+you\s+please\b", "", "redundant_phrases"),
            (r"\bcould\s+you\s+please\b", "", "redundant_phrases"),
        ]

        for pattern, replacement, name in redundant_patterns:
            new_optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
            if len(new_optimized) < len(optimized):
                optimized = new_optimized
                if name not in optimizations_applied:
                    optimizations_applied.append(name)

        # 3. Simplify instructions
        optimized = self._simplify_instructions(optimized, optimizations_applied)

        # 4. Use abbreviations (if aggressive)
        if self.aggressive:
            optimized = self._apply_abbreviations(optimized, optimizations_applied)

        # 5. Remove unnecessary context (if max_length specified)
        if max_length and len(optimized) > max_length:
            optimized = self._truncate_intelligently(optimized, max_length, optimizations_applied)

        optimized_length = len(optimized)
        reduction_percent = (
            ((original_length - optimized_length) / original_length * 100)
            if original_length > 0
            else 0.0
        )

        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            original_length=original_length,
            optimized_length=optimized_length,
            reduction_percent=reduction_percent,
            optimizations_applied=optimizations_applied,
        )

    def _simplify_instructions(self, prompt: str, optimizations_applied: List[str]) -> str:
        """Simplify instruction phrases."""
        simplifications = [
            (r"\bprovide\s+a\s+detailed\s+explanation\b", "Explain", "simplify_instructions"),
            (r"\bplease\s+provide\b", "Provide", "simplify_instructions"),
            (r"\bmake\s+sure\s+to\b", "", "simplify_instructions"),
            (r"\bensure\s+that\b", "", "simplify_instructions"),
            (r"\bit\s+would\s+be\s+great\s+if\s+you\s+could\b", "", "simplify_instructions"),
        ]

        optimized = prompt
        for pattern, replacement, name in simplifications:
            new_optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
            if len(new_optimized) < len(optimized):
                optimized = new_optimized
                if name not in optimizations_applied:
                    optimizations_applied.append(name)

        return optimized

    def _apply_abbreviations(self, prompt: str, optimizations_applied: List[str]) -> str:
        """Apply common abbreviations."""
        abbreviations = [
            (r"\bfor\s+example\b", "e.g.", "abbreviations"),
            (r"\bthat\s+is\b", "i.e.", "abbreviations"),
            (r"\bwith\s+respect\s+to\b", "wrt", "abbreviations"),
            (r"\bas\s+soon\s+as\s+possible\b", "ASAP", "abbreviations"),
        ]

        optimized = prompt
        for pattern, replacement, name in abbreviations:
            new_optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
            if len(new_optimized) < len(optimized):
                optimized = new_optimized
                if name not in optimizations_applied:
                    optimizations_applied.append(name)

        return optimized

    def _truncate_intelligently(
        self, prompt: str, max_length: int, optimizations_applied: List[str]
    ) -> str:
        """Truncate prompt intelligently (preserve important parts)."""
        if len(prompt) <= max_length:
            return prompt

        # Try to preserve the question/task part
        # Look for question markers and task indicators
        question_markers = ["Current step:", "step:", "Question:", "Task:", "Provide"]

        # Find the main question/task (usually near the end)
        main_part = prompt
        task_start = -1

        for marker in question_markers:
            marker_lower = marker.lower()
            prompt_lower = prompt.lower()
            idx = prompt_lower.rfind(marker_lower)
            if idx != -1:
                # Found marker, keep everything from marker onwards
                task_start = idx
                break

        if task_start != -1:
            # Keep from task marker onwards, plus some context before
            context_before = 200  # Keep 200 chars of context before task
            start_idx = max(0, task_start - context_before)
            main_part = prompt[start_idx:]

            # If main part fits, use it
            if len(main_part) <= max_length:
                optimizations_applied.append("intelligent_truncation")
                return main_part

        # Otherwise, preserve the END (most recent/important part)
        # Don't truncate from end - truncate from START to preserve task
        if len(prompt) > max_length:
            # Keep the last max_length chars (preserves task/question)
            optimizations_applied.append("truncation_preserve_end")
            return prompt[-max_length:]

        optimizations_applied.append("truncation")
        return prompt[: max_length - 3] + "..."


class LLMCache:
    """
    Cache for LLM responses to avoid redundant calls.

    Uses simple hash-based caching.
    """

    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached responses
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, prompt: str) -> Optional[Any]:
        """
        Get cached response.

        Args:
            prompt: Prompt to look up

        Returns:
            Cached response or None
        """
        cache_key = self._hash_prompt(prompt)

        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1
        return None

    def set(self, prompt: str, response: Any) -> None:
        """
        Cache a response.

        Args:
            prompt: Prompt
            response: Response to cache
        """
        cache_key = self._hash_prompt(prompt)

        # Evict oldest if at max size
        if len(self.cache) >= self.max_size:
            # Remove first item (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[cache_key] = response

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash key for prompt."""
        import hashlib

        # Normalize prompt (lowercase, remove extra whitespace)
        normalized = re.sub(r"\s+", " ", prompt.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
