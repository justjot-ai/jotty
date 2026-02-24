"""
Progressive Summarization Tree
===============================

Maintains a chain of summaries at multiple abstraction levels to avoid
lossy one-shot compression. When context exceeds budget:

    Level 0 (raw):    Full recent messages (preserved verbatim)
    Level 1 (detail): Summarized older blocks (~300 chars each)
    Level 2 (theme):  Merged level-1 summaries into themes (~500 chars)

This ensures recent work is always full-fidelity while older context
degrades gracefully into increasingly abstract summaries rather than
being hard-truncated.

Inspired by OpenClaw community proposal (#22251) for hierarchical
progressive summarization, adapted to Jotty's existing compression
infrastructure.

Usage::

    from Jotty.core.infrastructure.context.progressive_summarizer import (
        ProgressiveSummarizer,
    )

    ps = ProgressiveSummarizer(max_level0_chars=8000)
    ps.add_block("Phase 1: Built the API framework...")
    ps.add_block("Phase 2: Added authentication...")
    ps.add_block("Phase 3: Deployed to staging...")

    # Get context that fits within budget
    context = ps.build_context(budget_chars=4000)
    # Returns: level-2 themes + level-1 details + raw recent blocks

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SummaryBlock:
    """A block of content at any summarization level."""

    content: str
    level: int = 0  # 0=raw, 1=detailed summary, 2=theme summary
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_count: int = 1  # How many raw blocks this represents
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


class ProgressiveSummarizer:
    """Maintains a tree of progressively summarized context blocks.

    Blocks flow through levels:
        add_block() → level-0 (raw)
        When level-0 exceeds budget → oldest promoted to level-1 (summarized)
        When 3+ level-1 blocks exist → merged into level-2 (themes)

    Args:
        max_level0_chars: Max chars to keep at level 0 before promoting.
        max_level1_blocks: Max level-1 blocks before merging to level-2.
        summary_max_chars: Max chars per individual summary.
    """

    def __init__(
        self,
        max_level0_chars: int = 8000,
        max_level1_blocks: int = 5,
        summary_max_chars: int = 300,
    ) -> None:
        self._max_level0_chars = max_level0_chars
        self._max_level1_blocks = max_level1_blocks
        self._summary_max_chars = summary_max_chars

        self._level0: List[SummaryBlock] = []  # Raw blocks
        self._level1: List[SummaryBlock] = []  # Detail summaries
        self._level2: List[SummaryBlock] = []  # Theme summaries

    @property
    def total_blocks(self) -> int:
        return len(self._level0) + len(self._level1) + len(self._level2)

    def add_block(self, content: str) -> None:
        """Add a new raw content block. Triggers promotion if over budget."""
        if not content or not content.strip():
            return

        self._level0.append(SummaryBlock(content=content, level=0))
        self._maybe_promote()

    def _maybe_promote(self) -> None:
        """Promote oldest level-0 blocks to level-1 if over budget."""
        total_l0 = sum(b.char_count for b in self._level0)

        while total_l0 > self._max_level0_chars and len(self._level0) > 1:
            # Pop oldest raw block, summarize to level-1
            oldest = self._level0.pop(0)
            summary = self._summarize_block(oldest.content)
            self._level1.append(SummaryBlock(content=summary, level=1, source_count=1))
            total_l0 = sum(b.char_count for b in self._level0)

        # Merge level-1 blocks to level-2 when too many accumulate
        if len(self._level1) >= self._max_level1_blocks:
            self._merge_level1_to_level2()

    def _merge_level1_to_level2(self) -> None:
        """Merge all level-1 summaries into a single level-2 theme summary."""
        if not self._level1:
            return

        combined = "\n".join(b.content for b in self._level1)
        total_sources = sum(b.source_count for b in self._level1)

        theme = self._summarize_block(combined, max_chars=500)
        self._level2.append(SummaryBlock(content=theme, level=2, source_count=total_sources))

        self._level1.clear()
        logger.debug(
            "Merged %d level-1 blocks into level-2 theme (%d sources)",
            total_sources,
            total_sources,
        )

    def _summarize_block(self, text: str, max_chars: int = 0) -> str:
        """Create a concise summary of a text block.

        Uses pure extraction (no LLM) for speed and reliability.
        Extracts: first line + key findings + stats.
        """
        max_chars = max_chars or self._summary_max_chars
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        if not lines:
            return text[:max_chars]

        # First meaningful line as header
        header = lines[0][:120]

        # Extract key lines (results, conclusions, errors)
        key_markers = ("result", "error", "done", "found", "created", "success", "fail")
        key_lines = []
        for line in lines[1:]:
            if any(m in line.lower() for m in key_markers) and len(line) > 15:
                key_lines.append(line[:150])
                if len(key_lines) >= 3:
                    break

        # Build summary
        parts = [header]
        if key_lines:
            parts.extend(key_lines)
        parts.append(f"[{len(lines)} lines total]")

        summary = " | ".join(parts)
        return summary[:max_chars]

    def build_context(self, budget_chars: int = 8000) -> str:
        """Build context string fitting within budget.

        Order: level-2 themes → level-1 details → level-0 raw (most recent).
        Level-0 (raw recent) gets priority; older summaries fill remainder.

        Args:
            budget_chars: Maximum character budget for output.

        Returns:
            Formatted context string with section headers.
        """
        sections: List[str] = []
        chars_used = 0

        # Level-2 themes first (most compressed, cheapest)
        if self._level2:
            theme_text = "\n".join(f"- {b.content}" for b in self._level2)
            header = (
                f"[Context Themes ({sum(b.source_count for b in self._level2)} blocks summarized)]"
            )
            section = f"{header}\n{theme_text}"
            if chars_used + len(section) <= budget_chars:
                sections.append(section)
                chars_used += len(section)

        # Level-1 detail summaries
        if self._level1:
            detail_text = "\n".join(f"- {b.content}" for b in self._level1)
            header = f"[Recent Context ({len(self._level1)} blocks summarized)]"
            section = f"{header}\n{detail_text}"
            if chars_used + len(section) <= budget_chars:
                sections.append(section)
                chars_used += len(section)

        # Level-0 raw blocks (most recent, highest priority)
        for block in self._level0:
            if chars_used + block.char_count <= budget_chars:
                sections.append(block.content)
                chars_used += block.char_count
            else:
                # Partial inclusion of last block
                remaining = budget_chars - chars_used
                if remaining > 200:
                    sections.append(block.content[:remaining])
                    chars_used += remaining
                break

        return "\n\n".join(sections)

    def stats(self) -> dict:
        """Get summarizer statistics."""
        return {
            "level0_blocks": len(self._level0),
            "level0_chars": sum(b.char_count for b in self._level0),
            "level1_blocks": len(self._level1),
            "level1_chars": sum(b.char_count for b in self._level1),
            "level2_blocks": len(self._level2),
            "level2_chars": sum(b.char_count for b in self._level2),
            "total_source_blocks": (
                len(self._level0)
                + sum(b.source_count for b in self._level1)
                + sum(b.source_count for b in self._level2)
            ),
        }

    def clear(self) -> None:
        """Clear all levels."""
        self._level0.clear()
        self._level1.clear()
        self._level2.clear()


__all__ = ["ProgressiveSummarizer", "SummaryBlock"]
