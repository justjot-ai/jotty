"""
Incremental Memory Consolidation
=================================

Non-blocking, streaming memory consolidation.

KISS PRINCIPLE: Simple queue-based streaming (no complex scheduling).
DRY PRINCIPLE: Reuses existing consolidation logic.
"""

import asyncio
import logging
from typing import List, Optional, Any
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class IncrementalConsolidator:
    """
    Consolidate memories incrementally (no blocking).

    PROBLEM:
    Batch consolidation every 100 episodes → 10s blocking
    User-facing latency spikes hurt UX

    SOLUTION:
    Consolidate 1 memory per 10 episodes → 100ms each
    Background task processes queue continuously
    Zero user-facing latency

    ARCHITECTURE:
    1. Memories added to consolidation queue
    2. Background task processes queue (1 memory at a time)
    3. Yields to event loop between memories (non-blocking)
    """

    def __init__(self, batch_size: int = 1, delay_between_ms: int = 50) -> None:
        """
        Args:
            batch_size: Memories to consolidate per iteration (1 = true streaming)
            delay_between_ms: Delay between batches (ms) to avoid CPU saturation
        """
        self.batch_size = batch_size
        self.delay_between_ms = delay_between_ms

        self.queue: deque = deque()
        self.processing = False
        self.processed_count = 0
        self.failed_count = 0

        # DRY: Get existing consolidator
        self._base_consolidator = None

        logger.info(
            f"♻️  IncrementalConsolidator initialized "
            f"(batch_size={batch_size}, delay={delay_between_ms}ms)"
        )

    def _get_base_consolidator(self) -> Any:
        """Lazy-load base consolidator (DRY)."""
        if self._base_consolidator is None:
            try:
                from Jotty.core.memory import get_consolidator
                self._base_consolidator = get_consolidator()
            except ImportError:
                logger.warning("Base consolidator unavailable")
                self._base_consolidator = NoopConsolidator()
        return self._base_consolidator

    def enqueue_memory(self, memory: Any) -> Any:
        """
        Add memory to consolidation queue.

        Args:
            memory: MemoryEntry to consolidate
        """
        self.queue.append(memory)

        # Auto-start processing if not running
        if not self.processing:
            asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> Any:
        """Background task to process consolidation queue."""
        if self.processing:
            return  # Already running

        self.processing = True
        logger.debug("🔄 Started incremental consolidation background task")

        try:
            while self.queue:
                # Process batch
                batch = []
                for _ in range(min(self.batch_size, len(self.queue))):
                    if self.queue:
                        batch.append(self.queue.popleft())

                if batch:
                    await self._consolidate_batch(batch)

                # Yield to event loop (KISS - simple sleep)
                await asyncio.sleep(self.delay_between_ms / 1000.0)

        finally:
            self.processing = False
            logger.debug(
                f"✅ Incremental consolidation complete "
                f"(processed={self.processed_count}, failed={self.failed_count})"
            )

    async def _consolidate_batch(self, memories: List[Any]) -> Any:
        """Consolidate a small batch of memories."""
        consolidator = self._get_base_consolidator()

        for memory in memories:
            try:
                # Delegate to base consolidator (DRY)
                if hasattr(consolidator, 'consolidate_single'):
                    await consolidator.consolidate_single(memory)
                else:
                    # Fall back to batch consolidation with single item
                    await consolidator.consolidate(memories=[memory])

                self.processed_count += 1

            except Exception as e:
                logger.warning(f"Failed to consolidate memory: {e}")
                self.failed_count += 1

    def get_stats(self) -> dict:
        """Get consolidation statistics."""
        return {
            'queue_size': len(self.queue),
            'processing': self.processing,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'success_rate': (
                self.processed_count / (self.processed_count + self.failed_count)
                if (self.processed_count + self.failed_count) > 0
                else 0.0
            ),
        }

    async def flush(self, timeout: float = 30.0) -> Any:
        """
        Wait for queue to be fully processed.

        Args:
            timeout: Max wait time (seconds)
        """
        start = asyncio.get_event_loop().time()

        while self.queue and (asyncio.get_event_loop().time() - start) < timeout:
            await asyncio.sleep(0.1)

        if self.queue:
            logger.warning(
                f"Flush timeout: {len(self.queue)} memories still in queue"
            )


class NoopConsolidator:
    """Noop consolidator for fallback."""

    async def consolidate_single(self, memory: Any) -> Any:
        pass


# Singleton
_incremental_consolidator = None


def get_incremental_consolidator(
    batch_size: int = 1,
    delay_between_ms: int = 50
) -> IncrementalConsolidator:
    """Get or create incremental consolidator singleton."""
    global _incremental_consolidator
    if _incremental_consolidator is None:
        _incremental_consolidator = IncrementalConsolidator(
            batch_size=batch_size,
            delay_between_ms=delay_between_ms
        )
    return _incremental_consolidator


__all__ = [
    'IncrementalConsolidator',
    'get_incremental_consolidator',
]
