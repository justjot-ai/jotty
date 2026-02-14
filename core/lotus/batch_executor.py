"""
Batch Executor - Batch LLM Calls for Throughput Optimization

LOTUS Insight: 1 call with 100 items < 100 calls with 1 item each

Features:
- Collect operations into batches before LLM execution
- Automatic flush when batch is full or timeout reached
- Group by operation type for efficient batching
- Parallel batch execution

Cost Impact:
- Reduced API overhead
- Better throughput (5-10x)
- Lower latency per item
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar
from collections import defaultdict
from enum import Enum
import uuid

from .config import LotusConfig, BatchConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BatchItem:
    """Single item in a batch."""
    id: str
    operation_type: str
    prompt: str
    item: Any
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """Result for a batch execution."""
    batch_id: str
    operation_type: str
    items_count: int
    success_count: int
    error_count: int
    total_latency_ms: float
    avg_latency_per_item_ms: float
    results: List[Tuple[str, Any]]  # List of (item_id, result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "operation_type": self.operation_type,
            "items_count": self.items_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_per_item_ms": self.avg_latency_per_item_ms,
        }


class BatchExecutor:
    """
    Batched execution engine for LLM operations.

    Collects individual operations into batches and executes them together.
    Reduces API overhead and improves throughput.

    DRY: Reuses BatchConfig from LotusConfig.

    Usage:
        executor = BatchExecutor(config)

        # Queue items (returns futures)
        future1 = await executor.queue("filter", item1, prompt_fn)
        future2 = await executor.queue("filter", item2, prompt_fn)

        # Get results when ready
        result1 = await future1
        result2 = await future2

        # Or force flush
        await executor.flush()
    """

    def __init__(
        self,
        config: Optional[LotusConfig] = None,
        lm_provider: Optional[Any] = None,
    ):
        """
        Initialize batch executor.

        Args:
            config: LOTUS configuration
            lm_provider: Language model provider
        """
        self.config = config or LotusConfig()
        self.batch_config = self.config.batch
        self.lm_provider = lm_provider

        # Pending items grouped by operation type
        self._pending: Dict[str, List[BatchItem]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Flush timer
        self._flush_task: Optional[asyncio.Task] = None
        self._last_flush_time: Dict[str, float] = {}

        # Stats
        self._total_items = 0
        self._total_batches = 0
        self._total_latency_ms = 0.0

        if self.batch_config.enabled:
            logger.info(
                f"BatchExecutor initialized: max_batch_size={self.batch_config.max_batch_size}, "
                f"max_wait_ms={self.batch_config.max_wait_ms}"
            )

    async def queue(
        self,
        operation_type: str,
        item: Any,
        prompt: str,
    ) -> asyncio.Future:
        """
        Queue an item for batched execution.

        Args:
            operation_type: Type of operation (filter, map, etc.)
            item: The item to process
            prompt: The prompt for this item

        Returns:
            Future that resolves to the result
        """
        if not self.batch_config.enabled:
            # Execute immediately if batching disabled
            result = await self._execute_single(prompt)
            future = asyncio.Future()
            future.set_result(result)
            return future

        async with self._lock:
            # Create future for this item
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            # Create batch item
            batch_item = BatchItem(
                id=str(uuid.uuid4())[:8],
                operation_type=operation_type,
                prompt=prompt,
                item=item,
                future=future,
            )

            # Add to pending
            self._pending[operation_type].append(batch_item)
            self._total_items += 1

            # Check if batch is full
            if len(self._pending[operation_type]) >= self.batch_config.max_batch_size:
                # Schedule immediate flush for this operation type
                asyncio.create_task(self._flush_operation(operation_type))
            else:
                # Start/reset flush timer
                self._ensure_flush_timer(operation_type)

            return future

    async def queue_many(
        self,
        operation_type: str,
        items: List[Any],
        prompt_fn: Callable[[Any], str],
    ) -> List[asyncio.Future]:
        """
        Queue multiple items for batched execution.

        Args:
            operation_type: Type of operation
            items: Items to process
            prompt_fn: Function to generate prompt from item

        Returns:
            List of futures
        """
        futures = []
        for item in items:
            prompt = prompt_fn(item)
            future = await self.queue(operation_type, item, prompt)
            futures.append(future)
        return futures

    async def execute_batch(
        self,
        operation_type: str,
        items: List[Any],
        prompt_fn: Callable[[Any], str],
    ) -> List[Any]:
        """
        Execute a batch immediately (bypasses queue).

        Useful for cases where you have a complete batch ready.

        Args:
            operation_type: Type of operation
            items: Items to process
            prompt_fn: Function to generate prompt from item

        Returns:
            List of results in same order as items
        """
        if not items:
            return []

        prompts = [prompt_fn(item) for item in items]
        results = await self._execute_batch(prompts)

        return results

    def _ensure_flush_timer(self, operation_type: str) -> None:
        """Ensure flush timer is running for operation type."""
        now = time.time()
        last_flush = self._last_flush_time.get(operation_type, 0)

        # If no pending items or timer already scheduled, skip
        if not self._pending[operation_type]:
            return

        # Schedule flush after max_wait_ms
        async def delayed_flush():
            await asyncio.sleep(self.batch_config.max_wait_ms / 1000)
            await self._flush_operation(operation_type)

        # Only schedule if we don't have one pending
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(delayed_flush())

    async def _flush_operation(self, operation_type: str):
        """Flush pending items for a specific operation type."""
        async with self._lock:
            items = self._pending.pop(operation_type, [])
            self._last_flush_time[operation_type] = time.time()

        if not items:
            return

        batch_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.debug(
            f"Flushing batch {batch_id}: {len(items)} items of type '{operation_type}'"
        )

        try:
            # Extract prompts
            prompts = [item.prompt for item in items]

            # Execute batch
            results = await self._execute_batch(prompts)

            # Set results on futures
            for item, result in zip(items, results):
                if not item.future.done():
                    item.future.set_result(result)

            elapsed_ms = (time.time() - start_time) * 1000
            self._total_batches += 1
            self._total_latency_ms += elapsed_ms

            logger.debug(
                f"Batch {batch_id} complete: {len(items)} items in {elapsed_ms:.1f}ms "
                f"({elapsed_ms/len(items):.1f}ms/item)"
            )

        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {e}")

            # Set exception on all futures
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)

    async def flush(self):
        """Flush all pending items."""
        operation_types = list(self._pending.keys())
        for op_type in operation_types:
            await self._flush_operation(op_type)

    async def _execute_single(self, prompt: str) -> str:
        """Execute single prompt."""
        results = await self._execute_batch([prompt])
        return results[0] if results else ""

    async def _execute_batch(self, prompts: List[str]) -> List[str]:
        """
        Execute batch of prompts.

        DRY: Uses DSPy LM infrastructure.
        """
        if not prompts:
            return []

        try:
            import dspy

            lm = self.lm_provider or dspy.settings.lm

            if lm is None:
                logger.warning("No LM configured, returning empty results")
                return ["" for _ in prompts]

            # DSPy LM supports batch calls
            responses = lm(prompts)

            # Normalize response format
            if isinstance(responses, str):
                responses = [responses]
            elif hasattr(responses, '__iter__'):
                responses = [str(r) for r in responses]
            else:
                responses = [str(responses)]

            # Pad if needed
            while len(responses) < len(prompts):
                responses.append("")

            return responses[:len(prompts)]

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            # Return empty results with retry option
            return ["" for _ in prompts]

    async def _execute_batch_with_retry(
        self,
        prompts: List[str],
        max_retries: Optional[int] = None,
    ) -> List[str]:
        """Execute batch with retry logic."""
        max_retries = max_retries or self.batch_config.max_retries

        for attempt in range(max_retries):
            try:
                return await self._execute_batch(prompts)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = self.batch_config.retry_delay_ms / 1000
                    logger.warning(f"Batch attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

        return ["" for _ in prompts]

    def get_stats(self) -> Dict[str, Any]:
        """Get batch executor statistics."""
        pending_count = sum(len(items) for items in self._pending.values())

        return {
            "total_items": self._total_items,
            "total_batches": self._total_batches,
            "avg_batch_size": self._total_items / max(self._total_batches, 1),
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_per_batch_ms": self._total_latency_ms / max(self._total_batches, 1),
            "pending_items": pending_count,
            "pending_by_type": {k: len(v) for k, v in self._pending.items()},
        }

    async def close(self):
        """Close executor and flush pending items."""
        await self.flush()

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass


class ParallelBatchExecutor(BatchExecutor):
    """
    Extended batch executor with parallel batch execution.

    Executes multiple batches in parallel for maximum throughput.
    """

    async def execute_parallel(
        self,
        operation_type: str,
        items: List[Any],
        prompt_fn: Callable[[Any], str],
    ) -> List[Any]:
        """
        Execute items in parallel batches.

        Args:
            operation_type: Type of operation
            items: Items to process
            prompt_fn: Function to generate prompt from item

        Returns:
            List of results in same order as items
        """
        if not items:
            return []

        batch_size = self.batch_config.max_batch_size
        max_parallel = self.batch_config.max_parallel_batches

        # Split into batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

        all_results = []
        for i in range(0, len(batches), max_parallel):
            # Execute up to max_parallel batches concurrently
            parallel_batches = batches[i:i + max_parallel]

            tasks = [
                self.execute_batch(operation_type, batch, prompt_fn)
                for batch in parallel_batches
            ]

            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                all_results.extend(result)

        return all_results
