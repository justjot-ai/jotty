"""
Semantic Operators - Declarative LLM Data Operations

LOTUS-Inspired API for semantic data transformations:
- sem_filter: Filter rows using natural language condition
- sem_map: Map each row through LLM transformation
- sem_extract: Extract structured data from text
- sem_topk: Select top-k items by semantic criteria
- sem_join: Join datasets by semantic similarity

All operators automatically apply:
- Model cascading (cheap model first)
- Semantic caching (memoization)
- Batch processing (throughput optimization)

DRY: Operators compose together and reuse core optimization components.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import pandas as pd

from .batch_executor import BatchExecutor
from .config import LotusConfig
from .model_cascade import ModelCascade
from .semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class OperatorResult:
    """Result from a semantic operator."""

    data: Any
    stats: Dict[str, Any] = field(default_factory=dict)
    cost_estimate: float = 0.0
    cache_hits: int = 0
    items_processed: int = 0


class SemanticOperator(ABC):
    """
    Base class for semantic operators.

    All operators:
    1. Accept natural language instructions
    2. Apply optimization (cascade, cache, batch)
    3. Return transformed data

    DRY: Common optimization logic in base class.
    """

    def __init__(
        self,
        config: Optional[LotusConfig] = None,
        cascade: Optional[ModelCascade] = None,
        cache: Optional[SemanticCache] = None,
        batch_executor: Optional[BatchExecutor] = None,
    ) -> None:
        """
        Initialize operator with shared optimization components.

        DRY: All operators share the same optimization infrastructure.
        """
        self.config = config or LotusConfig()
        self.cascade = cascade or ModelCascade(self.config)
        self.cache = cache or SemanticCache(self.config)
        self.batch_executor = batch_executor or BatchExecutor(self.config)

    @abstractmethod
    async def execute(self, data: Any, instruction: str, **kwargs: Any) -> OperatorResult:
        """Execute the semantic operation."""
        pass

    def _build_prompt(
        self,
        instruction: str,
        item: Any,
        context: Optional[str] = None,
    ) -> str:
        """Build prompt for LLM call."""
        prompt = f"Instruction: {instruction}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Input: {item}\n\nOutput:"
        return prompt


class SemFilter(SemanticOperator):
    """
    Semantic filter operator.

    Filters data based on natural language condition.

    Usage:
        filter_op = SemFilter(config)
        result = await filter_op.execute(
            data=documents,
            instruction="Keep only positive reviews"
        )
    """

    async def execute(
        self, data: List[Any], instruction: str, use_cascade: bool = True, **kwargs: Any
    ) -> OperatorResult:
        """
        Filter data based on natural language condition.

        Args:
            data: List of items to filter
            instruction: Natural language filter condition
            use_cascade: Whether to use model cascade optimization

        Returns:
            OperatorResult with filtered data
        """
        if not data:
            return OperatorResult(data=[], items_processed=0)

        cache_hits = 0
        filtered = []
        to_process = []
        to_process_indices = []

        # Check cache first
        for i, item in enumerate(data):
            hit, result = self.cache.get(instruction, item)
            if hit:
                cache_hits += 1
                if result:  # True = keep
                    filtered.append(item)
            else:
                to_process.append(item)
                to_process_indices.append(i)

        # Process uncached items
        if to_process:
            if use_cascade:
                results = await self._filter_with_cascade(to_process, instruction)
            else:
                results = await self._filter_batch(to_process, instruction)

            # Cache and collect results
            for item, (keep, _) in zip(to_process, results):
                self.cache.put(instruction, item, keep)
                if keep:
                    filtered.append(item)

        stats = {
            "cache_hits": cache_hits,
            "items_processed": len(to_process),
            "items_filtered_out": len(data) - len(filtered),
            "filter_rate": (len(data) - len(filtered)) / max(len(data), 1),
        }

        return OperatorResult(
            data=filtered,
            stats=stats,
            cache_hits=cache_hits,
            items_processed=len(to_process),
        )

    async def _filter_with_cascade(
        self,
        items: List[Any],
        instruction: str,
    ) -> List[tuple]:
        """Filter using model cascade."""

        def prompt_fn(item: Any) -> Any:
            return self._build_prompt(
                f"Does this item satisfy the condition: {instruction}? Answer YES or NO with confidence.",
                item,
            )

        def parse_fn(response: str) -> Tuple:
            response_lower = response.lower()
            if "yes" in response_lower:
                confidence = 0.9 if "confident" in response_lower else 0.7
                return True, confidence
            elif "no" in response_lower:
                confidence = 0.9 if "confident" in response_lower else 0.7
                return False, confidence
            return None, 0.3

        cascade_results = await self.cascade.execute("filter", items, prompt_fn, parse_fn)

        return [(r.result, r.confidence) for r in cascade_results]

    async def _filter_batch(
        self,
        items: List[Any],
        instruction: str,
    ) -> List[tuple]:
        """Filter using batch execution (no cascade)."""

        def prompt_fn(item: Any) -> Any:
            return self._build_prompt(
                f"Does this item satisfy: {instruction}? Answer YES or NO.", item
            )

        results = await self.batch_executor.execute_batch("filter", items, prompt_fn)

        parsed = []
        for response in results:
            if "yes" in response.lower():
                parsed.append((True, 0.8))
            else:
                parsed.append((False, 0.8))

        return parsed


class SemMap(SemanticOperator):
    """
    Semantic map operator.

    Transforms each item using natural language instruction.

    Usage:
        map_op = SemMap(config)
        result = await map_op.execute(
            data=documents,
            instruction="Summarize this document in one sentence"
        )
    """

    async def execute(
        self, data: List[Any], instruction: str, output_column: Optional[str] = None, **kwargs: Any
    ) -> OperatorResult:
        """
        Map each item through LLM transformation.

        Args:
            data: List of items to transform
            instruction: Transformation instruction
            output_column: Column name for output (if using DataFrame)

        Returns:
            OperatorResult with transformed data
        """
        if not data:
            return OperatorResult(data=[], items_processed=0)

        cache_hits = 0
        results = []
        to_process = []
        to_process_indices = []

        # Check cache
        for i, item in enumerate(data):
            hit, result = self.cache.get(instruction, item)
            if hit:
                cache_hits += 1
                results.append((i, result))
            else:
                to_process.append(item)
                to_process_indices.append(i)

        # Process uncached
        if to_process:

            def prompt_fn(item: Any) -> Any:
                return self._build_prompt(instruction, item)

            batch_results = await self.batch_executor.execute_batch("map", to_process, prompt_fn)

            for idx, item, result in zip(to_process_indices, to_process, batch_results):
                self.cache.put(instruction, item, result)
                results.append((idx, result))

        # Sort by original index and extract results
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]

        stats = {
            "cache_hits": cache_hits,
            "items_processed": len(to_process),
        }

        return OperatorResult(
            data=final_results,
            stats=stats,
            cache_hits=cache_hits,
            items_processed=len(to_process),
        )


class SemExtract(SemanticOperator):
    """
    Semantic extraction operator.

    Extracts structured data from text based on schema.

    Usage:
        extract_op = SemExtract(config)
        result = await extract_op.execute(
            data=documents,
            instruction="Extract: name, date, amount",
            schema={"name": str, "date": str, "amount": float}
        )
    """

    async def execute(
        self,
        data: List[Any],
        instruction: str,
        schema: Optional[Dict[str, type]] = None,
        **kwargs: Any,
    ) -> OperatorResult:
        """
        Extract structured data from items.

        Args:
            data: List of items to extract from
            instruction: Extraction instruction (e.g., "Extract name, date, amount")
            schema: Optional schema for output validation

        Returns:
            OperatorResult with extracted data
        """
        if not data:
            return OperatorResult(data=[], items_processed=0)

        # Build extraction prompt
        schema_hint = ""
        if schema:
            schema_hint = f"\nOutput as JSON with fields: {list(schema.keys())}"

        cache_hits = 0
        results = []
        to_process = []
        to_process_indices = []

        # Check cache
        for i, item in enumerate(data):
            hit, result = self.cache.get(instruction, item)
            if hit:
                cache_hits += 1
                results.append((i, result))
            else:
                to_process.append(item)
                to_process_indices.append(i)

        # Process uncached
        if to_process:

            def prompt_fn(item: Any) -> Any:
                return self._build_prompt(instruction + schema_hint, item)

            batch_results = await self.batch_executor.execute_batch(
                "extract", to_process, prompt_fn
            )

            # Parse JSON results
            import json

            for idx, item, result in zip(to_process_indices, to_process, batch_results):
                try:
                    # Try to parse as JSON
                    parsed = json.loads(result)
                except Exception:
                    # Fall back to raw text
                    parsed = {"raw": result}

                self.cache.put(instruction, item, parsed)
                results.append((idx, parsed))

        # Sort and extract
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]

        return OperatorResult(
            data=final_results,
            stats={"cache_hits": cache_hits, "items_processed": len(to_process)},
            cache_hits=cache_hits,
            items_processed=len(to_process),
        )


class SemTopK(SemanticOperator):
    """
    Semantic top-k operator.

    Selects top-k items by semantic criteria.

    Usage:
        topk_op = SemTopK(config)
        result = await topk_op.execute(
            data=candidates,
            instruction="Most relevant to machine learning",
            k=10
        )
    """

    async def execute(
        self, data: List[Any], instruction: str, k: int = 10, **kwargs: Any
    ) -> OperatorResult:
        """
        Select top-k items by semantic criteria.

        Uses cascade to efficiently score and filter:
        1. Quick scoring pass with cheap model
        2. Re-rank top candidates with better model

        Args:
            data: List of items to rank
            instruction: Ranking criteria
            k: Number of items to return

        Returns:
            OperatorResult with top-k items
        """
        if not data or k <= 0:
            return OperatorResult(data=[], items_processed=0)

        if len(data) <= k:
            return OperatorResult(data=list(data), items_processed=0)

        # Score all items
        def prompt_fn(item: Any) -> Any:
            return self._build_prompt(
                f"Rate how well this item matches: {instruction}. "
                f"Respond with a score from 0-100.",
                item,
            )

        def parse_fn(response: str) -> tuple:
            try:
                # Extract number from response
                import re

                numbers = re.findall(r"\d+", response)
                if numbers:
                    score = int(numbers[0])
                    confidence = 0.8 if score > 0 else 0.5
                    return score, confidence
            except Exception:
                pass
            return 50, 0.3

        cascade_results = await self.cascade.execute("topk", data, prompt_fn, parse_fn)

        # Sort by score and take top k
        scored = [
            (item, r.result if r.result is not None else 0)
            for item, r in zip(data, cascade_results)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_k = [item for item, score in scored[:k]]

        return OperatorResult(
            data=top_k,
            stats={
                "total_scored": len(data),
                "top_k": k,
            },
            items_processed=len(data),
        )


class SemanticDataFrame:
    """
    Pandas-like DataFrame wrapper with semantic operators.

    Provides a fluent API for chaining semantic operations.

    Usage:
        sdf = SemanticDataFrame(df, config)
        result = await (
            sdf
            .sem_filter("positive sentiment")
            .sem_map("summarize in one line", output_col="summary")
            .sem_topk(10, "most informative")
            .execute()
        )
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, List[Dict], List[Any]],
        config: Optional[LotusConfig] = None,
        text_column: Optional[str] = None,
    ) -> None:
        """
        Initialize SemanticDataFrame.

        Args:
            data: Input data (DataFrame, list of dicts, or list of items)
            config: LOTUS configuration
            text_column: Column to use for semantic operations (if DataFrame)
        """
        self.config = config or LotusConfig()
        self.text_column = text_column

        # Normalize to list
        if isinstance(data, pd.DataFrame):
            self._df = data
            if text_column:
                self._items = data[text_column].tolist()
            else:
                self._items = data.to_dict("records")
        elif isinstance(data, list):
            self._df = None
            self._items = data
        else:
            self._df = None
            self._items = [data]

        # Shared operators (DRY: reuse across all operations)
        cascade = ModelCascade(self.config)
        cache = SemanticCache(self.config)
        batch_executor = BatchExecutor(self.config)

        self._filter_op = SemFilter(self.config, cascade, cache, batch_executor)
        self._map_op = SemMap(self.config, cascade, cache, batch_executor)
        self._extract_op = SemExtract(self.config, cascade, cache, batch_executor)
        self._topk_op = SemTopK(self.config, cascade, cache, batch_executor)

        # Operation queue for lazy execution
        self._operations: List[tuple] = []

    def sem_filter(self, condition: str) -> "SemanticDataFrame":
        """Add filter operation to queue."""
        self._operations.append(("filter", condition, {}))
        return self

    def sem_map(
        self,
        instruction: str,
        output_col: Optional[str] = None,
    ) -> "SemanticDataFrame":
        """Add map operation to queue."""
        self._operations.append(("map", instruction, {"output_col": output_col}))
        return self

    def sem_extract(
        self,
        instruction: str,
        schema: Optional[Dict] = None,
    ) -> "SemanticDataFrame":
        """Add extract operation to queue."""
        self._operations.append(("extract", instruction, {"schema": schema}))
        return self

    def sem_topk(self, k: int, criteria: str) -> "SemanticDataFrame":
        """Add top-k operation to queue."""
        self._operations.append(("topk", criteria, {"k": k}))
        return self

    async def execute(self) -> OperatorResult:
        """
        Execute all queued operations.

        Returns:
            OperatorResult with final data and aggregate stats
        """
        current_data = self._items
        total_stats = {
            "operations": [],
            "total_cache_hits": 0,
            "total_items_processed": 0,
        }

        for op_type, instruction, kwargs in self._operations:
            if op_type == "filter":
                result = await self._filter_op.execute(current_data, instruction)
            elif op_type == "map":
                result = await self._map_op.execute(current_data, instruction, **kwargs)
            elif op_type == "extract":
                result = await self._extract_op.execute(current_data, instruction, **kwargs)
            elif op_type == "topk":
                result = await self._topk_op.execute(current_data, instruction, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {op_type}")

            current_data = result.data
            total_stats["operations"].append(
                {
                    "type": op_type,
                    "instruction": instruction[:50],
                    "stats": result.stats,
                }
            )
            total_stats["total_cache_hits"] += result.cache_hits
            total_stats["total_items_processed"] += result.items_processed

        # Clear operation queue
        self._operations = []

        return OperatorResult(
            data=current_data,
            stats=total_stats,
            cache_hits=total_stats["total_cache_hits"],
            items_processed=total_stats["total_items_processed"],
        )

    def to_list(self) -> List[Any]:
        """Get current items as list (for sync access before execute)."""
        return self._items

    def __len__(self) -> int:
        return len(self._items)
