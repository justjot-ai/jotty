"""
LOTUS Optimizer - Unified Optimization Layer for Jotty v2

Brings together all optimization components:
- Model Cascade
- Semantic Cache
- Batch Executor
- Adaptive Validator

Single entry point for optimized LLM operations.

DRY: Composes all optimization components with shared configuration.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

from .config import LotusConfig, ModelTier
from .model_cascade import ModelCascade, CascadeResult
from .semantic_cache import SemanticCache
from .batch_executor import BatchExecutor, ParallelBatchExecutor
from .adaptive_validator import AdaptiveValidator, ValidationDecision

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStats:
    """Aggregated optimization statistics."""
    total_operations: int = 0
    cache_hits: int = 0
    cascade_proxy_resolved: int = 0
    cascade_oracle_resolved: int = 0
    validations_skipped: int = 0
    total_cost_estimate: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_operations, 1)

    @property
    def proxy_resolution_rate(self) -> float:
        total_cascade = self.cascade_proxy_resolved + self.cascade_oracle_resolved
        return self.cascade_proxy_resolved / max(total_cascade, 1)

    @property
    def validation_skip_rate(self) -> float:
        return self.validations_skipped / max(self.total_operations, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_operations": self.total_operations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "cascade_proxy_resolved": self.cascade_proxy_resolved,
            "cascade_oracle_resolved": self.cascade_oracle_resolved,
            "proxy_resolution_rate": self.proxy_resolution_rate,
            "validations_skipped": self.validations_skipped,
            "validation_skip_rate": self.validation_skip_rate,
            "total_cost_estimate": self.total_cost_estimate,
            "total_latency_ms": self.total_latency_ms,
        }


class LotusOptimizer:
    """
    Unified optimization layer for Jotty v2.

    Integrates all LOTUS-inspired optimizations:
    1. Model Cascade - Use cheap models when possible
    2. Semantic Cache - Memoize repeated operations
    3. Batch Executor - Batch LLM calls for throughput
    4. Adaptive Validator - Skip validation for trusted agents

    Usage:
        optimizer = LotusOptimizer(config)

        # Optimized execution
        result = await optimizer.execute(
            operation="filter",
            agent="my_agent",
            items=documents,
            instruction="Keep positive reviews",
            validate=True,
        )

        # Get optimization stats
        stats = optimizer.get_stats()
    """

    def __init__(self, config: Optional[LotusConfig] = None, lm_provider: Optional[Any] = None) -> None:
        """
        Initialize unified optimizer.

        Args:
            config: LOTUS configuration (uses defaults if None)
            lm_provider: Language model provider (optional)
        """
        self.config = config or LotusConfig()

        # Initialize all optimization components with shared config
        self.cascade = ModelCascade(self.config, lm_provider)
        self.cache = SemanticCache(self.config)
        self.batch_executor = ParallelBatchExecutor(self.config, lm_provider)
        self.validator = AdaptiveValidator(self.config)

        # Aggregated stats
        self.stats = OptimizationStats()

        logger.info("LotusOptimizer initialized with all optimization components")

    async def execute(
        self,
        operation: str,
        items: List[Any],
        instruction: str,
        agent: str = "default",
        prompt_fn: Optional[Callable[[Any], str]] = None,
        parse_fn: Optional[Callable[[str], Tuple[Any, float]]] = None,
        use_cascade: bool = True,
        use_cache: bool = True,
        validate: bool = True,
        validation_fn: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Execute operation with all optimizations applied.

        Args:
            operation: Operation type (filter, map, extract, etc.)
            items: Items to process
            instruction: Natural language instruction
            agent: Agent name (for validation tracking)
            prompt_fn: Custom prompt function (optional)
            parse_fn: Custom parse function for cascade (optional)
            use_cascade: Whether to use model cascade
            use_cache: Whether to use semantic cache
            validate: Whether to validate results
            validation_fn: Custom validation function

        Returns:
            List of processed results
        """
        if not items:
            return []

        start_time = time.time()
        self.stats.total_operations += len(items)

        # Default prompt function
        if prompt_fn is None:
            prompt_fn = lambda item: f"Instruction: {instruction}\nInput: {item}\nOutput:"

        # Default parse function
        if parse_fn is None:
            parse_fn = lambda response: (response.strip(), 0.8)

        # Phase 1: Check cache
        cached_results = {}
        items_to_process = []
        indices_to_process = []

        if use_cache:
            for i, item in enumerate(items):
                hit, result = self.cache.get(instruction, item)
                if hit:
                    cached_results[i] = result
                    self.stats.cache_hits += 1
                else:
                    items_to_process.append(item)
                    indices_to_process.append(i)
        else:
            items_to_process = items
            indices_to_process = list(range(len(items)))

        # Phase 2: Process uncached items
        processed_results = {}

        if items_to_process:
            if use_cascade:
                # Use model cascade for optimized execution
                cascade_results = await self.cascade.execute(
                    operation, items_to_process, prompt_fn, parse_fn
                )

                for idx, item, result in zip(indices_to_process, items_to_process, cascade_results):
                    processed_results[idx] = result.result

                    # Update cascade stats
                    if result.tier_used == ModelTier.FAST:
                        self.stats.cascade_proxy_resolved += 1
                    else:
                        self.stats.cascade_oracle_resolved += 1

                    # Cache the result
                    if use_cache:
                        self.cache.put(instruction, item, result.result)

            else:
                # Use batch executor without cascade
                batch_results = await self.batch_executor.execute_batch(
                    operation, items_to_process, prompt_fn
                )

                for idx, item, result in zip(indices_to_process, items_to_process, batch_results):
                    parsed_result, _ = parse_fn(result)
                    processed_results[idx] = parsed_result

                    if use_cache:
                        self.cache.put(instruction, item, parsed_result)

        # Combine results in original order
        all_results = []
        for i in range(len(items)):
            if i in cached_results:
                all_results.append(cached_results[i])
            elif i in processed_results:
                all_results.append(processed_results[i])
            else:
                all_results.append(None)

        # Phase 3: Adaptive validation
        if validate:
            decision = self.validator.should_validate(agent, operation)

            if decision.should_validate:
                # Run validation
                if validation_fn:
                    is_valid = validation_fn(all_results)
                    self.validator.record_result(agent, operation, is_valid)
                else:
                    # Default: assume valid
                    self.validator.record_result(agent, operation, True)
            else:
                # Skip validation
                self.stats.validations_skipped += 1
                self.validator.record_skip(agent, operation)

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats.total_latency_ms += elapsed_ms

        return all_results

    async def execute_filter(
        self,
        items: List[Any],
        condition: str,
        agent: str = "default",
    ) -> List[Any]:
        """
        Optimized filter operation.

        Args:
            items: Items to filter
            condition: Natural language filter condition
            agent: Agent name

        Returns:
            Filtered items
        """
        def prompt_fn(item: Any) -> Any:
            return f"Does this satisfy the condition: {condition}?\nItem: {item}\nAnswer YES or NO:"

        def parse_fn(response: str) -> Tuple:
            is_yes = "yes" in response.lower()
            confidence = 0.9 if is_yes or "no" in response.lower() else 0.5
            return is_yes, confidence

        results = await self.execute(
            operation="filter",
            items=items,
            instruction=condition,
            agent=agent,
            prompt_fn=prompt_fn,
            parse_fn=parse_fn,
        )

        # Filter to items where result is True
        return [item for item, keep in zip(items, results) if keep]

    async def execute_map(
        self,
        items: List[Any],
        instruction: str,
        agent: str = "default",
    ) -> List[Any]:
        """
        Optimized map operation.

        Args:
            items: Items to transform
            instruction: Transformation instruction
            agent: Agent name

        Returns:
            Transformed items
        """
        return await self.execute(
            operation="map",
            items=items,
            instruction=instruction,
            agent=agent,
            use_cascade=False,  # Map typically needs full model
        )

    async def execute_extract(
        self,
        items: List[Any],
        instruction: str,
        schema: Optional[Dict] = None,
        agent: str = "default",
    ) -> List[Dict]:
        """
        Optimized extraction operation.

        Args:
            items: Items to extract from
            instruction: Extraction instruction
            schema: Output schema
            agent: Agent name

        Returns:
            List of extracted dictionaries
        """
        import json

        schema_hint = ""
        if schema:
            schema_hint = f" Output as JSON with fields: {list(schema.keys())}"

        def parse_fn(response: str) -> Tuple:
            try:
                parsed = json.loads(response)
                return parsed, 0.9
            except Exception:
                return {"raw": response}, 0.5

        return await self.execute(
            operation="extract",
            items=items,
            instruction=instruction + schema_hint,
            agent=agent,
            parse_fn=parse_fn,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated optimization statistics."""
        return {
            "optimizer": self.stats.to_dict(),
            "cascade": self.cascade.get_stats(),
            "cache": self.cache.get_stats(),
            "batch_executor": self.batch_executor.get_stats(),
            "validator": self.validator.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = OptimizationStats()
        self.cascade.reset_stats()

    def get_cost_estimate(self) -> float:
        """Get total estimated cost."""
        cascade_stats = self.cascade.get_stats()
        return cascade_stats.get("total_cost", 0.0)

    def get_savings_estimate(self) -> Dict[str, float]:
        """
        Estimate savings from optimizations.

        Returns breakdown of savings from each optimization.
        """
        cascade_stats = self.cascade.get_stats()
        cache_stats = self.cache.get_stats()
        validator_stats = self.validator.get_stats()

        # Estimate what it would have cost without optimizations
        # Assume $0.01 per operation with Opus
        baseline_cost = self.stats.total_operations * 0.01

        # Actual cost
        actual_cost = cascade_stats.get("total_cost", 0)

        # Breakdown
        return {
            "baseline_cost": baseline_cost,
            "actual_cost": actual_cost,
            "total_savings": baseline_cost - actual_cost,
            "savings_from_cache": cache_stats.get("hits", 0) * 0.01,
            "savings_from_cascade": cascade_stats.get("cost_savings", 0),
            "savings_from_validation_skip": validator_stats.get("total_skips", 0) * 0.005,
        }

    async def close(self) -> Any:
        """Clean up resources."""
        await self.batch_executor.close()


# Factory function for easy creation
def create_optimizer(
    skip_threshold: float = 0.95,
    cache_enabled: bool = True,
    cascade_enabled: bool = True,
    batch_size: int = 20,
) -> LotusOptimizer:
    """
    Create a configured LotusOptimizer.

    Args:
        skip_threshold: Validation skip threshold
        cache_enabled: Whether to enable caching
        cascade_enabled: Whether to enable cascade
        batch_size: Maximum batch size

    Returns:
        Configured LotusOptimizer
    """
    config = LotusConfig()
    config.validation_skip_threshold = skip_threshold
    config.cache.enabled = cache_enabled
    config.batch.max_batch_size = batch_size

    return LotusOptimizer(config)
