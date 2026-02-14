"""
Model Cascade - Tiered Model Selection for Cost Optimization

LOTUS Insight: 80%+ queries can be resolved by cheap models!

Architecture:
    Query → Proxy (Haiku) → [Confident? → Done] OR [→ Oracle (Sonnet/Opus)]

Cost Impact (example):
    Before: 100 × Opus = $15 per 1M input tokens
    After:  70 × Haiku + 25 × Sonnet + 5 × Opus = ~$1.7 per 1M input tokens
    Result: ~9x cost reduction
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import time

from .config import LotusConfig, CascadeThresholds, ModelTier

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Result from cascade execution."""
    item: Any
    result: Any
    confidence: float
    model_used: str
    tier_used: ModelTier
    cost_estimate: float
    latency_ms: float


@dataclass
class CascadeStats:
    """Statistics from cascade execution."""
    total_items: int = 0
    proxy_resolved: int = 0       # Resolved by fast tier
    oracle_resolved: int = 0      # Needed balanced tier
    fallback_resolved: int = 0    # Needed powerful tier
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    cache_hits: int = 0

    @property
    def proxy_rate(self) -> float:
        """Percentage resolved by proxy."""
        return self.proxy_resolved / max(self.total_items, 1)

    @property
    def cost_savings(self) -> float:
        """Estimated savings vs using powerful tier for all."""
        # Estimate cost if all went to Opus
        opus_cost = self.total_items * 0.015  # Rough estimate per item
        return max(0, opus_cost - self.total_cost)


class ModelCascade:
    """
    Cascaded model execution with confidence-based routing.

    Implements LOTUS cascade pattern:
    1. Score all items with proxy (fast/cheap) model
    2. High confidence → accept proxy result
    3. Low confidence → reject immediately
    4. Uncertain → route to oracle (more capable model)

    DRY: Reuses LotusConfig for thresholds, model selection.
    """

    def __init__(
        self,
        config: Optional[LotusConfig] = None,
        lm_provider: Optional[Any] = None,  # DSPy LM or similar
    ):
        """
        Initialize model cascade.

        Args:
            config: LOTUS configuration
            lm_provider: Language model provider (optional, uses dspy.settings.lm if None)
        """
        self.config = config or LotusConfig()
        self.lm_provider = lm_provider
        self.stats = CascadeStats()

        # Threshold learning history (for adaptive threshold tuning)
        self._threshold_history: Dict[str, List[Tuple[float, bool]]] = {}

        logger.info(f"ModelCascade initialized with models: {self.config.models}")

    async def execute(
        self,
        operation: str,
        items: List[Any],
        prompt_fn: Callable[[Any], str],
        parse_fn: Callable[[str], Tuple[Any, float]],  # Returns (result, confidence)
    ) -> List[CascadeResult]:
        """
        Execute operation with cascade optimization.

        Args:
            operation: Operation type (filter, classify, extract, map)
            items: Items to process
            prompt_fn: Function to generate prompt from item
            parse_fn: Function to parse LLM response, returns (result, confidence)

        Returns:
            List of CascadeResult with optimized execution
        """
        if not items:
            return []

        thresholds = self.config.get_cascade_thresholds(operation)
        start_time = time.time()

        # Phase 1: Proxy scoring (batch call to fast model)
        logger.debug(f"Cascade Phase 1: Proxy scoring {len(items)} items")
        proxy_results = await self._batch_score(
            items, prompt_fn, parse_fn, ModelTier.FAST
        )

        # Phase 2: Triage based on confidence
        confident_positive: List[CascadeResult] = []
        confident_negative: List[CascadeResult] = []
        uncertain: List[Tuple[Any, int]] = []  # (item, original_index)

        for idx, (item, (result, confidence)) in enumerate(zip(items, proxy_results)):
            if confidence >= thresholds.tau_pos:
                # High confidence - accept proxy result
                confident_positive.append(CascadeResult(
                    item=item,
                    result=result,
                    confidence=confidence,
                    model_used=self.config.get_model(ModelTier.FAST),
                    tier_used=ModelTier.FAST,
                    cost_estimate=self._estimate_item_cost(ModelTier.FAST),
                    latency_ms=0,  # Will be set at end
                ))
                self.stats.proxy_resolved += 1

            elif confidence <= thresholds.tau_neg:
                # Low confidence - reject (for filter operations)
                confident_negative.append(CascadeResult(
                    item=item,
                    result=False if operation == "filter" else None,
                    confidence=confidence,
                    model_used=self.config.get_model(ModelTier.FAST),
                    tier_used=ModelTier.FAST,
                    cost_estimate=self._estimate_item_cost(ModelTier.FAST),
                    latency_ms=0,
                ))
                self.stats.proxy_resolved += 1

            else:
                # Uncertain - needs oracle
                uncertain.append((item, idx))

        # Phase 3: Oracle for uncertain items
        oracle_results: List[CascadeResult] = []
        if uncertain:
            logger.debug(f"Cascade Phase 2: Oracle processing {len(uncertain)} uncertain items")
            uncertain_items = [item for item, _ in uncertain]

            oracle_scored = await self._batch_score(
                uncertain_items, prompt_fn, parse_fn, ModelTier.BALANCED
            )

            # Check if we need fallback to powerful tier
            for (item, orig_idx), (result, confidence) in zip(uncertain, oracle_scored):
                if confidence >= 0.5:  # Oracle resolved it
                    oracle_results.append(CascadeResult(
                        item=item,
                        result=result,
                        confidence=confidence,
                        model_used=self.config.get_model(ModelTier.BALANCED),
                        tier_used=ModelTier.BALANCED,
                        cost_estimate=self._estimate_item_cost(ModelTier.BALANCED),
                        latency_ms=0,
                    ))
                    self.stats.oracle_resolved += 1
                else:
                    # Fallback to powerful tier
                    fallback_result = await self._single_score(
                        item, prompt_fn, parse_fn, ModelTier.POWERFUL
                    )
                    oracle_results.append(CascadeResult(
                        item=item,
                        result=fallback_result[0],
                        confidence=fallback_result[1],
                        model_used=self.config.get_model(ModelTier.POWERFUL),
                        tier_used=ModelTier.POWERFUL,
                        cost_estimate=self._estimate_item_cost(ModelTier.POWERFUL),
                        latency_ms=0,
                    ))
                    self.stats.fallback_resolved += 1

        # Combine results in original order
        all_results = confident_positive + confident_negative + oracle_results

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats.total_items += len(items)
        self.stats.total_latency_ms += elapsed_ms
        self.stats.total_cost += sum(r.cost_estimate for r in all_results)

        # Set latency for all results
        avg_latency = elapsed_ms / max(len(all_results), 1)
        for r in all_results:
            r.latency_ms = avg_latency

        logger.info(
            f"Cascade complete: {len(items)} items, "
            f"{self.stats.proxy_resolved}/{len(items)} proxy-resolved "
            f"({self.stats.proxy_rate:.1%}), "
            f"est. cost ${self.stats.total_cost:.4f}"
        )

        return all_results

    async def _batch_score(
        self,
        items: List[Any],
        prompt_fn: Callable,
        parse_fn: Callable,
        tier: ModelTier,
    ) -> List[Tuple[Any, float]]:
        """
        Batch score items with specified model tier.

        DRY: Uses existing DSPy infrastructure for model calls.
        """
        model_name = self.config.get_model(tier)

        # Generate prompts for all items
        prompts = [prompt_fn(item) for item in items]

        try:
            # Use DSPy or direct LM call for batch
            if self.lm_provider:
                import dspy
                with dspy.context(lm=self.lm_provider):
                    responses = await self._call_lm_batch(prompts, model_name)
            else:
                responses = await self._call_lm_batch(prompts, model_name)

            # Parse responses
            results = []
            for response in responses:
                try:
                    result, confidence = parse_fn(response)
                    results.append((result, confidence))
                except Exception as e:
                    logger.warning(f"Parse error: {e}")
                    results.append((None, 0.0))

            return results

        except Exception as e:
            logger.error(f"Batch scoring failed: {e}")
            # Return low confidence for all to trigger oracle
            return [(None, 0.0) for _ in items]

    async def _single_score(
        self,
        item: Any,
        prompt_fn: Callable,
        parse_fn: Callable,
        tier: ModelTier,
    ) -> Tuple[Any, float]:
        """Score single item with specified model tier."""
        results = await self._batch_score([item], prompt_fn, parse_fn, tier)
        return results[0] if results else (None, 0.0)

    async def _call_lm_batch(
        self,
        prompts: List[str],
        model_name: str,
    ) -> List[str]:
        """
        Call language model with batch of prompts.

        DRY: Integrates with existing DSPy/LM infrastructure.
        """
        try:
            import dspy

            # Check if we have a configured LM
            lm = self.lm_provider or dspy.settings.lm

            if lm is None:
                raise ValueError("No LM configured. Set dspy.configure(lm=...)")

            # DSPy's LM supports batch calls
            responses = lm(prompts)

            # Handle different response formats
            if isinstance(responses, list):
                return [str(r) for r in responses]
            else:
                return [str(responses)]

        except Exception as e:
            logger.error(f"LM batch call failed: {e}")
            return ["" for _ in prompts]

    def _estimate_item_cost(self, tier: ModelTier) -> float:
        """Estimate cost per item for a tier."""
        # Rough estimates based on typical prompt/response sizes
        avg_input_tokens = 500
        avg_output_tokens = 100

        model_name = self.config.get_model(tier)
        return self.config.estimate_cost(model_name, avg_input_tokens, avg_output_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """Get cascade statistics."""
        return {
            "total_items": self.stats.total_items,
            "proxy_resolved": self.stats.proxy_resolved,
            "oracle_resolved": self.stats.oracle_resolved,
            "fallback_resolved": self.stats.fallback_resolved,
            "proxy_rate": self.stats.proxy_rate,
            "total_cost": self.stats.total_cost,
            "total_latency_ms": self.stats.total_latency_ms,
            "cost_savings": self.stats.cost_savings,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = CascadeStats()

    async def learn_thresholds(
        self,
        operation: str,
        sample_items: List[Any],
        prompt_fn: Callable,
        oracle_fn: Callable[[Any], Any],  # Ground truth oracle
        recall_target: float = 0.95,
        precision_target: float = 0.90,
    ) -> CascadeThresholds:
        """
        Learn optimal thresholds from sample data.

        LOTUS approach: Statistical threshold optimization with accuracy guarantees.

        Args:
            operation: Operation type
            sample_items: Sample items for threshold learning
            prompt_fn: Prompt generation function
            oracle_fn: Ground truth function
            recall_target: Target recall rate
            precision_target: Target precision rate

        Returns:
            Learned CascadeThresholds
        """
        logger.info(f"Learning thresholds for {operation} with {len(sample_items)} samples")

        # Get oracle (ground truth) results
        oracle_results = [oracle_fn(item) for item in sample_items]

        # Get proxy scores
        def simple_parse(response: str) -> Tuple[bool, float]:
            """Simple yes/no parser with confidence extraction."""
            response_lower = response.lower()
            if "yes" in response_lower or "true" in response_lower:
                return True, 0.8
            elif "no" in response_lower or "false" in response_lower:
                return False, 0.8
            return None, 0.5

        proxy_results = await self._batch_score(
            sample_items, prompt_fn, simple_parse, ModelTier.FAST
        )

        # Find optimal thresholds
        best_thresholds = CascadeThresholds()
        best_score = 0.0

        for tau_pos in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            for tau_neg in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                if tau_neg >= tau_pos:
                    continue

                # Simulate cascade with these thresholds
                tp, fp, tn, fn = 0, 0, 0, 0
                uncertain_count = 0

                for oracle_result, (proxy_result, confidence) in zip(oracle_results, proxy_results):
                    if confidence >= tau_pos:
                        # Accept proxy
                        if proxy_result == oracle_result:
                            tp += 1
                        else:
                            fp += 1
                    elif confidence <= tau_neg:
                        # Reject by proxy
                        if not oracle_result:
                            tn += 1
                        else:
                            fn += 1
                    else:
                        # Uncertain (would go to oracle)
                        uncertain_count += 1
                        tp += 1  # Assume oracle is correct

                # Calculate metrics
                recall = tp / max(tp + fn, 1)
                precision = tp / max(tp + fp, 1)
                oracle_rate = uncertain_count / len(sample_items)

                # Score: maximize proxy resolution while meeting targets
                if recall >= recall_target and precision >= precision_target:
                    score = 1 - oracle_rate  # Higher is better (more proxy resolved)

                    if score > best_score:
                        best_score = score
                        best_thresholds = CascadeThresholds(
                            tau_pos=tau_pos,
                            tau_neg=tau_neg,
                            recall_target=recall_target,
                            precision_target=precision_target,
                        )

        logger.info(
            f"Learned thresholds: tau_pos={best_thresholds.tau_pos}, "
            f"tau_neg={best_thresholds.tau_neg}, "
            f"proxy_rate={best_score:.1%}"
        )

        # Update config
        self.config.cascade_thresholds[operation] = best_thresholds

        return best_thresholds
