"""
LOTUS Optimization Benchmark
============================

Demonstrates real cost and performance advantages of LOTUS optimization.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from .adaptive_validator import AdaptiveValidator
from .config import LotusConfig
from .semantic_cache import SemanticCache


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    total_items: int
    total_time_ms: float
    total_cost: float
    llm_calls: int
    cache_hits: int = 0
    validations_skipped: int = 0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Items: {self.total_items}\n"
            f"  Time: {self.total_time_ms:.1f}ms\n"
            f"  Cost: ${self.total_cost:.4f}\n"
            f"  LLM Calls: {self.llm_calls}\n"
            f"  Cache Hits: {self.cache_hits}\n"
            f"  Validations Skipped: {self.validations_skipped}"
        )


class MockLLM:
    """
    Mock LLM that simulates real costs and latencies.

    Pricing (per 1M tokens):
    - Haiku: $0.25 input, $1.25 output
    - Sonnet: $3.00 input, $15.00 output
    - Opus: $15.00 input, $75.00 output
    """

    COSTS = {
        "haiku": (0.25, 1.25),  # 60x cheaper than Opus
        "sonnet": (3.0, 15.0),  # 5x cheaper than Opus
        "opus": (15.0, 75.0),  # Most expensive
    }

    LATENCIES_MS = {
        "haiku": 50,  # Fast
        "sonnet": 150,  # Medium
        "opus": 500,  # Slow
    }

    def __init__(self) -> None:
        self.calls = []
        self.total_cost = 0.0
        self.total_latency = 0.0

    async def call(self, prompt: str, model: str = "sonnet") -> tuple:
        """Simulate LLM call with realistic cost and latency."""
        # Simulate tokens (rough estimate)
        input_tokens = len(prompt) // 4
        output_tokens = 50  # Average response

        # Calculate cost
        costs = self.COSTS.get(model, self.COSTS["sonnet"])
        cost = (input_tokens / 1_000_000) * costs[0] + (output_tokens / 1_000_000) * costs[1]

        # Simulate latency
        latency = self.LATENCIES_MS.get(model, 150)
        await asyncio.sleep(latency / 1000)  # Convert to seconds

        self.calls.append({"model": model, "cost": cost, "latency": latency})
        self.total_cost += cost
        self.total_latency += latency

        # Generate mock response
        response = f"Response for: {prompt[:50]}..."
        confidence = random.uniform(0.6, 0.95)

        return response, confidence

    def reset(self) -> None:
        self.calls = []
        self.total_cost = 0.0
        self.total_latency = 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": len(self.calls),
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency,
            "calls_by_model": {
                model: len([c for c in self.calls if c["model"] == model])
                for model in ["haiku", "sonnet", "opus"]
            },
        }


async def benchmark_without_optimization(
    items: List[str],
    llm: MockLLM,
) -> BenchmarkResult:
    """
    Baseline: No optimization - always use Opus, no caching.
    """
    llm.reset()
    start = time.time()

    for item in items:
        prompt = f"Process this item: {item}"
        await llm.call(prompt, model="opus")  # Always expensive model

    elapsed_ms = (time.time() - start) * 1000
    stats = llm.get_stats()

    return BenchmarkResult(
        name="WITHOUT OPTIMIZATION (Opus only)",
        total_items=len(items),
        total_time_ms=elapsed_ms,
        total_cost=stats["total_cost"],
        llm_calls=stats["total_calls"],
    )


async def benchmark_with_cascade(
    items: List[str],
    llm: MockLLM,
) -> BenchmarkResult:
    """
    With Model Cascade: Haiku first, Sonnet for uncertain, Opus for complex.
    """
    llm.reset()
    start = time.time()

    haiku_resolved = 0
    sonnet_resolved = 0
    opus_resolved = 0

    for item in items:
        prompt = f"Process this item: {item}"

        # Try Haiku first
        response, confidence = await llm.call(prompt, model="haiku")

        if confidence >= 0.85:
            haiku_resolved += 1
            continue

        # Try Sonnet for uncertain
        response, confidence = await llm.call(prompt, model="sonnet")

        if confidence >= 0.7:
            sonnet_resolved += 1
            continue

        # Fall back to Opus for complex
        response, confidence = await llm.call(prompt, model="opus")
        opus_resolved += 1

    elapsed_ms = (time.time() - start) * 1000
    stats = llm.get_stats()

    return BenchmarkResult(
        name=f"WITH CASCADE (H:{haiku_resolved} S:{sonnet_resolved} O:{opus_resolved})",
        total_items=len(items),
        total_time_ms=elapsed_ms,
        total_cost=stats["total_cost"],
        llm_calls=stats["total_calls"],
    )


async def benchmark_with_cache(
    items: List[str],
    llm: MockLLM,
    repeat_rate: float = 0.3,  # 30% repeat queries
) -> BenchmarkResult:
    """
    With Semantic Cache: Cache hits are free.
    """
    llm.reset()
    config = LotusConfig()
    cache = SemanticCache(config)

    start = time.time()

    # Create items with some repeats
    all_items = items.copy()
    num_repeats = int(len(items) * repeat_rate)
    all_items.extend(random.choices(items, k=num_repeats))
    random.shuffle(all_items)

    for item in all_items:
        instruction = "Process this item"

        # Check cache
        hit, result = cache.get(instruction, item)
        if hit:
            continue  # Free!

        # Cache miss - call LLM
        prompt = f"{instruction}: {item}"
        response, _ = await llm.call(prompt, model="sonnet")

        # Store in cache
        cache.put(instruction, item, response)

    elapsed_ms = (time.time() - start) * 1000
    stats = llm.get_stats()
    cache_stats = cache.get_stats()

    return BenchmarkResult(
        name=f"WITH CACHE ({repeat_rate:.0%} repeats)",
        total_items=len(all_items),
        total_time_ms=elapsed_ms,
        total_cost=stats["total_cost"],
        llm_calls=stats["total_calls"],
        cache_hits=cache_stats["hits"],
    )


async def benchmark_with_adaptive_validation(
    items: List[str],
    llm: MockLLM,
) -> BenchmarkResult:
    """
    With Adaptive Validation: Skip validation for trusted agents.
    """
    llm.reset()
    validator = AdaptiveValidator(
        skip_threshold=0.90,
        sample_rate=0.10,
        min_samples=5,
    )

    start = time.time()

    validations_run = 0
    validations_skipped = 0

    # Warm up: Build trust
    for i in range(10):
        validator.record_result("processor", "process", success=True)

    for item in items:
        prompt = f"Process: {item}"

        # Main processing (always runs)
        await llm.call(prompt, model="sonnet")

        # Check if validation needed
        decision = validator.should_validate("processor", "process")

        if decision.should_validate:
            # Run validation (extra LLM call)
            await llm.call(f"Validate: {item}", model="haiku")
            validator.record_result("processor", "process", success=True)
            validations_run += 1
        else:
            validator.record_skip("processor", "process")
            validations_skipped += 1

    elapsed_ms = (time.time() - start) * 1000
    stats = llm.get_stats()

    return BenchmarkResult(
        name="WITH ADAPTIVE VALIDATION",
        total_items=len(items),
        total_time_ms=elapsed_ms,
        total_cost=stats["total_cost"],
        llm_calls=stats["total_calls"],
        validations_skipped=validations_skipped,
    )


async def benchmark_full_lotus(
    items: List[str],
    llm: MockLLM,
    repeat_rate: float = 0.3,
) -> BenchmarkResult:
    """
    Full LOTUS: Cascade + Cache + Adaptive Validation combined.
    """
    llm.reset()
    config = LotusConfig()
    cache = SemanticCache(config)
    validator = AdaptiveValidator(skip_threshold=0.90, sample_rate=0.10, min_samples=5)

    start = time.time()

    # Create items with repeats
    all_items = items.copy()
    num_repeats = int(len(items) * repeat_rate)
    all_items.extend(random.choices(items, k=num_repeats))
    random.shuffle(all_items)

    # Warm up validator
    for i in range(10):
        validator.record_result("processor", "process", success=True)

    cache_hits = 0
    validations_skipped = 0
    haiku_calls = 0
    sonnet_calls = 0
    opus_calls = 0

    for item in all_items:
        instruction = "Process this item"

        # 1. Check cache first
        hit, result = cache.get(instruction, item)
        if hit:
            cache_hits += 1
            continue  # Free!

        prompt = f"{instruction}: {item}"

        # 2. Cascade: Try cheap model first
        response, confidence = await llm.call(prompt, model="haiku")
        haiku_calls += 1

        if confidence < 0.85:
            # Need better model
            response, confidence = await llm.call(prompt, model="sonnet")
            sonnet_calls += 1

            if confidence < 0.7:
                response, confidence = await llm.call(prompt, model="opus")
                opus_calls += 1

        # Cache the result
        cache.put(instruction, item, response)

        # 3. Adaptive validation
        decision = validator.should_validate("processor", "process")
        if decision.should_validate:
            await llm.call(f"Validate: {item}", model="haiku")
            validator.record_result("processor", "process", success=True)
        else:
            validator.record_skip("processor", "process")
            validations_skipped += 1

    elapsed_ms = (time.time() - start) * 1000
    stats = llm.get_stats()

    return BenchmarkResult(
        name=f"FULL LOTUS (H:{haiku_calls} S:{sonnet_calls} O:{opus_calls})",
        total_items=len(all_items),
        total_time_ms=elapsed_ms,
        total_cost=stats["total_cost"],
        llm_calls=stats["total_calls"],
        cache_hits=cache_hits,
        validations_skipped=validations_skipped,
    )


async def run_benchmarks() -> Any:
    """Run all benchmarks and display comparison."""
    print("=" * 70)
    print("LOTUS OPTIMIZATION BENCHMARK")
    print("=" * 70)

    # Generate test data
    items = [f"Document {i}: " + "x" * random.randint(100, 500) for i in range(50)]

    llm = MockLLM()

    print("\nTest Configuration:")
    print(f"  Items: {len(items)}")
    print("  Repeat rate for cache tests: 30%")
    print("  Models: Haiku ($0.25/1M), Sonnet ($3/1M), Opus ($15/1M)")
    print()

    # Run benchmarks
    results = []

    print("Running benchmarks...\n")

    # 1. Baseline (no optimization)
    result1 = await benchmark_without_optimization(items, llm)
    results.append(result1)
    print(f"1. {result1}\n")

    # 2. With cascade
    result2 = await benchmark_with_cascade(items, llm)
    results.append(result2)
    print(f"2. {result2}\n")

    # 3. With cache
    result3 = await benchmark_with_cache(items, llm, repeat_rate=0.3)
    results.append(result3)
    print(f"3. {result3}\n")

    # 4. With adaptive validation
    result4 = await benchmark_with_adaptive_validation(items, llm)
    results.append(result4)
    print(f"4. {result4}\n")

    # 5. Full LOTUS
    result5 = await benchmark_full_lotus(items, llm, repeat_rate=0.3)
    results.append(result5)
    print(f"5. {result5}\n")

    # Summary
    baseline_cost = result1.total_cost
    baseline_time = result1.total_time_ms

    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<45} {'Cost':>10} {'Savings':>10} {'Time':>10}")
    print("-" * 75)

    for r in results:
        savings = (1 - r.total_cost / baseline_cost) * 100 if baseline_cost > 0 else 0
        _time_reduction = (1 - r.total_time_ms / baseline_time) * 100 if baseline_time > 0 else 0
        print(f"{r.name:<45} ${r.total_cost:>8.4f} {savings:>9.1f}% {r.total_time_ms:>8.0f}ms")

    print("\n" + "=" * 70)

    # Calculate total advantage
    lotus_cost = result5.total_cost
    cost_reduction = (1 - lotus_cost / baseline_cost) * 100 if baseline_cost > 0 else 0
    cost_multiplier = baseline_cost / lotus_cost if lotus_cost > 0 else float("inf")

    print("\nFULL LOTUS ADVANTAGE:")
    print(f"  Cost reduction: {cost_reduction:.1f}% ({cost_multiplier:.1f}x cheaper)")
    print(f"  Cache hits: {result5.cache_hits} (free queries)")
    print(f"  Validations skipped: {result5.validations_skipped}")
    print(f"  Baseline cost: ${baseline_cost:.4f}")
    print(f"  LOTUS cost: ${lotus_cost:.4f}")
    print(f"  SAVINGS: ${baseline_cost - lotus_cost:.4f}")
    print("=" * 70)

    return results


def run_sync() -> Any:
    """Synchronous entry point."""
    return asyncio.run(run_benchmarks())


if __name__ == "__main__":
    run_sync()
