#!/usr/bin/env python3
"""
Multi-Strategy Benchmark Utility
=================================

High-level utility for comparing multiple merge strategies automatically.
Reduces boilerplate for common benchmarking scenarios.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
from .multi_swarm_coordinator import MultiSwarmCoordinator, SwarmResult, MergeStrategy
from ..observability import get_distributed_tracer
from ..learning import get_cost_aware_td_lambda
from ..safety import get_adaptive_threshold_manager


@dataclass
class StrategyResult:
    """Result from a single strategy execution."""
    strategy: MergeStrategy
    strategy_name: str
    result: SwarmResult
    execution_time: float
    cost: float
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Results from multi-strategy benchmark."""
    task: str
    num_swarms: int
    results: List[StrategyResult]
    total_cost: float
    total_time: float
    speedup: float
    best_strategy: StrategyResult

    def print_summary(self, verbose: bool = True):
        """Print formatted benchmark summary."""
        print("\n" + "="*80)
        print("MULTI-STRATEGY BENCHMARK RESULTS")
        print("="*80 + "\n")

        print(f"📋 Task: {self.task[:80]}...")
        print(f"🧠 Swarms: {self.num_swarms}")
        print(f"🎯 Strategies tested: {len(self.results)}")
        print()

        # Cost analysis
        print("💰 Cost Analysis:")
        print(f"   Total cost: ${self.total_cost:.6f}")
        print(f"   Average per strategy: ${self.total_cost / len(self.results):.6f}")
        print(f"   Total API calls: {self.num_swarms * len(self.results)}")
        print()

        # Performance
        print("⚡ Performance:")
        print(f"   Total execution time: {self.total_time:.2f}s")
        print(f"   Average per strategy: {self.total_time / len(self.results):.2f}s")
        print(f"   Speedup from parallelization: {self.speedup:.2f}x")
        print()

        # Strategy comparison
        print("🏆 Strategy Comparison:")
        print(f"{'Strategy':<15} {'Success':<8} {'Confidence':<12} {'Cost':<12} {'Time (s)':<10}")
        print(f"{'-'*60}")
        for r in self.results:
            print(f"{r.strategy_name:<15} "
                  f"{'✓' if r.result.success else '✗':<8} "
                  f"{r.result.confidence:<12.2f} "
                  f"${r.cost:<11.6f} "
                  f"{r.execution_time:<10.2f}")
        print()

        # Best strategy
        print(f"🎯 Best Strategy: {self.best_strategy.strategy_name}")
        print(f"   Cost-effectiveness: ${self.best_strategy.cost:.6f} / {self.best_strategy.result.confidence:.2f} confidence")
        print()

        if verbose:
            print("📄 Results Preview:\n")
            for r in self.results:
                preview = r.result.output[:150] if r.result.output else "No output"
                print(f"[{r.strategy_name}]")
                print(f"  {preview}...")
                print()


class MultiStrategyBenchmark:
    """
    High-level utility for benchmarking multiple merge strategies.

    Usage:
        swarms = SwarmAdapter.quick_swarms([...])
        benchmark = MultiStrategyBenchmark(
            swarms=swarms,
            task="Research AI trends",
            strategies=[MergeStrategy.VOTING, MergeStrategy.CONCATENATE]
        )
        results = await benchmark.run(auto_trace=True, auto_learn=True)
        results.print_summary()
    """

    def __init__(
        self,
        swarms: List[Any],
        task: str,
        strategies: Optional[List[MergeStrategy]] = None,
        coordinator: Optional[MultiSwarmCoordinator] = None
    ):
        """
        Initialize benchmark.

        Args:
            swarms: List of swarms to execute
            task: Task to execute
            strategies: List of strategies to test (default: all 5)
            coordinator: Optional coordinator instance (creates if not provided)
        """
        self.swarms = swarms
        self.task = task
        self.strategies = strategies or [
            MergeStrategy.VOTING,
            MergeStrategy.CONCATENATE,
            MergeStrategy.BEST_OF_N,
            MergeStrategy.ENSEMBLE,
            MergeStrategy.FIRST_SUCCESS,
        ]
        self.coordinator = coordinator or MultiSwarmCoordinator()

    async def run(
        self,
        auto_trace: bool = True,
        auto_learn: bool = True,
        auto_threshold: bool = True,
        verbose: bool = True
    ) -> BenchmarkResults:
        """
        Run benchmark across all strategies.

        Args:
            auto_trace: Automatically use distributed tracing
            auto_learn: Automatically use cost-aware learning
            auto_threshold: Automatically track adaptive thresholds
            verbose: Print progress messages

        Returns:
            BenchmarkResults with all results and analysis
        """
        # Initialize enhancements if requested
        tracer = get_distributed_tracer("benchmark") if auto_trace else None
        learner = get_cost_aware_td_lambda() if auto_learn else None
        threshold_mgr = get_adaptive_threshold_manager() if auto_threshold else None

        if verbose:
            print(f"\n🚀 Starting benchmark with {len(self.strategies)} strategies...")
            print(f"   Swarms: {len(self.swarms)}")
            print(f"   Auto-trace: {auto_trace}")
            print(f"   Auto-learn: {auto_learn}")
            print(f"   Auto-threshold: {auto_threshold}")
            print()

        results = []
        total_cost = 0.0

        # Main trace context
        trace_ctx = tracer.trace("multi_strategy_benchmark") if tracer else None
        main_trace_id = trace_ctx.__enter__() if trace_ctx else None

        try:
            for strategy in self.strategies:
                strategy_name = strategy.name

                if verbose:
                    print(f"▶ Testing {strategy_name}...")

                # Nested trace for strategy
                strategy_trace_ctx = tracer.trace(f"strategy_{strategy_name.lower()}") if tracer else None
                strategy_trace_id = strategy_trace_ctx.__enter__() if strategy_trace_ctx else None

                try:
                    start_time = time.time()

                    # Execute
                    result = await self.coordinator.execute_parallel(
                        swarms=self.swarms,
                        task=self.task,
                        merge_strategy=strategy
                    )

                    execution_time = time.time() - start_time
                    cost = result.metadata.get('cost_usd', 0.0)
                    total_cost += cost

                    # Auto-learning
                    if learner:
                        learner.update(
                            state={"strategy": strategy_name, "num_swarms": len(self.swarms)},
                            action={"merge_strategy": strategy_name},
                            reward=1.0 if result.success else 0.0,
                            next_state={"completed": True},
                            cost_usd=cost
                        )

                    # Auto-threshold tracking
                    if threshold_mgr:
                        cost_per_swarm = cost / len(self.swarms) if self.swarms else 0.0
                        threshold_mgr.record_observation(
                            f"swarm_cost_{strategy_name.lower()}",
                            cost_per_swarm,
                            cost_per_swarm > 0.01,
                            0.01
                        )

                    results.append(StrategyResult(
                        strategy=strategy,
                        strategy_name=strategy_name,
                        result=result,
                        execution_time=execution_time,
                        cost=cost,
                        trace_id=strategy_trace_id,
                        metadata={"auto_trace": auto_trace, "auto_learn": auto_learn}
                    ))

                    if verbose:
                        print(f"  ✓ {strategy_name}: {result.confidence:.2f} confidence, "
                              f"${cost:.6f}, {execution_time:.2f}s\n")

                finally:
                    if strategy_trace_ctx:
                        strategy_trace_ctx.__exit__(None, None, None)

        finally:
            if trace_ctx:
                trace_ctx.__exit__(None, None, None)

        # Calculate metrics
        total_time = sum(r.execution_time for r in results)
        sequential_estimate = (total_time / len(results)) * len(self.swarms) * len(results)
        speedup = sequential_estimate / total_time if total_time > 0 else 0.0

        # Find best strategy (cost-effectiveness)
        best = min(results, key=lambda r: r.cost / (r.result.confidence + 0.01))

        return BenchmarkResults(
            task=self.task,
            num_swarms=len(self.swarms),
            results=results,
            total_cost=total_cost,
            total_time=total_time,
            speedup=speedup,
            best_strategy=best
        )


# Facade function
def benchmark_strategies(
    swarms: List[Any],
    task: str,
    strategies: Optional[List[MergeStrategy]] = None,
    **kwargs
) -> "MultiStrategyBenchmark":
    """
    Quick function to create a benchmark.

    Usage:
        benchmark = benchmark_strategies(swarms, task)
        results = await benchmark.run()
        results.print_summary()
    """
    return MultiStrategyBenchmark(swarms, task, strategies, **kwargs)
