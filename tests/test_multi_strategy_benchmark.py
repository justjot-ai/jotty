#!/usr/bin/env python3
"""
Tests for Multi-Strategy Benchmark Utility
===========================================

Tests the high-level benchmarking utility that reduces boilerplate.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


@pytest.mark.asyncio
async def test_benchmark_imports():
    """Test that benchmark utilities are properly exported."""
    # Should be importable from orchestration
    from Jotty.core.orchestration import (
        MultiStrategyBenchmark,
        BenchmarkResults,
        StrategyResult,
        benchmark_strategies,
    )

    assert MultiStrategyBenchmark is not None
    assert BenchmarkResults is not None
    assert StrategyResult is not None
    assert benchmark_strategies is not None


@pytest.mark.asyncio
@patch('os.getenv')
@patch('anthropic.AsyncAnthropic')
async def test_benchmark_basic_usage(mock_anthropic_class, mock_getenv):
    """Test basic benchmark usage with mocked API."""
    from Jotty.core.orchestration import (
        SwarmAdapter,
        MultiStrategyBenchmark,
        MergeStrategy
    )

    # Mock API
    mock_getenv.return_value = "sk-test-key"

    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.usage = Mock(input_tokens=10, output_tokens=20)

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    mock_anthropic_class.return_value = mock_client

    # Create swarms
    swarms = SwarmAdapter.quick_swarms([
        ("Expert1", "Prompt 1"),
        ("Expert2", "Prompt 2"),
    ])

    # Create benchmark
    benchmark = MultiStrategyBenchmark(
        swarms=swarms,
        task="Test task",
        strategies=[MergeStrategy.VOTING, MergeStrategy.CONCATENATE]
    )

    # Run benchmark
    results = await benchmark.run(
        auto_trace=True,
        auto_learn=True,
        auto_threshold=True,
        verbose=False  # Suppress output in tests
    )

    # Verify results structure
    assert results.task == "Test task"
    assert results.num_swarms == 2
    assert len(results.results) == 2
    assert results.total_cost >= 0.0
    assert results.total_time >= 0.0
    assert results.speedup >= 0.0
    assert results.best_strategy is not None

    # Verify strategies were tested
    strategy_names = [r.strategy_name for r in results.results]
    assert "VOTING" in strategy_names
    assert "CONCATENATE" in strategy_names


@pytest.mark.asyncio
@patch('os.getenv')
@patch('anthropic.AsyncAnthropic')
async def test_benchmark_all_strategies(mock_anthropic_class, mock_getenv):
    """Test benchmark with all 5 merge strategies."""
    from Jotty.core.orchestration import SwarmAdapter, benchmark_strategies

    # Mock API
    mock_getenv.return_value = "sk-test-key"

    mock_response = Mock()
    mock_response.content = [Mock(text="Test")]
    mock_response.usage = Mock(input_tokens=10, output_tokens=10)

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    mock_anthropic_class.return_value = mock_client

    # Create swarms using facade function
    swarms = SwarmAdapter.quick_swarms([("S1", "P1"), ("S2", "P2")])

    # Use facade function (default: all 5 strategies)
    bench = benchmark_strategies(swarms, "Test task")

    results = await bench.run(verbose=False)

    # Should test all 5 strategies by default
    assert len(results.results) == 5


@pytest.mark.asyncio
async def test_benchmark_results_print_summary():
    """Test that BenchmarkResults.print_summary() works."""
    from Jotty.core.orchestration import (
        BenchmarkResults,
        StrategyResult,
        MergeStrategy,
        SwarmResult
    )
    import io
    import sys

    # Create mock results
    strategy_result = StrategyResult(
        strategy=MergeStrategy.VOTING,
        strategy_name="VOTING",
        result=SwarmResult(
            swarm_name="test",
            output="Test output",
            success=True,
            confidence=0.8
        ),
        execution_time=1.0,
        cost=0.001
    )

    results = BenchmarkResults(
        task="Test",
        num_swarms=2,
        results=[strategy_result],
        total_cost=0.001,
        total_time=1.0,
        speedup=2.0,
        best_strategy=strategy_result
    )

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        results.print_summary(verbose=True)
        output = sys.stdout.getvalue()

        # Verify output contains key sections
        assert "MULTI-STRATEGY BENCHMARK RESULTS" in output
        assert "Cost Analysis" in output
        assert "Performance" in output
        assert "Strategy Comparison" in output
        assert "Best Strategy" in output
        assert "VOTING" in output

    finally:
        sys.stdout = old_stdout


@pytest.mark.asyncio
async def test_benchmark_auto_integration_flags():
    """Test that auto-integration flags work correctly."""
    from Jotty.core.orchestration import MultiStrategyBenchmark, MergeStrategy

    # Create mock swarms
    class MockSwarm:
        name = "Mock"

        async def execute(self, task):
            from Jotty.core.orchestration import SwarmResult
            return SwarmResult(
                swarm_name=self.name,
                output="Mock output",
                success=True,
                confidence=0.9
            )

    swarms = [MockSwarm(), MockSwarm()]

    benchmark = MultiStrategyBenchmark(
        swarms=swarms,
        task="Test",
        strategies=[MergeStrategy.VOTING]  # Just one for speed
    )

    # Test with all auto flags enabled
    results = await benchmark.run(
        auto_trace=True,
        auto_learn=True,
        auto_threshold=True,
        verbose=False
    )

    assert results is not None
    assert len(results.results) == 1

    # Test with all auto flags disabled
    results2 = await benchmark.run(
        auto_trace=False,
        auto_learn=False,
        auto_threshold=False,
        verbose=False
    )

    assert results2 is not None
    assert len(results2.results) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
