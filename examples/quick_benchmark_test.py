"""
Quick Benchmark Test

A simplified benchmark test that runs quickly for testing purposes.
Use this for quick validation without full LLM calls.
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.evaluation import CustomBenchmark
from examples.benchmark_test import JottyBenchmarkWrapper
from core.foundation.data_structures import SwarmConfig


def create_simple_benchmark():
    """Create a very simple benchmark for quick testing."""
    return CustomBenchmark(
        name="simple_test",
        tasks=[
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "What is 3*3?", "answer": "9"},
        ]
    )


def main():
    """Run quick benchmark test."""
    print("=" * 60)
    print("Quick Benchmark Test")
    print("=" * 60)
    
    # Create benchmark
    benchmark = create_simple_benchmark()
    print(f"\n📊 Benchmark: {benchmark.name}")
    print(f"   Tasks: {len(benchmark.tasks)}")
    
    # Create wrapper
    config = SwarmConfig(random_seed=42)
    wrapper = JottyBenchmarkWrapper(config=config, use_multi_agent=False)
    
    # Run evaluation
    print("\n🚀 Running evaluation...")
    metrics = benchmark.evaluate(wrapper)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total Tasks: {metrics.total_tasks}")
    print(f"Successful: {metrics.successful_tasks}")
    print(f"Failed: {metrics.failed_tasks}")
    print(f"Pass Rate: {metrics.pass_rate:.2%}")
    print(f"Avg Execution Time: {metrics.avg_execution_time:.2f}s")
    
    # Print task results
    print("\nTask Results:")
    for result in metrics.results:
        status = "✅" if result.success else "❌"
        answer = result.answer or result.error or "No answer"
        print(f"  {status} {result.task_id}: {answer[:50]}")
    
    print("\n✅ Quick test complete!")


if __name__ == "__main__":
    main()
