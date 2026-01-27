"""
Evaluation Framework Examples

Demonstrates:
- Reproducibility (fixed seeds)
- Benchmark evaluation
- Evaluation protocol (multiple runs, variance tracking)
- Ablation studies
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.evaluation import (
    ReproducibilityConfig,
    set_reproducible_seeds,
    CustomBenchmark,
    EvaluationProtocol,
    AblationStudy,
    ComponentContribution,
    ComponentType
)
from core.foundation.data_structures import SwarmConfig


def example_reproducibility():
    """Example: Reproducibility with fixed seeds."""
    print("=== Example 1: Reproducibility ===\n")
    
    # Set seeds
    seeds = set_reproducible_seeds(random_seed=42, numpy_seed=42)
    print(f"✅ Set seeds: {seeds}")
    
    # Use config
    config = ReproducibilityConfig(
        random_seed=42,
        numpy_seed=42,
        enable_deterministic=True
    )
    print(f"✅ Reproducibility config: seed={config.random_seed}")
    
    # Verify reproducibility info
    from core.evaluation.reproducibility import get_reproducibility_info
    info = get_reproducibility_info()
    print(f"✅ Reproducibility info: {info}")


def example_custom_benchmark():
    """Example: Custom benchmark evaluation."""
    print("\n\n=== Example 2: Custom Benchmark ===\n")
    
    # Create custom benchmark
    benchmark = CustomBenchmark(
        name="test_benchmark",
        tasks=[
            {"id": "task1", "question": "What is 2+2?", "answer": "4"},
            {"id": "task2", "question": "What is Python?", "answer": "A programming language"},
            {"id": "task3", "question": "What is 10*5?", "answer": "50"},
        ]
    )
    
    # Simple agent mock
    class SimpleAgent:
        def run(self, question):
            # Mock agent that sometimes gets it right
            if "2+2" in question:
                return "4"
            elif "Python" in question:
                return "A programming language"
            elif "10*5" in question:
                return "50"
            return "Unknown"
    
    agent = SimpleAgent()
    
    # Evaluate
    metrics = benchmark.evaluate(agent)
    
    print(f"✅ Benchmark: {benchmark.name}")
    print(f"✅ Total tasks: {metrics.total_tasks}")
    print(f"✅ Successful: {metrics.successful_tasks}")
    print(f"✅ Pass rate: {metrics.pass_rate:.2%}")
    print(f"✅ Avg execution time: {metrics.avg_execution_time:.2f}s")


def example_evaluation_protocol():
    """Example: Evaluation protocol with multiple runs."""
    print("\n\n=== Example 3: Evaluation Protocol ===\n")
    
    # Create benchmark
    benchmark = CustomBenchmark(
        name="test_benchmark",
        tasks=[
            {"id": "task1", "question": "What is 2+2?", "answer": "4"},
            {"id": "task2", "question": "What is 3+3?", "answer": "6"},
        ]
    )
    
    # Simple agent
    class SimpleAgent:
        def run(self, question):
            if "2+2" in question:
                return "4"
            elif "3+3" in question:
                return "6"
            return "Unknown"
    
    agent = SimpleAgent()
    
    # Run evaluation protocol
    protocol = EvaluationProtocol(
        benchmark=benchmark,
        n_runs=3,
        random_seed=42
    )
    
    report = protocol.evaluate(agent, save_results=False)
    
    print(f"✅ Benchmark: {report.benchmark_name}")
    print(f"✅ Number of runs: {report.n_runs}")
    print(f"✅ Mean pass rate: {report.mean_pass_rate:.2%} ± {report.std_pass_rate:.2%}")
    print(f"✅ Mean cost: ${report.mean_cost:.4f} ± ${report.std_cost:.4f}")


def example_ablation_study():
    """Example: Ablation study."""
    print("\n\n=== Example 4: Ablation Study ===\n")
    
    # Create benchmark
    benchmark = CustomBenchmark(
        name="test_benchmark",
        tasks=[
            {"id": "task1", "question": "What is 2+2?", "answer": "4"},
        ]
    )
    
    # Agent factory
    def create_agent(config):
        class ConfigurableAgent:
            def __init__(self, config):
                self.config = config
            
            def run(self, question):
                # Mock: learning helps, memory doesn't
                if hasattr(config, 'enable_rl') and config.enable_rl:
                    return "4"  # Correct with learning
                return "5"  # Wrong without learning
        
        return ConfigurableAgent(config)
    
    # Create baseline config
    baseline_config = SwarmConfig(enable_rl=True)
    
    # Define components to test
    components = [
        {
            "name": "learning",
            "type": ComponentType.FEATURE,
            "disable": lambda c: setattr(c, 'enable_rl', False)
        },
    ]
    
    # Run ablation study
    study = AblationStudy(
        benchmark=benchmark,
        agent_factory=create_agent,
        components=components,
        n_runs=2,
        random_seed=42,
        baseline_config=baseline_config
    )
    
    result = study.run()
    
    print(f"✅ Study: {result.study_name}")
    print(f"✅ Baseline pass rate: {result.baseline_report.mean_pass_rate:.2%}")
    
    for contrib in result.component_contributions:
        print(f"✅ Component '{contrib.component_name}':")
        print(f"   Contribution: {contrib.contribution:.2%}")
        print(f"   Cost impact: ${contrib.cost_impact:.4f}")
    
    if result.recommendations:
        print(f"\n✅ Recommendations:")
        for rec in result.recommendations:
            print(f"   - {rec}")


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Framework Examples")
    print("=" * 60)
    
    example_reproducibility()
    example_custom_benchmark()
    example_evaluation_protocol()
    example_ablation_study()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
