"""
Test Evaluation Framework

Tests reproducibility, benchmarks, evaluation protocol, and ablation studies.
"""
import sys
import random
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
    ComponentType
)
from core.foundation.data_structures import SwarmConfig


def test_reproducibility():
    """Test reproducibility with fixed seeds."""
    print("=== Test 1: Reproducibility ===\n")
    
    try:
        # Set seeds
        seeds1 = set_reproducible_seeds(random_seed=42)
        random_value1 = random.random()
        
        # Reset and set same seed
        seeds2 = set_reproducible_seeds(random_seed=42)
        random_value2 = random.random()
        
        # Should be same
        assert random_value1 == random_value2, "Random values should be same with same seed"
        print(f"✅ Reproducibility: Same seed produces same random value")
        
        # Test config
        config = ReproducibilityConfig(random_seed=42)
        assert config.random_seed == 42
        print(f"✅ Reproducibility config: seed={config.random_seed}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_benchmark():
    """Test custom benchmark."""
    print("\n=== Test 2: Custom Benchmark ===\n")
    
    try:
        # Create benchmark
        benchmark = CustomBenchmark(
            name="test",
            tasks=[
                {"id": "task1", "question": "What is 2+2?", "answer": "4"},
            ]
        )
        
        # Simple agent
        class Agent:
            def run(self, question):
                return "4"
        
        # Evaluate
        metrics = benchmark.evaluate(Agent())
        
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 1
        assert metrics.pass_rate == 1.0
        
        print(f"✅ Benchmark evaluation: pass_rate={metrics.pass_rate:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_protocol():
    """Test evaluation protocol."""
    print("\n=== Test 3: Evaluation Protocol ===\n")
    
    try:
        benchmark = CustomBenchmark(
            name="test",
            tasks=[
                {"id": "task1", "question": "What is 2+2?", "answer": "4"},
            ]
        )
        
        class Agent:
            def run(self, question):
                return "4"
        
        protocol = EvaluationProtocol(
            benchmark=benchmark,
            n_runs=3,
            random_seed=42
        )
        
        report = protocol.evaluate(Agent(), save_results=False)
        
        assert report.n_runs == 3
        assert report.mean_pass_rate > 0
        assert len(report.runs) == 3
        
        print(f"✅ Evaluation protocol: {report.n_runs} runs, "
              f"pass_rate={report.mean_pass_rate:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ablation_study():
    """Test ablation study."""
    print("\n=== Test 4: Ablation Study ===\n")
    
    try:
        benchmark = CustomBenchmark(
            name="test",
            tasks=[
                {"id": "task1", "question": "What is 2+2?", "answer": "4"},
            ]
        )
        
        def create_agent(config):
            class Agent:
                def run(self, question):
                    if hasattr(config, 'enable_rl') and config.enable_rl:
                        return "4"
                    return "5"
            return Agent()
        
        baseline_config = SwarmConfig(enable_rl=True)
        
        components = [
            {
                "name": "learning",
                "type": ComponentType.FEATURE,
                "disable": lambda c: setattr(c, 'enable_rl', False)
            },
        ]
        
        study = AblationStudy(
            benchmark=benchmark,
            agent_factory=create_agent,
            components=components,
            n_runs=2,
            random_seed=42,
            baseline_config=baseline_config
        )
        
        result = study.run()
        
        assert len(result.component_contributions) == 1
        assert result.baseline_report.mean_pass_rate > 0
        
        print(f"✅ Ablation study: {len(result.component_contributions)} components tested")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Evaluation Framework Tests")
    print("=" * 60)
    
    tests = [
        test_reproducibility,
        test_custom_benchmark,
        test_evaluation_protocol,
        test_ablation_study,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Evaluation framework working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
