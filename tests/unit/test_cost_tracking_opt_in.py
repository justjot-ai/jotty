"""
Test Cost Tracking and Monitoring Opt-In Functionality

Verifies that:
1. Features work when enabled
2. Features don't break anything when disabled (default)
3. No performance impact when disabled
4. Backward compatibility maintained
"""

import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.foundation.data_structures import SwarmConfig
from core.llm.unified import UnifiedLLM
from core.monitoring import CostTracker, ExecutionStatus, MonitoringFramework


def test_cost_tracker_disabled():
    """Test that cost tracker doesn't break when disabled."""
    print("=== Test 1: Cost Tracker Disabled (Default) ===")

    # Create tracker with tracking disabled (default)
    tracker = CostTracker(enable_tracking=False)

    # Try to record calls - should not break
    try:
        record = tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )

        # Should return a record with cost=0.0
        assert record.cost == 0.0, f"Expected cost=0.0 when disabled, got {record.cost}"
        print("✅ Cost tracker disabled: Records calls but cost=0.0")

        # Get metrics - should return zeros
        metrics = tracker.get_metrics()
        assert metrics.total_cost == 0.0, f"Expected total_cost=0.0, got {metrics.total_cost}"
        assert metrics.total_calls == 0, f"Expected total_calls=0, got {metrics.total_calls}"
        print("✅ Cost tracker disabled: Metrics return zeros")

        return True
    except Exception as e:
        print(f"❌ Cost tracker disabled test failed: {e}")
        return False


def test_cost_tracker_enabled():
    """Test that cost tracker works when enabled."""
    print("\n=== Test 2: Cost Tracker Enabled ===")

    # Create tracker with tracking enabled
    tracker = CostTracker(enable_tracking=True)

    try:
        # Record a call
        record = tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )

        # Should have non-zero cost
        assert record.cost > 0.0, f"Expected cost > 0.0 when enabled, got {record.cost}"
        print(f"✅ Cost tracker enabled: Recorded call with cost=${record.cost:.6f}")

        # Get metrics - should have values
        metrics = tracker.get_metrics()
        assert metrics.total_cost > 0.0, f"Expected total_cost > 0.0, got {metrics.total_cost}"
        assert metrics.total_calls == 1, f"Expected total_calls=1, got {metrics.total_calls}"
        print(f"✅ Cost tracker enabled: Metrics show total_cost=${metrics.total_cost:.6f}")

        return True
    except Exception as e:
        print(f"❌ Cost tracker enabled test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_monitoring_disabled():
    """Test that monitoring doesn't break when disabled."""
    print("\n=== Test 3: Monitoring Disabled (Default) ===")

    # Create monitor with monitoring disabled (default)
    monitor = MonitoringFramework(enable_monitoring=False)

    try:
        # Try to start execution - should not break
        exec_metrics = monitor.start_execution("test_agent", "test_task")

        # Should return ExecutionMetrics but not track
        assert exec_metrics.agent_name == "test_agent"
        print("✅ Monitoring disabled: start_execution works")

        # Finish execution - should not break
        monitor.finish_execution(exec_metrics, status=ExecutionStatus.SUCCESS)
        print("✅ Monitoring disabled: finish_execution works")

        # Get performance metrics - should return zeros
        perf_metrics = monitor.get_performance_metrics()
        assert (
            perf_metrics.total_executions == 0
        ), f"Expected 0 executions, got {perf_metrics.total_executions}"
        print("✅ Monitoring disabled: get_performance_metrics returns zeros")

        return True
    except Exception as e:
        print(f"❌ Monitoring disabled test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_monitoring_enabled():
    """Test that monitoring works when enabled."""
    print("\n=== Test 4: Monitoring Enabled ===")

    # Create monitor with monitoring enabled
    monitor = MonitoringFramework(enable_monitoring=True)

    try:
        # Start execution
        exec_metrics = monitor.start_execution("test_agent", "test_task")
        assert exec_metrics.agent_name == "test_agent"
        print("✅ Monitoring enabled: start_execution works")

        # Finish execution
        monitor.finish_execution(
            exec_metrics,
            status=ExecutionStatus.SUCCESS,
            input_tokens=1000,
            output_tokens=500,
            cost=0.015,
        )
        print("✅ Monitoring enabled: finish_execution works")

        # Get performance metrics - should have values
        perf_metrics = monitor.get_performance_metrics()
        assert (
            perf_metrics.total_executions == 1
        ), f"Expected 1 execution, got {perf_metrics.total_executions}"
        assert (
            perf_metrics.successful_executions == 1
        ), f"Expected 1 success, got {perf_metrics.successful_executions}"
        print(f"✅ Monitoring enabled: Metrics show {perf_metrics.total_executions} execution(s)")

        return True
    except Exception as e:
        print(f"❌ Monitoring enabled test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_integration_no_tracker():
    """Test that LLM works without cost tracker (backward compatibility)."""
    print("\n=== Test 5: LLM Integration Without Tracker (Backward Compatible) ===")

    try:
        # Create LLM without cost tracker (default behavior)
        llm = UnifiedLLM(cost_tracker=None)

        # Should not break
        assert llm.cost_tracker is None
        print("✅ LLM without tracker: Initializes correctly")

        # Check that _track_cost method exists but handles None gracefully
        # We can't actually call generate() without a real provider, but we can check the method exists
        assert hasattr(llm, "_track_cost"), "Missing _track_cost method"
        print("✅ LLM without tracker: _track_cost method exists")

        return True
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_integration_with_tracker():
    """Test that LLM works with cost tracker."""
    print("\n=== Test 6: LLM Integration With Tracker ===")

    try:
        # Create tracker
        tracker = CostTracker(enable_tracking=True)

        # Create LLM with cost tracker
        llm = UnifiedLLM(cost_tracker=tracker)

        # Should have tracker
        assert llm.cost_tracker is not None
        assert llm.cost_tracker == tracker
        print("✅ LLM with tracker: Initializes correctly")

        # Check that _track_cost method exists
        assert hasattr(llm, "_track_cost"), "Missing _track_cost method"
        print("✅ LLM with tracker: _track_cost method exists")

        return True
    except Exception as e:
        print(f"❌ LLM integration with tracker test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_defaults():
    """Test that config defaults are correct (opt-in)."""
    print("\n=== Test 7: Config Defaults (Opt-In) ===")

    try:
        # Create config with defaults
        config = SwarmConfig()

        # Should be disabled by default
        assert (
            config.enable_cost_tracking == False
        ), f"Expected False, got {config.enable_cost_tracking}"
        assert config.enable_monitoring == False, f"Expected False, got {config.enable_monitoring}"
        assert (
            config.enable_efficiency_metrics == False
        ), f"Expected False, got {config.enable_efficiency_metrics}"
        print("✅ Config defaults: All features disabled by default (opt-in)")

        # Should be able to enable
        config.enable_cost_tracking = True
        config.enable_monitoring = True
        assert config.enable_cost_tracking == True
        assert config.enable_monitoring == True
        print("✅ Config defaults: Can enable features")

        return True
    except Exception as e:
        print(f"❌ Config defaults test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_impact_disabled():
    """Test that disabled features have no performance impact."""
    print("\n=== Test 8: Performance Impact When Disabled ===")

    import time

    try:
        # Create disabled tracker
        tracker = CostTracker(enable_tracking=False)

        # Measure time for many calls
        start = time.time()
        for _ in range(1000):
            tracker.record_llm_call(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=1000,
                output_tokens=500,
                success=True,
            )
        elapsed = time.time() - start

        # Should be very fast (< 0.1s for 1000 calls)
        assert elapsed < 0.1, f"Too slow: {elapsed:.4f}s for 1000 calls"
        print(f"✅ Performance: {elapsed:.4f}s for 1000 calls (disabled) - Fast enough")

        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Cost Tracking & Monitoring Opt-In Functionality")
    print("=" * 60)

    tests = [
        test_cost_tracker_disabled,
        test_cost_tracker_enabled,
        test_monitoring_disabled,
        test_monitoring_enabled,
        test_llm_integration_no_tracker,
        test_llm_integration_with_tracker,
        test_config_defaults,
        test_performance_impact_disabled,
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
        print("\n🎉 All tests passed! Opt-in functionality works correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
