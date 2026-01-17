#!/usr/bin/env python3
"""Test ExecutionManager integration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.foundation import JottyConfig
from core.orchestration.managers import ExecutionManager


def test_execution_manager():
    """Test ExecutionManager basic functionality."""
    print("🧪 Testing ExecutionManager Integration\n")
    print("="*60)

    config = JottyConfig()
    manager = ExecutionManager(config)

    # Test 1: Record successful execution
    print("\n🔵 Test 1: Record successful execution")
    manager.record_execution("Agent1", success=True, duration=1.5)
    stats = manager.get_stats()
    print(f"   Executions: {stats['total_executions']}, Successes: {stats['successes']}")
    print(f"   Success rate: {stats['success_rate']:.1%}, Avg duration: {stats['avg_duration']:.2f}s")
    assert stats['total_executions'] == 1
    assert stats['successes'] == 1
    assert stats['success_rate'] == 1.0
    assert abs(stats['avg_duration'] - 1.5) < 0.01
    print("   ✅ PASS")

    # Test 2: Record failed execution
    print("\n🟡 Test 2: Record failed execution")
    manager.record_execution("Agent2", success=False, duration=0.5)
    stats = manager.get_stats()
    print(f"   Executions: {stats['total_executions']}, Successes: {stats['successes']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    assert stats['total_executions'] == 2
    assert stats['successes'] == 1
    assert abs(stats['success_rate'] - 0.5) < 0.01
    print("   ✅ PASS")

    # Test 3: Multiple executions
    print("\n🟢 Test 3: Multiple executions")
    manager.record_execution("Agent3", success=True, duration=2.0)
    manager.record_execution("Agent4", success=True, duration=1.0)
    stats = manager.get_stats()
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Total duration: {stats['total_duration']:.2f}s")
    print(f"   Avg duration: {stats['avg_duration']:.2f}s")
    assert stats['total_executions'] == 4
    assert stats['successes'] == 3
    assert abs(stats['success_rate'] - 0.75) < 0.01
    print("   ✅ PASS")

    # Test 4: Reset stats
    print("\n🟣 Test 4: Reset statistics")
    manager.reset_stats()
    stats = manager.get_stats()
    print(f"   Executions after reset: {stats['total_executions']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    assert stats['total_executions'] == 0
    assert stats['successes'] == 0
    print("   ✅ PASS")

    print("\n" + "="*60)
    print("✅ All ExecutionManager tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test_execution_manager())
