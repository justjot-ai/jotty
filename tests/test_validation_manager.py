#!/usr/bin/env python3
"""Test ValidationManager integration."""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.foundation import JottyConfig
from core.orchestration.managers import ValidationManager


async def test_validation_manager():
    """Test ValidationManager basic functionality."""
    print("🧪 Testing ValidationManager Integration\n")
    print("="*60)

    config = JottyConfig()
    manager = ValidationManager(config)

    # Test 1: Validation with successful result (dict)
    print("\n🔵 Test 1: Successful result (dict)")
    result = {"success": True, "data": "test"}
    validation = await manager.run_reviewer(None, result, None)
    print(f"   Result: {validation.passed}, Reward: {validation.reward}, Feedback: {validation.feedback}")
    assert validation.passed == True
    assert validation.reward == 1.0
    print("   ✅ PASS")

    # Test 2: Validation with failed result (dict)
    print("\n🟡 Test 2: Failed result (dict)")
    result = {"success": False, "error": "Something went wrong"}
    validation = await manager.run_reviewer(None, result, None)
    print(f"   Result: {validation.passed}, Reward: {validation.reward}, Feedback: {validation.feedback}")
    assert validation.passed == False
    assert validation.reward == 0.0
    assert "Something went wrong" in validation.feedback
    print("   ✅ PASS")

    # Test 3: Validation with dspy.Prediction-like object
    print("\n🟢 Test 3: DSPy Prediction-like object")
    class MockPrediction:
        def __init__(self, success, reasoning="Test reasoning"):
            self.success = success
            self._reasoning = reasoning

    result = MockPrediction(success=True)
    validation = await manager.run_reviewer(None, result, None)
    print(f"   Result: {validation.passed}, Reward: {validation.reward}, Feedback: {validation.feedback}")
    assert validation.passed == True
    print("   ✅ PASS")

    # Test 4: Validation with default result
    print("\n🟣 Test 4: Default result (no success field)")
    result = "some string output"
    validation = await manager.run_reviewer(None, result, None)
    print(f"   Result: {validation.passed}, Reward: {validation.reward}, Feedback: {validation.feedback}")
    assert validation.passed == True
    assert validation.reward == 0.8
    print("   ✅ PASS")

    # Test 5: Check stats
    print("\n📊 Test 5: Validation statistics")
    stats = manager.get_stats()
    print(f"   Total validations: {stats['total_validations']}")
    print(f"   Approvals: {stats['approvals']}")
    print(f"   Approval rate: {stats['approval_rate']:.1%}")
    assert stats['total_validations'] == 4
    assert stats['approvals'] == 3
    assert abs(stats['approval_rate'] - 0.75) < 0.01
    print("   ✅ PASS")

    print("\n" + "="*60)
    print("✅ All ValidationManager tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(test_validation_manager()))
