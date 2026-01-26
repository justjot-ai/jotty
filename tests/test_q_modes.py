#!/usr/bin/env python3
"""Test both Q-value modes (simple and LLM) work correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.foundation import JottyConfig
from core.orchestration.managers import LearningManager

def test_simple_mode():
    """Test simple Q-value mode (average reward)."""
    print("\n🔵 Testing SIMPLE Q-value mode...")
    config = JottyConfig(
        enable_rl=True,
        q_value_mode="simple"
    )

    manager = LearningManager(config)

    # Record some experiences
    state = {"goal": "test", "actor": "TestAgent"}
    action = {"actor": "TestAgent", "task": "test task"}

    # Record 3 rewards: 0.5, 0.7, 0.9
    manager.record_outcome(state, action, 0.5)
    manager.record_outcome(state, action, 0.7)
    manager.record_outcome(state, action, 0.9)

    # Predict Q-value (should be average: (0.5 + 0.7 + 0.9) / 3 = 0.7)
    q_value, confidence, alternative = manager.predict_q_value(state, action)

    expected = 0.7
    if abs(q_value - expected) < 0.01:
        print(f"✅ Simple mode working: Q-value={q_value:.3f} (expected {expected:.3f})")
        return True
    else:
        print(f"❌ Simple mode FAILED: Q-value={q_value:.3f} (expected {expected:.3f})")
        return False

def test_llm_mode():
    """Test LLM Q-value mode (semantic prediction)."""
    print("\n🟢 Testing LLM Q-value mode...")
    config = JottyConfig(
        enable_rl=True,
        q_value_mode="llm"
    )

    manager = LearningManager(config)

    # Record some experiences
    state = {"goal": "test", "actor": "TestAgent"}
    action = {"actor": "TestAgent", "task": "test task"}

    manager.record_outcome(state, action, 0.8)

    # Predict Q-value (LLM-based, so we just check it returns a value)
    q_value, confidence, alternative = manager.predict_q_value(state, action)

    if q_value is not None:
        print(f"✅ LLM mode working: Q-value={q_value:.3f}, confidence={confidence:.3f}")
        return True
    else:
        print(f"❌ LLM mode FAILED: Q-value is None")
        return False

def main():
    print("🧪 Testing Q-Value Modes After Refactoring\n")
    print("="*60)

    # Test both modes
    simple_ok = test_simple_mode()
    llm_ok = test_llm_mode()

    print("\n" + "="*60)
    print("📊 RESULTS:")
    print(f"   Simple mode: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"   LLM mode:    {'✅ PASS' if llm_ok else '❌ FAIL'}")

    if simple_ok and llm_ok:
        print("\n✅ All Q-value modes working correctly!")
        return 0
    else:
        print("\n❌ Some Q-value modes failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
