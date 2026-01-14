#!/usr/bin/env python
"""
Test Claude CLI JSON Output with DSPy
======================================

Tests that Claude CLI with --output-format json works with DSPy signatures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_cli_wrapper import ClaudeCLILM
import dspy


def test_simple_signature():
    """Test a simple DSPy signature with Claude CLI JSON output."""
    print("\n" + "="*70)
    print("TEST: Claude CLI JSON Output with DSPy Signatures")
    print("="*70)

    # Configure DSPy with Claude CLI
    print("\n🔧 Configuring Claude CLI LM with JSON output...")
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)
    print("✓ DSPy configured")

    # Test 1: Simple prediction without ChainOfThought
    print("\n" + "-"*70)
    print("TEST 1: Direct LM call (no structured output)")
    print("-"*70)

    response = lm(prompt="What is 2+2? Answer in one word.")
    print(f"Response: {response}")
    print("✅ Direct LM call works!")

    # Test 2: Try ChainOfThought with structured output
    print("\n" + "-"*70)
    print("TEST 2: ChainOfThought with structured output")
    print("-"*70)

    class SimpleQuestion(dspy.Signature):
        """Answer a simple math question."""
        question = dspy.InputField()
        answer = dspy.OutputField(desc="Just the number")

    predictor = dspy.ChainOfThought(SimpleQuestion)

    try:
        result = predictor(question="What is 5+3?")
        print(f"✅ Structured output works!")
        print(f"   Question: What is 5+3?")
        print(f"   Reasoning: {result.reasoning}")
        print(f"   Answer: {result.answer}")
        return True
    except Exception as e:
        print(f"❌ Structured output failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_signature()

    if success:
        print("\n" + "="*70)
        print("🎉 SUCCESS! Claude CLI JSON output works with DSPy!")
        print("="*70)
        sys.exit(0)
    else:
        print("\n❌ Test failed")
        sys.exit(1)
