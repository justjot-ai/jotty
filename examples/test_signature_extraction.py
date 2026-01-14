#!/usr/bin/env python
"""
Test signature extraction for DSPy ChainOfThought
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_cli_wrapper import ClaudeCLILM
import dspy
from core import SwarmConfig, AgentSpec, Conductor


def test_signature_extraction():
    """Test that Conductor correctly extracts DSPy ChainOfThought signatures."""

    print("\n" + "="*70)
    print("TEST: Signature Extraction for DSPy ChainOfThought")
    print("="*70)

    # Configure DSPy
    print("\n1️⃣ Configuring DSPy...")
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)

    # Define signature
    print("\n2️⃣ Defining signature...")
    class SimpleTask(dspy.Signature):
        """Simple test signature."""
        code = dspy.InputField(desc="Python code")
        result = dspy.OutputField(desc="Analysis result")

    # Create agent
    print("\n3️⃣ Creating agent...")
    test_agent = AgentSpec(
        name="TestAgent",
        agent=dspy.ChainOfThought(SimpleTask),
        outputs=["result"]
    )

    # Create Conductor
    print("\n4️⃣ Creating Conductor...")
    actors = [test_agent]
    config = SwarmConfig(
        max_actor_iters=1,
        enable_rl=False  # Disable RL for testing
    )

    conductor = Conductor(
        actors=actors,
        metadata_provider=None,  # No metadata provider needed
        config=config,
        enable_data_registry=False
    )

    # Check extracted signature
    print("\n5️⃣ Checking extracted signature...")
    signature = conductor.actor_signatures.get("TestAgent", {})
    print(f"   Extracted signature: {signature}")

    if 'code' in signature:
        print("   ✅ SUCCESS: 'code' parameter found in signature!")
        print(f"   Details: {signature['code']}")
        return True
    elif 'kwargs' in signature:
        print("   ❌ FAIL: Found 'kwargs' instead of 'code'")
        print(f"   This means signature extraction fell back to forward() method")
        return False
    else:
        print(f"   ❌ FAIL: Unexpected signature: {list(signature.keys())}")
        return False


if __name__ == "__main__":
    success = test_signature_extraction()
    sys.exit(0 if success else 1)
