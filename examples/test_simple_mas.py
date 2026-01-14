#!/usr/bin/env python
"""
Simple MAS Test - Minimal working example
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_cli_wrapper import ClaudeCLILM
import dspy
from core import SwarmConfig, AgentSpec, Conductor


async def test_simple_mas():
    """Test a simple multi-agent system with one agent."""

    print("\n" + "="*70)
    print("SIMPLE MAS TEST")
    print("="*70)

    # Configure DSPy
    print("\n1️⃣ Configuring DSPy...")
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)

    # Define signature
    print("\n2️⃣ Defining signature...")
    class AnalyzeCode(dspy.Signature):
        """Analyze code and identify issues."""
        code = dspy.InputField(desc="Python code to analyze")
        issues = dspy.OutputField(desc="List of issues found")

    # Create agent
    print("\n3️⃣ Creating agent...")
    agent = AgentSpec(
        name="CodeAnalyzer",
        agent=dspy.ChainOfThought(AnalyzeCode),
        outputs=["issues"]
    )

    # Create Conductor
    print("\n4️⃣ Creating Conductor...")
    actors = [agent]
    config = SwarmConfig(
        max_actor_iters=1,
        enable_rl=False  # Disable RL for simple workflows
    )

    conductor = Conductor(
        actors=actors,
        metadata_provider=None,  # No metadata provider needed
        config=config,
        enable_data_registry=False
    )

    # Test code
    code_sample = '''
def calculate(x, y):
    result = x / y
    return result
'''

    print("\n5️⃣ Running analysis...")
    print(f"   Code: {code_sample.strip()}")

    try:
        # Run the conductor
        result = await conductor.run(
            goal="Analyze this code",
            code=code_sample,
            max_iterations=1
        )

        # Get outputs
        all_outputs = conductor.io_manager.get_all_outputs()
        print("\n6️⃣ Results:")

        if all_outputs:
            for agent_name, output in all_outputs.items():
                print(f"\n   Agent: {agent_name}")
                if hasattr(output, 'output_fields'):
                    for field_name, field_value in output.output_fields.items():
                        print(f"   {field_name}: {field_value}")
            print("\n✅ SUCCESS!")
            return True
        else:
            print("\n❌ No outputs generated")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple_mas())
    sys.exit(0 if success else 1)
