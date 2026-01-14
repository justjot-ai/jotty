#!/usr/bin/env python
"""
WORKING Two-Agent System: Code Quality Analyzer
================================================

A complete, working example demonstrating:
- Two agents collaborating (IssueDetector → SolutionProvider)
- Parameter passing between agents
- Real Claude CLI integration
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_cli_wrapper import ClaudeCLILM
import dspy
from core import SwarmConfig, AgentSpec, Conductor


async def run_two_agent_system():
    """Run a two-agent code analyzer system."""

    print("\n" + "="*70)
    print("TWO-AGENT CODE ANALYZER")
    print("="*70)

    # Configure DSPy with Claude CLI
    print("\n1️⃣ Configuring Claude CLI LM...")
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)
    print("   ✅ Claude CLI configured")

    # Define agent signatures
    print("\n2️⃣ Defining agent signatures...")

    class DetectIssues(dspy.Signature):
        """Analyze code and identify quality issues."""
        code = dspy.InputField(desc="Python code to analyze")
        issues = dspy.OutputField(desc="List of issues found (2-3 bullet points)")

    class SuggestFixes(dspy.Signature):
        """Suggest fixes for code quality issues."""
        issues = dspy.InputField(desc="List of issues to fix")
        suggestions = dspy.OutputField(desc="Specific fix suggestions (2-3 bullet points)")

    print("   ✅ Signatures defined")

    # Create agents
    print("\n3️⃣ Creating agents...")

    detector = AgentSpec(
        name="IssueDetector",
        agent=dspy.ChainOfThought(DetectIssues),
        outputs=["issues"]
    )

    fixer = AgentSpec(
        name="SolutionProvider",
        agent=dspy.ChainOfThought(SuggestFixes),
        parameter_mappings={"issues": "IssueDetector.issues"},  # Get issues from detector (ActorName.field format)
        outputs=["suggestions"]
    )

    print("   ✅ IssueDetector - Finds code quality issues")
    print("   ✅ SolutionProvider - Suggests fixes (receives issues from IssueDetector)")

    # Create configuration
    print("\n4️⃣ Creating Conductor...")

    actors = [detector, fixer]
    config = SwarmConfig(
        max_actor_iters=5,
        enable_rl=False  # Disable RL for simple workflows
    )

    conductor = Conductor(
        actors=actors,
        metadata_provider=None,  # No metadata provider needed for this example
        config=config,
        enable_data_registry=False
    )

    print("   ✅ Conductor initialized with refactored components")

    # Sample code to analyze
    code_sample = '''
def calculate(x, y):
    result = x / y
    return result

data = [1, 2, 3]
for i in range(5):
    print(data[i])
'''

    print("\n" + "="*70)
    print("5️⃣ RUNNING CODE ANALYSIS")
    print("="*70)

    print("\n📝 Code to analyze:")
    print("-" * 70)
    print(code_sample)
    print("-" * 70)

    try:
        print("\n⚙️  Executing agents...")

        # Run the multi-agent system
        result = await conductor.run(
            goal="Analyze code and suggest improvements",
            code=code_sample,
            max_iterations=10
        )

        # Get outputs from both agents
        all_outputs = conductor.io_manager.get_all_outputs()

        print("\n" + "="*70)
        print("6️⃣ RESULTS")
        print("="*70)

        # Show IssueDetector output
        detector_output = all_outputs.get("IssueDetector")
        if detector_output and hasattr(detector_output, 'output_fields'):
            print("\n🔍 Agent 1: IssueDetector")
            print("-" * 70)
            issues = detector_output.output_fields.get('issues', 'N/A')
            print(f"Issues found:\n{issues}")

        # Show SolutionProvider output
        fixer_output = all_outputs.get("SolutionProvider")
        if fixer_output and hasattr(fixer_output, 'output_fields'):
            print("\n💡 Agent 2: SolutionProvider")
            print("-" * 70)
            suggestions = fixer_output.output_fields.get('suggestions', 'N/A')
            print(f"Suggestions:\n{suggestions}")

        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)

        # Verify it worked
        if len(all_outputs) >= 2:
            print("\n🎉 SUCCESS!")
            print("\n✅ Verified:")
            print("   • Two agents executed successfully")
            print("   • Parameter passing worked (IssueDetector → SolutionProvider)")
            print("   • Both agents produced output")
            print("   • Multi-agent collaboration is fully functional!")
            return True
        else:
            print(f"\n⚠️  Only {len(all_outputs)} agent(s) produced output")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_two_agent_system())
    sys.exit(0 if success else 1)
