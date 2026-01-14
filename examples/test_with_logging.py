#!/usr/bin/env python
"""
Two-Agent System with Full Logging Enabled
===========================================

Demonstrates where Jotty writes logs and how to access them.
"""

import sys
import asyncio
import warnings
from pathlib import Path
from datetime import datetime

# Suppress DSPy forward() warning during signature introspection
warnings.filterwarnings('ignore', message='.*Calling module.forward.*')

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_cli_wrapper import ClaudeCLILM
import dspy
from core import SwarmConfig, AgentSpec, Conductor


async def run_with_logging():
    """Run a two-agent system with full logging enabled."""

    print("\n" + "="*70)
    print("TWO-AGENT CODE ANALYZER WITH LOGGING")
    print("="*70)

    # Configure DSPy
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)

    # Define signatures
    class DetectIssues(dspy.Signature):
        """Analyze code and identify quality issues."""
        code = dspy.InputField(desc="Python code to analyze")
        issues = dspy.OutputField(desc="List of issues found (2-3 bullet points)")

    class SuggestFixes(dspy.Signature):
        """Suggest fixes for code quality issues."""
        issues = dspy.InputField(desc="List of issues to fix")
        suggestions = dspy.OutputField(desc="Specific fix suggestions (2-3 bullet points)")

    # Create agents
    detector = AgentSpec(
        name="IssueDetector",
        agent=dspy.ChainOfThought(DetectIssues),
        outputs=["issues"]
    )

    fixer = AgentSpec(
        name="SolutionProvider",
        agent=dspy.ChainOfThought(SuggestFixes),
        parameter_mappings={"issues": "IssueDetector.issues"},  # ActorName.field format
        outputs=["suggestions"]
    )

    # Create configuration with logging enabled
    print("\n📁 Configuring logging...")

    config = SwarmConfig(
        max_actor_iters=5,
        enable_rl=False,  # Disable RL for simple workflows
        # Logging configuration
        output_base_dir="./outputs",
        create_run_folder=True,  # Create timestamped folders
        enable_beautified_logs=True,  # Human-readable logs
        enable_debug_logs=True,  # Raw debug logs
        log_level="INFO",
        # Profiling
        enable_profiling=True,  # Track execution times
        profiling_verbosity="summary",  # Show summary at end
        # Memory optimization for simple workflows
        rag_max_candidates=10,  # Limit LLM scoring (default: 50, too slow!)
        # Persistence
        persist_agent_outputs=True,
        persist_memories=True,
        auto_save_interval=1
    )

    # Create output directory name
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = f"./outputs/{timestamp}"

    print(f"   ✅ Output directory: {output_dir}")
    print(f"   ✅ Logs will be written to: {output_dir}/logs/")

    # Create conductor
    actors = [detector, fixer]

    conductor = Conductor(
        actors=actors,
        metadata_provider=None,  # No metadata provider needed
        config=config,
        enable_data_registry=False
    )

    # Sample code
    code_sample = '''
def calculate(x, y):
    result = x / y
    return result
'''

    print("\n" + "="*70)
    print("RUNNING ANALYSIS")
    print("="*70)

    try:
        # Run with explicit output_dir
        result = await conductor.run(
            goal="Analyze code and suggest improvements",
            code=code_sample,
            max_iterations=10,
            output_dir=output_dir  # Explicit output directory
        )

        # Get outputs
        all_outputs = conductor.io_manager.get_all_outputs()

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        for agent_name, output in all_outputs.items():
            print(f"\n🔍 {agent_name}")
            print("-" * 70)
            if hasattr(output, 'output_fields'):
                for field_name, field_value in output.output_fields.items():
                    print(f"{field_name}:")
                    print(f"{field_value}\n")

        print("\n" + "="*70)
        print("WHERE TO FIND THE LOGS")
        print("="*70)

        # Check what was actually created
        output_path = Path(output_dir)
        if output_path.exists():
            print(f"\n📁 Output directory created: {output_dir}/")

            # List all files
            for item in sorted(output_path.rglob("*")):
                if item.is_file():
                    size = item.stat().st_size
                    rel_path = item.relative_to(output_path)
                    print(f"   📄 {rel_path} ({size} bytes)")

            print(f"\n💡 To view logs:")
            print(f"   cat {output_dir}/logs/beautified.log")
            print(f"   cat {output_dir}/logs/debug.log")
        else:
            print(f"\n⚠️  Output directory not created: {output_dir}")
            print("   (Logs may be disabled or output_dir not set)")

        print("\n✅ COMPLETE!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_with_logging())
    sys.exit(0 if success else 1)
