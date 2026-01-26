#!/usr/bin/env python
"""
Test profiling with intentional 200s sleep to verify profiling captures overhead
"""
import sys
import asyncio
import time
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', message='.*Calling module.forward.*')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "examples"))

from examples.claude_cli_wrapper import ClaudeCLILM
import dspy
from core import SwarmConfig, AgentSpec, Conductor


async def run_with_sleep_profiling():
    """Run with intentional sleep to verify profiling works."""
    
    print("\n" + "="*70)
    print("PROFILING TEST WITH 200s SLEEP")
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
        parameter_mappings={"issues": "IssueDetector.issues"},
        outputs=["suggestions"]
    )
    
    # Create configuration with profiling enabled
    print("\n📁 Configuring profiling...")
    
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_with_sleep")
    output_dir = f"./outputs/{timestamp}"
    
    config = SwarmConfig(
        max_actor_iters=5,
        enable_rl=False,
        enable_beautified_logs=True,
        enable_debug_logs=True,
        log_level="INFO",
        enable_profiling=True,
        profiling_verbosity="summary",
        persist_agent_outputs=True,
        persist_memories=True,
        auto_save_interval=1
    )
    
    print(f"   ✅ Output directory: {output_dir}")
    
    # Create conductor
    actors = [detector, fixer]
    
    conductor = Conductor(
        actors=actors,
        metadata_provider=None,
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
    print("RUNNING ANALYSIS (with 200s sleep before first agent)")
    print("="*70)
    
    try:
        # Add 200 second sleep before running to simulate overhead
        print("\n⏱️  Sleeping for 200 seconds to test profiling...")
        print("   (This simulates the 92% overhead you're seeing)")
        
        # Instrument the sleep with profiling
        from core.utils.profiler import timed_block
        with timed_block("TestSleep200s", component="TestOverhead", name="200 second test sleep"):
            time.sleep(200)
        
        print("\n✅ Sleep complete, now running agents...")
        
        result = await conductor.run(
            goal="Analyze code and suggest improvements",
            code=code_sample,
            max_iterations=10,
            output_dir=output_dir
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
        print("CHECK PROFILING REPORTS")
        print("="*70)
        print(f"\nProfiling reports saved to: {output_dir}/profiling/")
        print("\nTo view:")
        print(f"  cat {output_dir}/profiling/execution_timeline.txt")
        print(f"  cat {output_dir}/profiling/profiling_data.json")
        print("\n✅ COMPLETE!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_with_sleep_profiling())
    sys.exit(0 if success else 1)
