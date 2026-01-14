"""
Example: Show where improvements are stored and how to access them.
"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import create_optimization_pipeline
from core.foundation.agent_config import AgentConfig


class SimpleAgent:
    def forward(self, task: str = None, teacher_output: str = None, **kwargs) -> str:
        result = type('Result', (), {})()
        if teacher_output:
            result._store = {"output": teacher_output}
        else:
            result._store = {"output": "Wrong answer"}
        return result


class TeacherAgent:
    def forward(self, **kwargs) -> str:
        gold_standard = kwargs.get('gold_standard', '')
        result = type('Result', (), {})()
        result._store = {"output": gold_standard}
        return result


async def demonstrate_improvement_storage():
    """Show where improvements are stored."""
    print("=" * 80)
    print("IMPROVEMENT STORAGE DEMONSTRATION")
    print("=" * 80)
    print()
    
    agents = [
        AgentConfig(name="agent", agent=SimpleAgent(), outputs=["output"]),
        AgentConfig(name="teacher", agent=TeacherAgent(), metadata={"is_teacher": True})
    ]
    
    async def evaluate(output, gold_standard, task, context):
        score = 1.0 if str(output) == str(gold_standard) else 0.0
        return {"score": score, "status": "CORRECT" if score == 1.0 else "INCORRECT"}
    
    # Create pipeline with improvement storage enabled
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,  # Save improvements to JSON
        output_path="./test_outputs/storage_demo"
    )
    pipeline.config.evaluation_function = evaluate
    
    print("📋 Running optimization...")
    result = await pipeline.optimize(
        task="Generate answer",
        context={},
        gold_standard="Correct answer"
    )
    
    print()
    print("=" * 80)
    print("WHERE IMPROVEMENTS ARE STORED")
    print("=" * 80)
    print()
    
    # 1. JSON File
    improvements_file = Path("./test_outputs/storage_demo/improvements.json")
    if improvements_file.exists():
        print("✅ 1. JSON FILE (Default Storage)")
        print(f"   Location: {improvements_file}")
        print()
        
        with open(improvements_file, 'r') as f:
            improvements = json.load(f)
        
        print(f"   Total Improvements: {len(improvements)}")
        print()
        
        for i, imp in enumerate(improvements, 1):
            print(f"   Improvement {i}:")
            print(f"     Iteration: {imp['iteration']}")
            print(f"     Task: {imp['task']}")
            print(f"     Student Output: {imp['student_output']}")
            print(f"     Teacher Output: {imp['teacher_output']}")
            print(f"     Learned Pattern: {imp['learned_pattern']}")
            print()
    else:
        print("❌ Improvements file not found")
    
    # 2. Summary File
    summary_file = Path("./test_outputs/storage_demo/improvements_summary.json")
    if summary_file.exists():
        print("✅ 2. SUMMARY FILE")
        print(f"   Location: {summary_file}")
        print()
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"   Optimization Complete: {summary['optimization_complete']}")
        print(f"   Total Iterations: {summary['total_iterations']}")
        print(f"   Total Improvements: {summary['total_improvements']}")
        print(f"   Score Improvement: {summary['summary']['score_improvement']}")
        print()
        print("   Learned Patterns:")
        for pattern in summary['summary']['learned_patterns']:
            print(f"     - {pattern}")
        print()
    
    # 3. DSPy Instructions (if enabled)
    print("📝 3. DSPY INSTRUCTIONS (Optional)")
    print("   To enable: config.update_dspy_instructions = True")
    print("   Updates: agent.instructions attribute")
    print("   Format: Adds learned patterns to agent instructions")
    print()
    
    # 4. Jotty Learned Instructions (if enabled)
    print("📝 4. JOTTY LEARNED INSTRUCTIONS (Optional)")
    print("   To enable: config.update_jotty_instructions = True + provide conductor")
    print("   Updates: conductor.learned_instructions['actor']")
    print("   Format: List of learned instruction strings")
    print()
    
    print("=" * 80)
    print("HOW TO USE IMPROVEMENTS")
    print("=" * 80)
    print()
    print("1. Load from JSON file:")
    print("   ```python")
    print("   import json")
    print("   with open('improvements.json', 'r') as f:")
    print("       improvements = json.load(f)")
    print("   ```")
    print()
    print("2. Extract learned patterns:")
    print("   ```python")
    print("   patterns = [imp['learned_pattern'] for imp in improvements]")
    print("   ```")
    print()
    print("3. Use in future runs:")
    print("   ```python")
    print("   # Add patterns to agent context or instructions")
    print("   context['learned_patterns'] = patterns")
    print("   ```")


if __name__ == "__main__":
    asyncio.run(demonstrate_improvement_storage())
