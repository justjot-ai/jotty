"""
Detailed Mermaid Diagram Learning Test

Shows step-by-step how agent learns to generate perfect Mermaid diagrams.
"""

import asyncio
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import create_optimization_pipeline
from core.foundation.agent_config import AgentConfig


class MermaidAgent:
    """Agent that starts with wrong Mermaid and learns."""
    def __init__(self):
        self.attempts = 0
        self.learned_patterns = []
    
    def forward(self, description: str = None, teacher_output: str = None, **kwargs) -> str:
        self.attempts += 1
        result = type('Result', (), {})()
        
        # If teacher output available, learn from it!
        if teacher_output:
            self.learned_patterns.append(teacher_output)
            result._store = {"output": teacher_output}
            return result
        
        # Use learned pattern if available
        if self.learned_patterns:
            result._store = {"output": self.learned_patterns[-1]}
            return result
        
        # Progressive mistakes (showing different types of errors)
        if self.attempts == 1:
            # Error 1: Missing node definitions
            result._store = {"output": "graph A --> B"}
        elif self.attempts == 2:
            # Error 2: Missing arrow
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]"}
        elif self.attempts == 3:
            # Error 3: Wrong arrow syntax
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]\n    A->B"}
        else:
            # Shouldn't reach here if learning works
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]\n    A --> B"}
        
        return result


class MermaidTeacher:
    """Teacher that provides perfect Mermaid diagrams."""
    def forward(self, **kwargs) -> str:
        gold = kwargs.get('gold_standard', '')
        result = type('Result', (), {})()
        result._store = {"output": gold}
        return result


async def test_detailed_mermaid_learning():
    """Detailed Mermaid learning test with visual output."""
    print("\n" + "=" * 80)
    print("MERMAID DIAGRAM GENERATION - DETAILED LEARNING TEST")
    print("=" * 80)
    print()
    
    agent = MermaidAgent()
    
    agents = [
        AgentConfig(
            name="mermaid_generator",
            agent=agent,
            outputs=["output"]
        ),
        AgentConfig(
            name="teacher",
            agent=MermaidTeacher(),
            metadata={"is_teacher": True}
        )
    ]
    
    async def evaluate_mermaid(output, gold_standard, task, context):
        """Detailed Mermaid evaluation."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Syntax checks
        has_graph = "graph" in output_str.lower() or "flowchart" in output_str.lower()
        has_nodes = "[" in output_str or "{" in output_str
        has_arrow = "-->" in output_str
        has_newlines = "\n" in output_str
        
        # Detailed feedback
        issues = []
        if not has_graph:
            issues.append("❌ Missing 'graph' or 'flowchart' declaration")
        if not has_nodes:
            issues.append("❌ Missing node definitions (use [label] or {label})")
        if not has_arrow:
            issues.append("❌ Missing arrow connection (use -->)")
        if not has_newlines:
            issues.append("⚠️  No line breaks (harder to read)")
        
        syntax_valid = has_graph and has_nodes and has_arrow
        matches_gold = output_str == gold_str
        
        if matches_gold:
            score = 1.0
            status = "CORRECT"
        elif syntax_valid:
            score = 0.5
            status = "PARTIAL"
        else:
            score = 0.0
            status = "INCORRECT"
        
        return {
            "score": score,
            "status": status,
            "difference": "\n".join(issues) if issues else None,
            "syntax_valid": syntax_valid,
            "matches_gold": matches_gold,
            "issues": issues
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        output_path="./test_outputs/mermaid_detailed"
    )
    pipeline.config.evaluation_function = evaluate_mermaid
    
    # Test with a simple diagram
    gold = """graph TD
    A[Start]
    B[Process]
    C[End]
    A --> B
    B --> C"""
    
    print("🎯 GOLD STANDARD:")
    print(gold)
    print()
    print("=" * 80)
    print("ITERATION-BY-ITERATION IMPROVEMENT")
    print("=" * 80)
    print()
    
    result = await pipeline.optimize(
        task="Generate workflow diagram",
        context={"description": "Start → Process → End"},
        gold_standard=gold
    )
    
    # Show detailed iteration breakdown
    for iteration in result['iterations']:
        iter_num = iteration['iteration']
        metadata = iteration.get('metadata', {})
        eval_result = metadata.get('evaluation_result', {})
        
        print(f"📊 ITERATION {iter_num}")
        print("-" * 80)
        
        # Get agent output
        agent_output = iteration.get('output', 'N/A')
        if not agent_output or agent_output == 'N/A':
            pipeline_result = metadata.get('pipeline_result', {})
            agent_output = pipeline_result.get('output', 'N/A')
        
        print(f"Agent Output:")
        print(f"```mermaid")
        print(f"{agent_output}")
        print(f"```")
        print()
        
        print(f"Evaluation:")
        print(f"  Score: {iteration['evaluation_score']:.2f} / 1.0")
        print(f"  Status: {iteration['evaluation_status']}")
        
        if eval_result.get('issues'):
            print(f"  Issues Found:")
            for issue in eval_result['issues']:
                print(f"    {issue}")
        
        if iteration.get('has_teacher_output'):
            teacher_eval = eval_result.get('teacher_evaluation', {})
            print(f"  🎓 Teacher Output Score: {teacher_eval.get('score', 0):.2f}")
            print(f"  ✅ Teacher provided correct diagram")
        
        print(f"  Success: {'✅ YES' if iteration['success'] else '❌ NO'}")
        print()
    
    # Show improvements
    print("=" * 80)
    print("LEARNED IMPROVEMENTS")
    print("=" * 80)
    print()
    
    improvements_file = Path("./test_outputs/mermaid_detailed/improvements.json")
    if improvements_file.exists():
        with open(improvements_file, 'r') as f:
            improvements = json.load(f)
        
        for i, imp in enumerate(improvements, 1):
            print(f"Improvement {i}:")
            print(f"  Task: {imp['task']}")
            print()
            print(f"  ❌ BEFORE (Student):")
            print(f"  ```mermaid")
            print(f"  {imp['student_output']}")
            print(f"  ```")
            print(f"  Score: {imp['student_score']:.2f}")
            print()
            print(f"  ✅ AFTER (Teacher):")
            print(f"  ```mermaid")
            print(f"  {imp['teacher_output']}")
            print(f"  ```")
            print(f"  Score: {imp['teacher_score']:.2f}")
            print()
            print(f"  📚 Learned Pattern:")
            print(f"  {imp['learned_pattern']}")
            print()
            print("-" * 80)
            print()
    
    # Final result
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print()
    print(f"✅ Optimization Complete: {result['optimization_complete']}")
    print(f"📊 Total Iterations: {result['total_iterations']}")
    print(f"🎯 Final Score: {result['final_result']['evaluation_score']:.2f} / 1.0")
    print()
    print("Final Mermaid Diagram:")
    print("```mermaid")
    print(result['final_result']['output'])
    print("```")
    print()
    
    # Verify it's valid Mermaid
    final_output = result['final_result']['output']
    if "graph" in final_output.lower() and "-->" in final_output and "[" in final_output:
        print("✅ VALID MERMAID SYNTAX!")
        print("   ✓ Has 'graph' declaration")
        print("   ✓ Has node definitions")
        print("   ✓ Has arrow connections")
    else:
        print("❌ Invalid Mermaid syntax")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_detailed_mermaid_learning())
