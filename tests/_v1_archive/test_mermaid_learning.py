"""
Test Mermaid Diagram Generation Learning

This test demonstrates:
1. Agent starts with wrong/invalid Mermaid syntax
2. Teacher provides correct Mermaid diagram
3. Agent learns to generate perfect Mermaid diagrams
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import create_optimization_pipeline
from core.foundation.agent_config import AgentConfig


class MermaidGeneratorAgent:
    """Agent that generates Mermaid diagrams (initially wrong)."""
    def __init__(self):
        self.attempts = 0
        self.learned_from_teacher = None
    
    def forward(self, description: str = None, teacher_output: str = None, **kwargs) -> str:
        self.attempts += 1
        result = type('Result', (), {})()
        
        # If teacher output is available, use it!
        if teacher_output:
            self.learned_from_teacher = teacher_output
            result._store = {"output": teacher_output}
            return result
        
        # Otherwise, use learned value if available
        if self.learned_from_teacher:
            result._store = {"output": self.learned_from_teacher}
            return result
        
        # First attempt: Wrong syntax - missing node definitions
        if self.attempts == 1:
            result._store = {"output": "graph A --> B"}  # Invalid: missing node definitions
        # Second attempt: Better but still wrong - missing arrow
        elif self.attempts == 2:
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]"}  # Missing arrow
        # Third attempt: Wrong arrow syntax
        elif self.attempts == 3:
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]\n    A->B"}  # Wrong arrow syntax (-> instead of -->)
        # Fourth attempt: Still learning
        else:
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]\n    A --> B"}  # Correct!
        
        return result


class MermaidTeacherAgent:
    """Teacher that always provides correct Mermaid diagrams."""
    def forward(self, **kwargs) -> str:
        gold_standard = kwargs.get('gold_standard', '')
        result = type('Result', (), {})()
        result._store = {"output": gold_standard}
        return result


async def test_mermaid_learning():
    """Test Mermaid diagram generation learning."""
    print("=" * 80)
    print("MERMAID DIAGRAM GENERATION - LEARNING TEST")
    print("=" * 80)
    print()
    
    mermaid_agent = MermaidGeneratorAgent()
    
    agents = [
        AgentConfig(
            name="mermaid_generator",
            agent=mermaid_agent,
            parameter_mappings={"description": "description"},
            outputs=["output"]
        ),
        AgentConfig(
            name="teacher",
            agent=MermaidTeacherAgent(),
            metadata={"is_teacher": True}
        )
    ]
    
    async def evaluate_mermaid(output, gold_standard, task, context):
        """Evaluate Mermaid diagram syntax and correctness."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check syntax validity
        has_graph = "graph" in output_str.lower() or "flowchart" in output_str.lower()
        has_nodes = "[" in output_str and "]" in output_str
        has_arrow = "-->" in output_str
        
        syntax_valid = has_graph and has_nodes and has_arrow
        matches_gold = output_str == gold_str
        
        if matches_gold:
            score = 1.0
            status = "CORRECT"
        elif syntax_valid:
            score = 0.5  # Valid syntax but not exact match
            status = "PARTIAL"
        else:
            score = 0.0  # Invalid syntax
            status = "INCORRECT"
        
        # Provide detailed feedback
        feedback = []
        if not has_graph:
            feedback.append("Missing 'graph' or 'flowchart' declaration")
        if not has_nodes:
            feedback.append("Missing node definitions (use [label] syntax)")
        if not has_arrow:
            feedback.append("Missing arrow (use --> syntax)")
        if not matches_gold:
            feedback.append(f"Output doesn't match gold standard")
        
        return {
            "score": score,
            "status": status,
            "difference": "; ".join(feedback) if feedback else None,
            "syntax_valid": syntax_valid,
            "matches_gold": matches_gold
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        output_path="./test_outputs/mermaid_learning"
    )
    pipeline.config.evaluation_function = evaluate_mermaid
    
    # Test Case 1: Simple flowchart
    print("📊 TEST CASE 1: Simple Flowchart")
    print("-" * 80)
    gold_standard_1 = "graph TD\n    A[Start]\n    B[End]\n    A --> B"
    
    print(f"Gold Standard:\n{gold_standard_1}")
    print()
    print("Expected Learning Process:")
    print("  1. Agent produces: 'graph A --> B' (invalid - missing nodes)")
    print("  2. Teacher provides: Correct diagram")
    print("  3. Agent learns: Correct syntax")
    print()
    
    result1 = await pipeline.optimize(
        task="Generate simple flowchart",
        context={"description": "Start to End flow"},
        gold_standard=gold_standard_1
    )
    
    print(f"Results:")
    print(f"  Optimization Complete: {result1['optimization_complete']}")
    print(f"  Total Iterations: {result1['total_iterations']}")
    print(f"  Final Score: {result1['final_result']['evaluation_score']}")
    print(f"  Final Output:\n{result1['final_result']['output']}")
    print()
    
    # Show improvements
    improvements_file = Path("./test_outputs/mermaid_learning/improvements.json")
    if improvements_file.exists():
        import json
        with open(improvements_file, 'r') as f:
            improvements = json.load(f)
        
        print(f"📚 Learned Improvements: {len(improvements)}")
        for imp in improvements:
            print(f"  - {imp['learned_pattern']}")
    print()
    
    # Test Case 2: More complex diagram
    print("=" * 80)
    print("📊 TEST CASE 2: Complex Flowchart")
    print("-" * 80)
    
    # Reset agent for new test
    mermaid_agent2 = MermaidGeneratorAgent()
    agents[0].agent = mermaid_agent2
    
    gold_standard_2 = """graph TD
    A[User Login]
    B{Valid?}
    C[Show Dashboard]
    D[Show Error]
    A --> B
    B -->|Yes| C
    B -->|No| D"""
    
    print(f"Gold Standard:\n{gold_standard_2}")
    print()
    
    result2 = await pipeline.optimize(
        task="Generate login flow diagram",
        context={"description": "User login with validation"},
        gold_standard=gold_standard_2
    )
    
    print(f"Results:")
    print(f"  Optimization Complete: {result2['optimization_complete']}")
    print(f"  Total Iterations: {result2['total_iterations']}")
    print(f"  Final Score: {result2['final_result']['evaluation_score']}")
    print(f"  Final Output:\n{result2['final_result']['output']}")
    print()
    
    # Show iteration-by-iteration learning
    print("=" * 80)
    print("ITERATION-BY-ITERATION LEARNING PROCESS")
    print("=" * 80)
    print()
    
    for iteration in result2['iterations']:
        iter_num = iteration['iteration']
        output = iteration.get('output', 'N/A')
        score = iteration['evaluation_score']
        status = iteration['evaluation_status']
        teacher_used = iteration.get('has_teacher_output', False)
        
        print(f"Iteration {iter_num}:")
        print(f"  Output: {repr(str(output)[:60])}")
        print(f"  Score: {score:.2f} / 1.0")
        print(f"  Status: {status}")
        print(f"  Teacher Used: {'✓ Yes' if teacher_used else '✗ No'}")
        
        # Show what was learned
        if teacher_used:
            print(f"  📚 Learning: Agent received teacher's correct output")
        if iteration['success']:
            print(f"  ✅ Success: Agent now produces correct output!")
        print()
    
    # Show final learned patterns
    print("=" * 80)
    print("LEARNED PATTERNS")
    print("=" * 80)
    print()
    
    if improvements_file.exists():
        with open(improvements_file, 'r') as f:
            all_improvements = json.load(f)
        
        print(f"Total Improvements Recorded: {len(all_improvements)}")
        print()
        for i, imp in enumerate(all_improvements, 1):
            print(f"Pattern {i}:")
            print(f"  Task: {imp['task']}")
            print(f"  Student: {imp['student_output'][:50]}...")
            print(f"  Teacher: {imp['teacher_output'][:50]}...")
            print(f"  Learned: {imp['learned_pattern']}")
            print()
    
    print("=" * 80)
    print("✅ MERMAID LEARNING TEST COMPLETE")
    print("=" * 80)
    
    return result1, result2


if __name__ == "__main__":
    asyncio.run(test_mermaid_learning())
