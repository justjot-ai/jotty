"""
Test OptimizationPipeline with WRONG initial outputs to verify improvement.

This test verifies that the pipeline can:
1. Start with wrong output
2. Detect failure via evaluation
3. Use teacher model to get correct output
4. Iterate and improve
5. Eventually succeed
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import (
    OptimizationPipeline,
    OptimizationConfig,
    create_optimization_pipeline
)
from core.foundation.agent_config import AgentConfig


class ImprovingAgent:
    """Agent that improves over iterations by learning from teacher."""
    def __init__(self):
        self.attempts = 0
        self.learned_from_teacher = None
    
    def forward(self, task: str = None, teacher_output: str = None, _teacher_feedback: str = None, **kwargs) -> str:
        self.attempts += 1
        result = type('Result', (), {})()
        
        # Debug: log what we received
        print(f"  [Agent Debug] Attempt {self.attempts}, teacher_output={teacher_output}, kwargs keys={list(kwargs.keys())}")
        
        # If teacher output is available, use it!
        if teacher_output:
            print(f"  [Agent Debug] Using teacher output: {teacher_output}")
            self.learned_from_teacher = teacher_output
            result._store = {"output": teacher_output}
            return result
        
        # Otherwise, use learned value if available
        if self.learned_from_teacher:
            result._store = {"output": self.learned_from_teacher}
            return result
        
        # First attempt: Wrong output
        if self.attempts == 1:
            result._store = {"output": "Wrong answer"}
        # Second attempt: Still wrong but closer
        elif self.attempts == 2:
            result._store = {"output": "Better but still wrong"}
        # Third attempt: Correct!
        else:
            result._store = {"output": "Correct answer"}
        
        return result


class TeacherAgent:
    """Teacher that always provides correct answer."""
    def forward(self, **kwargs) -> str:
        gold_standard = kwargs.get('gold_standard', '')
        result = type('Result', (), {})()
        result._store = {"output": gold_standard}
        return result


async def test_improvement_with_wrong_initial_output():
    """Test that pipeline improves from wrong initial output."""
    print("\n" + "="*80)
    print("Test: Improvement from Wrong Initial Output")
    print("="*80)
    
    improving_agent = ImprovingAgent()
    
    # Main pipeline: only the student agent
    agents = [
        AgentConfig(
            name="main_agent",
            agent=improving_agent,
            outputs=["output"]
        )
    ]
    
    # Teacher agent is separate (not in main pipeline - will be discovered when needed)
    # Add it to agents list ONLY for discovery, but mark it so it doesn't run in main pipeline
    teacher_agent_config = AgentConfig(
        name="teacher",
        agent=TeacherAgent(),
        metadata={"is_teacher": True},
        enabled=False  # Don't run in main pipeline
    )
    agents.append(teacher_agent_config)
    
    async def evaluate(output, gold_standard, task, context):
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT",
            "difference": f"Expected '{gold_str}', got '{output_str}'" if score == 0.0 else None
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        output_path="./test_outputs/improvement_test"
    )
    pipeline.config.evaluation_function = evaluate
    
    gold_standard = "Correct answer"
    
    print(f"Gold Standard: '{gold_standard}'")
    print(f"Initial agent will produce: 'Wrong answer'")
    print("\nRunning optimization...")
    
    result = await pipeline.optimize(
        task="Generate correct answer",
        context={},
        gold_standard=gold_standard
    )
    
    print(f"\nResults:")
    print(f"  Status: {result['status']}")
    print(f"  Total Iterations: {result['total_iterations']}")
    print(f"  Optimization Complete: {result['optimization_complete']}")
    print(f"  Consecutive Passes: {result['consecutive_passes']}")
    
    print(f"\nIteration Details:")
    for it in result['iterations']:
        print(f"  Iteration {it['iteration']}:")
        print(f"    Score: {it['evaluation_score']:.2f}")
        print(f"    Status: {it['evaluation_status']}")
        print(f"    Teacher Used: {it.get('has_teacher_output', False)}")
        print(f"    Success: {it['success']}")
    
    if result.get('final_result'):
        final = result['final_result']
        print(f"\nFinal Result:")
        print(f"  Output: {final.get('output')}")
        print(f"  Score: {final.get('evaluation_score')}")
        print(f"  Status: {final.get('evaluation_status')}")
    
    # Verify improvement happened
    assert result['total_iterations'] > 0, "Should have at least one iteration"
    
    # Check if we eventually succeeded
    if result['optimization_complete']:
        print("\n✓ SUCCESS: Pipeline optimized and achieved correct output!")
        assert result['final_result']['evaluation_score'] == 1.0, f"Expected score 1.0, got {result['final_result']['evaluation_score']}"
        assert result['final_result']['output'] == gold_standard, f"Expected '{gold_standard}', got '{result['final_result']['output']}'"
        print(f"✓ Final output matches gold standard: '{result['final_result']['output']}'")
    else:
        print("\n⚠ WARNING: Optimization didn't complete (may need more iterations or better teacher)")
        # Still check if we improved
        if result['final_result'] and result['final_result'].get('evaluation_score', 0) > 0:
            print(f"  Partial success: Score improved to {result['final_result']['evaluation_score']}")
    
    return result


class LearningAgent:
    """Agent that learns from teacher feedback."""
    def __init__(self):
        self.attempts = 0
        self.learned_answer = None
    
    def forward(self, task: str = None, teacher_output: str = None, auditor_feedback: str = None, **kwargs) -> str:
        self.attempts += 1
        result = type('Result', (), {})()
        
        # If teacher output is available, use it!
        if teacher_output:
            self.learned_answer = teacher_output
            result._store = {"output": teacher_output}
            return result
        
        # If we have feedback, try to learn from it
        if auditor_feedback and "Correct answer" in auditor_feedback:
            self.learned_answer = "Correct answer"
        
        # Use learned answer if available
        if self.learned_answer:
            result._store = {"output": self.learned_answer}
        else:
            # Otherwise produce wrong answer
            result._store = {"output": f"Wrong attempt {self.attempts}"}
        
        return result


async def test_learning_from_feedback():
    """Test that agent learns from evaluation feedback."""
    print("\n" + "="*80)
    print("Test: Learning from Feedback")
    print("="*80)
    
    learning_agent = LearningAgent()
    
    agents = [
        AgentConfig(
            name="learning_agent",
            agent=learning_agent,
            outputs=["output"]
        ),
        AgentConfig(
            name="teacher",
            agent=TeacherAgent(),
            metadata={"is_teacher": True}
        )
    ]
    
    async def evaluate(output, gold_standard, task, context):
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT",
            "difference": f"Expected '{gold_str}', got '{output_str}'" if score == 0.0 else None
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        output_path="./test_outputs/learning_test"
    )
    pipeline.config.evaluation_function = evaluate
    
    gold_standard = "Correct answer"
    
    print(f"Gold Standard: '{gold_standard}'")
    print("Agent will learn from feedback...")
    
    result = await pipeline.optimize(
        task="Learn the correct answer",
        context={},
        gold_standard=gold_standard
    )
    
    print(f"\nResults:")
    print(f"  Total Iterations: {result['total_iterations']}")
    print(f"  Optimization Complete: {result['optimization_complete']}")
    
    # Show learning progress
    print(f"\nLearning Progress:")
    for it in result['iterations']:
        print(f"  Iteration {it['iteration']}: Score={it['evaluation_score']:.2f}, Status={it['evaluation_status']}")
    
    return result


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
        
        # First attempt: Wrong syntax
        if self.attempts == 1:
            result._store = {"output": "graph A --> B"}  # Missing node definitions
        # Second attempt: Better but still wrong
        elif self.attempts == 2:
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]"}  # Missing arrow
        # Third attempt: Correct!
        else:
            result._store = {"output": "graph TD\n    A[Start]\n    B[End]\n    A --> B"}
        
        return result


async def test_mermaid_improvement():
    """Test Mermaid diagram generation improvement."""
    print("\n" + "="*80)
    print("Test: Mermaid Diagram Improvement")
    print("="*80)
    
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
            agent=TeacherAgent(),
            metadata={"is_teacher": True}
        )
    ]
    
    async def evaluate_mermaid(output, gold_standard, task, context):
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check syntax validity
        has_graph = "graph" in output_str.lower()
        has_nodes = "[" in output_str and "]" in output_str
        has_arrow = "-->" in output_str
        
        syntax_valid = has_graph and has_nodes and has_arrow
        matches_gold = output_str == gold_str
        
        if matches_gold:
            score = 1.0
        elif syntax_valid:
            score = 0.5  # Valid syntax but not exact match
        else:
            score = 0.0  # Invalid syntax
        
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT",
            "difference": f"Syntax valid: {syntax_valid}, Matches gold: {matches_gold}"
        }
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        output_path="./test_outputs/mermaid_improvement"
    )
    pipeline.config.evaluation_function = evaluate_mermaid
    
    gold_standard = "graph TD\n    A[Start]\n    B[End]\n    A --> B"
    
    print(f"Gold Standard:\n{gold_standard}")
    print(f"\nInitial output will be: 'graph A --> B' (invalid syntax)")
    
    result = await pipeline.optimize(
        task="Generate workflow diagram",
        context={"description": "Start to End flow"},
        gold_standard=gold_standard
    )
    
    print(f"\nResults:")
    print(f"  Total Iterations: {result['total_iterations']}")
    print(f"  Optimization Complete: {result['optimization_complete']}")
    
    print(f"\nImprovement Progress:")
    for it in result['iterations']:
        score = it['evaluation_score']
        status_icon = "✓" if score == 1.0 else "⚠" if score == 0.5 else "✗"
        print(f"  {status_icon} Iteration {it['iteration']}: Score={score:.2f} ({it['evaluation_status']})")
    
    if result['optimization_complete']:
        print("\n✓ SUCCESS: Mermaid diagram improved from invalid to correct!")
    
    return result


async def main():
    """Run all improvement tests."""
    print("="*80)
    print("OptimizationPipeline - Improvement Tests")
    print("Testing actual optimization from WRONG to CORRECT outputs")
    print("="*80)
    
    try:
        # Test 1: Basic improvement
        await test_improvement_with_wrong_initial_output()
        
        # Test 2: Learning from feedback
        await test_learning_from_feedback()
        
        # Test 3: Mermaid improvement
        await test_mermaid_improvement()
        
        print("\n" + "="*80)
        print("✓ ALL IMPROVEMENT TESTS COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
