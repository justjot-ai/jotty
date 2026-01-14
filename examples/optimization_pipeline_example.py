"""
Example: Using OptimizationPipeline for Iterative Improvement

This example demonstrates how to use the OptimizationPipeline to:
1. Run multiple agents in sequence
2. Evaluate outputs against gold standards
3. Use a teacher model when evaluation fails
4. Update knowledge base for learning
5. Iterate until success or max iterations

# ✅ GENERIC: Works with any agents, any domain
"""

import asyncio
import dspy
from pathlib import Path

# Import Jotty components
from jotty.core.jotty import (
    OptimizationPipeline,
    OptimizationConfig,
    AgentConfig,
    create_optimization_pipeline
)


# =============================================================================
# Example 1: Simple Optimization Pipeline
# =============================================================================

async def example_simple_optimization():
    """Simple example with basic agents."""
    
    # Define agents
    class Agent1(dspy.Module):
        def forward(self, task: str) -> str:
            return f"Agent1 processed: {task}"
    
    class Agent2(dspy.Module):
        def forward(self, input: str) -> str:
            return f"Agent2 enhanced: {input}"
    
    agents = [
        AgentConfig(
            name="agent1",
            agent=Agent1(),
            outputs=["result"]
        ),
        AgentConfig(
            name="agent2",
            agent=Agent2(),
            parameter_mappings={"input": "result"}
        )
    ]
    
    # Create pipeline
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        output_path="./outputs/optimization_example"
    )
    
    # Define evaluation function
    def evaluate(output, gold_standard, task, context):
        """Simple evaluation: check if output matches gold standard."""
        return {
            "score": 1.0 if output == gold_standard else 0.0,
            "status": "CORRECT" if output == gold_standard else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate
    
    # Run optimization
    result = await pipeline.optimize(
        task="Process this task",
        context={},
        gold_standard="Agent2 enhanced: Agent1 processed: Process this task"
    )
    
    print(f"Optimization complete: {result['optimization_complete']}")
    print(f"Iterations: {result['total_iterations']}")
    print(f"Final score: {result['final_result']['evaluation_score']}")


# =============================================================================
# Example 2: With Teacher Model
# =============================================================================

async def example_with_teacher():
    """Example with teacher model fallback."""
    
    class StudentAgent(dspy.Module):
        def forward(self, task: str) -> str:
            # Student makes a mistake
            return f"Student answer: {task} (incorrect)"
    
    class TeacherAgent(dspy.Module):
        def forward(self, task: str, student_output: str, gold_standard: str, evaluation_feedback: str) -> str:
            # Teacher provides correct answer
            return gold_standard
    
    agents = [
        AgentConfig(
            name="student",
            agent=StudentAgent(),
            outputs=["output"]
        ),
        AgentConfig(
            name="teacher",
            agent=TeacherAgent(),
            metadata={"is_teacher": True}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=2,
        enable_teacher_model=True,
        enable_kb_updates=False,  # No KB agent in this example
        output_path="./outputs/teacher_example"
    )
    
    def evaluate(output, gold_standard, task, context):
        return {
            "score": 1.0 if output == gold_standard else 0.0,
            "status": "CORRECT" if output == gold_standard else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Solve this problem",
        context={},
        gold_standard="Correct answer"
    )
    
    print(f"Teacher used: {any(it.get('has_teacher_output') for it in result['iterations'])}")
    print(f"Final output: {result['final_result']['output']}")


# =============================================================================
# Example 3: With Knowledge Base Updates
# =============================================================================

async def example_with_kb_updates():
    """Example with KB update agent."""
    
    class MainAgent(dspy.Module):
        def forward(self, task: str) -> str:
            return f"Generated: {task}"
    
    class TeacherAgent(dspy.Module):
        def forward(self, task: str, student_output: str, gold_standard: str) -> str:
            return gold_standard
    
    class KBUpdateAgent(dspy.Module):
        def forward(self, student_output: str, teacher_output: str, evaluation_result: dict) -> dict:
            # Analyze differences and update KB
            return {
                "status": "updated",
                "updates": [
                    f"Learned: Use '{teacher_output}' instead of '{student_output}'"
                ]
            }
    
    agents = [
        AgentConfig(
            name="main",
            agent=MainAgent(),
            outputs=["output"]
        ),
        AgentConfig(
            name="teacher",
            agent=TeacherAgent(),
            metadata={"is_teacher": True}
        ),
        AgentConfig(
            name="kb_updater",
            agent=KBUpdateAgent(),
            metadata={"is_kb_updater": True}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=1,
        enable_teacher_model=True,
        enable_kb_updates=True,
        output_path="./outputs/kb_example"
    )
    
    def evaluate(output, gold_standard, task, context):
        return {
            "score": 1.0 if output == gold_standard else 0.0,
            "status": "CORRECT" if output == gold_standard else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Generate output",
        context={},
        gold_standard="Expected output"
    )
    
    # Check KB updates
    for iteration in result['iterations']:
        if iteration.get('has_kb_updates'):
            print(f"Iteration {iteration['iteration']}: KB updated")


# =============================================================================
# Example 4: Integration with Jotty Conductor
# =============================================================================

async def example_with_conductor():
    """Example using OptimizationPipeline with Jotty Conductor."""
    
    from jotty.core.jotty import create_conductor, JottyConfig
    
    # Create Conductor with agents
    class Agent1(dspy.Module):
        def forward(self, query: str) -> str:
            return f"Agent1: {query}"
    
    class Agent2(dspy.Module):
        def forward(self, input: str) -> str:
            return f"Agent2: {input}"
    
    agents = [
        AgentConfig(
            name="agent1",
            agent=Agent1(),
            outputs=["result"]
        ),
        AgentConfig(
            name="agent2",
            agent=Agent2(),
            parameter_mappings={"input": "result"}
        )
    ]
    
    # Create Conductor
    conductor = create_conductor(
        agents=agents,
        config=JottyConfig(base_path="./outputs/conductor_example")
    )
    
    # Create OptimizationPipeline with Conductor
    pipeline = OptimizationPipeline(
        agents=agents,
        config=OptimizationConfig(
            max_iterations=3,
            required_pass_count=1,
            output_path="./outputs/conductor_example"
        ),
        conductor=conductor
    )
    
    def evaluate(output, gold_standard, task, context):
        return {
            "score": 1.0 if str(output) == gold_standard else 0.0,
            "status": "CORRECT" if str(output) == gold_standard else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Process query",
        context={"query": "test"},
        gold_standard="Agent2: Agent1: test"
    )
    
    print(f"Used Conductor: {pipeline.conductor is not None}")
    print(f"Result: {result['optimization_complete']}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("=" * 80)
    print("Example 1: Simple Optimization")
    print("=" * 80)
    await example_simple_optimization()
    
    print("\n" + "=" * 80)
    print("Example 2: With Teacher Model")
    print("=" * 80)
    await example_with_teacher()
    
    print("\n" + "=" * 80)
    print("Example 3: With KB Updates")
    print("=" * 80)
    await example_with_kb_updates()
    
    print("\n" + "=" * 80)
    print("Example 4: With Conductor")
    print("=" * 80)
    await example_with_conductor()


if __name__ == "__main__":
    # Configure DSPy
    dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))
    
    # Run examples
    asyncio.run(main())
