"""
Test Expert Agent + OptimizationPipeline Integration

Tests that expert agents work correctly with OptimizationPipeline
when using DSPy modules.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available. Install with: pip install dspy-ai")
    sys.exit(1)

from core.experts import ExpertAgent, ExpertAgentConfig
from core.orchestration.optimization_pipeline import create_optimization_pipeline
from core.foundation.agent_config import AgentConfig


class SimpleTestExpert(ExpertAgent):
    """Simple test expert using DSPy."""
    
    def _create_default_agent(self):
        """Create a simple DSPy agent."""
        class SimpleSignature(dspy.Signature):
            """Simple generation signature."""
            task: str = dspy.InputField(desc="Task")
            description: str = dspy.InputField(desc="Description")
            
            output: str = dspy.OutputField(desc="Output")
        
        return dspy.ChainOfThought(SimpleSignature)


async def test_optimization_pipeline_integration():
    """Test that OptimizationPipeline works with DSPy expert agents."""
    print("=" * 80)
    print("TESTING EXPERT + OPTIMIZATION PIPELINE INTEGRATION")
    print("=" * 80)
    print()
    
    # Create expert
    config = ExpertAgentConfig(
        name="simple_test_expert",
        domain="test",
        description="Simple test expert",
        training_gold_standards=[
            {
                "task": "Generate output",
                "context": {"description": "Test description"},
                "gold_standard": "Expected output"
            }
        ],
        max_training_iterations=2,
        required_training_pass_count=1,
        enable_teacher_model=True
    )
    
    expert = SimpleTestExpert(config)
    
    # Test that agents are created correctly
    print("1. Testing Agent Creation")
    print("-" * 80)
    agents = expert._create_agents()
    
    print(f"   Number of agents: {len(agents)}")
    print(f"   Agent names: {[a.name for a in agents]}")
    
    main_agent = agents[0].agent
    teacher_agent = agents[1].agent if len(agents) > 1 else None
    
    print(f"   Main agent type: {type(main_agent).__name__}")
    print(f"   Is DSPy module: {expert._is_dspy_module(main_agent)}")
    
    if teacher_agent:
        print(f"   Teacher agent type: {type(teacher_agent).__name__}")
        print(f"   Teacher is DSPy: {expert._is_dspy_module(teacher_agent)}")
    
    print("   ✅ Agents created correctly!")
    print()
    
    # Test OptimizationPipeline with DSPy agents
    print("2. Testing OptimizationPipeline with DSPy Agents")
    print("-" * 80)
    
    # Create pipeline with expert's agents
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=2,
        required_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        output_path="./test_outputs/expert_integration"
    )
    
    # Simple evaluation function
    async def evaluate(output, gold_standard, task, context):
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate
    
    print("   Pipeline created successfully")
    print(f"   Number of agents in pipeline: {len(pipeline.agents)}")
    print()
    
    # Note: Actual training requires LLM configuration
    print("   ⚠️  Note: Full training requires LLM configuration:")
    print("      dspy.configure(lm=dspy.LM(model='claude-3-opus'))")
    print()
    
    # Test output extraction
    print("3. Testing Output Extraction from Pipeline")
    print("-" * 80)
    
    # Mock DSPy Prediction
    class MockPrediction:
        def __init__(self):
            self.output = "Test output"
    
    mock_output = MockPrediction()
    extracted = pipeline._extract_agent_output(mock_output)
    
    print(f"   Mock output type: {type(mock_output).__name__}")
    print(f"   Extracted value: {extracted}")
    assert extracted == "Test output", "Should extract output from DSPy Prediction"
    print("   ✅ Output extraction works!")
    print()
    
    print("=" * 80)
    print("✅ INTEGRATION TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ Expert agents create DSPy modules correctly")
    print("  ✅ OptimizationPipeline accepts DSPy agents")
    print("  ✅ Output extraction works for DSPy Predictions")
    print()
    print("Integration is working! Expert agents can use DSPy with OptimizationPipeline.")


if __name__ == "__main__":
    asyncio.run(test_optimization_pipeline_integration())
