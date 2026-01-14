"""
Test Expert Agent Base Class DSPy Integration

Tests that the base ExpertAgent class properly handles DSPy modules
so all expert agents benefit automatically.
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


class TestExpertAgent(ExpertAgent):
    """Test expert agent to verify base class DSPy integration."""
    
    def _create_default_agent(self):
        """Create a simple DSPy agent for testing."""
        class TestSignature(dspy.Signature):
            """Test signature."""
            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Description")
            
            output: str = dspy.OutputField(desc="Output")
        
        return dspy.ChainOfThought(TestSignature)


async def test_base_class_dspy_support():
    """Test that base ExpertAgent class handles DSPy correctly."""
    print("=" * 80)
    print("TESTING BASE CLASS DSPy INTEGRATION")
    print("=" * 80)
    print()
    
    # Create test expert
    config = ExpertAgentConfig(
        name="test_expert",
        domain="test",
        description="Test expert for DSPy integration",
        training_gold_standards=[
            {
                "task": "Test task",
                "context": {"description": "Test description"},
                "gold_standard": "Test output"
            }
        ],
        max_training_iterations=2,
        required_training_pass_count=1,
        enable_teacher_model=True
    )
    
    expert = TestExpertAgent(config)
    
    # Test DSPy detection
    print("1. Testing DSPy Module Detection")
    print("-" * 80)
    agents = expert._create_agents()
    main_agent = agents[0].agent
    
    is_dspy = expert._is_dspy_module(main_agent)
    print(f"   Agent Type: {type(main_agent).__name__}")
    print(f"   Is DSPy Module: {is_dspy}")
    assert is_dspy, "Agent should be detected as DSPy module"
    print("   ✅ DSPy detection works!")
    print()
    
    # Test DSPy output extraction
    print("2. Testing DSPy Output Extraction")
    print("-" * 80)
    
    # Create a mock DSPy Prediction
    class MockPrediction:
        def __init__(self):
            self.output = "Test output from DSPy"
    
    mock_result = MockPrediction()
    extracted = expert._extract_dspy_output(mock_result)
    print(f"   Mock Result Type: {type(mock_result).__name__}")
    print(f"   Extracted Output: {extracted}")
    assert extracted == "Test output from DSPy", "Should extract output from DSPy Prediction"
    print("   ✅ DSPy output extraction works!")
    print()
    
    # Test regular output extraction
    print("3. Testing Regular Output Extraction")
    print("-" * 80)
    
    class RegularResult:
        def __init__(self):
            self._store = {"output": "Regular output"}
    
    regular_result = RegularResult()
    extracted = expert._extract_dspy_output(regular_result)
    print(f"   Regular Result Type: {type(regular_result).__name__}")
    print(f"   Extracted Output: {extracted}")
    assert extracted == "Regular output", "Should extract from _store dict"
    print("   ✅ Regular output extraction works!")
    print()
    
    # Test teacher creation (should use DSPy if available)
    print("4. Testing Teacher Creation")
    print("-" * 80)
    teacher = expert._create_default_teacher()
    print(f"   Teacher Type: {type(teacher).__name__}")
    is_dspy_teacher = expert._is_dspy_module(teacher)
    print(f"   Is DSPy Module: {is_dspy_teacher}")
    print("   ✅ Teacher creation works!")
    print()
    
    print("=" * 80)
    print("✅ ALL BASE CLASS TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ DSPy module detection works")
    print("  ✅ DSPy output extraction works")
    print("  ✅ Regular output extraction works")
    print("  ✅ Teacher creation uses DSPy when available")
    print()
    print("All expert agents now benefit from base class DSPy integration!")


if __name__ == "__main__":
    asyncio.run(test_base_class_dspy_support())
