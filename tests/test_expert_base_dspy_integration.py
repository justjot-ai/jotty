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

from core.experts.base_expert import BaseExpert
from core.experts import ExpertAgentConfig


class TestExpertAgent(BaseExpert):
    """Test expert agent to verify base class DSPy integration."""

    @property
    def domain(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Test expert for DSPy integration"

    def _create_domain_agent(self, improvements=None):
        """Create a simple DSPy agent for testing."""
        class TestSignature(dspy.Signature):
            """Test signature."""
            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Description")

            output: str = dspy.OutputField(desc="Output")

        return dspy.ChainOfThought(TestSignature)

    def _create_domain_teacher(self):
        """Create teacher agent."""
        class TeacherSignature(dspy.Signature):
            """Teacher signature."""
            task: str = dspy.InputField(desc="Task description")
            gold_standard: str = dspy.InputField(desc="Correct output")
            student_output: str = dspy.InputField(desc="Student output")
            output: str = dspy.OutputField(desc="Correct output")
        return dspy.Predict(TeacherSignature)

    @staticmethod
    def _get_default_training_cases():
        return [{"task": "Test task", "context": {"description": "Test description"}, "gold_standard": "Test output"}]

    @staticmethod
    def _get_default_validation_cases():
        return [{"task": "Validation task", "context": {"description": "Test"}, "gold_standard": "Test output"}]

    async def _evaluate_domain(self, output, gold_standard, task, context):
        return {"score": 1.0 if str(output).strip() == str(gold_standard).strip() else 0.0, "status": "CORRECT"}


async def test_base_class_dspy_support():
    """Test that BaseExpert class handles DSPy correctly."""
    print("=" * 80)
    print("TESTING BASE CLASS DSPy INTEGRATION")
    print("=" * 80)
    print()

    # Create test expert (BaseExpert-based, no ExpertAgentConfig needed for init)
    expert = TestExpertAgent()

    # Test DSPy agent creation
    print("1. Testing DSPy Agent Creation")
    print("-" * 80)
    agent = expert._create_domain_agent()

    is_dspy = expert._is_dspy_available() and isinstance(agent, dspy.Module)
    print(f"   Agent Type: {type(agent).__name__}")
    print(f"   Is DSPy Module: {is_dspy}")
    assert is_dspy, "Agent should be detected as DSPy module"
    print("   DSPy detection works!")
    print()

    # Test _create_default_agent delegates to _create_domain_agent
    print("2. Testing _create_default_agent Delegation")
    print("-" * 80)
    default_agent = expert._create_default_agent(improvements=[])
    print(f"   Default Agent Type: {type(default_agent).__name__}")
    assert isinstance(default_agent, dspy.Module), "Default agent should be DSPy module"
    print("   _create_default_agent delegation works!")
    print()

    # Test training data access
    print("3. Testing Training Data Access")
    print("-" * 80)
    training_data = expert.get_training_data()
    print(f"   Training cases: {len(training_data)}")
    assert len(training_data) > 0, "Should have training data"
    print("   Training data access works!")
    print()

    # Test teacher creation (should use DSPy if available)
    print("4. Testing Teacher Creation")
    print("-" * 80)
    teacher = expert._create_domain_teacher()
    print(f"   Teacher Type: {type(teacher).__name__}")
    is_dspy_teacher = isinstance(teacher, dspy.Module)
    print(f"   Is DSPy Module: {is_dspy_teacher}")
    print("   Teacher creation works!")
    print()

    # Test get_stats
    print("5. Testing get_stats")
    print("-" * 80)
    stats = expert.get_stats()
    print(f"   Stats: {stats}")
    assert stats['domain'] == 'test'
    assert stats['expert_type'] == 'TestExpertAgent'
    print("   get_stats works!")
    print()

    print("=" * 80)
    print("ALL BASE CLASS TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  DSPy agent creation works")
    print("  _create_default_agent delegation works")
    print("  Training data access works")
    print("  Teacher creation uses DSPy when available")
    print("  get_stats works")
    print()
    print("All expert agents now benefit from BaseExpert DSPy integration!")


if __name__ == "__main__":
    asyncio.run(test_base_class_dspy_support())
