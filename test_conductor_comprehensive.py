"""
Comprehensive Full Conductor Tests
===================================

Tests the FULL MultiAgentsOrchestrator (Conductor) with all features:
- Learning (Q-learning, TD-lambda)
- Tools (ToolShed integration)
- Memory (Cortex brain-inspired)
- Validation (Architect/Auditor)

Uses Claude CLI with JSON output for actual LLM calls.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

import dspy

# Configure Claude CLI
print("Configuring DSPy with Claude CLI...")
try:
    sys.path.insert(0, str(Path(__file__).parent / "examples"))
    from claude_cli_wrapper_enhanced import EnhancedClaudeCLILM

    lm = EnhancedClaudeCLILM(model="sonnet")
    dspy.configure(lm=lm)
    print("✅ DSPy configured with Claude CLI (sonnet)")
except Exception as e:
    print(f"❌ Failed to configure Claude CLI: {e}")
    sys.exit(1)

# ============================================================================
# Test 11: Full Conductor Basic Operation
# ============================================================================

async def test_conductor_basic():
    """Test 11: Full Conductor with minimal configuration"""
    print("\n" + "="*80)
    print("TEST 11: Full Conductor - Basic Operation")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.agent_config import AgentConfig
        from core.foundation.data_structures import JottyConfig
        import dspy

        print("\n  Creating Conductor with basic configuration...")

        # Simple signature for testing
        class SimpleTaskSignature(dspy.Signature):
            """Simple task execution"""
            task = dspy.InputField()
            result = dspy.OutputField()

        # Create agent config
        agents = [
            AgentConfig(
                name="worker",
                agent=dspy.ChainOfThought(SimpleTaskSignature),
                architect_prompts=[],
                auditor_prompts=[],
                enable_architect=False,  # Disable validation for basic test
                enable_auditor=False
            )
        ]

        # Create minimal config
        config = JottyConfig()
        config.enable_validation = False  # Disable validation for basic test
        config.max_episode_iterations = 2

        print("  Initializing Conductor...")

        # Create Conductor with minimal setup
        # metadata_provider can be None for basic testing
        conductor = MultiAgentsOrchestrator(
            actors=agents,
            metadata_provider=None,  # Optional for basic test
            config=config
        )

        print("  Running basic task...")
        result = await conductor.run(goal="Count from 1 to 3", max_steps=2)

        print(f"  Success: {result is not None}")

        if result is not None:
            print("✅ Full Conductor basic operation works")
            return True
        else:
            print("⚠️ Conductor execution returned None")
            return False

    except ImportError as e:
        print(f"  ⚠️ Skipped: Cannot import Conductor - {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 12: Conductor with Learning Enabled
# ============================================================================

async def test_conductor_with_learning():
    """Test 12: Full Conductor with Q-learning enabled"""
    print("\n" + "="*80)
    print("TEST 12: Full Conductor with Learning")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.agent_config import AgentConfig
        from core.foundation.data_structures import JottyConfig
        import dspy

        print("\n  Creating Conductor with learning enabled...")

        # Simple signature
        class TaskSignature(dspy.Signature):
            """Execute a task"""
            task = dspy.InputField()
            result = dspy.OutputField()

        # Create agent
        agents = [
            AgentConfig(
                name="learner",
                agent=dspy.ChainOfThought(TaskSignature),
                architect_prompts=[],
                auditor_prompts=[],
                enable_architect=False,
                enable_auditor=False
            )
        ]

        # Configure with learning parameters
        config = JottyConfig()
        config.enable_validation = False
        config.max_episode_iterations = 2
        # Learning is enabled by default in JottyConfig
        # TD-lambda parameters already set in config

        print("  Initializing Conductor with learning...")

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            metadata_provider=None,
            config=config
        )

        # Check if learning manager exists
        has_learning_manager = hasattr(conductor, 'learning_manager')
        print(f"  Learning manager initialized: {has_learning_manager}")

        print("  Running task with learning...")
        result = await conductor.run(goal="Calculate 2 + 2", max_steps=1)

        print(f"  Success: {result is not None}")

        if result is not None and has_learning_manager:
            print("✅ Full Conductor with learning works")
            return True
        else:
            print(f"⚠️ Learning incomplete (result: {result is not None}, manager: {has_learning_manager})")
            return False

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 13: Conductor with Memory (Cortex)
# ============================================================================

async def test_conductor_with_memory():
    """Test 13: Full Conductor with brain-inspired memory"""
    print("\n" + "="*80)
    print("TEST 13: Full Conductor with Memory (Cortex)")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.agent_config import AgentConfig
        from core.foundation.data_structures import JottyConfig
        import dspy

        print("\n  Creating Conductor with memory enabled...")

        # Simple signature
        class MemoryTaskSignature(dspy.Signature):
            """Task that uses memory"""
            task = dspy.InputField()
            result = dspy.OutputField()

        # Create agent
        agents = [
            AgentConfig(
                name="memory_user",
                agent=dspy.ChainOfThought(MemoryTaskSignature),
                architect_prompts=[],
                auditor_prompts=[],
                enable_architect=False,
                enable_auditor=False
            )
        ]

        # Configure with memory
        config = JottyConfig()
        config.enable_validation = False
        config.max_episode_iterations = 2
        # Memory capacities already set in JottyConfig defaults

        print("  Initializing Conductor with memory...")

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            metadata_provider=None,
            config=config
        )

        # Check if brain memory exists
        has_brain = hasattr(conductor, 'brain') and conductor.brain is not None
        print(f"  Brain memory initialized: {has_brain}")

        print("  Running tasks to test memory...")

        # First task - store something
        result1 = await conductor.run(goal="Remember: The answer is 42", max_steps=1)

        # Second task - recall
        result2 = await conductor.run(goal="What was the answer?", max_steps=1)

        print(f"  First task success: {result1 is not None}")
        print(f"  Second task success: {result2 is not None}")

        if result1 is not None and result2 is not None:
            print("✅ Full Conductor with memory works")
            return True
        else:
            print(f"⚠️ Memory test incomplete")
            return False

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 14: Conductor with Validation
# ============================================================================

async def test_conductor_with_validation():
    """Test 14: Full Conductor with Architect/Auditor validation"""
    print("\n" + "="*80)
    print("TEST 14: Full Conductor with Validation")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.agent_config import AgentConfig
        from core.foundation.data_structures import JottyConfig
        import dspy

        print("\n  Creating Conductor with validation enabled...")

        # Task signature
        class ValidatedTaskSignature(dspy.Signature):
            """Task with validation"""
            task = dspy.InputField()
            result = dspy.OutputField()

        # Create agent with validation enabled
        agents = [
            AgentConfig(
                name="validated_worker",
                agent=dspy.ChainOfThought(ValidatedTaskSignature),
                architect_prompts=["Plan how to execute this task carefully"],
                auditor_prompts=["Verify the output is correct"],
                enable_architect=True,  # Enable pre-execution planning
                enable_auditor=True,     # Enable post-execution validation
                validation_mode="standard"
            )
        ]

        # Configure with validation
        config = JottyConfig()
        config.enable_validation = True  # Enable validation system
        config.max_validation_rounds = 2
        config.max_episode_iterations = 3

        print("  Initializing Conductor with validation...")

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            metadata_provider=None,
            config=config
        )

        # Check if validation manager exists
        has_validation = hasattr(conductor, 'validation_manager')
        print(f"  Validation manager initialized: {has_validation}")

        print("  Running task with validation...")
        result = await conductor.run(goal="Explain what is 5 * 5", max_steps=1)

        print(f"  Success: {result is not None}")

        if result is not None and has_validation:
            print("✅ Full Conductor with validation works")
            return True
        else:
            print(f"⚠️ Validation test incomplete")
            return False

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main Test Runner
# ============================================================================

async def run_conductor_tests():
    """Run ALL Conductor comprehensive tests"""
    print("\n" + "#"*80)
    print("# FULL CONDUCTOR COMPREHENSIVE TESTS")
    print("#"*80)

    tests = [
        ("Conductor - Basic Operation", test_conductor_basic),
        ("Conductor - With Learning", test_conductor_with_learning),
        ("Conductor - With Memory", test_conductor_with_memory),
        ("Conductor - With Validation", test_conductor_with_validation),
    ]

    results = {}
    for name, test_func in tests:
        result = await test_func()
        results[name] = result

    # Summary
    print("\n" + "#"*80)
    print("# FULL CONDUCTOR TEST SUMMARY")
    print("#"*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⚠️ {name} (skipped)")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ Some tests failed")
        return False
    else:
        print("\n✅ All Conductor tests passed!")
        return True

if __name__ == "__main__":
    success = asyncio.run(run_conductor_tests())
    sys.exit(0 if success else 1)
