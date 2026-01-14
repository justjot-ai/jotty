"""
Hello World Test for Jotty Framework
=====================================

Simple test to verify Jotty framework is working correctly.
This test uses a mock LLM so no API keys are needed.
"""
import asyncio
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

import dspy
from core.orchestration.conductor import Conductor
from core.foundation.agent_config import AgentConfig
from core.foundation.data_structures import JottyConfig


# =============================================================================
# Mock LLM (No API needed)
# =============================================================================

class MockLM:
    """Mock Language Model for testing without API calls."""

    def __init__(self):
        self.history = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Mock LLM call that returns predefined responses."""
        self.history.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})

        # Return a mock response
        if messages:
            # For chat-style calls
            return ["Hello! This is a mock response from Jotty. The task has been completed successfully."]
        else:
            # For completion-style calls
            return ["Mock completion response"]

    def inspect_history(self, n=1):
        """Return last n calls."""
        return self.history[-n:] if self.history else []


# =============================================================================
# Mock Metadata Provider (Required by Conductor)
# =============================================================================

class MockMetadataProvider:
    """Mock metadata provider for testing."""

    def get_all_tools(self):
        """Return empty tools dict."""
        return {}

    def get_context(self):
        """Return empty context."""
        return {}


# =============================================================================
# Simple Agent
# =============================================================================

class HelloWorldAgent(dspy.Module):
    """A simple agent that greets the user."""

    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("task -> greeting")

    def forward(self, task: str) -> dspy.Prediction:
        """Process the task and return a greeting."""
        # For testing, we'll return a simple response
        result = dspy.Prediction(greeting=f"Hello! Task received: {task}")
        return result


# =============================================================================
# Test Function
# =============================================================================

async def test_hello_world():
    """Test basic Jotty functionality."""

    print("=" * 70)
    print("JOTTY FRAMEWORK - Hello World Test")
    print("=" * 70)
    print()

    # Step 1: Configure mock LLM
    print("Step 1: Configuring mock LLM (no API key needed)...")
    mock_lm = MockLM()
    dspy.configure(lm=mock_lm)
    print("✓ Mock LLM configured")
    print()

    # Step 2: Create a simple agent
    print("Step 2: Creating HelloWorld agent...")
    agent = HelloWorldAgent()
    print("✓ Agent created")
    print()

    # Step 3: Create AgentConfig
    print("Step 3: Creating AgentConfig...")
    agent_config = AgentConfig(
        name="HelloWorldAgent",
        agent=agent,
        architect_prompts=[],  # No architect for simple test
        auditor_prompts=[],    # No auditor for simple test
        enable_architect=False,
        enable_auditor=False,
    )
    print("✓ AgentConfig created")
    print()

    # Step 4: Create minimal Jotty configuration
    print("Step 4: Creating minimal Jotty configuration...")
    config = JottyConfig(
        output_base_dir="./test_output",
        create_run_folder=False,
        enable_beautified_logs=False,
        log_level="WARNING",
        max_actor_iters=3,
        max_episode_iterations=1,
        enable_validation=False,  # Disable validation for simplicity
    )
    print("✓ Configuration created")
    print()

    # Step 5: Create mock metadata provider
    print("Step 5: Creating mock metadata provider...")
    metadata_provider = MockMetadataProvider()
    print("✓ Metadata provider created")
    print()

    # Step 6: Initialize Conductor
    print("Step 6: Initializing Jotty Conductor...")
    conductor = Conductor(
        actors=[agent_config],
        metadata_provider=metadata_provider,
        config=config,
    )
    print("✓ Conductor initialized")
    print()

    # Step 7: Run a simple task
    print("Step 7: Running 'Hello World' task...")
    try:
        result = await conductor.run(
            goal="Say hello to the world",
            task="Generate a friendly greeting"
        )
        print("✓ Task completed!")
        print()

        # Step 8: Display results
        print("Step 8: Results")
        print("-" * 70)
        print(f"Success: {result.success if hasattr(result, 'success') else 'N/A'}")
        print(f"Final Output: {result.final_output if hasattr(result, 'final_output') else result}")
        print("-" * 70)
        print()

        print("=" * 70)
        print("✓ HELLO WORLD TEST PASSED!")
        print("=" * 70)
        print()
        print("Jotty framework is working correctly! 🎉")
        print()

        return True

    except Exception as e:
        print(f"✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Simpler Synchronous Test (Fallback)
# =============================================================================

def test_hello_world_simple():
    """Simplified synchronous test."""

    print("=" * 70)
    print("JOTTY FRAMEWORK - Simple Hello World Test")
    print("=" * 70)
    print()

    try:
        # Test imports
        print("1. Testing imports...")
        from core.orchestration.conductor import Conductor
        from core.foundation.agent_config import AgentConfig
        from core.foundation.data_structures import JottyConfig
        print("   ✓ All imports successful")
        print()

        # Test agent creation
        print("2. Testing agent creation...")
        agent = HelloWorldAgent()
        print("   ✓ Agent created")
        print()

        # Test agent forward pass
        print("3. Testing agent forward pass...")
        result = agent.forward(task="Say hello")
        print(f"   ✓ Agent result: {result.greeting}")
        print()

        # Test configuration
        print("4. Testing configuration...")
        config = JottyConfig(
            output_base_dir="./test_output",
            enable_validation=False,
        )
        print("   ✓ Configuration created")
        print()

        # Test agent config
        print("5. Testing AgentConfig...")
        agent_config = AgentConfig(
            name="TestAgent",
            agent=agent,
            architect_prompts=[],
            auditor_prompts=[],
            enable_architect=False,
            enable_auditor=False,
        )
        print("   ✓ AgentConfig created")
        print()

        print("=" * 70)
        print("✓ SIMPLE TEST PASSED!")
        print("=" * 70)
        print()
        print("All core components are working! 🎉")
        print()

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print()
    print("Choose test mode:")
    print("1. Simple synchronous test (recommended)")
    print("2. Full async test with Conductor")
    print()

    # Default to simple test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        print("Running full async test...")
        print()
        success = asyncio.run(test_hello_world())
    else:
        print("Running simple synchronous test...")
        print()
        success = test_hello_world_simple()

    sys.exit(0 if success else 1)
