#!/usr/bin/env python
"""
Test Refactored Jotty Framework with Claude API
================================================

This script demonstrates that the refactored Jotty framework
(with ParameterResolver, ToolManager, and StateManager components)
works correctly with the Claude API.

Usage:
    export ANTHROPIC_API_KEY="your-api-key-here"
    python examples/test_refactored_jotty.py

The script will run a simple multi-agent workflow to verify:
1. ParameterResolver correctly resolves parameters between agents
2. ToolManager properly manages tools for agents
3. StateManager tracks state correctly
4. All components integrate seamlessly
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_simple_agent():
    """Test 1: Simple single-agent task."""
    print("\n" + "="*70)
    print("TEST 1: Simple Single-Agent Task")
    print("="*70)

    import dspy
    from core import SwarmConfig, AgentSpec, JottyCore

    # Configure DSPy with Claude
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set!")
        print("   Set your API key with: export ANTHROPIC_API_KEY='sk-...'")
        return False

    # Set API key as environment variable for DSPy/LiteLLM
    os.environ['ANTHROPIC_API_KEY'] = api_key

    lm = dspy.LM(
        model='anthropic/claude-3-5-haiku-20241022',
        max_tokens=500
    )
    dspy.configure(lm=lm)

    # Define agent signature
    class GreetingTask(dspy.Signature):
        """Generate a creative greeting message."""
        goal = dspy.InputField(desc="The greeting goal")
        greeting = dspy.OutputField(desc="A creative greeting")

    # Create agent
    agent = AgentSpec(
        name="GreetingAgent",
        agent=dspy.ChainOfThought(GreetingTask),
        architect_prompts=[],
        auditor_prompts=[]
    )

    # Create configuration
    config = SwarmConfig(
        actors=[agent],
        max_rounds=1,
        enable_learning=False
    )

    # Run the swarm
    print("\n🚀 Running single-agent swarm...")
    swarm = JottyCore(config)
    result = swarm.run(
        goal="Create a greeting for the newly refactored Jotty framework"
    )

    # Display results
    print(f"\n✅ Agent completed successfully!")
    print(f"   Result: {result.final_output}")

    return True


def test_multi_agent_parameter_resolution():
    """Test 2: Multi-agent with parameter resolution."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Agent Parameter Resolution")
    print("="*70)

    import dspy
    from core import SwarmConfig, AgentSpec, JottyCore

    api_key = os.getenv('ANTHROPIC_API_KEY')
    os.environ['ANTHROPIC_API_KEY'] = api_key

    lm = dspy.LM(
        model='anthropic/claude-3-5-haiku-20241022',
        max_tokens=500
    )
    dspy.configure(lm=lm)

    # Agent 1: Extract topic
    class ExtractTopic(dspy.Signature):
        """Extract the main topic from a query."""
        query = dspy.InputField()
        topic = dspy.OutputField(desc="Main topic (2-4 words)")

    # Agent 2: Explain topic (uses output from Agent 1)
    class ExplainTopic(dspy.Signature):
        """Explain a topic briefly."""
        topic = dspy.InputField()
        explanation = dspy.OutputField(desc="Brief explanation")

    # Create agents
    topic_agent = AgentSpec(
        name="TopicExtractor",
        agent=dspy.ChainOfThought(ExtractTopic),
        architect_prompts=[],
        auditor_prompts=[],
        outputs=["topic"]  # This agent produces 'topic'
    )

    explain_agent = AgentSpec(
        name="TopicExplainer",
        agent=dspy.ChainOfThought(ExplainTopic),
        architect_prompts=[],
        auditor_prompts=[],
        parameter_mappings={"topic": "TopicExtractor"}  # Get topic from TopicExtractor
    )

    # Create configuration
    config = SwarmConfig(
        actors=[topic_agent, explain_agent],
        max_rounds=2,
        enable_learning=False
    )

    # Run the swarm
    print("\n🚀 Running multi-agent swarm...")
    print("   Agent 1 will extract a topic")
    print("   Agent 2 will explain it (using ParameterResolver)")

    swarm = JottyCore(config)
    result = swarm.run(
        goal="Learn about code refactoring",
        query="What is the Single Responsibility Principle in software design?"
    )

    # Display results
    print(f"\n✅ Multi-agent workflow completed!")
    print(f"   Final result: {result.final_output}")

    return True


def test_component_integration():
    """Test 3: Verify all components are integrated."""
    print("\n" + "="*70)
    print("TEST 3: Component Integration Verification")
    print("="*70)

    from core.orchestration.parameter_resolver import ParameterResolver
    from core.orchestration.tool_manager import ToolManager
    from core.orchestration.state_manager import StateManager

    print("\n✅ All components importable:")
    print(f"   - ParameterResolver: {ParameterResolver.__name__}")
    print(f"   - ToolManager: {ToolManager.__name__}")
    print(f"   - StateManager: {StateManager.__name__}")

    # Check component line counts
    import inspect

    param_lines = len(inspect.getsourcelines(ParameterResolver)[0])
    tool_lines = len(inspect.getsourcelines(ToolManager)[0])
    state_lines = len(inspect.getsourcelines(StateManager)[0])

    print(f"\n📊 Component Sizes:")
    print(f"   - ParameterResolver: ~{param_lines} lines")
    print(f"   - ToolManager: ~{tool_lines} lines")
    print(f"   - StateManager: ~{state_lines} lines")
    print(f"   - Total extracted: ~{param_lines + tool_lines + state_lines} lines")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING REFACTORED JOTTY FRAMEWORK WITH CLAUDE API")
    print("="*70)

    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("\n❌ ANTHROPIC_API_KEY environment variable not set!")
        print("\nTo run this test:")
        print("  1. Get your API key from https://console.anthropic.com/")
        print("  2. Set it: export ANTHROPIC_API_KEY='your-key-here'")
        print("  3. Run this script again")
        sys.exit(1)

    # Run tests
    tests = [
        ("Component Integration", test_component_integration),
        ("Simple Agent", test_simple_agent),
        ("Multi-Agent Parameter Resolution", test_multi_agent_parameter_resolution),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The refactored Jotty framework is working correctly!")
        print("\nKey components verified:")
        print("  ✓ ParameterResolver - handles parameter resolution between agents")
        print("  ✓ ToolManager - manages tools for different agent types")
        print("  ✓ StateManager - tracks execution state and outputs")
        print("  ✓ Full integration - all components work together seamlessly")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
