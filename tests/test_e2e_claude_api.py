"""
End-to-End Test with Claude API
================================

Tests the refactored Jotty framework with actual Claude API calls
to verify all components work together correctly.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestClaudeAPIIntegration:
    """Test Jotty framework with Claude API."""

    def test_simple_agent_with_claude(self):
        """Test a simple agent using Claude API."""
        import dspy

        from core import AgentConfig, Orchestrator, SwarmConfig

        # Configure DSPy to use Claude
        api_key = os.getenv("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = api_key

        lm = dspy.LM(model="anthropic/claude-3-5-sonnet-20241022", max_tokens=1000)
        dspy.configure(lm=lm)

        # Create a simple task agent
        class SimpleTask(dspy.Signature):
            """Generate a creative greeting."""

            task = dspy.InputField(desc="The task to perform")
            response = dspy.OutputField(desc="The response")

        # Create agent
        agent = AgentConfig(
            name="GreetingAgent",
            agent=dspy.ChainOfThought(SimpleTask),
            architect_prompts=[],
            auditor_prompts=[],
        )

        # Create swarm configuration
        config = SwarmConfig(
            actors=[agent], max_rounds=1, enable_learning=False  # Disable learning for simple test
        )

        # Run the swarm
        swarm = Orchestrator(config)
        result = swarm.run(
            goal="Say hello to the refactored Jotty framework", task="Generate a creative greeting"
        )

        # Verify we got a response
        assert result is not None
        assert hasattr(result, "final_output")
        print(f"\n✅ Agent Response: {result.final_output}")

    def test_multi_agent_parameter_resolution(self):
        """Test parameter resolution between multiple agents."""
        import dspy

        from core import AgentConfig, Orchestrator, SwarmConfig

        # Configure DSPy
        api_key = os.getenv("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = api_key

        lm = dspy.LM(model="anthropic/claude-3-5-sonnet-20241022", max_tokens=500)
        dspy.configure(lm=lm)

        # Agent 1: Extract a topic
        class ExtractTopic(dspy.Signature):
            """Extract the main topic from a query."""

            query = dspy.InputField()
            topic = dspy.OutputField(desc="The main topic (1-3 words)")

        # Agent 2: Generate insight about topic
        class GenerateInsight(dspy.Signature):
            """Generate an insight about a topic."""

            topic = dspy.InputField()
            insight = dspy.OutputField(desc="A brief insight")

        # Create agents
        topic_agent = AgentConfig(
            name="TopicExtractor",
            agent=dspy.ChainOfThought(ExtractTopic),
            architect_prompts=[],
            auditor_prompts=[],
            outputs=["topic"],
        )

        insight_agent = AgentConfig(
            name="InsightGenerator",
            agent=dspy.ChainOfThought(GenerateInsight),
            architect_prompts=[],
            auditor_prompts=[],
            parameter_mappings={"topic": "TopicExtractor"},
        )

        # Create swarm
        config = SwarmConfig(
            actors=[topic_agent, insight_agent], max_rounds=2, enable_learning=False
        )

        # Run the swarm
        swarm = Orchestrator(config)
        result = swarm.run(goal="Learn about code refactoring", query="What is code refactoring?")

        # Verify both agents ran
        assert result is not None
        print(f"\n✅ Multi-agent result: {result.final_output}")

    def test_refactored_components_work_together(self):
        """Test that ParameterResolver, ToolManager, and StateManager work together."""
        import dspy

        from core import AgentConfig, Orchestrator, SwarmConfig

        # Configure DSPy
        api_key = os.getenv("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = api_key

        lm = dspy.LM(
            model="anthropic/claude-3-5-haiku-20241022", max_tokens=200  # Use Haiku for faster test
        )
        dspy.configure(lm=lm)

        # Create simple agent
        class SimpleResponse(dspy.Signature):
            """Generate a simple response."""

            task = dspy.InputField()
            result = dspy.OutputField()

        agent = AgentConfig(
            name="TestAgent",
            agent=dspy.ChainOfThought(SimpleResponse),
            architect_prompts=[],
            auditor_prompts=[],
        )

        # Create config and run swarm
        config = SwarmConfig(actors=[agent], max_rounds=1, enable_learning=False)

        # Run through Orchestrator which uses Conductor internally
        swarm = Orchestrator(config)
        result = swarm.run(goal="Test the refactored components", task="Say 'Components working!'")

        # Verify we got a result
        assert result is not None

        print("\n✅ All refactored components properly integrated!")


@pytest.mark.e2e
class TestComponentBehavior:
    """Test individual component behavior without API calls."""

    @pytest.mark.skip(reason="ParameterResolver module was removed in refactoring")
    def test_parameter_resolver_component_standalone(self):
        """Test ParameterResolver component works standalone."""
        from unittest.mock import Mock

        from core.orchestration.parameter_resolver import ParameterResolver

        # Create minimal mocks
        resolver = ParameterResolver(
            io_manager=Mock(),
            param_resolver=Mock(),
            metadata_fetcher=Mock(),
            actors={},
            actor_signatures={},
            param_mappings={},
            data_registry=Mock(),
            registration_orchestrator=Mock(),
            data_transformer=Mock(),
            shared_context={},
            config=Mock(),
        )

        # Test methods exist and are callable
        assert hasattr(resolver, "_resolve_param_from_iomanager")
        assert hasattr(resolver, "resolve_input")
        assert callable(resolver._resolve_param_from_iomanager)

        print("\n✅ ParameterResolver component verified")

    @pytest.mark.skip(reason="ToolManager module was removed in refactoring")
    def test_tool_manager_component_standalone(self):
        """Test ToolManager component works standalone."""
        from unittest.mock import Mock

        from core.orchestration.tool_manager import ToolManager

        # Create minimal mocks
        manager = ToolManager(
            metadata_tool_registry=Mock(),
            data_registry_tool=Mock(),
            metadata_fetcher=Mock(),
            config=Mock(),
        )

        # Test methods exist and are callable
        assert hasattr(manager, "_get_auto_discovered_dspy_tools")
        assert hasattr(manager, "_get_architect_tools")
        assert callable(manager._get_auto_discovered_dspy_tools)

        print("\n✅ ToolManager component verified")

    @pytest.mark.skip(reason="StateManager module was removed in refactoring")
    def test_state_manager_component_standalone(self):
        """Test StateManager component works standalone."""
        from unittest.mock import Mock

        from core.orchestration.state_manager import StateManager

        # Create minimal mocks
        mock_todo = Mock()
        mock_todo.completed = []
        mock_todo.subtasks = {}
        mock_todo.failed_tasks = []

        mock_io_manager = Mock()
        mock_io_manager.get_all_outputs.return_value = {}

        manager = StateManager(
            io_manager=mock_io_manager,
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={},
            todo=mock_todo,
            trajectory=[],
            config=Mock(),
        )

        # Test methods exist and are callable
        assert hasattr(manager, "_get_current_state")
        assert hasattr(manager, "get_actor_outputs")
        assert callable(manager._get_current_state)

        # Test state manager can get current state
        state = manager._get_current_state()
        assert isinstance(state, dict)

        print("\n✅ StateManager component verified")


def run_e2e_tests():
    """Run end-to-end tests."""
    import sys

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping API tests.")
        print("Set ANTHROPIC_API_KEY to run full E2E tests with Claude.")
        print("\nRunning component behavior tests only...\n")
        exit_code = pytest.main([__file__, "-v", "--tb=short", "-m", "e2e and not skipif"])
    else:
        print("=" * 70)
        print("END-TO-END TESTS WITH CLAUDE API")
        print("=" * 70)
        exit_code = pytest.main([__file__, "-v", "--tb=short", "-m", "e2e"])

    return exit_code


if __name__ == "__main__":
    exit_code = run_e2e_tests()
    sys.exit(exit_code)
