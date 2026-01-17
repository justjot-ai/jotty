"""
Integration test for AgentLifecycleManager (Phase 3.3).

Tests:
- Agent wrapping decisions
- Annotation loading
- Tool filtering for agents
- Wrapped agent tracking
- Statistics tracking
- Backward compatibility with deprecated ActorLifecycleManager
"""
import sys
from pathlib import Path
import json
import tempfile
import warnings

# Add Jotty to path
jotty_root = Path(__file__).parent
sys.path.insert(0, str(jotty_root))

from core.orchestration.managers.agent_lifecycle_manager import (
    AgentLifecycleManager,
    ActorLifecycleManager  # Deprecated
)
from core.foundation.data_structures import JottyConfig


class MockAgentConfig:
    """Mock agent config for testing."""
    def __init__(self, name, architect_prompts=None, auditor_prompts=None,
                 architect_tools=None, auditor_tools=None):
        self.name = name
        self.architect_prompts = architect_prompts or []
        self.auditor_prompts = auditor_prompts or []
        self.architect_tools = architect_tools
        self.auditor_tools = auditor_tools
        self.agent = f"MockAgent({name})"


class MockToolDiscoveryManager:
    """Mock tool discovery manager for testing."""
    def filter_tools_for_planner(self, all_tools):
        return [t for t in all_tools if "planner" in t.lower()]

    def filter_tools_for_reviewer(self, all_tools):
        return [t for t in all_tools if "reviewer" in t.lower()]


def test_agent_wrapping_decisions():
    """Test should_wrap_agent logic."""
    print("\n" + "="*70)
    print("TEST 1: Agent Wrapping Decisions")
    print("="*70)

    config = JottyConfig()
    manager = AgentLifecycleManager(config)

    # Test agent with validation prompts - should wrap
    agent1 = MockAgentConfig(
        "Agent1",
        architect_prompts=["prompt1", "prompt2"]
    )
    should_wrap = manager.should_wrap_agent(agent1)
    print(f"✅ Agent with architect_prompts should wrap: {should_wrap}")
    assert should_wrap is True

    # Test agent with auditor prompts - should wrap
    agent2 = MockAgentConfig(
        "Agent2",
        auditor_prompts=["prompt1"]
    )
    should_wrap = manager.should_wrap_agent(agent2)
    print(f"✅ Agent with auditor_prompts should wrap: {should_wrap}")
    assert should_wrap is True

    # Test agent with tools - should wrap
    agent3 = MockAgentConfig(
        "Agent3",
        architect_tools=["tool1", "tool2"]
    )
    should_wrap = manager.should_wrap_agent(agent3)
    print(f"✅ Agent with architect_tools should wrap: {should_wrap}")
    assert should_wrap is True

    # Test agent without validation or tools - should NOT wrap
    agent4 = MockAgentConfig("Agent4")
    should_wrap = manager.should_wrap_agent(agent4)
    print(f"✅ Agent without validation/tools should NOT wrap: {should_wrap}")
    assert should_wrap is False

    print("✅ TEST PASSED: Agent wrapping decisions work")


def test_annotation_loading():
    """Test annotation loading from JSON file."""
    print("\n" + "="*70)
    print("TEST 2: Annotation Loading")
    print("="*70)

    config = JottyConfig()
    manager = AgentLifecycleManager(config)

    # Create a temporary annotations file
    annotations_data = {
        "Agent1": {"type": "classifier", "confidence": 0.9},
        "Agent2": {"type": "extractor", "confidence": 0.85}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(annotations_data, f)
        temp_path = f.name

    try:
        # Test successful load
        annotations = manager.load_annotations(temp_path)
        print(f"✅ Loaded {len(annotations)} annotations")
        assert len(annotations) == 2
        assert "Agent1" in annotations
        assert annotations["Agent1"]["type"] == "classifier"

        # Test get_annotations_for_agent
        agent1_annotations = manager.get_annotations_for_agent("Agent1")
        print(f"✅ Agent1 annotations: {agent1_annotations}")
        assert agent1_annotations["confidence"] == 0.9

        # Test missing agent
        agent3_annotations = manager.get_annotations_for_agent("Agent3")
        print(f"✅ Missing agent returns empty dict: {agent3_annotations}")
        assert agent3_annotations == {}

    finally:
        # Clean up temp file
        Path(temp_path).unlink()

    # Test missing file
    annotations = manager.load_annotations("/nonexistent/path.json")
    print(f"✅ Missing file returns empty dict: {len(annotations)} annotations")
    assert annotations == {}

    # Test None path
    annotations = manager.load_annotations(None)
    print(f"✅ None path returns empty dict: {len(annotations)} annotations")
    assert annotations == {}

    print("✅ TEST PASSED: Annotation loading works")


def test_tool_filtering():
    """Test tool filtering for agents."""
    print("\n" + "="*70)
    print("TEST 3: Tool Filtering for Agents")
    print("="*70)

    config = JottyConfig()
    tool_discovery = MockToolDiscoveryManager()
    manager = AgentLifecycleManager(config, tool_discovery)

    all_tools = ["planner_tool1", "planner_tool2", "reviewer_tool1", "other_tool"]

    # Test architect role with no tools specified - should auto-filter
    agent1 = MockAgentConfig("Agent1", architect_tools=None)
    architect_tools = manager.filter_tools_for_agent(all_tools, agent1, "architect")
    print(f"✅ Auto-filtered architect tools: {architect_tools}")
    assert len(architect_tools) == 2
    assert "planner_tool1" in architect_tools

    # Test auditor role with no tools specified - should auto-filter
    agent2 = MockAgentConfig("Agent2", auditor_tools=None)
    auditor_tools = manager.filter_tools_for_agent(all_tools, agent2, "auditor")
    print(f"✅ Auto-filtered auditor tools: {auditor_tools}")
    assert len(auditor_tools) == 1
    assert "reviewer_tool1" in auditor_tools

    # Test with explicit tools - should use those
    agent3 = MockAgentConfig("Agent3", architect_tools=["custom_tool1", "custom_tool2"])
    explicit_tools = manager.filter_tools_for_agent(all_tools, agent3, "architect")
    print(f"✅ Explicit tools used: {explicit_tools}")
    assert len(explicit_tools) == 2
    assert "custom_tool1" in explicit_tools

    # Test unknown role - should return empty list
    unknown_tools = manager.filter_tools_for_agent(all_tools, agent1, "unknown")
    print(f"✅ Unknown role returns empty list: {unknown_tools}")
    assert unknown_tools == []

    print("✅ TEST PASSED: Tool filtering works")


def test_wrapped_agent_tracking():
    """Test tracking of wrapped agents."""
    print("\n" + "="*70)
    print("TEST 4: Wrapped Agent Tracking")
    print("="*70)

    config = JottyConfig()
    manager = AgentLifecycleManager(config)

    # Test marking agent as wrapped
    manager.mark_agent_wrapped("Agent1")
    print(f"✅ Marked Agent1 as wrapped")

    # Test checking if agent is wrapped
    is_wrapped = manager.is_agent_wrapped("Agent1")
    print(f"✅ Agent1 is wrapped: {is_wrapped}")
    assert is_wrapped is True

    # Test checking unwrapped agent
    is_wrapped = manager.is_agent_wrapped("Agent2")
    print(f"✅ Agent2 is NOT wrapped: {is_wrapped}")
    assert is_wrapped is False

    # Mark multiple agents
    manager.mark_agent_wrapped("Agent2")
    manager.mark_agent_wrapped("Agent3")
    print(f"✅ Marked Agent2 and Agent3 as wrapped")

    # Check stats
    stats = manager.get_stats()
    print(f"✅ Stats: {stats}")
    assert stats["total_wrapped_agents"] == 3
    assert "Agent1" in stats["wrapped_agents"]
    assert "Agent2" in stats["wrapped_agents"]
    assert "Agent3" in stats["wrapped_agents"]

    print("✅ TEST PASSED: Wrapped agent tracking works")


def test_statistics_tracking():
    """Test statistics tracking."""
    print("\n" + "="*70)
    print("TEST 5: Statistics Tracking")
    print("="*70)

    config = JottyConfig()
    tool_discovery = MockToolDiscoveryManager()
    manager = AgentLifecycleManager(config, tool_discovery)

    # Create temp annotations file
    annotations_data = {"Agent1": {"type": "test"}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(annotations_data, f)
        temp_path = f.name

    try:
        # Load annotations
        manager.load_annotations(temp_path)

        # Mark some agents as wrapped
        manager.mark_agent_wrapped("Agent1")
        manager.mark_agent_wrapped("Agent2")

        # Get stats
        stats = manager.get_stats()
        print(f"📊 Stats: {stats}")

        assert stats["total_wrapped_agents"] == 2
        assert len(stats["wrapped_agents"]) == 2
        assert stats["annotations_loaded"] == 1
        assert stats["has_tool_discovery"] is True

        # Reset stats
        manager.reset_stats()
        stats = manager.get_stats()
        print(f"📊 Stats after reset: {stats}")
        assert stats["total_wrapped_agents"] == 0
        assert len(stats["wrapped_agents"]) == 0

    finally:
        Path(temp_path).unlink()

    print("✅ TEST PASSED: Statistics tracking works")


def test_backward_compatibility():
    """Test backward compatibility with deprecated ActorLifecycleManager."""
    print("\n" + "="*70)
    print("TEST 6: Backward Compatibility (Deprecated ActorLifecycleManager)")
    print("="*70)

    config = JottyConfig()

    # Test that ActorLifecycleManager still works but raises deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create instance - should trigger deprecation warning
        manager = ActorLifecycleManager(config)

        # Check deprecation warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ActorLifecycleManager is deprecated" in str(w[0].message)
        print(f"✅ Deprecation warning raised: {w[0].message}")

    # Test deprecated methods still work but raise warnings
    agent = MockAgentConfig("TestAgent", architect_prompts=["prompt1"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test should_wrap_actor (deprecated)
        result = manager.should_wrap_actor(agent)
        assert result is True
        assert any("should_wrap_actor() is deprecated" in str(warning.message) for warning in w)
        print(f"✅ should_wrap_actor() works with deprecation warning")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test mark_actor_wrapped (deprecated)
        manager.mark_actor_wrapped("TestAgent")
        assert manager.is_agent_wrapped("TestAgent")  # New method works
        assert any("mark_actor_wrapped() is deprecated" in str(warning.message) for warning in w)
        print(f"✅ mark_actor_wrapped() works with deprecation warning")

    print("✅ TEST PASSED: Backward compatibility maintained")


def run_all_tests():
    """Run all agent lifecycle manager tests."""
    print("\n" + "🧪 "*35)
    print("AGENT LIFECYCLE MANAGER INTEGRATION TESTS (Phase 3.3)")
    print("🧪 "*35)

    try:
        test_agent_wrapping_decisions()
        test_annotation_loading()
        test_tool_filtering()
        test_wrapped_agent_tracking()
        test_statistics_tracking()
        test_backward_compatibility()

        print("\n" + "✅ "*35)
        print("ALL AGENT LIFECYCLE MANAGER TESTS PASSED!")
        print("✅ "*35)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
