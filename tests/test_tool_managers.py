"""
Integration test for ToolDiscoveryManager and ToolExecutionManager (Phase 2.5).

Tests:
- Tool auto-discovery
- Tool filtering for Planner/Reviewer
- Tool execution with caching
- Cache hit rate tracking
"""
import sys
from pathlib import Path
import json

# Add Jotty to path
jotty_root = Path(__file__).parent
sys.path.insert(0, str(jotty_root))

from core.orchestration.managers.tool_discovery_manager import ToolDiscoveryManager
from core.orchestration.managers.tool_execution_manager import ToolExecutionManager
from core.foundation.data_structures import JottyConfig


class MockTool:
    """Mock DSPy tool for testing."""
    def __init__(self, name, for_architect=False, for_auditor=False):
        self.name = name
        self._jotty_for_architect = for_architect
        self._jotty_for_auditor = for_auditor


class MockToolRegistry:
    """Mock MetadataToolRegistry for testing."""
    def __init__(self):
        self.tools = {
            "get_metadata": {"description": "Get metadata"},
            "validate_data": {"description": "Validate data"},
            "fetch_schema": {"description": "Fetch schema"}
        }
        self.call_count = {}

    def list_tools(self):
        return list(self.tools.keys())

    def get_tool_info(self, tool_name):
        return self.tools.get(tool_name, {})

    def call_tool(self, tool_name, **kwargs):
        """Mock tool call."""
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1
        return {
            "success": True,
            "tool": tool_name,
            "params": kwargs,
            "result": f"Result from {tool_name}"
        }


def test_tool_discovery_manager():
    """Test ToolDiscoveryManager basic functionality."""
    print("\n" + "="*70)
    print("TEST 1: ToolDiscoveryManager - Basic Functionality")
    print("="*70)

    config = JottyConfig()
    tool_registry = MockToolRegistry()
    manager = ToolDiscoveryManager(config, tool_registry)

    # Test list_tools
    tools = manager.list_tools()
    print(f"✅ Discovered {len(tools)} tools: {tools}")
    assert len(tools) == 3
    assert "get_metadata" in tools

    # Test get_tool_info
    info = manager.get_tool_info("get_metadata")
    print(f"✅ Tool info for 'get_metadata': {info}")
    assert info['description'] == "Get metadata"

    print("✅ TEST PASSED: ToolDiscoveryManager basic functionality works")


def test_tool_filtering():
    """Test tool filtering for Planner/Reviewer."""
    print("\n" + "="*70)
    print("TEST 2: ToolDiscoveryManager - Tool Filtering")
    print("="*70)

    config = JottyConfig()
    manager = ToolDiscoveryManager(config)

    # Create mock tools
    all_tools = [
        MockTool("tool1", for_architect=True, for_auditor=False),
        MockTool("tool2", for_architect=False, for_auditor=True),
        MockTool("tool3", for_architect=True, for_auditor=True),
        MockTool("tool4", for_architect=False, for_auditor=False)
    ]

    # Filter for Planner
    planner_tools = manager.filter_tools_for_planner(all_tools)
    print(f"✅ Planner tools: {[t.name for t in planner_tools]}")
    assert len(planner_tools) == 2  # tool1, tool3

    # Filter for Reviewer
    reviewer_tools = manager.filter_tools_for_reviewer(all_tools)
    print(f"✅ Reviewer tools: {[t.name for t in reviewer_tools]}")
    assert len(reviewer_tools) == 2  # tool2, tool3

    print("✅ TEST PASSED: Tool filtering works correctly")


def test_tool_execution_with_cache():
    """Test ToolExecutionManager with caching."""
    print("\n" + "="*70)
    print("TEST 3: ToolExecutionManager - Execution with Caching")
    print("="*70)

    config = JottyConfig()
    tool_registry = MockToolRegistry()
    shared_scratchpad = {}
    manager = ToolExecutionManager(config, tool_registry, shared_scratchpad)

    # First call - should execute
    result1 = manager.call_tool_with_cache("get_metadata", table="users")
    print(f"✅ First call result: {result1}")
    assert result1['success'] is True
    assert result1['tool'] == "get_metadata"

    # Second call with same params - should hit cache
    result2 = manager.call_tool_with_cache("get_metadata", table="users")
    print(f"✅ Second call result (cached): {result2}")
    assert result1 == result2

    # Check stats
    stats = manager.get_stats()
    print(f"📊 Execution stats: {stats}")
    assert stats['total_executions'] == 2
    assert stats['cache_hits'] == 1
    assert stats['cache_hit_rate'] == 0.5

    # Verify tool was only called once
    assert tool_registry.call_count['get_metadata'] == 1

    print("✅ TEST PASSED: Tool caching works correctly")


def test_cache_key_uniqueness():
    """Test that different parameters create different cache keys."""
    print("\n" + "="*70)
    print("TEST 4: ToolExecutionManager - Cache Key Uniqueness")
    print("="*70)

    config = JottyConfig()
    tool_registry = MockToolRegistry()
    shared_scratchpad = {}
    manager = ToolExecutionManager(config, tool_registry, shared_scratchpad)

    # Different params should not hit cache
    result1 = manager.call_tool_with_cache("get_metadata", table="users")
    result2 = manager.call_tool_with_cache("get_metadata", table="orders")

    print(f"✅ Call with table='users': {result1['params']}")
    print(f"✅ Call with table='orders': {result2['params']}")

    stats = manager.get_stats()
    print(f"📊 Stats: {stats}")
    assert stats['total_executions'] == 2
    assert stats['cache_hits'] == 0  # No cache hits

    # Verify tool was called twice
    assert tool_registry.call_count['get_metadata'] == 2

    print("✅ TEST PASSED: Different parameters create unique cache keys")


def test_enhanced_tool_description():
    """Test enhanced tool description building."""
    print("\n" + "="*70)
    print("TEST 5: ToolExecutionManager - Enhanced Tool Description")
    print("="*70)

    config = JottyConfig()
    manager = ToolExecutionManager(config)

    tool_info = {
        "description": "Fetch data from source",
        "signature": {
            "parameters": {
                "source": {"annotation": "str", "required": True},
                "limit": {"annotation": "int", "required": False}
            }
        }
    }

    desc = manager.build_enhanced_tool_description("fetch_data", tool_info)
    print(f"✅ Enhanced description:\n{desc}")

    assert "Fetch data from source" in desc
    assert "source" in desc
    assert "REQUIRED" in desc
    assert "optional" in desc
    assert "auto-resolved" in desc.lower()

    print("✅ TEST PASSED: Enhanced tool descriptions generated correctly")


def test_stats_reset():
    """Test statistics reset."""
    print("\n" + "="*70)
    print("TEST 6: ToolExecutionManager - Stats Reset")
    print("="*70)

    config = JottyConfig()
    tool_registry = MockToolRegistry()
    shared_scratchpad = {}
    manager = ToolExecutionManager(config, tool_registry, shared_scratchpad)

    # Make some calls
    manager.call_tool_with_cache("get_metadata", table="users")
    manager.call_tool_with_cache("get_metadata", table="users")  # Cache hit

    stats_before = manager.get_stats()
    print(f"📊 Stats before reset: {stats_before}")
    assert stats_before['total_executions'] == 2

    # Reset stats
    manager.reset_stats()
    stats_after = manager.get_stats()
    print(f"📊 Stats after reset: {stats_after}")
    assert stats_after['total_executions'] == 0
    assert stats_after['cache_hits'] == 0

    print("✅ TEST PASSED: Stats reset works correctly")


def run_all_tests():
    """Run all tool manager tests."""
    print("\n" + "🧪 "*35)
    print("TOOL MANAGERS INTEGRATION TESTS (Phase 2.5)")
    print("🧪 "*35)

    try:
        test_tool_discovery_manager()
        test_tool_filtering()
        test_tool_execution_with_cache()
        test_cache_key_uniqueness()
        test_enhanced_tool_description()
        test_stats_reset()

        print("\n" + "✅ "*35)
        print("ALL TOOL MANAGER TESTS PASSED!")
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
