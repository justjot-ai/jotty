"""
Integration test for StateActionManager (Phase 3.4).

Tests:
- Current state extraction
- Available actions enumeration
- State representation for Q-learning
- Statistics tracking
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_root = Path(__file__).parent
sys.path.insert(0, str(jotty_root))

from core.orchestration.managers.state_action_manager import StateActionManager
from core.foundation.data_structures import JottyConfig, TaskStatus


class MockTodo:
    """Mock TODO manager for testing."""
    def __init__(self):
        self.completed = ["task1", "task2"]
        self.failed_tasks = ["task3"]
        self.root_task = "Complete the project"
        self.subtasks = {
            "task4": type('obj', (object,), {'status': TaskStatus.PENDING})(),
            "task5": type('obj', (object,), {'status': TaskStatus.PENDING})(),
        }


class MockIOManager:
    """Mock IO manager for testing."""
    def get_all_outputs(self):
        return {
            "Fetcher": type('obj', (object,), {
                'output_fields': {"data": [1, 2, 3], "status": "success"}
            })(),
            "Processor": type('obj', (object,), {
                'output_fields': {"result": "processed"}
            })(),
        }


class MockContextGuard:
    """Mock context guard for testing."""
    def __init__(self):
        self.buffers = {
            "CRITICAL": [("ROOT_GOAL", "Test project goal", 100)],
            "HIGH": [],
        }


class MockActorConfig:
    """Mock actor config for testing."""
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled


def test_state_extraction_basic():
    """Test basic state extraction."""
    print("\n" + "="*70)
    print("TEST 1: Basic State Extraction")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock data
    todo = MockTodo()
    trajectory = [
        {"actor": "Fetcher", "passed": True, "tool_calls": []},
        {"actor": "Processor", "passed": False, "tool_calls": []},
    ]

    # Get state
    state = manager.get_current_state(
        todo=todo,
        trajectory=trajectory
    )

    print(f"✅ State extracted: {list(state.keys())}")
    assert 'todo' in state
    assert 'trajectory_length' in state
    assert 'recent_outcomes' in state

    # Check TODO stats
    assert state['todo']['completed'] == 2
    assert state['todo']['pending'] == 2
    assert state['todo']['failed'] == 1

    # Check trajectory stats
    assert state['trajectory_length'] == 2
    assert state['recent_outcomes'] == [True, False]

    print("✅ TEST PASSED: Basic state extraction works")


def test_state_extraction_with_context():
    """Test state extraction with shared context."""
    print("\n" + "="*70)
    print("TEST 2: State Extraction with Context")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock data
    todo = MockTodo()
    trajectory = []
    shared_context = {
        "query": "Get sales data for last month",
        "table_names": ["sales", "products"],
        "filters": {"region": "US"},
        "resolved_terms": {"sales_amount": "total_revenue"}
    }

    # Get state
    state = manager.get_current_state(
        todo=todo,
        trajectory=trajectory,
        shared_context=shared_context
    )

    print(f"✅ State with context: {list(state.keys())}")
    assert 'query' in state
    assert state['query'] == "Get sales data for last month"
    assert 'tables' in state
    assert state['tables'] == ["sales", "products"]
    assert 'filters' in state
    assert state['filters'] == {"region": "US"}
    assert 'resolved_terms' in state
    assert "sales_amount" in state['resolved_terms']

    print("✅ TEST PASSED: State extraction with context works")


def test_state_extraction_with_errors():
    """Test state extraction with error patterns."""
    print("\n" + "="*70)
    print("TEST 3: State Extraction with Error Patterns")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock data with errors
    todo = MockTodo()
    trajectory = [
        {
            "actor": "Fetcher",
            "passed": False,
            "error": "COLUMN_NOT_FOUND: Column 'date' cannot be resolved"
        },
        {
            "actor": "Fetcher",
            "passed": False,
            "error": "COLUMN_NOT_FOUND: Column 'timestamp' cannot be resolved"
        },
        {
            "actor": "Fetcher",
            "passed": True,
            "tool_calls": [{
                "success": True,
                "query": "SELECT * FROM table WHERE dl_last_updated > '2024-01-01'"
            }]
        },
    ]

    # Get state
    state = manager.get_current_state(
        todo=todo,
        trajectory=trajectory
    )

    print(f"✅ State with errors: {list(state.keys())}")
    assert 'errors' in state
    assert len(state['errors']) == 2
    assert state['errors'][0]['type'] == 'COLUMN_NOT_FOUND'
    assert state['errors'][0]['column'] == 'date'

    assert 'columns_tried' in state
    assert 'date' in state['columns_tried']
    assert 'timestamp' in state['columns_tried']

    assert 'working_column' in state
    assert state['working_column'] == 'dl_last_updated'

    assert 'error_resolution' in state
    assert 'dl_last_updated' in state['error_resolution']

    print("✅ TEST PASSED: Error pattern extraction works")


def test_state_extraction_with_tools():
    """Test state extraction with tool usage patterns."""
    print("\n" + "="*70)
    print("TEST 4: State Extraction with Tool Usage")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock data with tool calls
    todo = MockTodo()
    trajectory = [
        {
            "actor": "Fetcher",
            "passed": True,
            "tool_calls": [
                {"tool": "databricks_query", "success": True},
                {"tool": "s3_fetch", "success": True}
            ]
        },
        {
            "actor": "Processor",
            "passed": False,
            "tool_calls": [
                {"tool": "transform_data", "success": False}
            ]
        },
    ]

    # Get state
    state = manager.get_current_state(
        todo=todo,
        trajectory=trajectory
    )

    print(f"✅ State with tools: {list(state.keys())}")
    assert 'tool_calls' in state
    assert len(state['tool_calls']) == 3

    assert 'successful_tools' in state
    assert 'databricks_query' in state['successful_tools']
    assert 's3_fetch' in state['successful_tools']

    assert 'failed_tools' in state
    assert 'transform_data' in state['failed_tools']

    print("✅ TEST PASSED: Tool usage pattern extraction works")


def test_state_extraction_with_io_manager():
    """Test state extraction with IO manager outputs."""
    print("\n" + "="*70)
    print("TEST 5: State Extraction with IO Manager")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock data
    todo = MockTodo()
    trajectory = []
    io_manager = MockIOManager()

    # Get state
    state = manager.get_current_state(
        todo=todo,
        trajectory=trajectory,
        io_manager=io_manager
    )

    print(f"✅ State with IO manager: {list(state.keys())}")
    assert 'actor_outputs' in state
    assert 'Fetcher' in state['actor_outputs']
    assert 'data' in state['actor_outputs']['Fetcher']
    assert 'status' in state['actor_outputs']['Fetcher']

    assert 'Processor' in state['actor_outputs']
    assert 'result' in state['actor_outputs']['Processor']

    print("✅ TEST PASSED: IO manager output extraction works")


def test_state_extraction_with_context_guard():
    """Test state extraction with context guard."""
    print("\n" + "="*70)
    print("TEST 6: State Extraction with Context Guard")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock data
    todo = MockTodo()
    trajectory = []
    context_guard = MockContextGuard()

    # Get state
    state = manager.get_current_state(
        todo=todo,
        trajectory=trajectory,
        context_guard=context_guard
    )

    print(f"✅ State with context guard: {list(state.keys())}")
    assert 'query' in state
    assert state['query'] == "Test project goal"

    print("✅ TEST PASSED: Context guard extraction works")


def test_available_actions():
    """Test available actions enumeration."""
    print("\n" + "="*70)
    print("TEST 7: Available Actions Enumeration")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Create mock actors
    actors = {
        "Fetcher": MockActorConfig("Fetcher", enabled=True),
        "Processor": MockActorConfig("Processor", enabled=True),
        "Reporter": MockActorConfig("Reporter", enabled=False),
    }

    # Get available actions
    actions = manager.get_available_actions(actors)

    print(f"✅ Available actions: {actions}")
    assert len(actions) == 3

    # Check action structure
    assert actions[0]['actor'] == "Fetcher"
    assert actions[0]['action'] == "execute"
    assert actions[0]['enabled'] is True

    assert actions[2]['actor'] == "Reporter"
    assert actions[2]['enabled'] is False

    print("✅ TEST PASSED: Available actions enumeration works")


def test_statistics_tracking():
    """Test statistics tracking."""
    print("\n" + "="*70)
    print("TEST 8: Statistics Tracking")
    print("="*70)

    config = JottyConfig()
    manager = StateActionManager(config)

    # Get stats
    stats = manager.get_stats()
    print(f"📊 Stats: {stats}")
    assert stats["manager_initialized"] is True

    # Reset stats
    manager.reset_stats()
    print(f"📊 Stats after reset: {stats}")

    print("✅ TEST PASSED: Statistics tracking works")


def run_all_tests():
    """Run all state-action manager tests."""
    print("\n" + "🧪 "*35)
    print("STATE-ACTION MANAGER INTEGRATION TESTS (Phase 3.4)")
    print("🧪 "*35)

    try:
        test_state_extraction_basic()
        test_state_extraction_with_context()
        test_state_extraction_with_errors()
        test_state_extraction_with_tools()
        test_state_extraction_with_io_manager()
        test_state_extraction_with_context_guard()
        test_available_actions()
        test_statistics_tracking()

        print("\n" + "✅ "*35)
        print("ALL STATE-ACTION MANAGER TESTS PASSED!")
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
