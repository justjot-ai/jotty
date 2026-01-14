"""
Test StateManager Component in Isolation
========================================

Tests the extracted StateManager component independently
before integrating it into Conductor.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestStateManagerIsolation:
    """Test StateManager works in isolation."""

    def test_can_import_state_manager(self):
        """Test StateManager can be imported."""
        from core.orchestration.state_manager import StateManager
        assert StateManager is not None

    def test_can_create_state_manager_instance(self):
        """Test StateManager can be instantiated with mocks."""
        from core.orchestration.state_manager import StateManager

        # Create mock dependencies
        mock_io_manager = Mock()
        mock_data_registry = Mock()
        mock_metadata_provider = Mock()
        mock_context_guard = Mock()
        mock_shared_context = {}
        mock_todo = Mock()
        mock_todo.completed = []
        mock_todo.subtasks = {}
        mock_todo.failed_tasks = []
        mock_trajectory = []
        mock_config = Mock()

        # Create instance
        manager = StateManager(
            io_manager=mock_io_manager,
            data_registry=mock_data_registry,
            metadata_provider=mock_metadata_provider,
            context_guard=mock_context_guard,
            shared_context=mock_shared_context,
            todo=mock_todo,
            trajectory=mock_trajectory,
            config=mock_config
        )

        assert manager is not None
        assert manager.io_manager == mock_io_manager
        assert manager.data_registry == mock_data_registry
        assert manager.config == mock_config

    def test_get_current_state_basic(self):
        """Test _get_current_state returns state dict."""
        from core.orchestration.state_manager import StateManager

        # Create minimal mocks
        mock_todo = Mock()
        mock_todo.completed = []
        mock_todo.subtasks = {}
        mock_todo.failed_tasks = []
        mock_todo.root_task = "Test task"

        # Mock io_manager to return empty dict for get_all_outputs
        mock_io_manager = Mock()
        mock_io_manager.get_all_outputs.return_value = {}

        manager = StateManager(
            io_manager=mock_io_manager,
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={"query": "test query"},
            todo=mock_todo,
            trajectory=[],
            config=Mock()
        )

        state = manager._get_current_state()

        assert isinstance(state, dict)
        assert 'todo' in state
        assert 'trajectory_length' in state

    def test_get_available_actions_returns_list(self):
        """Test _get_available_actions returns action list."""
        from core.orchestration.state_manager import StateManager

        # Create manager with mock actors
        manager = StateManager(
            io_manager=Mock(),
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={},
            todo=Mock(completed=[], subtasks={}, failed_tasks=[]),
            trajectory=[],
            config=Mock(),
            actors={'actor1': Mock(), 'actor2': Mock()}
        )

        actions = manager._get_available_actions()

        assert isinstance(actions, list)
        assert len(actions) == 2

    def test_detect_output_type(self):
        """Test _detect_output_type identifies different types."""
        from core.orchestration.state_manager import StateManager

        manager = StateManager(
            io_manager=Mock(),
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={},
            todo=Mock(completed=[], subtasks={}, failed_tasks=[]),
            trajectory=[],
            config=Mock()
        )

        # Test string output
        assert manager._detect_output_type("test") == "text"

        # Test dict output
        assert manager._detect_output_type({"key": "value"}) == "json"

    def test_get_actor_outputs_returns_dict(self):
        """Test get_actor_outputs returns outputs dictionary."""
        from core.orchestration.state_manager import StateManager

        # Create trajectory with actor outputs
        trajectory = [
            {'actor': 'actor1', 'actor_output': 'output1'},
            {'actor': 'actor2', 'actor_output': 'output2'}
        ]

        manager = StateManager(
            io_manager=Mock(),
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={},
            todo=Mock(completed=[], subtasks={}, failed_tasks=[]),
            trajectory=trajectory,
            config=Mock()
        )

        outputs = manager.get_actor_outputs()

        assert isinstance(outputs, dict)
        assert outputs['actor1'] == 'output1'
        assert outputs['actor2'] == 'output2'

    def test_get_output_from_actor_finds_value(self):
        """Test get_output_from_actor retrieves specific actor output."""
        from core.orchestration.state_manager import StateManager

        # Create trajectory with actor outputs
        trajectory = [
            {'actor': 'actor1', 'actor_output': {'field1': 'value1', 'field2': 'value2'}},
            {'actor': 'actor2', 'actor_output': 'output2'}
        ]

        manager = StateManager(
            io_manager=Mock(),
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={},
            todo=Mock(completed=[], subtasks={}, failed_tasks=[]),
            trajectory=trajectory,
            config=Mock()
        )

        # Test getting full output
        output = manager.get_output_from_actor('actor1')
        assert output == {'field1': 'value1', 'field2': 'value2'}

        # Test getting specific field
        field_value = manager.get_output_from_actor('actor1', 'field1')
        assert field_value == 'value1'

    def test_generate_preview_for_string(self):
        """Test _generate_preview creates preview for string output."""
        from core.orchestration.state_manager import StateManager

        manager = StateManager(
            io_manager=Mock(),
            data_registry=Mock(),
            metadata_provider=Mock(),
            context_guard=None,
            shared_context={},
            todo=Mock(completed=[], subtasks={}, failed_tasks=[]),
            trajectory=[],
            config=Mock()
        )

        preview = manager._generate_preview("This is a test output")

        assert isinstance(preview, str)
        assert len(preview) > 0


@pytest.mark.unit
class TestStateManagerMethods:
    """Test individual StateManager methods."""

    def test_has_all_expected_methods(self):
        """Test StateManager has all expected methods."""
        from core.orchestration.state_manager import StateManager

        # Check all methods exist
        assert hasattr(StateManager, '_get_current_state')
        assert hasattr(StateManager, '_get_available_actions')
        assert hasattr(StateManager, '_introspect_actor_signature')
        assert hasattr(StateManager, '_detect_output_type')
        assert hasattr(StateManager, '_extract_schema')
        assert hasattr(StateManager, '_generate_preview')
        assert hasattr(StateManager, '_generate_tags')
        assert hasattr(StateManager, '_register_output_in_registry')
        assert hasattr(StateManager, '_register_output_in_registry_fallback')
        assert hasattr(StateManager, '_should_inject_registry_tool')
        assert hasattr(StateManager, 'get_actor_outputs')
        assert hasattr(StateManager, 'get_output_from_actor')


def run_state_manager_tests():
    """Run all StateManager tests."""
    print("="*70)
    print("STATE MANAGER ISOLATION TESTS")
    print("="*70)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_state_manager_tests()
    sys.exit(exit_code)
