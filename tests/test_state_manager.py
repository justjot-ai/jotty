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
@pytest.mark.skip(reason="core.orchestration.state_manager module was removed; StateManager no longer exists")
class TestStateManagerIsolation:
    """Test StateManager works in isolation.

    SKIPPED: The StateManager class was removed from core.orchestration.
    Its functionality was replaced by SwarmStateManager in
    core.orchestration.swarm_state_manager (tested below).
    """

    def test_can_import_state_manager(self):
        pass

    def test_can_create_state_manager_instance(self):
        pass

    def test_get_current_state_basic(self):
        pass

    def test_get_available_actions_returns_list(self):
        pass

    def test_detect_output_type(self):
        pass

    def test_get_actor_outputs_returns_dict(self):
        pass

    def test_get_output_from_actor_finds_value(self):
        pass

    def test_generate_preview_for_string(self):
        pass


@pytest.mark.unit
@pytest.mark.skip(reason="core.orchestration.state_manager module was removed; StateManager no longer exists")
class TestStateManagerMethods:
    """Test individual StateManager methods.

    SKIPPED: The StateManager class was removed from core.orchestration.
    Its functionality was replaced by SwarmStateManager.
    """

    def test_has_all_expected_methods(self):
        pass


import json
import tempfile

# =============================================================================
# Import SwarmStateManager classes with try/except guard
# =============================================================================

try:
    from Jotty.core.intelligence.orchestration.swarm_state_manager import (
        AgentStateTracker,
        SwarmStateManager,
    )
    SWARM_STATE_MANAGER_AVAILABLE = True
except ImportError:
    SWARM_STATE_MANAGER_AVAILABLE = False


def _make_mock_task_board(completed=None, pending=None, failed=None, root_task=None):
    """Create a mock SwarmTaskBoard for testing."""
    board = MagicMock()
    board.completed_tasks = completed or []
    board.failed_tasks = failed or []

    # Build subtasks dict with status mocks
    subtasks = {}
    for task_id in (completed or []):
        st = MagicMock()
        st.status.name = "COMPLETED"
        subtasks[task_id] = st
    for task_id in (pending or []):
        st = MagicMock()
        st.status.name = "PENDING"
        subtasks[task_id] = st
    for task_id in (failed or []):
        st = MagicMock()
        st.status.name = "FAILED"
        subtasks[task_id] = st
    board.subtasks = subtasks
    board.root_task = root_task or "Default root task"
    return board


def _make_swarm_state_manager(**kwargs):
    """Create a SwarmStateManager with sensible mock defaults."""
    defaults = {
        'swarm_task_board': _make_mock_task_board(),
        'swarm_memory': MagicMock(),
        'io_manager': None,
        'data_registry': None,
        'shared_context': None,
        'context_guard': None,
        'config': None,
        'agents': None,
        'agent_signatures': None,
    }
    defaults.update(kwargs)
    return SwarmStateManager(**defaults)


# =============================================================================
# TestAgentStateTrackerDeep
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not SWARM_STATE_MANAGER_AVAILABLE, reason="SwarmStateManager not importable")
class TestAgentStateTrackerDeep:
    """Deep tests for AgentStateTracker."""

    def test_initialization(self):
        """AgentStateTracker stores agent_name and initializes empty collections."""
        tracker = AgentStateTracker("test_agent")
        assert tracker.agent_name == "test_agent"
        assert tracker.outputs == []
        assert tracker.errors == []
        assert tracker.trajectory == []
        assert tracker.validation_results == []
        assert tracker.stats['total_executions'] == 0
        assert tracker.stats['successful_executions'] == 0
        assert tracker.stats['failed_executions'] == 0

    def test_initial_tool_usage_structure(self):
        """AgentStateTracker initializes tool_usage with successful and failed dicts."""
        tracker = AgentStateTracker("agent")
        assert 'successful' in tracker.tool_usage
        assert 'failed' in tracker.tool_usage
        assert tracker.tool_usage['successful'] == {}
        assert tracker.tool_usage['failed'] == {}

    def test_record_output_stores_entry(self):
        """record_output appends output with type and timestamp."""
        tracker = AgentStateTracker("agent")
        tracker.record_output({"result": "data"}, "json")

        assert len(tracker.outputs) == 1
        entry = tracker.outputs[0]
        assert entry['output'] == {"result": "data"}
        assert entry['type'] == "json"
        assert 'timestamp' in entry

    def test_record_output_increments_successful(self):
        """record_output increments total_executions and successful_executions."""
        tracker = AgentStateTracker("agent")
        tracker.record_output("ok")
        assert tracker.stats['total_executions'] == 1
        assert tracker.stats['successful_executions'] == 1
        assert tracker.stats['failed_executions'] == 0

    def test_record_output_infers_type_from_class(self):
        """record_output uses type(output).__name__ when output_type is None."""
        tracker = AgentStateTracker("agent")
        tracker.record_output([1, 2, 3])
        assert tracker.outputs[0]['type'] == "list"

    def test_record_error_stores_entry(self):
        """record_error appends error with type, context, and timestamp."""
        tracker = AgentStateTracker("agent")
        tracker.record_error("timeout", "TimeoutError", {"retry": 1})

        assert len(tracker.errors) == 1
        entry = tracker.errors[0]
        assert entry['error'] == "timeout"
        assert entry['type'] == "TimeoutError"
        assert entry['context'] == {"retry": 1}
        assert 'timestamp' in entry

    def test_record_error_increments_failed(self):
        """record_error increments total_executions and failed_executions."""
        tracker = AgentStateTracker("agent")
        tracker.record_error("fail")
        assert tracker.stats['total_executions'] == 1
        assert tracker.stats['failed_executions'] == 1
        assert tracker.stats['successful_executions'] == 0

    def test_record_error_default_type(self):
        """record_error defaults error_type to 'Unknown' when not provided."""
        tracker = AgentStateTracker("agent")
        tracker.record_error("something broke")
        assert tracker.errors[0]['type'] == "Unknown"

    def test_record_error_default_context(self):
        """record_error defaults context to empty dict when not provided."""
        tracker = AgentStateTracker("agent")
        tracker.record_error("err")
        assert tracker.errors[0]['context'] == {}

    def test_record_tool_call_successful(self):
        """record_tool_call tracks successful calls in tool_usage['successful']."""
        tracker = AgentStateTracker("agent")
        tracker.record_tool_call("web_search", True)
        assert tracker.tool_usage['successful']['web_search'] == 1
        assert tracker.stats['successful_tool_calls'] == 1
        assert tracker.stats['total_tool_calls'] == 1

    def test_record_tool_call_failed(self):
        """record_tool_call tracks failed calls in tool_usage['failed']."""
        tracker = AgentStateTracker("agent")
        tracker.record_tool_call("db_query", False)
        assert tracker.tool_usage['failed']['db_query'] == 1
        assert tracker.stats['failed_tool_calls'] == 1
        assert tracker.stats['total_tool_calls'] == 1

    def test_record_tool_call_increments_count(self):
        """record_tool_call increments count for repeated tool usage."""
        tracker = AgentStateTracker("agent")
        tracker.record_tool_call("calculator", True)
        tracker.record_tool_call("calculator", True)
        tracker.record_tool_call("calculator", False)
        assert tracker.tool_usage['successful']['calculator'] == 2
        assert tracker.tool_usage['failed']['calculator'] == 1
        assert tracker.stats['total_tool_calls'] == 3

    def test_record_trajectory_step(self):
        """record_trajectory_step adds timestamp and agent_name to step."""
        tracker = AgentStateTracker("my_agent")
        step = {"action": "search", "query": "AI trends"}
        tracker.record_trajectory_step(step)

        assert len(tracker.trajectory) == 1
        recorded = tracker.trajectory[0]
        assert recorded['agent'] == "my_agent"
        assert 'timestamp' in recorded
        assert recorded['action'] == "search"

    def test_record_validation(self):
        """record_validation stores type, passed, confidence, and feedback."""
        tracker = AgentStateTracker("agent")
        tracker.record_validation("architect", True, 0.95, "looks good")

        assert len(tracker.validation_results) == 1
        entry = tracker.validation_results[0]
        assert entry['type'] == "architect"
        assert entry['passed'] is True
        assert entry['confidence'] == 0.95
        assert entry['feedback'] == "looks good"
        assert 'timestamp' in entry

    def test_get_state_returns_comprehensive_dict(self):
        """get_state returns dict with stats, outputs, errors, tool_usage, etc."""
        tracker = AgentStateTracker("agent")
        tracker.record_output("data1")
        tracker.record_error("err1")
        tracker.record_tool_call("search", True)

        state = tracker.get_state()

        assert state['agent_name'] == "agent"
        assert 'stats' in state
        assert 'recent_outputs' in state
        assert 'recent_errors' in state
        assert 'tool_usage' in state
        assert 'recent_trajectory' in state
        assert 'recent_validation' in state
        assert 'success_rate' in state
        assert 'tool_success_rate' in state

    def test_get_state_success_rate_calculation(self):
        """get_state computes success_rate as successful/total executions."""
        tracker = AgentStateTracker("agent")
        tracker.record_output("ok")
        tracker.record_output("ok")
        tracker.record_error("fail")

        state = tracker.get_state()
        # 2 successful / 3 total
        assert abs(state['success_rate'] - 2 / 3) < 0.01

    def test_get_state_tool_success_rate_calculation(self):
        """get_state computes tool_success_rate as successful/total tool calls."""
        tracker = AgentStateTracker("agent")
        tracker.record_tool_call("t1", True)
        tracker.record_tool_call("t2", False)
        tracker.record_tool_call("t3", True)

        state = tracker.get_state()
        # 2 successful / 3 total
        assert abs(state['tool_success_rate'] - 2 / 3) < 0.01

    def test_get_state_zero_executions_success_rate(self):
        """get_state returns 0.0 success_rate when no executions recorded."""
        tracker = AgentStateTracker("agent")
        state = tracker.get_state()
        assert state['success_rate'] == 0.0
        assert state['tool_success_rate'] == 0.0

    def test_get_error_patterns_extracts_from_last_10(self):
        """get_error_patterns extracts patterns from last 10 errors."""
        tracker = AgentStateTracker("agent")
        for i in range(15):
            tracker.record_error(f"error_{i}", "TypeA")

        patterns = tracker.get_error_patterns()
        # Only last 10 errors processed
        assert len(patterns) == 10

    def test_get_error_patterns_includes_frequency(self):
        """get_error_patterns includes frequency count for each error type."""
        tracker = AgentStateTracker("agent")
        tracker.record_error("err1", "TypeA")
        tracker.record_error("err2", "TypeA")
        tracker.record_error("err3", "TypeB")

        patterns = tracker.get_error_patterns()
        type_a_patterns = [p for p in patterns if p['type'] == 'TypeA']
        assert len(type_a_patterns) > 0
        assert type_a_patterns[0]['frequency'] == 2

    def test_get_error_patterns_truncates_message(self):
        """get_error_patterns truncates message_pattern to 100 chars."""
        tracker = AgentStateTracker("agent")
        long_error = "x" * 200
        tracker.record_error(long_error, "LongError")

        patterns = tracker.get_error_patterns()
        assert len(patterns[0]['message_pattern']) == 100

    def test_get_successful_patterns_tool_usage(self):
        """get_successful_patterns returns tools used >= 2 times successfully."""
        tracker = AgentStateTracker("agent")
        tracker.record_tool_call("search", True)
        tracker.record_tool_call("search", True)
        tracker.record_tool_call("calc", True)  # only once

        patterns = tracker.get_successful_patterns()
        tool_patterns = [p for p in patterns if p['type'] == 'tool_usage']
        tool_names = [p['tool'] for p in tool_patterns]
        assert 'search' in tool_names
        assert 'calc' not in tool_names

    def test_get_successful_patterns_validation(self):
        """get_successful_patterns includes validation passes."""
        tracker = AgentStateTracker("agent")
        tracker.record_validation("architect", True, 0.9, "ok")
        tracker.record_validation("auditor", True, 0.8, "fine")

        patterns = tracker.get_successful_patterns()
        val_patterns = [p for p in patterns if p['type'] == 'validation']
        assert len(val_patterns) == 1
        assert val_patterns[0]['count'] == 2


# =============================================================================
# TestSwarmStateManagerDeep
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not SWARM_STATE_MANAGER_AVAILABLE, reason="SwarmStateManager not importable")
class TestSwarmStateManagerDeep:
    """Deep tests for SwarmStateManager."""

    def test_initialization_with_task_board_and_memory(self):
        """SwarmStateManager stores swarm_task_board and swarm_memory."""
        board = _make_mock_task_board()
        memory = MagicMock()
        manager = SwarmStateManager(swarm_task_board=board, swarm_memory=memory)

        assert manager.swarm_task_board is board
        assert manager.swarm_memory is memory
        assert manager.agent_trackers == {}
        assert manager.swarm_trajectory == []

    def test_initialization_defaults(self):
        """SwarmStateManager defaults optional params to empty/None."""
        manager = _make_swarm_state_manager()
        assert manager.shared_context == {}
        assert manager.context_guard is None
        assert manager.config is None
        assert manager.agents == {}
        assert manager.agent_signatures == {}

    def test_get_agent_tracker_creates_if_not_exists(self):
        """get_agent_tracker creates a new tracker if agent name is new."""
        manager = _make_swarm_state_manager()
        tracker = manager.get_agent_tracker("new_agent")

        assert isinstance(tracker, AgentStateTracker)
        assert tracker.agent_name == "new_agent"
        assert "new_agent" in manager.agent_trackers

    def test_get_agent_tracker_returns_existing(self):
        """get_agent_tracker returns the same tracker on repeat calls."""
        manager = _make_swarm_state_manager()
        tracker1 = manager.get_agent_tracker("agent_x")
        tracker2 = manager.get_agent_tracker("agent_x")
        assert tracker1 is tracker2

    def test_get_current_state_task_progress(self):
        """get_current_state includes task_progress with correct counts."""
        board = _make_mock_task_board(
            completed=["t1", "t2"],
            pending=["t3"],
            failed=["t4"],
        )
        manager = _make_swarm_state_manager(swarm_task_board=board)

        state = manager.get_current_state()

        assert state['task_progress']['completed'] == 2
        assert state['task_progress']['pending'] == 1
        assert state['task_progress']['failed'] == 1
        assert state['task_progress']['total'] == 4

    def test_get_current_state_query_from_shared_context(self):
        """get_current_state extracts query from shared_context."""
        manager = _make_swarm_state_manager(
            shared_context={"query": "analyze stock trends"}
        )

        state = manager.get_current_state()

        assert state['query'] == "analyze stock trends"
        assert state['goal'] == "analyze stock trends"

    def test_get_current_state_goal_from_shared_context(self):
        """get_current_state extracts goal if query is absent in shared_context."""
        manager = _make_swarm_state_manager(
            shared_context={"goal": "build a dashboard"}
        )

        state = manager.get_current_state()

        assert state['query'] == "build a dashboard"

    def test_get_current_state_query_from_task_board_root(self):
        """get_current_state falls back to task_board.root_task for query."""
        board = _make_mock_task_board(root_task="Root goal from board")
        manager = _make_swarm_state_manager(swarm_task_board=board, shared_context={})

        state = manager.get_current_state()

        assert state.get('query') == "Root goal from board"

    def test_get_current_state_error_patterns(self):
        """get_current_state extracts errors from trajectory (last 5)."""
        manager = _make_swarm_state_manager()
        for i in range(8):
            manager.swarm_trajectory.append({
                'error': f"Error {i}",
                'agent': f"agent_{i}",
            })

        state = manager.get_current_state()

        assert 'errors' in state
        assert len(state['errors']) == 5  # last 5

    def test_get_current_state_no_errors_when_trajectory_clean(self):
        """get_current_state omits errors key when trajectory has no errors."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory.append({'success': True, 'agent': 'a'})

        state = manager.get_current_state()

        assert 'errors' not in state

    def test_get_current_state_tool_usage_patterns(self):
        """get_current_state populates successful_tools and failed_tools."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory.append({
            'tool_calls': [
                {'tool': 'web_search', 'success': True},
                {'tool': 'db_query', 'success': False},
            ]
        })

        state = manager.get_current_state()

        assert 'successful_tools' in state
        assert 'web_search' in state['successful_tools']
        assert 'failed_tools' in state
        assert 'db_query' in state['failed_tools']

    def test_get_current_state_agent_states(self):
        """get_current_state includes agent_states from trackers."""
        manager = _make_swarm_state_manager()
        tracker = manager.get_agent_tracker("researcher")
        tracker.record_output("finding1")

        state = manager.get_current_state()

        assert 'agent_states' in state
        assert 'researcher' in state['agent_states']

    def test_get_current_state_trajectory_length(self):
        """get_current_state includes trajectory_length."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [{"a": 1}, {"b": 2}, {"c": 3}]

        state = manager.get_current_state()

        assert state['trajectory_length'] == 3

    def test_get_current_state_recent_outcomes(self):
        """get_current_state includes recent_outcomes from last 5 trajectory steps."""
        manager = _make_swarm_state_manager()
        for success in [True, False, True, True, False, True]:
            manager.swarm_trajectory.append({'success': success})

        state = manager.get_current_state()

        assert len(state['recent_outcomes']) == 5

    def test_get_current_state_attempts_and_success(self):
        """get_current_state includes attempts count and overall success flag."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [
            {'success': False},
            {'success': True},
        ]

        state = manager.get_current_state()

        assert state['attempts'] == 2
        assert state['success'] is True

    def test_get_current_state_no_success_when_all_fail(self):
        """get_current_state reports success=False when no steps succeeded."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [{'success': False}]

        state = manager.get_current_state()

        assert state['success'] is False

    def test_get_current_state_current_agent(self):
        """get_current_state sets current_agent from last trajectory step."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [
            {'agent': 'first_agent'},
            {'agent': 'last_agent'},
        ]

        state = manager.get_current_state()

        assert state['current_agent'] == 'last_agent'

    def test_get_current_state_validation_context(self):
        """get_current_state picks up architect_confidence and validation_passed."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [
            {'architect_confidence': 0.92, 'validation_passed': True},
        ]

        state = manager.get_current_state()

        assert state['architect_confidence'] == 0.92
        assert state['validation_passed'] is True

    def test_get_current_state_metadata_tables(self):
        """get_current_state extracts table_names from shared_context."""
        manager = _make_swarm_state_manager(
            shared_context={"query": "q", "table_names": ["orders", "users"]}
        )

        state = manager.get_current_state()

        assert state.get('tables') == ["orders", "users"]

    def test_get_current_state_metadata_filters(self):
        """get_current_state extracts filters from shared_context."""
        manager = _make_swarm_state_manager(
            shared_context={"query": "q", "filters": {"date": "2026-01"}}
        )

        state = manager.get_current_state()

        assert state.get('filters') == {"date": "2026-01"}

    def test_get_agent_state_returns_specific_agent(self):
        """get_agent_state returns the state dict for a specific agent."""
        manager = _make_swarm_state_manager()
        tracker = manager.get_agent_tracker("coder")
        tracker.record_output("code output")
        tracker.record_tool_call("compiler", True)

        agent_state = manager.get_agent_state("coder")

        assert agent_state['agent_name'] == "coder"
        assert agent_state['stats']['successful_executions'] == 1

    def test_get_agent_state_creates_tracker_if_missing(self):
        """get_agent_state creates a new tracker if agent not yet tracked."""
        manager = _make_swarm_state_manager()
        state = manager.get_agent_state("brand_new_agent")
        assert state['agent_name'] == "brand_new_agent"
        assert state['stats']['total_executions'] == 0

    def test_get_state_summary_human_readable(self):
        """get_state_summary returns a human-readable string."""
        board = _make_mock_task_board(
            completed=["t1"],
            pending=["t2", "t3"],
            failed=[],
        )
        manager = _make_swarm_state_manager(
            swarm_task_board=board,
            shared_context={"query": "Build a dashboard for stock prices"},
        )

        summary = manager.get_state_summary()

        assert isinstance(summary, str)
        assert "Swarm State Summary" in summary
        assert "1 completed" in summary
        assert "2 pending" in summary
        assert "Goal:" in summary

    def test_get_state_summary_includes_agents(self):
        """get_state_summary includes agent stats when trackers exist."""
        manager = _make_swarm_state_manager()
        tracker = manager.get_agent_tracker("researcher")
        tracker.record_output("r1")
        tracker.record_output("r2")

        summary = manager.get_state_summary()

        assert "researcher" in summary
        assert "2/2 successful" in summary

    def test_record_swarm_step_appends_to_trajectory(self):
        """record_swarm_step adds step with timestamp to swarm_trajectory."""
        manager = _make_swarm_state_manager()
        step = {"action": "search", "success": True}
        manager.record_swarm_step(step)

        assert len(manager.swarm_trajectory) == 1
        assert 'timestamp' in manager.swarm_trajectory[0]
        assert manager.swarm_trajectory[0]['action'] == "search"

    def test_record_swarm_step_propagates_to_agent_tracker(self):
        """record_swarm_step propagates to agent tracker when agent is specified."""
        manager = _make_swarm_state_manager()
        step = {"agent": "coder", "action": "compile", "success": True}
        manager.record_swarm_step(step)

        assert "coder" in manager.agent_trackers
        tracker = manager.agent_trackers["coder"]
        assert len(tracker.trajectory) == 1

    def test_record_swarm_step_no_agent_no_propagation(self):
        """record_swarm_step does not create tracker when no agent in step."""
        manager = _make_swarm_state_manager()
        step = {"action": "init", "success": True}
        manager.record_swarm_step(step)

        assert len(manager.agent_trackers) == 0

    def test_get_available_actions_returns_list(self):
        """get_available_actions returns action dicts for each configured agent."""
        agent_configs = {
            "researcher": MagicMock(enabled=True),
            "coder": MagicMock(enabled=False),
        }
        manager = _make_swarm_state_manager(agents=agent_configs)

        actions = manager.get_available_actions()

        assert len(actions) == 2
        names = [a['agent'] for a in actions]
        assert "researcher" in names
        assert "coder" in names

    def test_get_available_actions_includes_enabled_flag(self):
        """get_available_actions includes the enabled flag from agent config."""
        agent_configs = {
            "active_agent": MagicMock(enabled=True),
        }
        manager = _make_swarm_state_manager(agents=agent_configs)

        actions = manager.get_available_actions()

        assert actions[0]['enabled'] is True
        assert actions[0]['action'] == 'execute'

    def test_get_available_actions_empty_when_no_agents(self):
        """get_available_actions returns empty list when no agents configured."""
        manager = _make_swarm_state_manager(agents={})
        actions = manager.get_available_actions()
        assert actions == []

    def test_save_state_creates_json_file(self):
        """save_state writes state to a JSON file."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [{"agent": "a", "success": True}]
        tracker = manager.get_agent_tracker("test_agent")
        tracker.record_output("test output")

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "state.json"
            manager.save_state(file_path)

            assert file_path.exists()
            with open(file_path, 'r') as f:
                data = json.load(f)

            assert 'swarm_state' in data
            assert 'agent_states' in data
            assert 'timestamp' in data
            assert len(data['swarm_state']['trajectory']) == 1

    def test_load_state_restores_trajectory(self):
        """load_state restores swarm_trajectory from saved file."""
        manager = _make_swarm_state_manager()
        manager.swarm_trajectory = [{"agent": "a", "success": True}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "state.json"
            manager.save_state(file_path)

            # Create a fresh manager and load
            manager2 = _make_swarm_state_manager()
            assert manager2.swarm_trajectory == []
            manager2.load_state(file_path)
            assert len(manager2.swarm_trajectory) == 1

    def test_load_state_restores_agent_stats(self):
        """load_state restores agent tracker stats from saved file."""
        manager = _make_swarm_state_manager()
        tracker = manager.get_agent_tracker("coder")
        tracker.record_output("code1")
        tracker.record_output("code2")
        tracker.record_error("fail1")

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "state.json"
            manager.save_state(file_path)

            manager2 = _make_swarm_state_manager()
            manager2.load_state(file_path)

            loaded_tracker = manager2.get_agent_tracker("coder")
            assert loaded_tracker.stats['successful_executions'] == 2
            assert loaded_tracker.stats['failed_executions'] == 1

    def test_load_state_restores_tool_usage(self):
        """load_state restores swarm_tool_usage from saved file."""
        manager = _make_swarm_state_manager()
        manager.swarm_tool_usage = {
            'successful': {'search': 5},
            'failed': {'db_query': 2}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "state.json"
            manager.save_state(file_path)

            manager2 = _make_swarm_state_manager()
            manager2.load_state(file_path)

            assert manager2.swarm_tool_usage['successful']['search'] == 5
            assert manager2.swarm_tool_usage['failed']['db_query'] == 2

    def test_save_load_roundtrip_error_patterns(self):
        """save_state/load_state roundtrips swarm_error_patterns."""
        manager = _make_swarm_state_manager()
        manager.swarm_error_patterns = [{"type": "TimeoutError", "count": 3}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "state.json"
            manager.save_state(file_path)

            manager2 = _make_swarm_state_manager()
            manager2.load_state(file_path)

            assert manager2.swarm_error_patterns == [{"type": "TimeoutError", "count": 3}]

    def test_get_current_state_query_truncated_to_200_chars(self):
        """get_current_state truncates query to 200 characters."""
        long_query = "x" * 300
        manager = _make_swarm_state_manager(
            shared_context={"query": long_query}
        )

        state = manager.get_current_state()

        assert len(state['query']) == 200

    def test_get_current_state_resolved_terms(self):
        """get_current_state extracts resolved_terms keys from shared_context."""
        manager = _make_swarm_state_manager(
            shared_context={
                "query": "q",
                "resolved_terms": {"revenue": "total_sales", "profit": "net_income"}
            }
        )

        state = manager.get_current_state()

        assert 'resolved_terms' in state
        assert set(state['resolved_terms']) == {"revenue", "profit"}

    def test_multiple_agents_tracked_independently(self):
        """Multiple agents tracked independently with correct stats."""
        manager = _make_swarm_state_manager()

        tracker_a = manager.get_agent_tracker("agent_a")
        tracker_a.record_output("output_a")
        tracker_a.record_output("output_a2")

        tracker_b = manager.get_agent_tracker("agent_b")
        tracker_b.record_error("error_b")

        state_a = manager.get_agent_state("agent_a")
        state_b = manager.get_agent_state("agent_b")

        assert state_a['stats']['successful_executions'] == 2
        assert state_a['stats']['failed_executions'] == 0
        assert state_b['stats']['successful_executions'] == 0
        assert state_b['stats']['failed_executions'] == 1

    def test_swarm_trajectory_isolation(self):
        """SwarmStateManager swarm_trajectory is independent from agent trajectories."""
        manager = _make_swarm_state_manager()

        manager.swarm_trajectory.append({"global": "step1"})
        tracker = manager.get_agent_tracker("agent_x")
        tracker.record_trajectory_step({"local": "step1"})

        assert len(manager.swarm_trajectory) == 1
        assert manager.swarm_trajectory[0].get("global") == "step1"
        assert len(tracker.trajectory) == 1
        assert tracker.trajectory[0].get("local") == "step1"


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
