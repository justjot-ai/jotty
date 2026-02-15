"""
SwarmTaskBoard / Swarm Roadmap Unit Tests
==========================================

Comprehensive unit tests for all classes in core/orchestration/swarm_roadmap.py:

- TaskStatus enum
- TrajectoryStep dataclass
- SubtaskState dataclass
- SwarmTaskBoard (long-horizon task management)
- AgenticState (rich state representation)
- DecomposedQFunction (multi-objective Q-learning)
- ThoughtLevelCredit (reasoning step credit assignment)

All tests use mocks -- no real LLM calls, no API keys, runs offline and fast (<1s each).
"""

import json
import pytest
import hashlib
from datetime import datetime, timedelta
from dataclasses import fields
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Try importing DSPy -- tests skip if unavailable
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Try importing the modules under test
try:
    from Jotty.core.intelligence.orchestration.swarm_roadmap import (
        TaskStatus,
        TrajectoryStep,
        SubtaskState,
        SwarmTaskBoard,
        AgenticState,
        DecomposedQFunction,
        ThoughtLevelCredit,
        TodoItem,
    )
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="swarm_roadmap module not importable")


# =============================================================================
# HELPERS
# =============================================================================

def _make_trajectory_step(step_idx=0, action_type="thought", action_content="test",
                          observation="ok", reward=0.5, context_summary="ctx",
                          activated_memories=None):
    """Helper to create a TrajectoryStep with sensible defaults."""
    return TrajectoryStep(
        step_idx=step_idx,
        timestamp=datetime.now(),
        action_type=action_type,
        action_content=action_content,
        context_summary=context_summary,
        activated_memories=activated_memories or [],
        observation=observation,
        reward=reward,
    )


def _make_agentic_state(**kwargs):
    """Helper to create an AgenticState with sensible defaults."""
    defaults = {
        "state_id": "test-state",
        "agent_name": "test-agent",
        "episode_id": "ep-001",
        "task_description": "test task",
    }
    defaults.update(kwargs)
    return AgenticState(**defaults)


def _make_board_with_tasks():
    """Helper to create a SwarmTaskBoard with 3 sequential tasks."""
    board = SwarmTaskBoard(root_task="Build feature")
    board.add_task("t1", "Design", actor="designer")
    board.add_task("t2", "Implement", actor="developer", depends_on=["t1"])
    board.add_task("t3", "Test", actor="tester", depends_on=["t2"])
    return board


# =============================================================================
# TestTaskStatus
# =============================================================================

class TestTaskStatus:
    """Tests for the TaskStatus enum."""

    @pytest.mark.unit
    def test_pending_value(self):
        assert TaskStatus.PENDING.value == "pending"

    @pytest.mark.unit
    def test_in_progress_value(self):
        assert TaskStatus.IN_PROGRESS.value == "in_progress"

    @pytest.mark.unit
    def test_completed_value(self):
        assert TaskStatus.COMPLETED.value == "completed"

    @pytest.mark.unit
    def test_failed_value(self):
        assert TaskStatus.FAILED.value == "failed"

    @pytest.mark.unit
    def test_blocked_value(self):
        assert TaskStatus.BLOCKED.value == "blocked"

    @pytest.mark.unit
    def test_skipped_value(self):
        assert TaskStatus.SKIPPED.value == "skipped"

    @pytest.mark.unit
    def test_all_required_statuses_exist(self):
        names = {s.name for s in TaskStatus}
        for expected in ("PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "BLOCKED", "SKIPPED"):
            assert expected in names


# =============================================================================
# TestTrajectoryStep
# =============================================================================

class TestTrajectoryStep:
    """Tests for the TrajectoryStep dataclass."""

    @pytest.mark.unit
    def test_creation_with_all_fields(self):
        ts = datetime.now()
        step = TrajectoryStep(
            step_idx=0, timestamp=ts, action_type="thought",
            action_content="reason about X", context_summary="ctx",
            activated_memories=["m1"], observation="done", reward=0.8,
        )
        assert step.step_idx == 0
        assert step.action_type == "thought"
        assert step.action_content == "reason about X"
        assert step.observation == "done"
        assert step.reward == 0.8

    @pytest.mark.unit
    def test_default_predicted_outcome(self):
        step = _make_trajectory_step()
        assert step.predicted_outcome is None

    @pytest.mark.unit
    def test_default_prediction_confidence(self):
        step = _make_trajectory_step()
        assert step.prediction_confidence == 0.0

    @pytest.mark.unit
    def test_default_actual_divergence(self):
        step = _make_trajectory_step()
        assert step.actual_divergence == 0.0

    @pytest.mark.unit
    def test_activated_memories_list(self):
        step = _make_trajectory_step(activated_memories=["mem_a", "mem_b"])
        assert step.activated_memories == ["mem_a", "mem_b"]

    @pytest.mark.unit
    def test_context_summary_stored(self):
        step = _make_trajectory_step(context_summary="important context")
        assert step.context_summary == "important context"

    @pytest.mark.unit
    def test_negative_reward(self):
        step = _make_trajectory_step(reward=-1.0)
        assert step.reward == -1.0

    @pytest.mark.unit
    def test_various_action_types(self):
        for atype in ("thought", "tool_call", "decision", "output"):
            step = _make_trajectory_step(action_type=atype)
            assert step.action_type == atype


# =============================================================================
# TestSubtaskState
# =============================================================================

class TestSubtaskState:
    """Tests for the SubtaskState dataclass."""

    @pytest.mark.unit
    def test_creation_with_required_fields(self):
        st = SubtaskState(task_id="t1", description="Do something")
        assert st.task_id == "t1"
        assert st.description == "Do something"

    @pytest.mark.unit
    def test_default_status_pending(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.status == TaskStatus.PENDING

    @pytest.mark.unit
    def test_default_attempts_zero(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.attempts == 0

    @pytest.mark.unit
    def test_default_max_attempts_three(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.max_attempts == 3

    @pytest.mark.unit
    def test_default_progress_zero(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.progress == 0.0

    @pytest.mark.unit
    def test_default_priority_one(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.priority == 1.0

    @pytest.mark.unit
    def test_default_estimated_reward(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.estimated_reward == 0.5

    @pytest.mark.unit
    def test_default_confidence(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.confidence == 0.5

    @pytest.mark.unit
    def test_default_depends_on_empty(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.depends_on == []

    @pytest.mark.unit
    def test_default_blocks_empty(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.blocks == []

    @pytest.mark.unit
    def test_default_failure_reasons_empty(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.failure_reasons == []

    @pytest.mark.unit
    def test_can_start_no_dependencies(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.can_start(set()) is True

    @pytest.mark.unit
    def test_can_start_with_unmet_dependencies(self):
        st = SubtaskState(task_id="t2", description="x", depends_on=["t1"])
        assert st.can_start(set()) is False

    @pytest.mark.unit
    def test_can_start_with_all_met_dependencies(self):
        st = SubtaskState(task_id="t2", description="x", depends_on=["t1"])
        assert st.can_start({"t1"}) is True

    @pytest.mark.unit
    def test_can_start_partial_dependencies(self):
        st = SubtaskState(task_id="t3", description="x", depends_on=["t1", "t2"])
        assert st.can_start({"t1"}) is False

    @pytest.mark.unit
    def test_start_sets_in_progress(self):
        st = SubtaskState(task_id="t1", description="x")
        st.start()
        assert st.status == TaskStatus.IN_PROGRESS

    @pytest.mark.unit
    def test_start_sets_started_at(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.started_at is None
        st.start()
        assert st.started_at is not None
        assert isinstance(st.started_at, datetime)

    @pytest.mark.unit
    def test_start_increments_attempts(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.attempts == 0
        st.start()
        assert st.attempts == 1

    @pytest.mark.unit
    def test_complete_sets_completed(self):
        st = SubtaskState(task_id="t1", description="x")
        st.start()
        st.complete()
        assert st.status == TaskStatus.COMPLETED

    @pytest.mark.unit
    def test_complete_sets_completed_at(self):
        st = SubtaskState(task_id="t1", description="x")
        st.complete()
        assert st.completed_at is not None

    @pytest.mark.unit
    def test_complete_sets_progress_to_one(self):
        st = SubtaskState(task_id="t1", description="x")
        st.complete()
        assert st.progress == 1.0

    @pytest.mark.unit
    def test_complete_with_result_dict(self):
        st = SubtaskState(task_id="t1", description="x")
        result = {"output": "success", "score": 0.95}
        st.complete(result=result)
        assert st.result == result

    @pytest.mark.unit
    def test_fail_sets_error(self):
        st = SubtaskState(task_id="t1", description="x", max_attempts=1)
        st.start()  # attempts=1
        st.fail("something broke")
        assert st.error == "something broke"

    @pytest.mark.unit
    def test_fail_sets_failed_when_max_attempts_reached(self):
        st = SubtaskState(task_id="t1", description="x", max_attempts=1)
        st.start()  # attempts becomes 1, equals max_attempts
        st.fail("broke")
        assert st.status == TaskStatus.FAILED

    @pytest.mark.unit
    def test_fail_sets_pending_when_retries_remain(self):
        st = SubtaskState(task_id="t1", description="x", max_attempts=3)
        st.start()  # attempts=1
        st.fail("broke")
        # attempts=1 < max_attempts=3 => PENDING (retry)
        assert st.status == TaskStatus.PENDING

    @pytest.mark.unit
    def test_estimated_duration_default(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.estimated_duration == 60.0

    @pytest.mark.unit
    def test_predicted_next_task_default_none(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.predicted_next_task is None

    @pytest.mark.unit
    def test_predicted_duration_default_none(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.predicted_duration is None

    @pytest.mark.unit
    def test_predicted_reward_default_none(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.predicted_reward is None

    @pytest.mark.unit
    def test_intermediary_values_default_empty(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.intermediary_values == {}

    @pytest.mark.unit
    def test_actor_default_empty_string(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.actor == ""

    @pytest.mark.unit
    def test_result_default_none(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.result is None

    @pytest.mark.unit
    def test_error_default_none(self):
        st = SubtaskState(task_id="t1", description="x")
        assert st.error is None


# =============================================================================
# TestSwarmTaskBoard
# =============================================================================

class TestSwarmTaskBoard:
    """Tests for SwarmTaskBoard (long-horizon task management)."""

    @pytest.mark.unit
    def test_creation_with_defaults(self):
        board = SwarmTaskBoard()
        assert board.root_task == ""
        assert board.subtasks == {}
        assert board.execution_order == []
        assert board.current_task_id is None
        assert board.completed_tasks == set()
        assert board.failed_tasks == set()

    @pytest.mark.unit
    def test_creation_with_root_task(self):
        board = SwarmTaskBoard(root_task="Build AI system")
        assert board.root_task == "Build AI system"

    @pytest.mark.unit
    def test_todo_id_auto_generated(self):
        board = SwarmTaskBoard()
        assert board.todo_id != ""
        assert len(board.todo_id) == 32  # md5 hex digest

    @pytest.mark.unit
    def test_add_task_basic(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Design system")
        assert "t1" in board.subtasks
        assert board.subtasks["t1"].description == "Design system"

    @pytest.mark.unit
    def test_add_task_with_dependencies(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Design")
        board.add_task("t2", "Implement", depends_on=["t1"])
        assert board.subtasks["t2"].depends_on == ["t1"]

    @pytest.mark.unit
    def test_add_task_with_actor_and_priority(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Design", actor="architect", priority=2.0)
        assert board.subtasks["t1"].actor == "architect"
        assert board.subtasks["t1"].priority == 2.0

    @pytest.mark.unit
    def test_add_task_with_estimated_duration(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Quick task", estimated_duration=10.0)
        assert board.subtasks["t1"].estimated_duration == 10.0

    @pytest.mark.unit
    def test_add_task_with_max_attempts(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Critical", max_attempts=1)
        assert board.subtasks["t1"].max_attempts == 1

    @pytest.mark.unit
    def test_add_task_updates_execution_order(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "First")
        board.add_task("t2", "Second")
        assert board.execution_order == ["t1", "t2"]

    @pytest.mark.unit
    def test_add_task_does_not_duplicate_in_execution_order(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "First")
        board.add_task("t1", "First updated")
        assert board.execution_order.count("t1") == 1

    @pytest.mark.unit
    def test_add_task_updates_blocks(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Design")
        board.add_task("t2", "Implement", depends_on=["t1"])
        assert "t2" in board.subtasks["t1"].blocks

    @pytest.mark.unit
    def test_get_next_task_returns_first_available(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "First")
        board.add_task("t2", "Second")
        task = board.get_next_task()
        assert task is not None
        assert task.task_id == "t1"

    @pytest.mark.unit
    def test_get_next_task_respects_dependencies(self):
        board = _make_board_with_tasks()
        task = board.get_next_task()
        # Only t1 should be available (t2 depends on t1, t3 depends on t2)
        assert task.task_id == "t1"

    @pytest.mark.unit
    def test_get_next_task_returns_none_when_all_blocked(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A", depends_on=["t0"])  # t0 doesn't exist
        task = board.get_next_task()
        assert task is None

    @pytest.mark.unit
    def test_get_next_task_returns_none_when_all_completed(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Task")
        board.start_task("t1")
        board.complete_task("t1")
        task = board.get_next_task()
        assert task is None

    @pytest.mark.unit
    def test_get_next_task_with_q_predictor_exploit_mode(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Low Q", actor="low_agent")
        board.add_task("t2", "High Q", actor="high_agent")

        mock_predictor = Mock()
        # Returns (q_value, extra1, extra2)
        def predict_side_effect(state, action, goal):
            if action["actor"] == "high_agent":
                return (0.9, None, None)
            return (0.1, None, None)
        mock_predictor.predict_q_value = Mock(side_effect=predict_side_effect)

        mock_state = Mock()
        with patch("random.random", return_value=0.99):  # Exploit
            task = board.get_next_task(
                q_predictor=mock_predictor,
                current_state=mock_state,
                goal="test goal",
                epsilon=0.1,
            )
        assert task is not None
        assert task.actor == "high_agent"

    @pytest.mark.unit
    def test_get_next_task_with_q_predictor_explore_mode(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A", actor="agent_a")
        board.add_task("t2", "B", actor="agent_b")

        mock_predictor = Mock()
        mock_predictor.predict_q_value = Mock(return_value=(0.5, None, None))
        mock_state = Mock()

        with patch("random.random", return_value=0.01):  # Explore (< epsilon)
            task = board.get_next_task(
                q_predictor=mock_predictor,
                current_state=mock_state,
                goal="test",
                epsilon=0.5,
            )
        # Should still return a task (randomly chosen)
        assert task is not None

    @pytest.mark.unit
    def test_get_next_task_fallback_no_predictor(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        task = board.get_next_task(q_predictor=None)
        assert task is not None
        assert task.task_id == "t1"

    @pytest.mark.unit
    def test_unblock_ready_tasks_count(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B", depends_on=["t1"])
        # Manually set t2 to BLOCKED
        board.subtasks["t2"].status = TaskStatus.BLOCKED
        # Complete t1
        board.start_task("t1")
        board.complete_task("t1")
        unblocked = board.unblock_ready_tasks()
        assert unblocked == 1
        assert board.subtasks["t2"].status == TaskStatus.PENDING

    @pytest.mark.unit
    def test_unblock_ready_tasks_none_ready(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B", depends_on=["t1"])
        board.subtasks["t2"].status = TaskStatus.BLOCKED
        # t1 not completed, so t2 stays blocked
        unblocked = board.unblock_ready_tasks()
        assert unblocked == 0

    @pytest.mark.unit
    def test_start_task_sets_status(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        assert board.subtasks["t1"].status == TaskStatus.IN_PROGRESS

    @pytest.mark.unit
    def test_start_task_sets_current_task_id(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        assert board.current_task_id == "t1"

    @pytest.mark.unit
    def test_start_task_increments_attempts(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        assert board.subtasks["t1"].attempts == 1

    @pytest.mark.unit
    def test_complete_task_sets_status(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        board.complete_task("t1")
        assert board.subtasks["t1"].status == TaskStatus.COMPLETED

    @pytest.mark.unit
    def test_complete_task_adds_to_completed_set(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        board.complete_task("t1")
        assert "t1" in board.completed_tasks

    @pytest.mark.unit
    def test_complete_task_with_result(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        result = {"output": "done"}
        board.complete_task("t1", result=result)
        assert board.subtasks["t1"].result == result

    @pytest.mark.unit
    def test_complete_task_clears_current_task_id(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        board.complete_task("t1")
        assert board.current_task_id is None

    @pytest.mark.unit
    def test_fail_task_sets_error(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A", max_attempts=1)
        board.start_task("t1")
        board.fail_task("t1", "network error")
        assert board.subtasks["t1"].error == "network error"

    @pytest.mark.unit
    def test_fail_task_adds_to_failed_set_when_exhausted(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A", max_attempts=1)
        board.start_task("t1")  # attempts = 1
        board.fail_task("t1", "error")
        assert "t1" in board.failed_tasks

    @pytest.mark.unit
    def test_fail_task_retries_when_attempts_remain(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A", max_attempts=3)
        board.start_task("t1")  # attempts = 1
        board.fail_task("t1", "error")
        assert "t1" not in board.failed_tasks
        assert board.subtasks["t1"].status == TaskStatus.PENDING

    @pytest.mark.unit
    def test_checkpoint_creates_snapshot(self):
        board = SwarmTaskBoard(root_task="Test")
        board.add_task("t1", "A")
        board.start_task("t1")
        cp = board.checkpoint()
        assert cp["todo_id"] == board.todo_id
        assert cp["current_task_id"] == "t1"
        assert "t1" in cp["subtask_states"]
        assert len(board.checkpoints) == 1

    @pytest.mark.unit
    def test_checkpoint_records_timestamp(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.checkpoint()
        assert board.last_checkpoint is not None

    @pytest.mark.unit
    def test_restore_from_checkpoint(self):
        board = SwarmTaskBoard(root_task="Test")
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        board.start_task("t1")
        board.complete_task("t1")
        cp = board.checkpoint()

        # Now modify state
        board.start_task("t2")
        board.complete_task("t2")

        # Restore
        board.restore_from_checkpoint(cp)
        assert "t1" in board.completed_tasks
        assert "t2" not in board.completed_tasks
        assert board.subtasks["t1"].status == TaskStatus.COMPLETED
        assert board.subtasks["t2"].status == TaskStatus.PENDING

    @pytest.mark.unit
    def test_get_progress_summary_returns_string(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        summary = board.get_progress_summary()
        assert isinstance(summary, str)
        assert "Task List Progress" in summary

    @pytest.mark.unit
    def test_get_progress_summary_shows_counts(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        board.start_task("t1")
        board.complete_task("t1")
        summary = board.get_progress_summary()
        assert "1/2 completed" in summary

    @pytest.mark.unit
    def test_should_replan_no_tasks(self):
        board = SwarmTaskBoard()
        should, reason = board.should_replan(elapsed_time=100)
        assert should is False
        assert "No tasks" in reason

    @pytest.mark.unit
    def test_should_replan_deadline_exceeded(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        should, reason = board.should_replan(elapsed_time=400, global_deadline=300)
        assert should is True
        assert "DEADLINE_EXCEEDED" in reason

    @pytest.mark.unit
    def test_should_replan_behind_schedule(self):
        board = SwarmTaskBoard()
        for i in range(10):
            board.add_task(f"t{i}", f"Task {i}")
        # 60% of time passed, 0% complete
        should, reason = board.should_replan(
            elapsed_time=180, global_deadline=300, success_threshold=0.7
        )
        assert should is True
        assert "BEHIND_SCHEDULE" in reason

    @pytest.mark.unit
    def test_should_replan_high_failure_rate(self):
        board = SwarmTaskBoard()
        for i in range(4):
            board.add_task(f"t{i}", f"Task {i}", max_attempts=1)
        # Fail 2 out of 4 = 50% > 30%
        board.start_task("t0")
        board.fail_task("t0", "err")
        board.start_task("t1")
        board.fail_task("t1", "err")
        should, reason = board.should_replan(elapsed_time=10, global_deadline=300)
        assert should is True
        assert "HIGH_FAILURE_RATE" in reason

    @pytest.mark.unit
    def test_should_replan_on_track(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        board.complete_task("t1")
        should, reason = board.should_replan(elapsed_time=10, global_deadline=300)
        assert should is False
        assert "On track" in reason

    @pytest.mark.unit
    def test_replan_returns_list_of_observations(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        actions = board.replan(observation="test observation")
        assert isinstance(actions, list)

    @pytest.mark.unit
    def test_replan_skips_tasks_with_failed_deps(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A", max_attempts=1)
        board.add_task("t2", "B", depends_on=["t1"])
        board.start_task("t1")
        board.fail_task("t1", "error")
        actions = board.replan()
        skip_actions = [a for a in actions if a.startswith("SKIP")]
        assert len(skip_actions) > 0
        assert board.subtasks["t2"].status == TaskStatus.SKIPPED

    @pytest.mark.unit
    def test_replan_adds_observation_to_risk_factors(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.replan(observation="Unexpected latency")
        assert any("Unexpected latency" in r for r in board.risk_factors)

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        board = SwarmTaskBoard(root_task="Test root")
        board.add_task("t1", "Design", actor="arch")
        d = board.to_dict()
        assert d["root_task"] == "Test root"
        assert "t1" in d["subtasks"]
        assert d["subtasks"]["t1"]["description"] == "Design"
        assert d["execution_order"] == ["t1"]
        assert isinstance(d["completed_tasks"], list)
        assert isinstance(d["failed_tasks"], list)

    @pytest.mark.unit
    def test_get_state_summary_returns_string(self):
        board = SwarmTaskBoard(root_task="Build app")
        board.add_task("t1", "Design", actor="designer")
        summary = board.get_state_summary()
        assert isinstance(summary, str)
        assert "Build app" in summary
        assert "0/1 completed" in summary

    @pytest.mark.unit
    def test_get_state_summary_shows_in_progress(self):
        board = SwarmTaskBoard(root_task="Test")
        board.add_task("t1", "Working", actor="worker")
        board.start_task("t1")
        summary = board.get_state_summary()
        assert "In Progress" in summary
        assert "Working" in summary

    @pytest.mark.unit
    def test_get_state_summary_shows_failed(self):
        board = SwarmTaskBoard(root_task="Test")
        board.add_task("t1", "Fails", actor="worker", max_attempts=1)
        board.start_task("t1")
        board.fail_task("t1", "crashed")
        summary = board.get_state_summary()
        assert "Failed" in summary

    @pytest.mark.unit
    def test_update_q_value(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.update_q_value("t1", 0.8, 0.9)
        assert board.subtasks["t1"].estimated_reward == 0.8
        assert board.subtasks["t1"].confidence == 0.9

    @pytest.mark.unit
    def test_update_q_value_clamps_to_range(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.update_q_value("t1", 1.5, -0.5)
        assert board.subtasks["t1"].estimated_reward == 1.0
        assert board.subtasks["t1"].confidence == 0.0

    @pytest.mark.unit
    def test_update_q_value_nonexistent_task(self):
        board = SwarmTaskBoard()
        # Should not raise
        board.update_q_value("nonexistent", 0.5, 0.5)

    @pytest.mark.unit
    def test_record_intermediary_values(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.record_intermediary_values("t1", {"llm_calls": 3, "time": 1.5})
        assert board.subtasks["t1"].intermediary_values["llm_calls"] == 3
        assert board.subtasks["t1"].intermediary_values["time"] == 1.5

    @pytest.mark.unit
    def test_record_intermediary_values_updates_existing(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.record_intermediary_values("t1", {"llm_calls": 3})
        board.record_intermediary_values("t1", {"llm_calls": 5, "tokens": 100})
        assert board.subtasks["t1"].intermediary_values["llm_calls"] == 5
        assert board.subtasks["t1"].intermediary_values["tokens"] == 100

    @pytest.mark.unit
    def test_record_intermediary_values_nonexistent_task(self):
        board = SwarmTaskBoard()
        # Should not raise
        board.record_intermediary_values("nonexistent", {"x": 1})

    @pytest.mark.unit
    def test_predict_next(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.predict_next("t1", "t2", duration=30.0, reward=0.8)
        assert board.subtasks["t1"].predicted_next_task == "t2"
        assert board.subtasks["t1"].predicted_duration == 30.0
        assert board.subtasks["t1"].predicted_reward == 0.8

    @pytest.mark.unit
    def test_predict_next_partial(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.predict_next("t1", "t2")
        assert board.subtasks["t1"].predicted_next_task == "t2"
        assert board.subtasks["t1"].predicted_duration is None
        assert board.subtasks["t1"].predicted_reward is None

    @pytest.mark.unit
    def test_predict_next_nonexistent_task(self):
        board = SwarmTaskBoard()
        # Should not raise
        board.predict_next("nonexistent", "t2")

    @pytest.mark.unit
    def test_get_task_by_id_existing(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        task = board.get_task_by_id("t1")
        assert task is not None
        assert task.task_id == "t1"

    @pytest.mark.unit
    def test_get_task_by_id_missing(self):
        board = SwarmTaskBoard()
        task = board.get_task_by_id("nonexistent")
        assert task is None

    @pytest.mark.unit
    def test_items_property_backward_compat(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        assert board.items is board.subtasks
        assert "t1" in board.items

    @pytest.mark.unit
    def test_completed_property_backward_compat(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.start_task("t1")
        board.complete_task("t1")
        assert board.completed is board.completed_tasks
        assert "t1" in board.completed

    @pytest.mark.unit
    def test_chain_of_three_tasks_with_dependencies(self):
        board = _make_board_with_tasks()
        # Only t1 available
        task = board.get_next_task()
        assert task.task_id == "t1"

        # Complete t1, now t2 available
        board.start_task("t1")
        board.complete_task("t1")
        task = board.get_next_task()
        assert task.task_id == "t2"

        # Complete t2, now t3 available
        board.start_task("t2")
        board.complete_task("t2")
        task = board.get_next_task()
        assert task.task_id == "t3"

        # Complete t3, nothing left
        board.start_task("t3")
        board.complete_task("t3")
        task = board.get_next_task()
        assert task is None

    @pytest.mark.unit
    def test_parallel_tasks_no_dependencies(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        board.add_task("t3", "C")
        # All three should be available since no dependencies
        available = [
            tid for tid, t in board.subtasks.items()
            if t.status == TaskStatus.PENDING and t.can_start(board.completed_tasks)
        ]
        assert len(available) == 3

    @pytest.mark.unit
    def test_diamond_dependency_pattern(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Start")
        board.add_task("t2a", "Branch A", depends_on=["t1"])
        board.add_task("t2b", "Branch B", depends_on=["t1"])
        board.add_task("t3", "Join", depends_on=["t2a", "t2b"])

        # Only t1 available first
        task = board.get_next_task()
        assert task.task_id == "t1"

        # Complete t1 -> t2a and t2b available
        board.start_task("t1")
        board.complete_task("t1")
        available = [
            tid for tid, t in board.subtasks.items()
            if t.status == TaskStatus.PENDING and t.can_start(board.completed_tasks)
        ]
        assert set(available) == {"t2a", "t2b"}

        # Complete only t2a -> t3 NOT yet available
        board.start_task("t2a")
        board.complete_task("t2a")
        assert not board.subtasks["t3"].can_start(board.completed_tasks)

        # Complete t2b -> t3 available
        board.start_task("t2b")
        board.complete_task("t2b")
        assert board.subtasks["t3"].can_start(board.completed_tasks)

    @pytest.mark.unit
    def test_transition_probs_tracking(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        board.start_task("t1")
        board.complete_task("t1")
        board.start_task("t2")
        board.complete_task("t2")
        # After 2 completions, transition_probs should have an entry
        assert len(board.transition_probs) > 0

    @pytest.mark.unit
    def test_execution_order_tracking(self):
        board = SwarmTaskBoard()
        board.add_task("alpha", "First")
        board.add_task("beta", "Second")
        board.add_task("gamma", "Third")
        assert board.execution_order == ["alpha", "beta", "gamma"]

    @pytest.mark.unit
    def test_estimated_remaining_steps_after_completion(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        assert board.estimated_remaining_steps == 2
        board.start_task("t1")
        board.complete_task("t1")
        assert board.estimated_remaining_steps == 1

    @pytest.mark.unit
    def test_completion_probability_increases(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.add_task("t2", "B")
        initial_prob = board.completion_probability
        board.start_task("t1")
        board.complete_task("t1")
        assert board.completion_probability > initial_prob

    @pytest.mark.unit
    def test_start_task_nonexistent_no_error(self):
        board = SwarmTaskBoard()
        board.start_task("nonexistent")  # Should not raise

    @pytest.mark.unit
    def test_complete_task_nonexistent_no_error(self):
        board = SwarmTaskBoard()
        board.complete_task("nonexistent")  # Should not raise

    @pytest.mark.unit
    def test_fail_task_nonexistent_no_error(self):
        board = SwarmTaskBoard()
        board.fail_task("nonexistent", "err")  # Should not raise

    @pytest.mark.unit
    def test_checkpoints_list_grows(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.checkpoint()
        board.checkpoint()
        assert len(board.checkpoints) == 2

    @pytest.mark.unit
    def test_risk_factors_default_empty(self):
        board = SwarmTaskBoard()
        assert board.risk_factors == []

    @pytest.mark.unit
    def test_should_replan_stuck_tasks(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "Stuck task")
        board.subtasks["t1"].status = TaskStatus.IN_PROGRESS
        board.subtasks["t1"].attempts = 5  # >3
        should, reason = board.should_replan(elapsed_time=10, global_deadline=300)
        assert should is True
        assert "STUCK_TASKS" in reason

    @pytest.mark.unit
    def test_get_next_task_q_predictor_failure_fallback(self):
        """When Q-predictor raises, fall back to execution order."""
        board = SwarmTaskBoard()
        board.add_task("t1", "A", actor="agent_a")
        board.add_task("t2", "B", actor="agent_b")

        mock_predictor = Mock()
        mock_predictor.predict_q_value = Mock(side_effect=Exception("LLM error"))
        mock_state = Mock()

        with patch("random.random", return_value=0.99):  # Exploit mode
            task = board.get_next_task(
                q_predictor=mock_predictor,
                current_state=mock_state,
                goal="test",
                epsilon=0.1,
            )
        # Falls back to fixed order
        assert task is not None
        assert task.task_id == "t1"

    @pytest.mark.unit
    def test_to_dict_includes_checkpoints(self):
        board = SwarmTaskBoard()
        board.add_task("t1", "A")
        board.checkpoint()
        d = board.to_dict()
        assert "checkpoints" in d
        assert len(d["checkpoints"]) == 1


# =============================================================================
# TestAgenticState
# =============================================================================

class TestAgenticState:
    """Tests for the AgenticState dataclass."""

    @pytest.mark.unit
    def test_creation_with_defaults(self):
        state = AgenticState()
        assert state.agent_name == ""
        assert state.episode_id == ""
        assert state.task_description == ""

    @pytest.mark.unit
    def test_state_id_auto_generated(self):
        state = AgenticState()
        assert state.state_id != ""
        assert len(state.state_id) == 32  # md5 hex

    @pytest.mark.unit
    def test_state_id_preserved_when_provided(self):
        state = AgenticState(state_id="custom-id")
        assert state.state_id == "custom-id"

    @pytest.mark.unit
    def test_agent_name_field(self):
        state = _make_agentic_state(agent_name="researcher")
        assert state.agent_name == "researcher"

    @pytest.mark.unit
    def test_episode_id_field(self):
        state = _make_agentic_state(episode_id="ep-42")
        assert state.episode_id == "ep-42"

    @pytest.mark.unit
    def test_task_description_default(self):
        state = AgenticState()
        assert state.task_description == ""

    @pytest.mark.unit
    def test_task_decomposition_default(self):
        state = AgenticState()
        assert state.task_decomposition == []

    @pytest.mark.unit
    def test_trajectory_default_empty(self):
        state = AgenticState()
        assert state.trajectory == []

    @pytest.mark.unit
    def test_reasoning_trace_default_empty(self):
        state = AgenticState()
        assert state.reasoning_trace == []

    @pytest.mark.unit
    def test_tool_calls_default_empty(self):
        state = AgenticState()
        assert state.tool_calls == []

    @pytest.mark.unit
    def test_add_trajectory_step_adds_entry(self):
        state = _make_agentic_state()
        state.add_trajectory_step("thought", "I should search", "found results", 0.5)
        assert len(state.trajectory) == 1
        assert state.trajectory[0].action_type == "thought"
        assert state.trajectory[0].action_content == "I should search"

    @pytest.mark.unit
    def test_add_trajectory_step_multiple_entries(self):
        state = _make_agentic_state()
        state.add_trajectory_step("thought", "step1", "obs1", 0.3)
        state.add_trajectory_step("tool_call", "step2", "obs2", 0.7)
        assert len(state.trajectory) == 2
        assert state.trajectory[0].step_idx == 0
        assert state.trajectory[1].step_idx == 1

    @pytest.mark.unit
    def test_add_trajectory_step_updates_last_updated(self):
        state = _make_agentic_state()
        before = state.last_updated
        import time
        time.sleep(0.01)
        state.add_trajectory_step("thought", "x", "y", 0.0)
        assert state.last_updated >= before

    @pytest.mark.unit
    def test_add_reasoning_step_appends(self):
        state = _make_agentic_state()
        state.add_reasoning_step("First thought")
        state.add_reasoning_step("Second thought")
        assert state.reasoning_trace == ["First thought", "Second thought"]

    @pytest.mark.unit
    def test_add_tool_call_appends(self):
        state = _make_agentic_state()
        state.add_tool_call("web_search", {"query": "test"}, "results", True)
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["tool"] == "web_search"
        assert state.tool_calls[0]["success"] is True

    @pytest.mark.unit
    def test_add_tool_call_records_step_idx(self):
        state = _make_agentic_state()
        state.add_trajectory_step("thought", "x", "y", 0.0)
        state.add_tool_call("calc", {}, "42", True)
        assert state.tool_calls[0]["step_idx"] == 1  # len(trajectory) at time of call

    @pytest.mark.unit
    def test_to_key_returns_string(self):
        state = _make_agentic_state()
        key = state.to_key()
        assert isinstance(key, str)
        assert "test-agent" in key
        assert "test task" in key

    @pytest.mark.unit
    def test_to_key_includes_trajectory_summary(self):
        state = _make_agentic_state()
        state.add_trajectory_step("thought", "x", "y", 0.0)
        key = state.to_key()
        assert "steps:1" in key
        assert "last:thought" in key

    @pytest.mark.unit
    def test_to_llm_summary_returns_formatted(self):
        state = _make_agentic_state()
        summary = state.to_llm_summary()
        assert "Agent: test-agent" in summary
        assert "Task: test task" in summary

    @pytest.mark.unit
    def test_to_llm_summary_includes_reasoning(self):
        state = _make_agentic_state()
        state.add_reasoning_step("analyzed data")
        summary = state.to_llm_summary()
        assert "analyzed data" in summary

    @pytest.mark.unit
    def test_to_llm_summary_includes_tools(self):
        state = _make_agentic_state()
        state.add_tool_call("search", {}, "ok", True)
        summary = state.to_llm_summary()
        assert "search" in summary

    @pytest.mark.unit
    def test_to_llm_summary_includes_prediction(self):
        state = _make_agentic_state()
        state.predicted_outcome = "success expected"
        summary = state.to_llm_summary()
        assert "success expected" in summary

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        state = _make_agentic_state()
        state.add_trajectory_step("thought", "reasoning", "found it", 0.8)
        state.add_reasoning_step("key insight")
        d = state.to_dict()
        assert d["state_id"] == "test-state"
        assert d["agent_name"] == "test-agent"
        assert d["episode_id"] == "ep-001"
        assert len(d["trajectory"]) == 1
        assert d["reasoning_trace"] == ["key insight"]
        assert "predictions" in d
        assert "created_at" in d
        assert "last_updated" in d

    @pytest.mark.unit
    def test_from_dict_roundtrip(self):
        state = _make_agentic_state()
        state.predicted_next_action = "search"
        state.action_confidence = 0.9
        state.predicted_outcome = "will find results"
        state.predicted_reward = 0.7
        state.uncertainty = 0.3
        d = state.to_dict()
        restored = AgenticState.from_dict(d)
        assert restored.state_id == state.state_id
        assert restored.agent_name == state.agent_name
        assert restored.episode_id == state.episode_id
        assert restored.predicted_next_action == "search"
        assert restored.action_confidence == 0.9
        assert restored.predicted_outcome == "will find results"
        assert restored.predicted_reward == 0.7
        assert restored.uncertainty == 0.3

    @pytest.mark.unit
    def test_from_dict_with_empty_data(self):
        state = AgenticState.from_dict({})
        assert state.agent_name == ""
        assert state.task_description == ""
        assert state.action_confidence == 0.5

    @pytest.mark.unit
    def test_predicted_next_action_default(self):
        state = AgenticState()
        assert state.predicted_next_action == ""

    @pytest.mark.unit
    def test_action_confidence_default(self):
        state = AgenticState()
        assert state.action_confidence == 0.5

    @pytest.mark.unit
    def test_uncertainty_default(self):
        state = AgenticState()
        assert state.uncertainty == 0.5

    @pytest.mark.unit
    def test_active_causal_chains_default(self):
        state = AgenticState()
        assert state.active_causal_chains == []

    @pytest.mark.unit
    def test_created_at_timestamp(self):
        before = datetime.now()
        state = AgenticState()
        after = datetime.now()
        assert before <= state.created_at <= after

    @pytest.mark.unit
    def test_last_updated_timestamp(self):
        state = AgenticState()
        assert isinstance(state.last_updated, datetime)

    @pytest.mark.unit
    def test_activated_memories_default(self):
        state = AgenticState()
        assert state.activated_memories == []

    @pytest.mark.unit
    def test_memory_relevance_scores_default(self):
        state = AgenticState()
        assert state.memory_relevance_scores == {}

    @pytest.mark.unit
    def test_intervention_effects_default(self):
        state = AgenticState()
        assert state.intervention_effects == {}

    @pytest.mark.unit
    def test_subtask_completion_default(self):
        state = AgenticState()
        assert state.subtask_completion == {}

    @pytest.mark.unit
    def test_current_subtask_idx_default(self):
        state = AgenticState()
        assert state.current_subtask_idx == 0


# =============================================================================
# TestDecomposedQFunction
# =============================================================================

class TestDecomposedQFunction:
    """Tests for the DecomposedQFunction multi-objective Q-learning."""

    @pytest.mark.unit
    def test_initialization_with_defaults(self):
        qf = DecomposedQFunction()
        assert qf.weights["task"] == 0.5
        assert qf.weights["explore"] == 0.2
        assert qf.weights["causal"] == 0.15
        assert qf.weights["safety"] == 0.15

    @pytest.mark.unit
    def test_initialization_with_custom_config(self):
        qf = DecomposedQFunction(config={"task_weight": 0.8, "default_value": 0.0})
        assert qf.weights["task"] == 0.8
        assert qf.default_value == 0.0

    @pytest.mark.unit
    def test_get_q_value_unknown_state_returns_default(self):
        qf = DecomposedQFunction(config={"default_value": 0.5})
        state = _make_agentic_state()
        val = qf.get_q_value(state, "proceed", objective="task")
        assert val == 0.5

    @pytest.mark.unit
    def test_get_q_value_for_specific_objective(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        key = (state.to_key(), "search")
        qf.q_task[key] = 0.9
        val = qf.get_q_value(state, "search", objective="task")
        assert val == 0.9

    @pytest.mark.unit
    def test_get_q_value_no_objective_returns_combined(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        key = (state.to_key(), "search")
        qf.q_task[key] = 1.0
        qf.q_explore[key] = 1.0
        qf.q_causal[key] = 1.0
        qf.q_safety[key] = 1.0
        val = qf.get_q_value(state, "search")
        # All 1.0, weights sum to 1.0 => combined = 1.0
        assert abs(val - 1.0) < 1e-6

    @pytest.mark.unit
    def test_get_combined_value_uses_weights(self):
        qf = DecomposedQFunction(config={
            "task_weight": 1.0,
            "explore_weight": 0.0,
            "causal_weight": 0.0,
            "safety_weight": 0.0,
            "default_value": 0.0,
        })
        state = _make_agentic_state()
        key = (state.to_key(), "act")
        qf.q_task[key] = 0.8
        combined = qf.get_combined_value(state, "act")
        assert abs(combined - 0.8) < 1e-6

    @pytest.mark.unit
    def test_update_modifies_q_values(self):
        qf = DecomposedQFunction(config={"default_value": 0.5})
        state = _make_agentic_state(agent_name="a1", task_description="t1")
        next_state = _make_agentic_state(agent_name="a1", task_description="t1")
        next_state.add_trajectory_step("thought", "x", "y", 0.0)

        rewards = {"task": 1.0, "explore": 0.5, "causal": 0.3, "safety": 0.8}
        qf.update(state, "proceed", rewards, next_state)

        key = (state.to_key(), "proceed")
        # Q-values should have been modified from the default
        assert qf.q_task[key] != 0.5

    @pytest.mark.unit
    def test_adjust_weights_exploration(self):
        qf = DecomposedQFunction()
        qf.adjust_weights("exploration")
        assert qf.weights["explore"] == 0.4
        assert qf.weights["task"] == 0.3

    @pytest.mark.unit
    def test_adjust_weights_exploitation(self):
        qf = DecomposedQFunction()
        qf.adjust_weights("exploitation")
        assert qf.weights["task"] == 0.6
        assert qf.weights["explore"] == 0.1

    @pytest.mark.unit
    def test_adjust_weights_safety_critical(self):
        qf = DecomposedQFunction()
        qf.adjust_weights("safety_critical")
        assert qf.weights["safety"] == 0.5

    @pytest.mark.unit
    def test_adjust_weights_unknown_phase_no_change(self):
        qf = DecomposedQFunction()
        original = qf.weights.copy()
        qf.adjust_weights("unknown_phase")
        assert qf.weights == original

    @pytest.mark.unit
    def test_get_action_ranking_sorts_by_value(self):
        qf = DecomposedQFunction(config={"default_value": 0.0})
        state = _make_agentic_state()
        key_a = (state.to_key(), "action_a")
        key_b = (state.to_key(), "action_b")
        qf.q_task[key_a] = 0.9
        qf.q_task[key_b] = 0.1
        ranking = qf.get_action_ranking(state, ["action_a", "action_b"])
        assert ranking[0][0] == "action_a"
        assert ranking[0][1] > ranking[1][1]

    @pytest.mark.unit
    def test_get_action_ranking_empty_actions(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        ranking = qf.get_action_ranking(state, [])
        assert ranking == []

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        key = (state.to_key(), "act")
        qf.q_task[key] = 0.7
        d = qf.to_dict()
        assert "q_task" in d
        assert "weights" in d
        assert "alphas" in d

    @pytest.mark.unit
    def test_from_dict_deserialization(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        key = (state.to_key(), "act")
        qf.q_task[key] = 0.77
        qf.weights["task"] = 0.9
        d = qf.to_dict()
        restored = DecomposedQFunction.from_dict(d)
        assert restored.weights["task"] == 0.9
        # Check that q_task was restored
        assert len(restored.q_task) == 1

    @pytest.mark.unit
    def test_from_dict_empty_data(self):
        qf = DecomposedQFunction.from_dict({})
        assert qf.q_task == {}
        assert qf.q_explore == {}

    @pytest.mark.unit
    def test_multiple_objectives_stored(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        key = (state.to_key(), "act")
        qf.q_task[key] = 0.8
        qf.q_explore[key] = 0.6
        qf.q_causal[key] = 0.4
        qf.q_safety[key] = 0.9
        assert qf.get_q_value(state, "act", "task") == 0.8
        assert qf.get_q_value(state, "act", "explore") == 0.6
        assert qf.get_q_value(state, "act", "causal") == 0.4
        assert qf.get_q_value(state, "act", "safety") == 0.9

    @pytest.mark.unit
    def test_alpha_learning_rates_defaults(self):
        qf = DecomposedQFunction()
        assert qf.alphas["task"] == 0.05
        assert qf.alphas["explore"] == 0.1
        assert qf.alphas["causal"] == 0.08
        assert qf.alphas["safety"] == 0.03

    @pytest.mark.unit
    def test_alpha_learning_rates_custom(self):
        qf = DecomposedQFunction(config={"alpha_task": 0.2, "alpha_safety": 0.01})
        assert qf.alphas["task"] == 0.2
        assert qf.alphas["safety"] == 0.01

    @pytest.mark.unit
    def test_get_possible_actions_returns_list(self):
        qf = DecomposedQFunction()
        state = _make_agentic_state()
        actions = qf._get_possible_actions(state)
        assert isinstance(actions, list)
        assert len(actions) > 0

    @pytest.mark.unit
    def test_from_dict_roundtrip(self):
        qf = DecomposedQFunction(config={"task_weight": 0.7})
        state = _make_agentic_state()
        key = (state.to_key(), "test_action")
        qf.q_task[key] = 0.88
        qf.q_explore[key] = 0.33
        d = qf.to_dict()
        restored = DecomposedQFunction.from_dict(d)
        # Weights preserved
        assert restored.weights["task"] == 0.7
        # Q-values preserved
        assert len(restored.q_task) == 1
        assert len(restored.q_explore) == 1


# =============================================================================
# TestThoughtLevelCredit
# =============================================================================

class TestThoughtLevelCredit:
    """Tests for the ThoughtLevelCredit reasoning step credit assignment."""

    @pytest.mark.unit
    def test_initialization_with_default_config(self):
        tlc = ThoughtLevelCredit()
        assert tlc.temporal_weight == 0.3
        assert tlc.tool_weight == 0.4
        assert tlc.decision_weight == 0.3

    @pytest.mark.unit
    def test_initialization_with_custom_config(self):
        tlc = ThoughtLevelCredit(config={
            "temporal_weight": 0.5,
            "tool_weight": 0.3,
            "decision_weight": 0.2,
        })
        assert tlc.temporal_weight == 0.5
        assert tlc.tool_weight == 0.3

    @pytest.mark.unit
    def test_assign_credit_empty_trace(self):
        tlc = ThoughtLevelCredit()
        credits = tlc.assign_credit([], [], 1.0)
        assert credits == {}

    @pytest.mark.unit
    def test_assign_credit_single_step(self):
        tlc = ThoughtLevelCredit()
        credits = tlc.assign_credit(["analyzed the data"], [], 1.0)
        assert len(credits) > 0
        assert 0 in credits

    @pytest.mark.unit
    def test_assign_credit_with_reasoning_and_tools(self):
        tlc = ThoughtLevelCredit()
        trace = [
            "I need to search for information",
            "The search returned useful results",
            "Based on the results, the answer is X",
        ]
        tools = [
            {"tool": "search", "success": True, "step_idx": 0},
        ]
        credits = tlc.assign_credit(trace, tools, 1.0)
        assert len(credits) > 0
        # All steps should have credit
        for i in range(len(trace)):
            assert i in credits

    @pytest.mark.unit
    def test_assign_credit_temporal_weighting(self):
        tlc = ThoughtLevelCredit(config={
            "temporal_weight": 1.0,
            "tool_weight": 0.0,
            "decision_weight": 0.0,
        })
        trace = ["step 0 " * 100, "step 1 " * 100, "step 2 " * 100]
        credits = tlc.assign_credit(trace, [], 1.0)
        # Later steps should get more credit (temporal weighting)
        assert credits[2] > credits[0]

    @pytest.mark.unit
    def test_assign_credit_tool_usage_weighting(self):
        tlc = ThoughtLevelCredit()
        trace = [
            "I will use the calculator" * 5,  # Not linked without LM
            "The result is 42" * 5,  # Not linked without LM
        ]
        tools = [{"tool": "calculator", "success": True}]
        credits = tlc.assign_credit(trace, tools, 1.0)
        # Without LM, tool linking returns None, so tool credits go nowhere special
        assert len(credits) > 0

    @pytest.mark.unit
    def test_assign_credit_decision_point_weighting(self):
        tlc = ThoughtLevelCredit()
        # Create trace where some steps match decision heuristics
        trace = [
            "This is a very long analysis step with lots of detail about the problem " * 10,
            "Decision: use approach A",  # Short, follows long -> decision
            "Very long detailed explanation again with many words " * 10,
            "Execute the plan",  # Short again
        ]
        credits = tlc.assign_credit(trace, [], 1.0)
        assert len(credits) == 4

    @pytest.mark.unit
    def test_assign_credit_sums_to_absolute_outcome(self):
        tlc = ThoughtLevelCredit()
        trace = ["step1 is detailed analysis here", "step2 short", "final conclusion here"]
        credits = tlc.assign_credit(trace, [], 0.8)
        total = sum(credits.values())
        assert abs(total - 0.8) < 1e-6

    @pytest.mark.unit
    def test_assign_credit_negative_outcome(self):
        tlc = ThoughtLevelCredit()
        trace = ["first thought was wrong", "second thought also wrong"]
        credits = tlc.assign_credit(trace, [], -1.0)
        # With negative outcome, total is negative, so normalization is skipped
        # (normalization only applies when total > 0)
        # All individual credits should be <= 0
        for v in credits.values():
            assert v <= 0

    @pytest.mark.unit
    def test_get_step_value_summary_no_credits(self):
        tlc = ThoughtLevelCredit()
        summary = tlc.get_step_value_summary({}, [])
        assert summary == "No credits assigned"

    @pytest.mark.unit
    def test_get_step_value_summary_with_credits(self):
        tlc = ThoughtLevelCredit()
        credits = {0: 0.3, 1: 0.7}
        trace = ["first step", "second step"]
        summary = tlc.get_step_value_summary(credits, trace)
        assert "Step 0" in summary
        assert "Step 1" in summary
        assert "0.300" in summary
        assert "0.700" in summary

    @pytest.mark.unit
    def test_identify_decision_steps_final_third(self):
        tlc = ThoughtLevelCredit()
        # 10 steps, steps 7-9 should be in final third (0.7 threshold)
        trace = [f"step {i} with some content here" * 5 for i in range(10)]
        decisions = tlc._identify_decision_steps(trace)
        # At least the last few steps should be identified
        assert 9 in decisions or 8 in decisions or 7 in decisions

    @pytest.mark.unit
    def test_identify_decision_steps_short_statements(self):
        tlc = ThoughtLevelCredit()
        trace = [
            "This is a really long statement " * 20,
            "Short decision made here",  # 24 chars, within 20-150
        ]
        decisions = tlc._identify_decision_steps(trace)
        assert 1 in decisions

    @pytest.mark.unit
    def test_find_linked_thought_no_lm_returns_none(self):
        tlc = ThoughtLevelCredit()
        result = tlc._find_linked_thought(["step1", "step2"], "tool_name")
        assert result is None

    @pytest.mark.unit
    def test_lm_config_stored(self):
        mock_lm = Mock()
        tlc = ThoughtLevelCredit(config={"lm": mock_lm})
        assert tlc.lm is mock_lm


# =============================================================================
# TestTodoItem
# =============================================================================

class TestTodoItem:
    """Tests for the TodoItem dataclass."""

    @pytest.mark.unit
    def test_creation(self):
        item = TodoItem(
            id="item1", description="Do thing", actor="agent",
            status="pending", priority=0.8, estimated_reward=0.5,
        )
        assert item.id == "item1"
        assert item.description == "Do thing"
        assert item.actor == "agent"

    @pytest.mark.unit
    def test_defaults(self):
        item = TodoItem(
            id="item1", description="Do thing", actor="agent",
            status="pending", priority=0.8, estimated_reward=0.5,
        )
        assert item.dependencies == []
        assert item.attempts == 0
        assert item.max_attempts == 5
        assert item.failure_reasons == []
        assert item.completion_time is None

    @pytest.mark.unit
    def test_with_dependencies(self):
        item = TodoItem(
            id="item2", description="After item1", actor="agent",
            status="pending", priority=0.5, estimated_reward=0.3,
            dependencies=["item1"],
        )
        assert item.dependencies == ["item1"]
