"""
Tests for Persistence SharedContext and Job Queue Components
=============================================================

Comprehensive unit tests covering:
- SharedContext (core/persistence/shared_context.py)
- TaskPriority enum (core/job_queue/task.py)
- Task dataclass (core/job_queue/task.py)
- TaskQueue abstract class (core/job_queue/task_queue.py)
- MemoryTaskQueue (core/job_queue/memory_queue.py)
- SQLiteTaskQueue (core/job_queue/sqlite_queue.py)

All tests use mocks and tmp_path -- NO real LLM calls.
"""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# Guarded imports with skip markers
# ---------------------------------------------------------------------------

try:
    from Jotty.core.infrastructure.persistence.shared_context import SharedContext
    SHARED_CONTEXT_AVAILABLE = True
except ImportError:
    SHARED_CONTEXT_AVAILABLE = False

try:
    from Jotty.core.infrastructure.job_queue.task import Task, TaskPriority
    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False

try:
    from Jotty.core.infrastructure.job_queue.task_queue import TaskQueue
    TASK_QUEUE_AVAILABLE = True
except ImportError:
    TASK_QUEUE_AVAILABLE = False

try:
    from Jotty.core.infrastructure.job_queue.memory_queue import MemoryTaskQueue
    MEMORY_QUEUE_AVAILABLE = True
except ImportError:
    MEMORY_QUEUE_AVAILABLE = False

try:
    from Jotty.core.infrastructure.job_queue.sqlite_queue import SQLiteTaskQueue
    SQLITE_QUEUE_AVAILABLE = True
except ImportError:
    SQLITE_QUEUE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper: run async in sync context
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# 1. SharedContext Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextInit:
    """Tests for SharedContext.__init__."""

    def test_init_creates_empty_data_dict(self):
        ctx = SharedContext()
        assert ctx.data == {}

    def test_init_data_is_dict_type(self):
        ctx = SharedContext()
        assert isinstance(ctx.data, dict)

    def test_init_lock_is_created(self):
        ctx = SharedContext()
        assert ctx._lock is not None


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextSetGet:
    """Tests for SharedContext.set and .get methods."""

    def test_set_and_get_string_value(self):
        ctx = SharedContext()
        ctx.set("key1", "value1")
        assert ctx.get("key1") == "value1"

    def test_set_and_get_dict_value(self):
        ctx = SharedContext()
        data = {"nested": "data", "count": 42}
        ctx.set("config", data)
        assert ctx.get("config") == data

    def test_set_and_get_list_value(self):
        ctx = SharedContext()
        items = [1, 2, 3, "four"]
        ctx.set("items", items)
        assert ctx.get("items") == items

    def test_get_missing_key_returns_none(self):
        ctx = SharedContext()
        assert ctx.get("nonexistent") is None

    def test_set_overwrites_existing_key(self):
        ctx = SharedContext()
        ctx.set("key", "original")
        ctx.set("key", "updated")
        assert ctx.get("key") == "updated"

    def test_set_and_get_none_value(self):
        ctx = SharedContext()
        ctx.set("nullable", None)
        # get() returns None both for missing and None-valued keys
        assert ctx.get("nullable") is None

    def test_set_multiple_keys(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        ctx.set("c", 3)
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2
        assert ctx.get("c") == 3


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextGetAll:
    """Tests for SharedContext.get_all."""

    def test_get_all_empty_context(self):
        ctx = SharedContext()
        assert ctx.get_all() == {}

    def test_get_all_returns_copy(self):
        ctx = SharedContext()
        ctx.set("x", 10)
        all_data = ctx.get_all()
        assert all_data == {"x": 10}
        # Modifying the copy should not affect original
        all_data["x"] = 999
        assert ctx.get("x") == 10

    def test_get_all_with_multiple_items(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", "two")
        ctx.set("c", [3])
        result = ctx.get_all()
        assert result == {"a": 1, "b": "two", "c": [3]}


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextKeys:
    """Tests for SharedContext.keys."""

    def test_keys_empty(self):
        ctx = SharedContext()
        assert ctx.keys() == []

    def test_keys_returns_list_of_keys(self):
        ctx = SharedContext()
        ctx.set("alpha", 1)
        ctx.set("beta", 2)
        keys = ctx.keys()
        assert sorted(keys) == ["alpha", "beta"]

    def test_keys_returns_list_type(self):
        ctx = SharedContext()
        ctx.set("k", "v")
        assert isinstance(ctx.keys(), list)


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextHas:
    """Tests for SharedContext.has."""

    def test_has_existing_key(self):
        ctx = SharedContext()
        ctx.set("present", "yes")
        assert ctx.has("present") is True

    def test_has_missing_key(self):
        ctx = SharedContext()
        assert ctx.has("absent") is False

    def test_has_after_clear(self):
        ctx = SharedContext()
        ctx.set("temp", 1)
        ctx.clear()
        assert ctx.has("temp") is False


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextClear:
    """Tests for SharedContext.clear."""

    def test_clear_empties_all_data(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        ctx.clear()
        assert ctx.get_all() == {}
        assert ctx.keys() == []

    def test_clear_on_empty_context(self):
        ctx = SharedContext()
        ctx.clear()
        assert ctx.get_all() == {}


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextSummary:
    """Tests for SharedContext.summary."""

    def test_summary_empty(self):
        ctx = SharedContext()
        summary = ctx.summary()
        assert "SharedContext" in summary
        assert "0 items" in summary

    def test_summary_with_items(self):
        ctx = SharedContext()
        ctx.set("foo", 1)
        ctx.set("bar", 2)
        summary = ctx.summary()
        assert "SharedContext" in summary
        assert "2 items" in summary
        assert "foo" in summary
        assert "bar" in summary


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextRepr:
    """Tests for SharedContext.__repr__."""

    def test_repr_delegates_to_summary(self):
        ctx = SharedContext()
        ctx.set("key1", "val")
        assert repr(ctx) == ctx.summary()


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextContains:
    """Tests for SharedContext.__contains__."""

    def test_contains_existing_key(self):
        ctx = SharedContext()
        ctx.set("item", 42)
        assert "item" in ctx

    def test_contains_missing_key(self):
        ctx = SharedContext()
        assert "missing" not in ctx


@pytest.mark.unit
@pytest.mark.skipif(not SHARED_CONTEXT_AVAILABLE, reason="SharedContext not importable")
class TestSharedContextThreadSafety:
    """Tests for SharedContext thread-safety."""

    def test_concurrent_set_get_operations(self):
        ctx = SharedContext()
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    ctx.set(f"thread_{thread_id}_key_{i}", f"value_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Each of 10 threads wrote 10 keys = 100 keys total
        assert len(ctx.keys()) == 100

    def test_concurrent_set_and_read(self):
        ctx = SharedContext()
        errors = []

        def writer():
            try:
                for i in range(50):
                    ctx.set(f"w_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    ctx.get_all()
                    ctx.keys()
                    ctx.has("w_0")
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        writer_thread.start()
        for rt in reader_threads:
            rt.start()

        writer_thread.join()
        for rt in reader_threads:
            rt.join()

        assert len(errors) == 0


# ===========================================================================
# 2. TaskPriority Enum Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task/TaskPriority not importable")
class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_critical_value(self):
        assert TaskPriority.CRITICAL.value == 1

    def test_high_value(self):
        assert TaskPriority.HIGH.value == 2

    def test_medium_value(self):
        assert TaskPriority.MEDIUM.value == 3

    def test_low_value(self):
        assert TaskPriority.LOW.value == 4

    def test_optional_value(self):
        assert TaskPriority.OPTIONAL.value == 5

    def test_ordering_by_value(self):
        priorities = [
            TaskPriority.OPTIONAL,
            TaskPriority.LOW,
            TaskPriority.MEDIUM,
            TaskPriority.HIGH,
            TaskPriority.CRITICAL,
        ]
        sorted_priorities = sorted(priorities, key=lambda p: p.value)
        assert sorted_priorities == [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.MEDIUM,
            TaskPriority.LOW,
            TaskPriority.OPTIONAL,
        ]

    def test_enum_has_five_members(self):
        assert len(TaskPriority) == 5


# ===========================================================================
# 3. Task Dataclass Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task not importable")
class TestTaskInit:
    """Tests for Task.__init__ and __post_init__."""

    def test_required_fields(self):
        task = Task(task_id="T-001", title="Test Task")
        assert task.task_id == "T-001"
        assert task.title == "Test Task"

    def test_defaults_for_optional_fields(self):
        task = Task(task_id="T-002", title="Defaults")
        assert task.description == ""
        assert task.category == ""
        assert task.priority == 3
        assert task.tags is None
        assert task.progress_percent == 0
        assert task.pid is None
        assert task.worktree_path is None
        assert task.git_branch is None
        assert task.log_file is None
        assert task.error_message is None
        assert task.retry_count == 0
        assert task.depends_on is None
        assert task.blocks is None
        assert task.created_by == "system"
        assert task.assigned_to is None
        assert task.context_files is None
        assert task.task_content is None
        assert task.filename is None
        assert task.reference_apps is None
        assert task.estimated_effort is None
        assert task.metadata == {}

    def test_post_init_sets_created_at(self):
        before = datetime.now()
        task = Task(task_id="T-003", title="Time")
        after = datetime.now()
        assert task.created_at is not None
        assert before <= task.created_at <= after

    def test_post_init_preserves_existing_created_at(self):
        specific_time = datetime(2025, 6, 15, 12, 0, 0)
        task = Task(task_id="T-004", title="PresetTime", created_at=specific_time)
        assert task.created_at == specific_time

    def test_post_init_status_defaults_to_backlog(self):
        task = Task(task_id="T-005", title="Status")
        assert task.status == "backlog"

    def test_post_init_none_status_becomes_backlog(self):
        task = Task(task_id="T-006", title="NullStatus", status=None)
        assert task.status == "backlog"

    def test_agent_type_default(self):
        task = Task(task_id="T-007", title="Agent")
        assert task.agent_type == "claude"

    def test_agent_type_none_becomes_claude(self):
        task = Task(task_id="T-008", title="NullAgent", agent_type=None)
        assert task.agent_type == "claude"


@pytest.mark.unit
@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task not importable")
class TestTaskToDict:
    """Tests for Task.to_dict."""

    def test_to_dict_contains_all_fields(self):
        task = Task(task_id="T-010", title="DictTest")
        d = task.to_dict()
        expected_keys = {
            'task_id', 'title', 'description', 'category', 'priority',
            'tags', 'status', 'progress_percent', 'pid', 'worktree_path',
            'git_branch', 'log_file', 'created_at', 'started_at',
            'completed_at', 'last_heartbeat', 'estimated_hours',
            'actual_hours', 'error_message', 'retry_count', 'max_retries',
            'depends_on', 'blocks', 'created_by', 'assigned_to',
            'agent_type', 'context_files', 'task_content', 'filename',
            'reference_apps', 'estimated_effort', 'metadata',
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_datetime_as_isoformat(self):
        now = datetime(2025, 7, 1, 10, 30, 0)
        task = Task(task_id="T-011", title="ISO", created_at=now)
        d = task.to_dict()
        assert d['created_at'] == now.isoformat()

    def test_to_dict_none_datetime_fields(self):
        task = Task(task_id="T-012", title="NullDates")
        d = task.to_dict()
        assert d['started_at'] is None
        assert d['completed_at'] is None
        assert d['last_heartbeat'] is None

    def test_to_dict_preserves_values(self):
        task = Task(
            task_id="T-013",
            title="FullTask",
            description="A full task",
            category="testing",
            priority=2,
            status="pending",
            progress_percent=50,
            retry_count=1,
            created_by="user",
        )
        d = task.to_dict()
        assert d['task_id'] == "T-013"
        assert d['title'] == "FullTask"
        assert d['description'] == "A full task"
        assert d['category'] == "testing"
        assert d['priority'] == 2
        assert d['status'] == "pending"
        assert d['progress_percent'] == 50
        assert d['retry_count'] == 1
        assert d['created_by'] == "user"


@pytest.mark.unit
@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task not importable")
class TestTaskFromDict:
    """Tests for Task.from_dict."""

    def test_from_dict_roundtrip(self):
        original = Task(
            task_id="T-020",
            title="Roundtrip",
            description="Round-trip test",
            category="test",
            priority=1,
            status="in_progress",
        )
        d = original.to_dict()
        restored = Task.from_dict(d)
        assert restored.task_id == original.task_id
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.category == original.category
        assert restored.priority == original.priority
        assert restored.status == original.status

    def test_from_dict_datetime_iso_parsing(self):
        d = {
            'task_id': 'T-021',
            'title': 'ISO Parse',
            'created_at': '2025-06-15T12:00:00',
        }
        task = Task.from_dict(d)
        assert task.created_at == datetime(2025, 6, 15, 12, 0, 0)

    def test_from_dict_datetime_z_suffix(self):
        d = {
            'task_id': 'T-022',
            'title': 'Z Suffix',
            'created_at': '2025-06-15T12:00:00Z',
        }
        task = Task.from_dict(d)
        assert task.created_at is not None
        # Z gets replaced with +00:00 for timezone-aware parsing
        assert task.created_at.year == 2025
        assert task.created_at.month == 6
        assert task.created_at.day == 15

    def test_from_dict_defaults_for_missing_fields(self):
        d = {
            'task_id': 'T-023',
            'title': 'Minimal',
        }
        task = Task.from_dict(d)
        assert task.description == ""
        assert task.category == ""
        assert task.priority == 3
        assert task.status == "backlog"
        assert task.progress_percent == 0
        assert task.retry_count == 0
        assert task.created_by == "system"
        assert task.agent_type == "claude"
        assert task.metadata == {}

    def test_from_dict_with_datetime_object(self):
        now = datetime.now()
        d = {
            'task_id': 'T-024',
            'title': 'DatetimeObj',
            'created_at': now,
        }
        task = Task.from_dict(d)
        assert task.created_at == now

    def test_from_dict_with_none_created_at(self):
        d = {
            'task_id': 'T-025',
            'title': 'NoDate',
            'created_at': None,
        }
        task = Task.from_dict(d)
        # from_dict sets created_at to None, then __post_init__ sets it to now
        assert task.created_at is not None


# ===========================================================================
# 4. TaskQueue Abstract Class Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TASK_QUEUE_AVAILABLE, reason="TaskQueue not importable")
class TestTaskQueueAbstract:
    """Tests for TaskQueue abstract class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            TaskQueue()

    def test_declares_abstract_methods(self):
        abstract_methods = set()
        for name in dir(TaskQueue):
            method = getattr(TaskQueue, name, None)
            if callable(method) and getattr(method, '__isabstractmethod__', False):
                abstract_methods.add(name)

        expected = {
            'enqueue', 'dequeue', 'get_task', 'update_status',
            'heartbeat', 'get_running_count', 'get_running_count_by_agent',
            'get_stats', 'get_tasks_by_status', 'get_running_tasks',
            'get_by_filename', 'update_task_priority', 'update_task_metadata',
            'delete_task', 'create_task', 'reset_task_to_backlog',
            'validate_pids', 'export_to_json',
        }
        assert expected.issubset(abstract_methods)


# ===========================================================================
# 5. MemoryTaskQueue Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueEnqueue:
    """Tests for MemoryTaskQueue.enqueue."""

    def test_enqueue_returns_task_id(self):
        queue = MemoryTaskQueue()
        task = Task(task_id="MQ-001", title="Enqueue Test")
        result = run_async(queue.enqueue(task))
        assert result == "MQ-001"

    def test_enqueue_stores_task(self):
        queue = MemoryTaskQueue()
        task = Task(task_id="MQ-002", title="Stored")
        run_async(queue.enqueue(task))
        retrieved = run_async(queue.get_task("MQ-002"))
        assert retrieved is not None
        assert retrieved.title == "Stored"


@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueDequeue:
    """Tests for MemoryTaskQueue.dequeue."""

    def test_dequeue_returns_none_on_empty(self):
        queue = MemoryTaskQueue()
        result = run_async(queue.dequeue())
        assert result is None

    def test_dequeue_returns_pending_task_by_priority(self):
        queue = MemoryTaskQueue()
        t1 = Task(task_id="MQ-D1", title="Low", priority=4, status="pending")
        t2 = Task(task_id="MQ-D2", title="High", priority=1, status="pending")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        result = run_async(queue.dequeue())
        assert result is not None
        assert result.task_id == "MQ-D2"

    def test_dequeue_ignores_non_pending(self):
        queue = MemoryTaskQueue()
        t1 = Task(task_id="MQ-D3", title="Backlog", status="backlog")
        run_async(queue.enqueue(t1))
        result = run_async(queue.dequeue())
        assert result is None


@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueUpdateStatus:
    """Tests for MemoryTaskQueue.update_status."""

    def test_update_status_changes_status(self):
        queue = MemoryTaskQueue()
        task = Task(task_id="MQ-U1", title="Update", status="pending")
        run_async(queue.enqueue(task))
        result = run_async(queue.update_status("MQ-U1", "in_progress", pid=1234))
        assert result is True
        updated = run_async(queue.get_task("MQ-U1"))
        assert updated.status == "in_progress"
        assert updated.pid == 1234

    def test_update_status_nonexistent_task(self):
        queue = MemoryTaskQueue()
        result = run_async(queue.update_status("NOPE", "completed"))
        assert result is False

    def test_update_status_completed_clears_pid(self):
        queue = MemoryTaskQueue()
        task = Task(task_id="MQ-U2", title="Complete", status="in_progress", pid=999)
        run_async(queue.enqueue(task))
        run_async(queue.update_status("MQ-U2", "completed"))
        updated = run_async(queue.get_task("MQ-U2"))
        assert updated.status == "completed"
        assert updated.pid is None
        assert updated.completed_at is not None

    def test_update_status_failed_clears_pid(self):
        queue = MemoryTaskQueue()
        task = Task(task_id="MQ-U3", title="Fail", status="in_progress", pid=888)
        run_async(queue.enqueue(task))
        run_async(queue.update_status("MQ-U3", "failed", error="Boom"))
        updated = run_async(queue.get_task("MQ-U3"))
        assert updated.status == "failed"
        assert updated.pid is None
        assert updated.error_message == "Boom"


@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueGetTask:
    """Tests for MemoryTaskQueue.get_task."""

    def test_get_existing_task(self):
        queue = MemoryTaskQueue()
        task = Task(task_id="MQ-G1", title="Get Me")
        run_async(queue.enqueue(task))
        result = run_async(queue.get_task("MQ-G1"))
        assert result is not None
        assert result.task_id == "MQ-G1"

    def test_get_nonexistent_task(self):
        queue = MemoryTaskQueue()
        result = run_async(queue.get_task("NONE"))
        assert result is None


@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueListTasks:
    """Tests for MemoryTaskQueue.get_tasks_by_status."""

    def test_list_by_status(self):
        queue = MemoryTaskQueue()
        t1 = Task(task_id="MQ-L1", title="Pending1", status="pending")
        t2 = Task(task_id="MQ-L2", title="Backlog1", status="backlog")
        t3 = Task(task_id="MQ-L3", title="Pending2", status="pending")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        run_async(queue.enqueue(t3))
        pending = run_async(queue.get_tasks_by_status("pending"))
        assert len(pending) == 2
        assert all(t.status == "pending" for t in pending)

    def test_list_empty_status(self):
        queue = MemoryTaskQueue()
        result = run_async(queue.get_tasks_by_status("completed"))
        assert result == []


@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueStatistics:
    """Tests for MemoryTaskQueue.get_stats."""

    def test_stats_empty_queue(self):
        queue = MemoryTaskQueue()
        stats = run_async(queue.get_stats())
        assert stats['pending'] == 0
        assert stats['in_progress'] == 0
        assert stats['completed'] == 0
        assert stats['failed'] == 0
        assert stats['active_pids'] == 0

    def test_stats_with_tasks(self):
        queue = MemoryTaskQueue()
        t1 = Task(task_id="MQ-S1", title="P", status="pending")
        t2 = Task(task_id="MQ-S2", title="IP", status="in_progress", pid=100)
        t3 = Task(task_id="MQ-S3", title="C", status="completed")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        run_async(queue.enqueue(t3))
        stats = run_async(queue.get_stats())
        assert stats['pending'] == 1
        assert stats['in_progress'] == 1
        assert stats['completed'] == 1
        assert stats['active_pids'] == 1
        assert 100 in stats['pids']


@pytest.mark.unit
@pytest.mark.skipif(
    not (MEMORY_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="MemoryTaskQueue or Task not importable",
)
class TestMemoryTaskQueueClear:
    """Tests for MemoryTaskQueue.delete_task (clear-like behavior)."""

    def test_delete_all_tasks(self):
        queue = MemoryTaskQueue()
        t1 = Task(task_id="MQ-C1", title="Del1")
        t2 = Task(task_id="MQ-C2", title="Del2")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        run_async(queue.delete_task("MQ-C1"))
        run_async(queue.delete_task("MQ-C2"))
        assert run_async(queue.get_task("MQ-C1")) is None
        assert run_async(queue.get_task("MQ-C2")) is None

    def test_delete_nonexistent_returns_false(self):
        queue = MemoryTaskQueue()
        result = run_async(queue.delete_task("NOPE"))
        assert result is False


# ===========================================================================
# 6. SQLiteTaskQueue Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueEnqueue:
    """Tests for SQLiteTaskQueue.enqueue."""

    def test_enqueue_returns_task_id(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-001", title="SQLite Enqueue")
        result = run_async(queue.enqueue(task))
        assert result == "SQ-001"

    def test_enqueue_stores_task(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-002", title="Stored")
        run_async(queue.enqueue(task))
        retrieved = run_async(queue.get_task("SQ-002"))
        assert retrieved is not None
        assert retrieved.title == "Stored"


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueDequeue:
    """Tests for SQLiteTaskQueue.dequeue."""

    def test_dequeue_empty_returns_none(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        result = run_async(queue.dequeue())
        assert result is None

    def test_dequeue_returns_highest_priority_pending(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        t1 = Task(task_id="SQ-D1", title="Low", priority=5, status="pending")
        t2 = Task(task_id="SQ-D2", title="High", priority=1, status="pending")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        result = run_async(queue.dequeue())
        assert result is not None
        assert result.task_id == "SQ-D2"


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueUpdateStatus:
    """Tests for SQLiteTaskQueue.update_status."""

    def test_update_status_in_progress(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-U1", title="Update", status="pending")
        run_async(queue.enqueue(task))
        result = run_async(queue.update_status("SQ-U1", "in_progress", pid=5678))
        assert result is True
        updated = run_async(queue.get_task("SQ-U1"))
        assert updated.status == "in_progress"

    def test_update_status_completed(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-U2", title="Complete", status="in_progress")
        run_async(queue.enqueue(task))
        run_async(queue.update_status("SQ-U2", "completed"))
        updated = run_async(queue.get_task("SQ-U2"))
        assert updated.status == "completed"
        assert updated.pid is None

    def test_update_status_failed_with_error(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-U3", title="Fail", status="in_progress")
        run_async(queue.enqueue(task))
        run_async(queue.update_status("SQ-U3", "failed", error="Crashed"))
        updated = run_async(queue.get_task("SQ-U3"))
        assert updated.status == "failed"
        assert updated.error_message == "Crashed"


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueGetTask:
    """Tests for SQLiteTaskQueue.get_task."""

    def test_get_existing(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-G1", title="FindMe")
        run_async(queue.enqueue(task))
        result = run_async(queue.get_task("SQ-G1"))
        assert result is not None
        assert result.title == "FindMe"

    def test_get_nonexistent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        result = run_async(queue.get_task("MISSING"))
        assert result is None


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueListTasks:
    """Tests for SQLiteTaskQueue.get_tasks_by_status."""

    def test_list_by_status(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        t1 = Task(task_id="SQ-L1", title="P1", status="pending")
        t2 = Task(task_id="SQ-L2", title="B1", status="backlog")
        t3 = Task(task_id="SQ-L3", title="P2", status="pending")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        run_async(queue.enqueue(t3))
        pending = run_async(queue.get_tasks_by_status("pending"))
        assert len(pending) == 2

    def test_list_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        result = run_async(queue.get_tasks_by_status("failed"))
        assert result == []


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueStatistics:
    """Tests for SQLiteTaskQueue.get_stats."""

    def test_stats_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        stats = run_async(queue.get_stats())
        assert stats['pending'] == 0
        assert stats['completed'] == 0

    def test_stats_with_tasks(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        t1 = Task(task_id="SQ-S1", title="P", status="pending")
        t2 = Task(task_id="SQ-S2", title="C", status="completed")
        run_async(queue.enqueue(t1))
        run_async(queue.enqueue(t2))
        stats = run_async(queue.get_stats())
        assert stats['pending'] == 1
        assert stats['completed'] == 1


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueClear:
    """Tests for SQLiteTaskQueue.delete_task."""

    def test_delete_task(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task = Task(task_id="SQ-C1", title="DeleteMe")
        run_async(queue.enqueue(task))
        result = run_async(queue.delete_task("SQ-C1"))
        assert result is True
        assert run_async(queue.get_task("SQ-C1")) is None

    def test_delete_nonexistent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        result = run_async(queue.delete_task("NOPE"))
        assert result is False


@pytest.mark.unit
@pytest.mark.skipif(
    not (SQLITE_QUEUE_AVAILABLE and TASK_AVAILABLE),
    reason="SQLiteTaskQueue or Task not importable",
)
class TestSQLiteTaskQueueCreateTask:
    """Tests for SQLiteTaskQueue.create_task."""

    def test_create_task_returns_id(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task_id = run_async(queue.create_task(title="Created", description="desc"))
        assert task_id is not None
        assert task_id.startswith("TASK-")

    def test_create_task_retrievable(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        queue = SQLiteTaskQueue(db_path)
        task_id = run_async(queue.create_task(title="Retrieve", priority=2))
        task = run_async(queue.get_task(task_id))
        assert task is not None
        assert task.title == "Retrieve"
        assert task.priority == 2
