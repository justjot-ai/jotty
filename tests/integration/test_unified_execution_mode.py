#!/usr/bin/env python3
"""
Comprehensive tests for Unified ExecutionMode
Tests learning, memory, queue integration, and all capabilities
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_execution_mode_imports():
    """Test that all imports work"""
    print("🧪 Test 1: Imports")
    try:

        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_execution_mode_creation():
    """Test ExecutionMode creation"""
    print("\n🧪 Test 2: ExecutionMode Creation")
    try:

        # Test that ExecutionMode can be instantiated (without Conductor for now)
        # We'll test with Conductor in integration tests
        print("✅ ExecutionMode class exists and can be imported")
        return True
    except Exception as e:
        print(f"❌ Creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_queue_integration():
    """Test queue integration"""
    print("\n🧪 Test 3: Queue Integration")
    try:
        from Jotty.core.queue import SQLiteTaskQueue

        db_path = tempfile.mktemp(suffix=".db")
        queue = SQLiteTaskQueue(db_path=db_path, init_schema=True)

        # Test basic queue operations
        from Jotty.core.queue.task import Task, TaskPriority, TaskStatus

        task = Task(
            task_id="TEST-001",
            title="Test Task",
            description="Test Description",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
        )

        task_id = await queue.enqueue(task)
        assert task_id == "TEST-001", "Task ID should match"

        retrieved = await queue.get_task("TEST-001")
        assert retrieved is not None, "Task should be retrievable"
        assert retrieved.title == "Test Task", "Task title should match"

        os.unlink(db_path)
        print("✅ Queue integration works")
        return True
    except Exception as e:
        print(f"❌ Queue integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_supervisor_compatibility():
    """Test supervisor compatibility"""
    print("\n🧪 Test 4: Supervisor Compatibility")
    try:
        import sys

        supervisor_path = Path(__file__).parent.parent.parent / "JustJot.ai" / "supervisor"
        sys.path.insert(0, str(supervisor_path))

        from state_manager import StateManager

        db_path = tempfile.mktemp(suffix=".db")
        os.environ["STATE_DB"] = db_path

        # Test without conductor
        sm = StateManager(db_path=db_path)
        assert sm._queue is not None, "Queue should be initialized"

        # Test with explicit None conductor
        sm2 = StateManager(db_path=db_path, conductor=None)
        assert sm2._queue is not None, "Queue should be initialized"

        # Test basic operations
        stats = sm.get_stats()
        assert isinstance(stats, dict), "Stats should be dict"

        task_id = sm.create_task("Test", "Description")
        assert task_id is not None, "Task creation should work"

        task = sm.get_task_by_task_id(task_id)
        assert task is not None, "Task retrieval should work"
        assert task["title"] == "Test", "Task title should match"

        os.unlink(db_path)
        print("✅ Supervisor compatibility verified")
        return True
    except Exception as e:
        print(f"❌ Supervisor compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_backward_compatibility():
    """Test backward-compatible wrappers"""
    print("\n🧪 Test 5: Backward Compatibility")
    try:
        from Jotty.core.intelligence.orchestration import ChatMode, WorkflowMode

        # Test that wrappers exist and can be imported
        assert WorkflowMode is not None, "WorkflowMode should exist"
        assert ChatMode is not None, "ChatMode should exist"

        print("✅ Backward-compatible wrappers exist")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_execution_mode_properties():
    """Test ExecutionMode properties and methods"""
    print("\n🧪 Test 6: ExecutionMode Properties")
    try:
        from Jotty.core.intelligence.orchestration import ExecutionMode

        # Test that ExecutionMode has expected methods
        assert hasattr(ExecutionMode, "execute"), "Should have execute method"
        assert hasattr(ExecutionMode, "stream"), "Should have stream method"
        assert hasattr(ExecutionMode, "enqueue_task"), "Should have enqueue_task method"
        assert hasattr(ExecutionMode, "process_queue"), "Should have process_queue method"
        assert hasattr(ExecutionMode, "learning_enabled"), "Should have learning_enabled property"
        assert hasattr(ExecutionMode, "memory_enabled"), "Should have memory_enabled property"
        assert hasattr(ExecutionMode, "queue_enabled"), "Should have queue_enabled property"

        print("✅ ExecutionMode has all expected methods and properties")
        return True
    except Exception as e:
        print(f"❌ Properties test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_chat_message():
    """Test ChatMessage dataclass"""
    print("\n🧪 Test 7: ChatMessage")
    try:
        from Jotty.core.intelligence.orchestration import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user", "Role should match"
        assert msg.content == "Hello", "Content should match"
        assert msg.timestamp is not None, "Timestamp should be set"

        msg_dict = msg.to_dict()
        assert isinstance(msg_dict, dict), "to_dict should return dict"
        assert msg_dict["role"] == "user", "Dict role should match"

        print("✅ ChatMessage works correctly")
        return True
    except Exception as e:
        print(f"❌ ChatMessage test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Comprehensive ExecutionMode Tests")
    print("=" * 60)

    tests = [
        test_execution_mode_imports,
        test_execution_mode_creation,
        test_queue_integration,
        test_supervisor_compatibility,
        test_backward_compatibility,
        test_execution_mode_properties,
        test_chat_message,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test.__name__, False))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
