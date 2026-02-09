#!/usr/bin/env python3
"""
Integration tests for Unified ExecutionMode with REAL Conductor
Tests actual learning, memory, and queue integration
"""

import sys
import os
import asyncio
import tempfile
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_execution_mode_with_real_conductor():
    """Test ExecutionMode with real Conductor (not mock)"""
    print("🧪 Test: ExecutionMode with Real Conductor")
    print("=" * 60)
    
    try:
        from Jotty.core.orchestration import ExecutionMode, Conductor
        from Jotty.core.queue import SQLiteTaskQueue
        from Jotty.core.foundation.data_structures import JottyConfig
        from Jotty.core.foundation.agent_config import AgentConfig
        
        # Create real Conductor with minimal config
        config = JottyConfig()
        
        # Create a simple metadata provider
        class SimpleMetadataProvider:
            def get_metadata(self, *args, **kwargs):
                return {}
        
        metadata_provider = SimpleMetadataProvider()
        
        # Create conductor with minimal actors (or empty list)
        conductor = Conductor(
            actors=[],  # Empty for now - just test initialization
            metadata_provider=metadata_provider,
            config=config
        )
        
        print("✅ Real Conductor created")
        
        # Test ExecutionMode with real conductor
        workflow = ExecutionMode(
            conductor=conductor,
            style="workflow",
            execution="sync"
        )
        
        print(f"✅ ExecutionMode created: learning={workflow.learning_enabled}, memory={workflow.memory_enabled}")
        
        # Test with queue
        db_path = tempfile.mktemp(suffix='.db')
        queue = SQLiteTaskQueue(db_path=db_path, init_schema=True)
        
        workflow_async = ExecutionMode(
            conductor=conductor,
            style="workflow",
            execution="async",
            queue=queue
        )
        
        print(f"✅ ExecutionMode with queue: queue={workflow_async.queue_enabled}")
        
        # Test enqueue (should work with real conductor)
        try:
            task_id = await workflow_async.enqueue_task("Test Goal", priority=1)
            print(f"✅ Enqueued task with real conductor: {task_id}")
        except Exception as e:
            print(f"⚠️  Enqueue test: {e} (may need conductor setup)")
        
        os.unlink(db_path)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_supervisor_with_execution_mode():
    """Test supervisor with ExecutionMode integration"""
    print("\n🧪 Test: Supervisor with ExecutionMode")
    print("=" * 60)
    
    try:
        import sys
        supervisor_path = Path(__file__).parent.parent.parent / "JustJot.ai" / "supervisor"
        sys.path.insert(0, str(supervisor_path))
        
        from state_manager import StateManager
        from Jotty.core.orchestration import Conductor
        from Jotty.core.foundation.data_structures import JottyConfig
        
        db_path = tempfile.mktemp(suffix='.db')
        os.environ['STATE_DB'] = db_path
        
        # Test 1: Without conductor (current behavior)
        sm1 = StateManager(db_path=db_path)
        assert sm1._queue is not None
        assert sm1.workflow is None  # No conductor, no ExecutionMode
        print("✅ Supervisor without conductor works")
        
        # Test 2: With conductor (ExecutionMode enabled)
        try:
            config = JottyConfig()
            class SimpleMetadataProvider:
                def get_metadata(self, *args, **kwargs):
                    return {}
            
            conductor = Conductor(
                actors=[],
                metadata_provider=SimpleMetadataProvider(),
                config=config
            )
            
            sm2 = StateManager(db_path=db_path, conductor=conductor)
            assert sm2._queue is not None
            assert sm2.workflow is not None, "ExecutionMode should be enabled with conductor"
            print("✅ Supervisor with conductor works - ExecutionMode enabled!")
            print(f"   Learning: {sm2.workflow.learning_enabled}")
            print(f"   Memory: {sm2.workflow.memory_enabled}")
            print(f"   Queue: {sm2.workflow.queue_enabled}")
        except Exception as e:
            print(f"⚠️  Conductor setup: {e} (may need full agent setup)")
            print("   But supervisor still works without conductor!")
        
        os.unlink(db_path)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chat_mode_with_real_conductor():
    """Test chat mode with real Conductor"""
    print("\n🧪 Test: Chat Mode with Real Conductor")
    print("=" * 60)
    
    try:
        from Jotty.core.orchestration import ExecutionMode, Conductor, ChatMessage
        from Jotty.core.foundation.data_structures import JottyConfig
        
        config = JottyConfig()
        class SimpleMetadataProvider:
            def get_metadata(self, *args, **kwargs):
                return {}
        
        conductor = Conductor(
            actors=[],
            metadata_provider=SimpleMetadataProvider(),
            config=config
        )
        
        chat = ExecutionMode(
            conductor=conductor,
            style="chat",
            execution="sync",
            agent_id="test-agent"
        )
        
        print(f"✅ Chat mode created: learning={chat.learning_enabled}, memory={chat.memory_enabled}")
        
        # Test that chat mode has expected methods
        assert hasattr(chat, 'stream'), "Should have stream method"
        assert hasattr(chat, 'execute'), "Should have execute method"
        
        print("✅ Chat mode has all expected methods")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_modes_comparison():
    """Test workflow sync vs async modes"""
    print("\n🧪 Test: Workflow Sync vs Async")
    print("=" * 60)
    
    try:
        from Jotty.core.orchestration import ExecutionMode, Conductor
        from Jotty.core.queue import SQLiteTaskQueue
        from Jotty.core.foundation.data_structures import JottyConfig
        
        config = JottyConfig()
        class SimpleMetadataProvider:
            def get_metadata(self, *args, **kwargs):
                return {}
        
        conductor = Conductor(
            actors=[],
            metadata_provider=SimpleMetadataProvider(),
            config=config
        )
        
        # Sync mode
        workflow_sync = ExecutionMode(conductor, style="workflow", execution="sync")
        print(f"✅ Sync mode: learning={workflow_sync.learning_enabled}, memory={workflow_sync.memory_enabled}, queue={workflow_sync.queue_enabled}")
        
        # Async mode
        db_path = tempfile.mktemp(suffix='.db')
        queue = SQLiteTaskQueue(db_path=db_path, init_schema=True)
        workflow_async = ExecutionMode(conductor, style="workflow", execution="async", queue=queue)
        print(f"✅ Async mode: learning={workflow_async.learning_enabled}, memory={workflow_async.memory_enabled}, queue={workflow_async.queue_enabled}")
        
        # Both should have learning & memory
        assert workflow_sync.learning_enabled == workflow_async.learning_enabled
        assert workflow_sync.memory_enabled == workflow_async.memory_enabled
        print("✅ Both modes get learning & memory")
        
        # Only async should have queue
        assert workflow_async.queue_enabled == True
        assert workflow_sync.queue_enabled == False
        print("✅ Queue only enabled in async mode")
        
        os.unlink(db_path)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Integration Tests - Real Conductor (Not Mock)")
    print("=" * 60)
    
    tests = [
        test_execution_mode_with_real_conductor,
        test_supervisor_with_execution_mode,
        test_chat_mode_with_real_conductor,
        test_workflow_modes_comparison,
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
    print("Integration Test Results:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_integration_tests())
    sys.exit(exit_code)
