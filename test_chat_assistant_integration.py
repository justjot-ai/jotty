#!/usr/bin/env python3
"""
Test ChatAssistant Integration with State Manager

This test simulates the full integration:
1. Create a mock state manager (like SQLiteTaskQueue)
2. Initialize ChatAPI with ChatAssistant
3. Query for tasks
4. Verify A2UI widgets are returned
"""

import sys
import asyncio
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.api.chat_api import ChatAPI
from core.orchestration.conductor import Conductor
from core.foundation.data_structures import JottyConfig


class MockStateManager:
    """Mock state manager that simulates SQLiteTaskQueue."""

    async def get_tasks_by_status(self, status: str):
        """Mock task query by status."""
        mock_tasks = {
            'backlog': [
                MockTask(
                    task_id='TASK-20260118-00001',
                    title='Test minimal supervisor with real agent',
                    description='Verify supervisor works with real agents',
                    status='backlog',
                    priority=3,
                    created_at='2026-01-18T10:00:00Z'
                )
            ],
            'completed': [
                MockTask(
                    task_id='TASK-20260117-00001',
                    title='Enhancements in Supervisor Dashboard',
                    status='completed'
                ),
                MockTask(
                    task_id='TASK-20260116-00002',
                    title='Enhancement of Browse Page',
                    status='completed'
                )
            ],
            'in_progress': []
        }
        return mock_tasks.get(status, [])

    async def export_to_json(self):
        """Mock export for fallback testing."""
        return {
            'task_details': {
                'TASK-20260118-00001': {
                    'task_id': 'TASK-20260118-00001',
                    'title': 'Test minimal supervisor with real agent',
                    'status': 'backlog',
                    'priority': 3,
                    'created_at': '2026-01-18T10:00:00Z'
                }
            },
            'backlog_tasks': ['TASK-20260118-00001'],
            'completed_task_files': [],
            'in_progress_tasks': []
        }


class MockTask:
    """Mock task object."""
    def __init__(self, task_id, title, status, description='', priority=3, created_at=None):
        self.task_id = task_id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.created_at = created_at


async def test_chat_assistant_integration():
    """Test ChatAssistant with state manager integration."""
    print("=" * 70)
    print("ChatAssistant Integration Test")
    print("=" * 70)

    # 1. Create mock state manager
    print("\n1. Creating mock state manager...")
    state_manager = MockStateManager()
    print("   ✅ Mock state manager created")

    # 2. Create minimal conductor (no agents needed - ChatAssistant auto-registers!)
    print("\n2. Creating Conductor...")

    # Create minimal metadata provider
    class MinimalMetadataProvider:
        def get_tools(self, actor_name=None):
            return []
        def get_tool_names(self):
            return []

    conductor = Conductor(
        actors=[],  # Empty! ChatAssistant will be auto-registered
        metadata_provider=MinimalMetadataProvider(),
        config=JottyConfig(),
        enable_data_registry=False
    )
    print(f"   ✅ Conductor created with {len(conductor.actors)} actors")

    # 3. Create ChatAPI with ChatAssistant (auto-registration)
    print("\n3. Creating ChatAPI with agent_id='ChatAssistant'...")
    print("   (This will auto-register ChatAssistant agent)")
    chat_api = ChatAPI(
        conductor=conductor,
        agent_id="ChatAssistant",
        state_manager=state_manager  # ← KEY: Pass state_manager!
    )
    print(f"   ✅ ChatAPI created, conductor now has {len(conductor.actors)} actors")
    print(f"   Actors: {list(conductor.actors.keys())}")

    # 4. Test backlog query
    print("\n4. Testing query: 'How many tasks in backlog?'")
    try:
        result = await chat_api.send(
            message="How many tasks in backlog?",
            history=[]
        )

        print("\n   Response structure:")
        print(f"   - Type: {type(result)}")
        print(f"   - Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

        # Check if A2UI format
        if isinstance(result, dict) and 'role' in result and 'content' in result:
            print("\n   ✅ A2UI FORMAT DETECTED!")
            print(f"   - Role: {result['role']}")
            print(f"   - Content blocks: {len(result['content'])}")

            # Inspect content
            for i, block in enumerate(result['content']):
                block_type = block.get('type', 'unknown')
                print(f"\n   Content Block {i+1}:")
                print(f"   - Type: {block_type}")

                if block_type == 'card':
                    print(f"   - Title: {block.get('title')}")
                    print(f"   - Subtitle: {block.get('subtitle')}")
                    body = block.get('body', {})
                    if isinstance(body, dict) and body.get('type') == 'list':
                        items = body.get('items', [])
                        print(f"   - List items: {len(items)}")
                        for item in items:
                            print(f"     • {item.get('title')} ({item.get('status')})")

                elif block_type == 'list':
                    items = block.get('items', [])
                    print(f"   - Items: {len(items)}")
                    for item in items:
                        print(f"     • {item.get('title')} ({item.get('status')})")

            print("\n   ✅ SUCCESS: ChatAssistant returned A2UI widgets!")
            return True

        else:
            print("\n   ❌ UNEXPECTED FORMAT:")
            import json
            print(json.dumps(result, indent=2))
            return False

    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming():
    """Test streaming with A2UI widgets."""
    print("\n" + "=" * 70)
    print("Streaming Test")
    print("=" * 70)

    class MinimalMetadataProvider:
        def get_tools(self, actor_name=None):
            return []
        def get_tool_names(self):
            return []

    state_manager = MockStateManager()
    conductor = Conductor(
        actors=[],
        metadata_provider=MinimalMetadataProvider(),
        config=JottyConfig(),
        enable_data_registry=False
    )
    chat_api = ChatAPI(
        conductor=conductor,
        agent_id="ChatAssistant",
        state_manager=state_manager
    )

    print("\nStreaming query: 'Show completed tasks'")
    print("\nEvents received:")

    try:
        event_count = 0
        a2ui_widget_received = False

        async for event in chat_api.stream(message="Show completed tasks", history=[]):
            event_count += 1
            event_type = event.get('type', 'unknown')
            print(f"   {event_count}. {event_type}")

            if event_type == 'a2ui_widget':
                a2ui_widget_received = True
                content = event.get('content', [])
                print(f"      → A2UI widget with {len(content)} blocks")

        if a2ui_widget_received:
            print("\n   ✅ SUCCESS: A2UI widget event received in stream!")
            return True
        else:
            print("\n   ⚠️  WARNING: No A2UI widget event in stream")
            return False

    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🧪 Testing ChatAssistant Integration with State Manager\n")

    # Test 1: Basic integration
    test1 = await test_chat_assistant_integration()

    # Test 2: Streaming
    test2 = await test_streaming()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Basic Integration: {'PASSED' if test1 else 'FAILED'}")
    print(f"✅ Streaming: {'PASSED' if test2 else 'FAILED'}")
    print("=" * 70)

    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED! A2UI integration is working!")
        print("\n📋 Next: Deploy to supervisor container and test with real UI")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
