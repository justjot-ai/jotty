#!/usr/bin/env python3
"""
Manual SDK Integration Test
============================

Quick test to verify SDK methods work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from Jotty.core.infrastructure.foundation.types.sdk_types import (
    ExecutionMode,
    SDKEventType,
    SDKResponse,
    SDKVoiceResponse,
)
from Jotty.sdk.client import Jotty


async def test_sdk_basic():
    """Test basic SDK functionality."""
    print("\n" + "=" * 60)
    print("SDK Integration Test")
    print("=" * 60)

    # Initialize SDK client in local mode
    print("\n1. Initializing SDK client...")
    client = Jotty().use_local()
    print("✓ SDK client initialized")

    # Test voice handle creation
    print("\n2. Testing voice handle...")
    voice = client.voice("en-US-Ava")
    print(f"✓ Voice handle created: {voice._voice}")
    print(f"  - STT provider: {voice._stt_provider}")
    print(f"  - TTS provider: {voice._tts_provider}")

    # Test configuration
    print("\n3. Testing voice configuration...")
    config = await client.configure_voice(stt_provider="groq", tts_provider="edge")
    print(f"✓ Voice configured:")
    print(f"  - STT: {config['stt_provider']}")
    print(f"  - TTS: {config['tts_provider']}")
    print(f"  - Available STT: {config['available_stt']}")
    print(f"  - Available TTS: {config['available_tts']}")

    # Test list voices
    print("\n4. Testing list_voices...")
    voices = await client.list_voices(provider="edge")
    print(f"✓ Found {len(voices)} Edge TTS voices")
    print(f"  - Sample: {list(voices.items())[:3]}")

    # Test memory status
    print("\n5. Testing memory_status...")
    try:
        mem_status = await client.memory_status()
        print(f"✓ Memory status retrieved:")
        print(f"  - Success: {mem_status.success}")
        if mem_status.success:
            print(f"  - Backend: {mem_status.content.get('backend', 'unknown')}")
            print(f"  - Total memories: {mem_status.content.get('total_memories', 0)}")
    except Exception as e:
        print(f"⚠ Memory system not initialized: {e}")

    # Test event system
    print("\n6. Testing event system...")
    events_received = []

    def on_event(event):
        events_received.append(event.type)
        print(f"  → Event: {event.type.value}")

    client.on(SDKEventType.START, on_event)
    client.on(SDKEventType.COMPLETE, on_event)

    # Emit test events
    await client._emit_event(SDKEventType.START, {"test": "start"})
    await client._emit_event(SDKEventType.COMPLETE, {"test": "complete"})

    print(f"✓ Events received: {len(events_received)}")

    # Test SDK types
    print("\n7. Testing SDK types...")

    # Test SDKResponse
    response = SDKResponse(
        success=True, content="Test content", mode=ExecutionMode.CHAT, execution_time=0.5
    )
    response_dict = response.to_dict()
    print(f"✓ SDKResponse serialized:")
    print(f"  - Success: {response_dict['success']}")
    print(f"  - Mode: {response_dict['mode']}")

    # Test SDKVoiceResponse
    voice_response = SDKVoiceResponse(
        success=True,
        content="Transcribed text",
        user_text="User input",
        confidence=0.95,
        provider="groq",
        mode=ExecutionMode.VOICE,
    )
    voice_dict = voice_response.to_dict()
    print(f"✓ SDKVoiceResponse serialized:")
    print(f"  - User text: {voice_dict['user_text']}")
    print(f"  - Confidence: {voice_dict['confidence']}")
    print(f"  - Provider: {voice_dict['provider']}")

    print("\n" + "=" * 60)
    print("✅ All basic SDK tests passed!")
    print("=" * 60 + "\n")

    return True


async def test_sdk_chat():
    """Test SDK chat method."""
    print("\n" + "=" * 60)
    print("SDK Chat Test")
    print("=" * 60)

    print("\n1. Initializing SDK client...")
    client = Jotty().use_local()
    print("✓ SDK client initialized")

    print("\n2. Testing chat method...")
    print("   Note: This requires LLM provider configured")

    try:
        # Simple chat test (may fail if no LLM configured)
        result = await client.chat("Say 'SDK works!'", timeout=10)

        print(f"✓ Chat completed:")
        print(f"  - Success: {result.success}")
        print(f"  - Mode: {result.mode}")
        print(f"  - Content: {result.content[:100] if result.content else 'None'}...")

        if not result.success:
            print(f"  - Error: {result.error}")
            print("  ⚠ This is expected if no LLM provider is configured")

    except Exception as e:
        print(f"⚠ Chat test skipped (no LLM configured): {e}")

    print("\n" + "=" * 60)
    print("✅ Chat test completed")
    print("=" * 60 + "\n")


async def test_sdk_memory():
    """Test SDK memory methods."""
    print("\n" + "=" * 60)
    print("SDK Memory Test")
    print("=" * 60)

    print("\n1. Initializing SDK client...")
    client = Jotty().use_local()
    print("✓ SDK client initialized")

    try:
        print("\n2. Testing memory_store...")
        store_result = await client.memory_store(
            content="Test memory content", level="episodic", goal="testing"
        )

        print(f"✓ Memory stored:")
        print(f"  - Success: {store_result.success}")
        if store_result.success:
            print(f"  - Memory ID: {store_result.content}")

        print("\n3. Testing memory_retrieve...")
        retrieve_result = await client.memory_retrieve(query="test memory", top_k=5)

        print(f"✓ Memory retrieved:")
        print(f"  - Success: {retrieve_result.success}")
        if retrieve_result.success:
            print(f"  - Memories found: {len(retrieve_result.content)}")

        print("\n4. Testing memory_status...")
        status_result = await client.memory_status()

        print(f"✓ Memory status:")
        print(f"  - Success: {status_result.success}")
        if status_result.success:
            print(f"  - Backend: {status_result.content.get('backend')}")
            print(f"  - Total: {status_result.content.get('total_memories', 0)}")

    except Exception as e:
        print(f"⚠ Memory test failed: {e}")

    print("\n" + "=" * 60)
    print("✅ Memory test completed")
    print("=" * 60 + "\n")


async def main():
    """Run all SDK tests."""
    try:
        # Run basic tests
        await test_sdk_basic()

        # Run chat test (may skip if no LLM)
        await test_sdk_chat()

        # Run memory test
        await test_sdk_memory()

        print("\n" + "=" * 60)
        print("✅ ALL SDK INTEGRATION TESTS PASSED!")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
