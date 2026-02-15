"""
Phase 1 SDK Tests
=================

Tests for Phase 1A-1G SDK additions:
- Voice methods (STT, TTS, voice_chat, voice_stream)
- Swarm methods (swarm, swarm_stream)
- Configuration (configure_lm, configure_voice)
- Memory methods (memory_store, memory_retrieve, memory_status)
- Document methods (upload_document, search_documents, chat_with_documents)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import SDK (use Jotty.sdk for proper package resolution)
from Jotty.sdk.client import Jotty, VoiceHandle
from Jotty.core.infrastructure.foundation.types.sdk_types import (
    SDKResponse, SDKVoiceResponse, SDKEvent, SDKEventType, ExecutionMode
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sdk_client():
    """Create SDK client in local mode."""
    client = Jotty().use_local()
    return client


@pytest.fixture
def mock_voice_processor():
    """Mock voice processor for STT/TTS."""
    mock = Mock()
    mock.speech_to_text = AsyncMock(return_value={
        "success": True,
        "text": "Hello world",
        "confidence": 0.95,
        "provider": "groq"
    })
    mock.text_to_speech = AsyncMock(return_value=b"fake_audio_data")
    return mock


@pytest.fixture
def mock_memory():
    """Mock memory system."""
    mock = Mock()
    mock.store = Mock(return_value="mem-123")
    mock.retrieve = Mock(return_value=[
        Mock(content="Previous task", level="episodic", relevance=0.9, metadata={})
    ])
    mock.status = Mock(return_value={
        "backend": "full",
        "total_memories": 42
    })
    return mock


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for swarm."""
    mock = Mock()
    mock.run = AsyncMock(return_value={
        "success": True,
        "final_output": "Swarm result",
        "agents_used": ["researcher", "coder"],
        "steps_executed": 3,
        "metadata": {}
    })
    return mock


# =============================================================================
# VOICE TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_voice_handle_creation(sdk_client):
    """Test VoiceHandle creation via fluent API."""
    voice = sdk_client.voice("en-US-Ava")
    assert isinstance(voice, VoiceHandle)
    assert voice._voice == "en-US-Ava"
    assert voice._stt_provider == "auto"
    assert voice._tts_provider == "auto"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stt_local_mode(sdk_client):
    """Test STT in local mode delegates to modalities."""
    with patch('Jotty.core.interface.modalities.voice.speech_to_text', new_callable=AsyncMock) as mock_stt:
        mock_stt.return_value = "Transcribed text"

        result = await sdk_client.stt(
            audio_data=b"fake_audio",
            mime_type="audio/webm",
            provider="groq"
        )

        assert isinstance(result, SDKVoiceResponse)
        assert result.success
        assert result.user_text == "Transcribed text"
        assert result.mode == ExecutionMode.VOICE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tts_local_mode(sdk_client):
    """Test TTS in local mode delegates to modalities."""
    with patch('Jotty.core.interface.modalities.voice.text_to_speech', new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = b"audio_bytes"

        audio = await sdk_client.tts(
            text="Hello world",
            voice="en-US-Ava",
            provider="edge"
        )

        assert isinstance(audio, bytes)
        assert audio == b"audio_bytes"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_voice_chat_pipeline(sdk_client):
    """Test voice_chat executes STT → chat → TTS pipeline."""
    with patch.object(sdk_client, 'stt') as mock_stt, \
         patch.object(sdk_client, 'chat') as mock_chat, \
         patch.object(sdk_client, 'tts') as mock_tts:

        # Mock STT result
        mock_stt.return_value = SDKVoiceResponse(
            success=True,
            content="User question",
            user_text="User question",
            confidence=0.95,
            mode=ExecutionMode.VOICE
        )

        # Mock chat result
        mock_chat.return_value = SDKResponse(
            success=True,
            content="Assistant answer"
        )

        # Mock TTS result
        mock_tts.return_value = b"response_audio"

        result = await sdk_client.voice_chat(
            audio_data=b"input_audio",
            mime_type="audio/webm"
        )

        assert result.success
        assert result.user_text == "User question"
        assert result.content == "Assistant answer"
        assert result.audio_data == b"response_audio"

        # Verify pipeline execution
        mock_stt.assert_called_once()
        mock_chat.assert_called_once_with("User question", mode="standard")
        mock_tts.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_voices(sdk_client):
    """Test list_voices returns available voices."""
    voices = await sdk_client.list_voices(provider="edge")

    # Should return EdgeTTS voices dict
    assert isinstance(voices, dict)
    # EdgeTTS has multiple voices
    assert len(voices) > 0


# =============================================================================
# SWARM TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_swarm_local_mode(sdk_client, mock_orchestrator):
    """Test swarm execution in local mode."""
    with patch('Jotty.core.intelligence.orchestration.Orchestrator', return_value=mock_orchestrator):
        result = await sdk_client.swarm(
            goal="Research AI trends",
            swarm_type="research"
        )

        assert result.success
        assert result.content == "Swarm result"
        assert result.mode == ExecutionMode.SWARM
        assert result.agents_used == ["researcher", "coder"]
        assert result.steps_executed == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_swarm_stream(sdk_client, mock_orchestrator):
    """Test swarm streaming execution."""
    with patch('Jotty.core.intelligence.orchestration.Orchestrator', return_value=mock_orchestrator):
        events = []
        async for event in sdk_client.swarm_stream(
            goal="Build a web scraper",
            swarm_type="coding"
        ):
            events.append(event)

        # Should emit START and COMPLETE events
        assert len(events) >= 2
        assert events[0].type == SDKEventType.START
        assert events[-1].type == SDKEventType.COMPLETE


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_configure_lm(sdk_client):
    """Test LM configuration via SDK."""
    with patch('Jotty.core.intelligence.orchestration.llm_providers.provider_manager.configure_dspy_lm') as mock_config:
        result = await sdk_client.configure_lm(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )

        assert result["success"]
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-3-5-sonnet-20241022"

        mock_config.assert_called_once_with(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_configure_voice(sdk_client):
    """Test voice provider configuration."""
    result = await sdk_client.configure_voice(
        stt_provider="groq",
        tts_provider="edge"
    )

    assert result["stt_provider"] == "groq"
    assert result["tts_provider"] == "edge"
    assert "groq" in result["available_stt"]
    assert "edge" in result["available_tts"]


# =============================================================================
# MEMORY TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_store(sdk_client, mock_memory):
    """Test storing to memory via SDK."""
    with patch('Jotty.core.intelligence.memory.facade.get_memory_system', return_value=mock_memory):
        result = await sdk_client.memory_store(
            content="Task completed successfully",
            level="episodic",
            goal="research"
        )

        assert result.success
        assert result.content == "mem-123"

        mock_memory.store.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_retrieve(sdk_client, mock_memory):
    """Test retrieving from memory via SDK."""
    with patch('Jotty.core.intelligence.memory.facade.get_memory_system', return_value=mock_memory):
        result = await sdk_client.memory_retrieve(
            query="How to complete task?",
            top_k=5
        )

        assert result.success
        assert len(result.content) == 1
        assert result.content[0]["content"] == "Previous task"
        assert result.content[0]["relevance"] == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_status(sdk_client, mock_memory):
    """Test memory status via SDK."""
    with patch('Jotty.core.intelligence.memory.facade.get_memory_system', return_value=mock_memory):
        result = await sdk_client.memory_status()

        assert result.success
        assert result.content["total_memories"] == 42


# =============================================================================
# DOCUMENT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_document(sdk_client, mock_memory):
    """Test document upload via SDK."""
    with patch('Jotty.core.intelligence.memory.facade.get_memory_system', return_value=mock_memory):
        result = await sdk_client.upload_document(
            file_content=b"Document content here",
            filename="report.txt"
        )

        assert result.success
        assert "doc_id" in result.content
        assert result.content["filename"] == "report.txt"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_documents(sdk_client, mock_memory):
    """Test document search via SDK."""
    with patch('Jotty.core.intelligence.memory.facade.get_memory_system', return_value=mock_memory):
        result = await sdk_client.search_documents(
            query="machine learning",
            top_k=3
        )

        assert result.success
        # Delegates to memory_retrieve
        mock_memory.retrieve.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_with_documents(sdk_client, mock_memory):
    """Test RAG chat via SDK."""
    with patch('Jotty.core.intelligence.memory.facade.get_memory_system', return_value=mock_memory), \
         patch.object(sdk_client, 'chat') as mock_chat:

        mock_chat.return_value = SDKResponse(
            success=True,
            content="Based on the documents..."
        )

        result = await sdk_client.chat_with_documents(
            message="What are the key findings?",
            doc_ids=["doc-1", "doc-2"]
        )

        assert result.success
        mock_chat.assert_called_once()
        # Verify chat was called with document context
        chat_message = mock_chat.call_args[0][0]
        assert "doc-1" in chat_message or "doc-2" in chat_message


# =============================================================================
# SDK EVENT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_sdk_event_sequence_numbers():
    """Test SDKEvent has sequence numbers."""
    event1 = SDKEvent(type=SDKEventType.START, seq=0)
    event2 = SDKEvent(type=SDKEventType.STREAM, seq=1, data="chunk")
    event3 = SDKEvent(type=SDKEventType.COMPLETE, seq=2)

    assert event1.seq == 0
    assert event2.seq == 1
    assert event3.seq == 2

    # Test serialization
    event_dict = event2.to_dict()
    assert event_dict["seq"] == 1
    assert event_dict["data"] == "chunk"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sdk_voice_response_serialization():
    """Test SDKVoiceResponse serialization."""
    response = SDKVoiceResponse(
        success=True,
        content="Hello",
        user_text="User input",
        audio_data=b"audio",
        audio_format="audio/mp3",
        confidence=0.95,
        provider="groq",
        mode=ExecutionMode.VOICE
    )

    response_dict = response.to_dict()
    assert response_dict["success"]
    assert response_dict["user_text"] == "User input"
    assert response_dict["confidence"] == 0.95
    assert response_dict["provider"] == "groq"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
