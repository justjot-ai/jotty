"""
Tests for streaming thinking events in LLM providers.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass


@pytest.mark.unit
class TestThinkingBlock:
    """Test ThinkingBlock dataclass."""

    def test_thinking_block_creation(self):
        from Jotty.core.intelligence.orchestration.llm_providers.types import ThinkingBlock

        block = ThinkingBlock(thinking="Let me think...")
        assert block.thinking == "Let me think..."
        assert block.type == "thinking"

    def test_thinking_block_custom_type(self):
        from Jotty.core.intelligence.orchestration.llm_providers.types import ThinkingBlock

        block = ThinkingBlock(thinking="hmm", type="custom")
        assert block.type == "custom"


@pytest.mark.unit
class TestAnthropicStreamingThinking:
    """Test Anthropic provider streaming with thinking events."""

    @pytest.mark.asyncio
    async def test_text_delta_dispatched_to_stream_callback(self):
        from Jotty.core.intelligence.orchestration.llm_providers.anthropic import (
            AnthropicProvider,
        )

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-sonnet-4-20250514"

        # Mock stream events
        @dataclass
        class Delta:
            type: str
            text: str = ""
            thinking: str = ""

        @dataclass
        class Event:
            type: str
            delta: Delta = None

        text_event = Event(
            type="content_block_delta",
            delta=Delta(type="text_delta", text="Hello world"),
        )

        # Mock final response
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello world"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter([text_event]))
        mock_stream.get_final_message.return_value = mock_response

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        provider._client = mock_client

        stream_chunks = []
        thinking_chunks = []

        def on_text(text):
            stream_chunks.append(text)

        def on_thinking(text):
            thinking_chunks.append(text)

        result, full_content = await provider.call_streaming(
            messages=[],
            tools=[],
            system="test",
            stream_callback=on_text,
            thinking_callback=on_thinking,
        )

        assert stream_chunks == ["Hello world"]
        assert thinking_chunks == []
        assert full_content == "Hello world"

    @pytest.mark.asyncio
    async def test_thinking_delta_dispatched_to_thinking_callback(self):
        from Jotty.core.intelligence.orchestration.llm_providers.anthropic import (
            AnthropicProvider,
        )

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-sonnet-4-20250514"

        @dataclass
        class Delta:
            type: str
            text: str = ""
            thinking: str = ""

        @dataclass
        class Event:
            type: str
            delta: Delta = None

        thinking_event = Event(
            type="content_block_delta",
            delta=Delta(type="thinking_delta", thinking="I need to reason..."),
        )
        text_event = Event(
            type="content_block_delta",
            delta=Delta(type="text_delta", text="Answer"),
        )

        mock_block_thinking = MagicMock()
        mock_block_thinking.type = "thinking"
        mock_block_thinking.thinking = "I need to reason..."

        mock_block_text = MagicMock()
        mock_block_text.type = "text"
        mock_block_text.text = "Answer"

        mock_response = MagicMock()
        mock_response.content = [mock_block_thinking, mock_block_text]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter([thinking_event, text_event]))
        mock_stream.get_final_message.return_value = mock_response

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        provider._client = mock_client

        stream_chunks = []
        thinking_chunks = []

        result, full_content = await provider.call_streaming(
            messages=[],
            tools=[],
            system="test",
            stream_callback=lambda t: stream_chunks.append(t),
            thinking_callback=lambda t: thinking_chunks.append(t),
        )

        assert stream_chunks == ["Answer"]
        assert thinking_chunks == ["I need to reason..."]
        assert full_content == "Answer"

    @pytest.mark.asyncio
    async def test_thinking_without_callback_is_silent(self):
        """No thinking_callback → thinking events silently ignored."""
        from Jotty.core.intelligence.orchestration.llm_providers.anthropic import (
            AnthropicProvider,
        )

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "claude-sonnet-4-20250514"

        @dataclass
        class Delta:
            type: str
            text: str = ""
            thinking: str = ""

        @dataclass
        class Event:
            type: str
            delta: Delta = None

        thinking_event = Event(
            type="content_block_delta",
            delta=Delta(type="thinking_delta", thinking="Thinking..."),
        )
        text_event = Event(
            type="content_block_delta",
            delta=Delta(type="text_delta", text="Output"),
        )

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Output"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=5)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter([thinking_event, text_event]))
        mock_stream.get_final_message.return_value = mock_response

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        provider._client = mock_client

        stream_chunks = []

        # No thinking_callback passed → default None
        result, full_content = await provider.call_streaming(
            messages=[],
            tools=[],
            system="test",
            stream_callback=lambda t: stream_chunks.append(t),
        )

        assert stream_chunks == ["Output"]
        assert full_content == "Output"


@pytest.mark.unit
class TestParseResponseThinking:
    """Test _parse_response handles ThinkingBlock."""

    def test_parse_response_includes_thinking_block(self):
        from Jotty.core.intelligence.orchestration.llm_providers.anthropic import (
            AnthropicProvider,
        )
        from Jotty.core.intelligence.orchestration.llm_providers.types import ThinkingBlock

        provider = AnthropicProvider.__new__(AnthropicProvider)

        mock_thinking = MagicMock()
        mock_thinking.type = "thinking"
        mock_thinking.thinking = "Deep thought"

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Answer"

        mock_response = MagicMock()
        mock_response.content = [mock_thinking, mock_text]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        result = provider._parse_response(mock_response)
        assert len(result.content) == 2
        assert isinstance(result.content[0], ThinkingBlock)
        assert result.content[0].thinking == "Deep thought"
        assert result.content[1].text == "Answer"


@pytest.mark.unit
class TestOpenAIThinkingCallbackCompat:
    """Test OpenAI provider accepts thinking_callback param."""

    @pytest.mark.asyncio
    async def test_openai_accepts_thinking_callback(self):
        from Jotty.core.intelligence.orchestration.llm_providers.openai import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.model = "gpt-4o"

        # Mock client
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hi"
        mock_chunk.choices[0].delta.tool_calls = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        provider._client = mock_client

        chunks = []
        result, content = await provider.call_streaming(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            system="test",
            stream_callback=lambda t: chunks.append(t),
            thinking_callback=lambda t: None,  # Should be accepted and ignored
        )
        assert "Hi" in content


@pytest.mark.unit
class TestGoogleThinkingCallbackCompat:
    """Test Google provider accepts thinking_callback param."""

    def test_google_call_streaming_signature(self):
        """Verify GoogleProvider.call_streaming accepts thinking_callback."""
        import inspect
        from Jotty.core.intelligence.orchestration.llm_providers.google import GoogleProvider

        sig = inspect.signature(GoogleProvider.call_streaming)
        assert "thinking_callback" in sig.parameters
        # Should have default of None
        assert sig.parameters["thinking_callback"].default is None
