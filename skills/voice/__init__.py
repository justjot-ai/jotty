"""
Voice Skill
===========

Multi-provider voice capabilities for Jotty.
Supports ElevenLabs (cloud) and Piper/Whisper (local).
"""

from .config import VoiceConfig, get_config, set_config
from .tools import (  # Convenience functions
    close_stream_tool,
    get_stream_chunk_tool,
    list_voices_tool,
    stream_voice,
    stream_voice_tool,
    text_to_voice,
    text_to_voice_tool,
    voice_to_text,
    voice_to_text_tool,
)

__all__ = [
    # Tools
    "voice_to_text_tool",
    "text_to_voice_tool",
    "stream_voice_tool",
    "get_stream_chunk_tool",
    "close_stream_tool",
    "list_voices_tool",
    # Convenience functions
    "voice_to_text",
    "text_to_voice",
    "stream_voice",
    # Config
    "VoiceConfig",
    "get_config",
    "set_config",
]
