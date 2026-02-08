"""
Voice Skill
===========

Multi-provider voice capabilities for Jotty.
Supports ElevenLabs (cloud) and Piper/Whisper (local).
"""

from .tools import (
    voice_to_text_tool,
    text_to_voice_tool,
    stream_voice_tool,
    get_stream_chunk_tool,
    close_stream_tool,
    list_voices_tool,
    # Convenience functions
    voice_to_text,
    text_to_voice,
    stream_voice,
)

from .config import VoiceConfig, get_config, set_config

__all__ = [
    # Tools
    'voice_to_text_tool',
    'text_to_voice_tool',
    'stream_voice_tool',
    'get_stream_chunk_tool',
    'close_stream_tool',
    'list_voices_tool',
    # Convenience functions
    'voice_to_text',
    'text_to_voice',
    'stream_voice',
    # Config
    'VoiceConfig',
    'get_config',
    'set_config',
]
