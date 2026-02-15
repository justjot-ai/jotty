"""
Voice Skill Configuration
=========================

Configuration for multi-provider voice capabilities.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VoiceConfig:
    """Voice skill configuration."""

    # Provider selection
    local_mode: bool = False  # Force local-only inference
    default_tts_provider: str = "auto"  # "elevenlabs", "local", "auto"
    default_stt_provider: str = "auto"  # "whisper", "local", "auto"

    # ElevenLabs config
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_default_voice: str = "Rachel"
    elevenlabs_model_id: str = "eleven_monolingual_v1"

    # OpenAI Whisper config
    openai_api_key: Optional[str] = None
    whisper_model: str = "whisper-1"

    # Local Whisper.cpp config
    whisper_cpp_path: Optional[str] = None
    whisper_model_path: Optional[str] = None
    whisper_language: str = "en"

    # Local Piper TTS config
    piper_path: Optional[str] = None
    piper_voice_path: Optional[str] = None

    # Audio settings
    sample_rate: int = 22050
    audio_format: str = "mp3"  # Default output format

    # Streaming settings
    chunk_size: int = 1024
    stream_buffer_size: int = 4096

    def __post_init__(self):
        """Load from environment if not set."""
        self.elevenlabs_api_key = self.elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.whisper_cpp_path = self.whisper_cpp_path or os.getenv("WHISPER_CPP_PATH")
        self.whisper_model_path = self.whisper_model_path or os.getenv("WHISPER_MODEL_PATH")
        self.piper_path = self.piper_path or os.getenv("PIPER_PATH")
        self.piper_voice_path = self.piper_voice_path or os.getenv("PIPER_VOICE_PATH")
        self.local_mode = self.local_mode or os.getenv("JOTTY_LOCAL_MODE", "").lower() == "true"

    @property
    def has_elevenlabs(self) -> bool:
        """Check if ElevenLabs is available."""
        return bool(self.elevenlabs_api_key) and not self.local_mode

    @property
    def has_whisper_api(self) -> bool:
        """Check if OpenAI Whisper API is available."""
        return bool(self.openai_api_key) and not self.local_mode

    @property
    def has_local_whisper(self) -> bool:
        """Check if local Whisper is available (Python package or whisper.cpp)."""
        # Check Python whisper package first (openai-whisper)
        try:
            import whisper  # noqa: F401

            return True
        except ImportError:
            pass
        # Fall back to whisper.cpp binary
        if self.whisper_cpp_path and self.whisper_model_path:
            return Path(self.whisper_cpp_path).exists() and Path(self.whisper_model_path).exists()
        return False

    @property
    def has_local_piper(self) -> bool:
        """Check if local Piper TTS is available."""
        # Check if piper-tts is installed as Python package
        try:
            import piper  # noqa: F401

            return True
        except ImportError:
            pass
        # Check if custom piper binary exists
        if self.piper_path and self.piper_voice_path:
            return Path(self.piper_path).exists() and Path(self.piper_voice_path).exists()
        return False


# Global config instance
_config: Optional[VoiceConfig] = None


def get_config() -> VoiceConfig:
    """Get voice configuration singleton."""
    global _config
    if _config is None:
        _config = VoiceConfig()
    return _config


def set_config(config: VoiceConfig) -> None:
    """Set voice configuration."""
    global _config
    _config = config
