"""
Voice Providers
===============

Multi-provider voice support with intelligent auto-selection.

STT Providers (priority order):
1. Groq Whisper (fast, free tier) - RECOMMENDED
2. OpenAI Whisper (reliable, paid)
3. Local Whisper (offline, privacy)

TTS Providers (priority order):
1. Edge TTS (free, high quality) - RECOMMENDED
2. OpenAI TTS (paid, high quality)
3. ElevenLabs (paid, ultra high quality)
4. Local Piper (offline, privacy)
"""

import logging
import os
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Import config with fallback
try:
    from ..config import get_config
except ImportError:
    try:
        import importlib.util
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("voice_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            get_config = config_module.get_config
    except Exception:
        # Minimal fallback config
        class MinimalConfig:
            has_elevenlabs = False
            has_local_piper = False
            has_whisper_api = False
            has_local_whisper = False
            has_groq_whisper = lambda self: bool(os.getenv("GROQ_API_KEY"))
            has_edge_tts = True  # edge-tts is free, no API key needed

        def get_config():
            return MinimalConfig()


class VoiceProviderBase:
    """Base class for voice providers."""

    name: str = "base"

    async def text_to_speech(
        self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> dict:
        """Convert text to speech."""
        raise NotImplementedError

    async def speech_to_text(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Convert speech to text."""
        raise NotImplementedError

    async def stream_speech(
        self, text: str, voice_id: Optional[str] = None, chunk_size: int = 1024
    ) -> AsyncIterator[bytes]:
        """Stream text-to-speech audio."""
        raise NotImplementedError


def get_tts_provider(provider: str = "auto") -> VoiceProviderBase:
    """
    Get TTS provider based on selection and availability.

    Priority (auto mode):
    1. Edge TTS (free, high quality) - BEST DEFAULT
    2. OpenAI TTS (paid, very high quality)
    3. ElevenLabs (paid, ultra high quality)
    4. Local Piper (offline)
    """
    from .edge_tts import EdgeTTSProvider
    from .elevenlabs import ElevenLabsProvider
    from .local import LocalProvider
    from .whisper import WhisperProvider  # Has OpenAI TTS

    config = get_config()

    if provider == "auto":
        # Priority: edge-tts (free) > OpenAI > ElevenLabs > local
        try:
            # Try Edge TTS first (free, no API key needed)
            return EdgeTTSProvider()
        except Exception:
            pass

        if config.has_whisper_api:  # OpenAI has TTS too
            return WhisperProvider()
        elif config.has_elevenlabs:
            return ElevenLabsProvider()
        elif config.has_local_piper:
            return LocalProvider()
        else:
            # Fallback to Edge TTS (should always work)
            logger.warning("No preferred TTS provider, using Edge TTS (free)")
            return EdgeTTSProvider()

    elif provider == "edge" or provider == "edge-tts":
        return EdgeTTSProvider()

    elif provider == "openai" or provider == "whisper":
        if not config.has_whisper_api:
            raise RuntimeError("OpenAI not available. Set OPENAI_API_KEY")
        return WhisperProvider()

    elif provider == "elevenlabs":
        if not config.has_elevenlabs:
            raise RuntimeError("ElevenLabs not available. Set ELEVENLABS_API_KEY")
        return ElevenLabsProvider()

    elif provider == "local":
        if not config.has_local_piper:
            raise RuntimeError("Local TTS not available. Install piper-tts")
        return LocalProvider()

    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


def get_stt_provider(provider: str = "auto") -> VoiceProviderBase:
    """
    Get STT provider based on selection and availability.

    Priority (auto mode):
    1. Groq Whisper (fast, free tier) - BEST DEFAULT
    2. OpenAI Whisper (reliable, paid)
    3. Local Whisper (offline)
    """
    from .groq_whisper import GroqWhisperProvider
    from .local import LocalProvider
    from .whisper import WhisperProvider

    config = get_config()

    if provider == "auto":
        # Priority: Groq (fast+free) > OpenAI > local
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return GroqWhisperProvider(groq_key)
        elif config.has_whisper_api:
            return WhisperProvider()
        elif config.has_local_whisper:
            return LocalProvider()
        else:
            raise RuntimeError(
                "No STT provider available. Set GROQ_API_KEY or OPENAI_API_KEY or install whisper.cpp"
            )

    elif provider == "groq" or provider == "groq-whisper":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("Groq not available. Set GROQ_API_KEY")
        return GroqWhisperProvider(groq_key)

    elif provider == "openai" or provider == "whisper":
        if not config.has_whisper_api:
            raise RuntimeError("OpenAI Whisper not available. Set OPENAI_API_KEY")
        return WhisperProvider()

    elif provider == "local":
        if not config.has_local_whisper:
            raise RuntimeError("Local STT not available. Install whisper.cpp")
        return LocalProvider()

    else:
        raise ValueError(f"Unknown STT provider: {provider}")


__all__ = [
    "VoiceProviderBase",
    "get_tts_provider",
    "get_stt_provider",
]
