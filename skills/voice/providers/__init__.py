"""
Voice Providers
===============

Multi-provider voice support following OpenClaw patterns.
Auto-selects based on availability and configuration.
"""

from typing import AsyncIterator, Optional

# Import config with fallback for standalone loading
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
        # Provide a minimal fallback config
        class MinimalConfig:
            has_elevenlabs = False
            has_local_piper = False
            has_whisper_api = False
            has_local_whisper = False

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
    """Get TTS provider based on selection and availability."""
    from .elevenlabs import ElevenLabsProvider
    from .local import LocalProvider

    config = get_config()

    if provider == "auto":
        # Auto-select based on availability
        if config.has_elevenlabs:
            return ElevenLabsProvider()
        elif config.has_local_piper:
            return LocalProvider()
        else:
            raise RuntimeError(
                "No TTS provider available. Install piper-tts or set ELEVENLABS_API_KEY"
            )

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
    """Get STT provider based on selection and availability."""
    from .local import LocalProvider
    from .whisper import WhisperProvider

    config = get_config()

    if provider == "auto":
        # Auto-select based on availability
        if config.has_whisper_api:
            return WhisperProvider()
        elif config.has_local_whisper:
            return LocalProvider()
        else:
            raise RuntimeError(
                "No STT provider available. Set OPENAI_API_KEY or install whisper.cpp"
            )

    elif provider == "whisper":
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
