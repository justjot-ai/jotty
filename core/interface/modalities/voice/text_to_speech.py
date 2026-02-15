"""
Text-to-Speech (TTS)
====================

Unified TTS interface using multiple providers with auto-selection.

Priority (auto mode):
1. Edge TTS (free, high quality)
2. OpenAI TTS (paid, very high quality)
3. ElevenLabs (paid, ultra high quality)
4. Local Piper (offline)
"""

import logging
from typing import Any, AsyncIterator, Dict, Optional

from .providers import get_tts_provider

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-speech synthesizer with multi-provider support.

    Auto-selects best available provider or use specific provider.
    """

    def __init__(self, platform: str = "generic", provider: str = "auto"):
        """
        Initialize TTS synthesizer.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
            provider: TTS provider (auto, edge, openai, elevenlabs, local)
        """
        self.platform = platform
        self.provider_name = provider
        self._provider = None

    def _get_provider(self):
        """Get or create provider instance."""
        if self._provider is None:
            self._provider = get_tts_provider(self.provider_name)
        return self._provider

    async def synthesize(self, text: str, voice: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Synthesize text to speech.

        Args:
            text: Text to convert to speech
            voice: Voice ID/name (provider-specific)
            **kwargs: Provider-specific options (speed, pitch, etc.)

        Returns:
            Dict with success, audio_base64 or audio_path, format, provider, voice_id
        """
        provider = self._get_provider()
        return await provider.text_to_speech(text, voice, **kwargs)

    async def stream(
        self, text: str, voice: Optional[str] = None, chunk_size: int = 1024
    ) -> AsyncIterator[bytes]:
        """
        Stream text-to-speech audio in chunks.

        Args:
            text: Text to convert to speech
            voice: Voice ID/name
            chunk_size: Size of audio chunks

        Yields:
            Audio data chunks
        """
        provider = self._get_provider()
        async for chunk in provider.stream_speech(text, voice, chunk_size):
            yield chunk


async def text_to_speech(
    text: str,
    platform: str = "generic",
    provider: str = "auto",
    voice: Optional[str] = None,
    **kwargs,
) -> bytes:
    """
    Convenience function to convert text to speech.

    Args:
        text: Text to convert
        platform: Platform name
        provider: TTS provider (auto, edge, openai, elevenlabs, local)
        voice: Voice ID/name
        **kwargs: Provider-specific options

    Returns:
        Audio data as bytes

    Raises:
        RuntimeError: If synthesis fails
    """
    tts = TextToSpeech(platform, provider)
    result = await tts.synthesize(text, voice, **kwargs)

    if not result.get("success"):
        raise RuntimeError(result.get("error", "Unknown TTS error"))

    # Return audio bytes
    if "audio_base64" in result:
        import base64

        return base64.b64decode(result["audio_base64"])
    elif "audio_path" in result:
        from pathlib import Path

        return Path(result["audio_path"]).read_bytes()
    else:
        raise RuntimeError("No audio data in TTS result")
