"""
Speech-to-Text (STT)
====================

Unified STT interface using multiple providers with auto-selection.

Priority (auto mode):
1. Groq Whisper (fast, free tier)
2. OpenAI Whisper (reliable, paid)
3. Local Whisper (offline)
"""

import logging
from typing import Any, Dict, Optional

from .providers import get_stt_provider

logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Speech-to-text converter with multi-provider support.

    Auto-selects best available provider or use specific provider.
    """

    def __init__(self, platform: str = "generic", provider: str = "auto"):
        """
        Initialize STT converter.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
            provider: STT provider (auto, groq, whisper, local)
        """
        self.platform = platform
        self.provider_name = provider
        self._provider = None

    def _get_provider(self):
        """Get or create provider instance."""
        if self._provider is None:
            self._provider = get_stt_provider(self.provider_name)
        return self._provider

    async def transcribe(
        self, audio_file: str, language: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_file: Path to audio file or file-like object
            language: Language code (e.g., 'en', 'es', 'fr')
            **kwargs: Provider-specific options

        Returns:
            Dict with success, text, language, provider, model
        """
        provider = self._get_provider()
        return await provider.speech_to_text(audio_file, language, **kwargs)


async def speech_to_text(
    audio_file: str,
    platform: str = "generic",
    provider: str = "auto",
    language: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function to convert speech to text.

    Args:
        audio_file: Path to audio file
        platform: Platform name
        provider: STT provider (auto, groq, whisper, local)
        language: Language code
        **kwargs: Provider-specific options

    Returns:
        Transcribed text

    Raises:
        RuntimeError: If transcription fails
    """
    stt = SpeechToText(platform, provider)
    result = await stt.transcribe(audio_file, language, **kwargs)

    if not result.get("success"):
        raise RuntimeError(result.get("error", "Unknown STT error"))

    return result["text"]
