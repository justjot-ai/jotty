"""
Edge TTS Provider
=================

Free TTS using Microsoft Edge's neural voices.
No API key required, high quality neural voices.
"""

import base64
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class EdgeTTSProvider:
    """Microsoft Edge TTS provider (free, no API key)."""

    name = "edge_tts"

    # Available voices (high quality neural voices)
    VOICES = {
        # English
        "en-us-female": "en-US-AvaNeural",  # Natural, warm
        "en-us-male": "en-US-AndrewNeural",  # Professional
        "en-gb-female": "en-GB-SoniaNeural",  # British
        "en-gb-male": "en-GB-RyanNeural",  # British male
        "en-au-female": "en-AU-NatashaNeural",  # Australian
        # Spanish
        "es-es-female": "es-ES-ElviraNeural",
        "es-mx-female": "es-MX-DaliaNeural",
        # French
        "fr-fr-female": "fr-FR-DeniseNeural",
        # German
        "de-de-female": "de-DE-KatjaNeural",
        # Italian
        "it-it-female": "it-IT-ElsaNeural",
        # Portuguese
        "pt-br-female": "pt-BR-FranciscaNeural",
        # Chinese
        "zh-cn-female": "zh-CN-XiaoxiaoNeural",
        # Japanese
        "ja-jp-female": "ja-JP-NanamiNeural",
        # Korean
        "ko-kr-female": "ko-KR-SunHiNeural",
        # Hindi
        "hi-in-female": "hi-IN-SwaraNeural",
        # Arabic
        "ar-sa-female": "ar-SA-ZariyahNeural",
    }

    # Language to voice mapping for auto-detection
    LANGUAGE_VOICES = {
        "en": "en-US-AvaNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "it": "it-IT-ElsaNeural",
        "pt": "pt-BR-FranciscaNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "hi": "hi-IN-SwaraNeural",
        "ar": "ar-SA-ZariyahNeural",
    }

    async def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        output_path: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Dict[str, Any]:
        """
        Convert text to speech using Edge TTS.

        Args:
            text: Text to synthesize
            voice_id: Voice ID (e.g., "en-US-AvaNeural") or friendly name
            output_path: Optional path to save audio
            rate: Speech rate (-50% to +100%)
            pitch: Pitch adjustment
            volume: Volume adjustment

        Returns:
            Dict with success, audio data, and metadata
        """
        try:
            import asyncio

            import edge_tts
        except ImportError:
            return {
                "success": False,
                "error": "edge-tts package not installed. Install with: pip install edge-tts",
            }

        # Resolve voice
        voice = self._resolve_voice(voice_id)

        try:
            # Generate speech
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)

            # Stream to bytes
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            result = {"success": True, "format": "mp3", "provider": self.name, "voice_id": voice}

            if output_path:
                Path(output_path).write_bytes(audio_data)
                result["audio_path"] = output_path
            else:
                result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")

            return result

        except Exception as e:
            logger.error(f"Edge TTS error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stream_speech(
        self, text: str, voice_id: Optional[str] = None, chunk_size: int = 1024
    ) -> AsyncIterator[bytes]:
        """
        Stream text-to-speech audio in chunks.

        Args:
            text: Text to synthesize
            voice_id: Voice ID or friendly name
            chunk_size: Size of audio chunks to yield

        Yields:
            Audio data chunks
        """
        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed")
            return

        voice = self._resolve_voice(voice_id)

        try:
            communicate = edge_tts.Communicate(text, voice)

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"Edge TTS streaming error: {e}", exc_info=True)
            raise

    async def speech_to_text(self, audio_path: str, language: Optional[str] = None):
        """Edge TTS doesn't provide STT."""
        return {
            "success": False,
            "error": "Edge TTS does not provide speech-to-text. Use Whisper, Groq, or Deepgram.",
        }

    def _resolve_voice(self, voice_id: Optional[str]) -> str:
        """Resolve friendly name or language code to full voice ID."""
        if not voice_id:
            return "en-US-AvaNeural"  # Default

        # Check if it's a friendly name
        if voice_id.lower() in self.VOICES:
            return self.VOICES[voice_id.lower()]

        # Check if it's a language code
        if voice_id.lower() in self.LANGUAGE_VOICES:
            return self.LANGUAGE_VOICES[voice_id.lower()]

        # Assume it's already a full voice ID
        return voice_id
