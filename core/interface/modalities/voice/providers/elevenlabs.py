"""
ElevenLabs Voice Provider
=========================

Cloud TTS provider using ElevenLabs API.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class ElevenLabsProvider:
    """ElevenLabs TTS provider."""

    name = "elevenlabs"

    # Voice ID mappings for common voice names
    VOICE_MAP = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "adam": "pNInz6obpgDQGcFmaJgB",
        "josh": "TxGEqnHWrfWFTfGW9XjX",
        "bella": "EXAVITQu4vr4xnSDxMaL",
        "elli": "MF3mGyEYCl7XYWbV9V6O",
        "sam": "yoZ06aMxZJJ28mfd3POQ",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"

    def _get_voice_id(self, voice_id: Optional[str]) -> str:
        """Resolve voice name to ID."""
        if not voice_id:
            return self.VOICE_MAP.get("rachel", "21m00Tcm4TlvDq8ikWAM")
        # Check if it's a name we know
        voice_lower = voice_id.lower()
        if voice_lower in self.VOICE_MAP:
            return self.VOICE_MAP[voice_lower]
        # Assume it's already a voice ID
        return voice_id

    async def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        output_path: Optional[str] = None,
        model_id: str = "eleven_monolingual_v1",
    ) -> Dict[str, Any]:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to synthesize
            voice_id: Voice ID or name
            output_path: Optional path to save audio
            model_id: ElevenLabs model ID

        Returns:
            Dict with success, audio data, and metadata
        """
        try:
            import httpx
        except ImportError:
            return {
                "success": False,
                "error": "httpx package not installed. Install with: pip install httpx",
            }

        if not self.api_key:
            return {"success": False, "error": "ELEVENLABS_API_KEY not set"}

        voice = self._get_voice_id(voice_id)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech/{voice}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": model_id,
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                    timeout=60.0,
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"ElevenLabs API error: {response.status_code} - {response.text}",
                    }

                audio_data = response.content

                result = {
                    "success": True,
                    "format": "mp3",
                    "provider": self.name,
                    "voice_id": voice,
                }

                if output_path:
                    Path(output_path).write_bytes(audio_data)
                    result["audio_path"] = output_path
                else:
                    result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")

                return result

        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stream_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        chunk_size: int = 1024,
        model_id: str = "eleven_monolingual_v1",
    ) -> AsyncIterator[bytes]:
        """
        Stream text-to-speech audio.

        Args:
            text: Text to synthesize
            voice_id: Voice ID or name
            chunk_size: Size of audio chunks
            model_id: ElevenLabs model ID

        Yields:
            Audio data chunks
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx package not installed")
            return

        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY not set")
            return

        voice = self._get_voice_id(voice_id)

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/text-to-speech/{voice}/stream",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": model_id,
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"ElevenLabs stream error: {response.status_code}")
                        return

                    async for chunk in response.aiter_bytes(chunk_size):
                        yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs stream error: {e}", exc_info=True)

    async def speech_to_text(
        self, audio_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """ElevenLabs doesn't provide STT - use Whisper instead."""
        return {
            "success": False,
            "error": "ElevenLabs does not provide speech-to-text. Use Whisper provider.",
        }
