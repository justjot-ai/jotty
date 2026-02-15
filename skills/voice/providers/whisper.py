"""
OpenAI Whisper Voice Provider
=============================

Cloud STT provider using OpenAI Whisper API.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WhisperProvider:
    """OpenAI Whisper STT provider."""

    name = "whisper"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def speech_to_text(
        self, audio_path: str, language: Optional[str] = None, model: str = "whisper-1"
    ) -> Dict[str, Any]:
        """
        Convert speech to text using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            model: Whisper model to use

        Returns:
            Dict with success, transcribed text, and metadata
        """
        try:
            import openai
        except ImportError:
            return {
                "success": False,
                "error": "openai package not installed. Install with: pip install openai",
            }

        if not self.api_key:
            return {"success": False, "error": "OPENAI_API_KEY not set"}

        audio_file = Path(audio_path)
        if not audio_file.exists():
            return {"success": False, "error": f"Audio file not found: {audio_path}"}

        try:
            client = openai.OpenAI(api_key=self.api_key)

            with open(audio_path, "rb") as f:
                kwargs = {
                    "model": model,
                    "file": f,
                }
                if language:
                    kwargs["language"] = language

                response = client.audio.transcriptions.create(**kwargs)

            return {
                "success": True,
                "text": response.text,
                "language": language or "auto",
                "provider": self.name,
                "model": model,
            }

        except Exception as e:
            logger.error(f"Whisper STT error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def text_to_speech(
        self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """OpenAI TTS using their newer TTS API."""
        try:
            import base64

            import openai
        except ImportError:
            return {
                "success": False,
                "error": "openai package not installed. Install with: pip install openai",
            }

        if not self.api_key:
            return {"success": False, "error": "OPENAI_API_KEY not set"}

        # OpenAI TTS voice options
        voice = voice_id or "alloy"  # alloy, echo, fable, onyx, nova, shimmer

        try:
            client = openai.OpenAI(api_key=self.api_key)

            response = client.audio.speech.create(model="tts-1", voice=voice, input=text)

            audio_data = response.content

            result = {"success": True, "format": "mp3", "provider": self.name, "voice_id": voice}

            if output_path:
                Path(output_path).write_bytes(audio_data)
                result["audio_path"] = output_path
            else:
                result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")

            return result

        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stream_speech(
        self, text: str, voice_id: Optional[str] = None, chunk_size: int = 1024
    ):
        """OpenAI TTS streaming is not supported in the same way."""
        # For now, generate full audio and yield in chunks
        result = await self.text_to_speech(text, voice_id)
        if result.get("success") and result.get("audio_base64"):
            import base64

            audio_data = base64.b64decode(result["audio_base64"])
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i : i + chunk_size]
