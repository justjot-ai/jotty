"""
Groq Whisper Provider
====================

Fast cloud STT using Groq's Whisper API.
Groq provides fast, free-tier Whisper transcription.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GroqWhisperProvider:
    """Groq Whisper STT provider (fast, free tier)."""

    name = "groq_whisper"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    async def speech_to_text(
        self, audio_path: str, language: Optional[str] = None, model: str = "whisper-large-v3"
    ) -> Dict[str, Any]:
        """
        Convert speech to text using Groq Whisper API.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            model: Whisper model to use (whisper-large-v3)

        Returns:
            Dict with success, transcribed text, and metadata
        """
        try:
            from groq import Groq
        except ImportError:
            return {
                "success": False,
                "error": "groq package not installed. Install with: pip install groq",
            }

        if not self.api_key:
            return {"success": False, "error": "GROQ_API_KEY not set"}

        audio_file = Path(audio_path)
        if not audio_file.exists():
            return {"success": False, "error": f"Audio file not found: {audio_path}"}

        try:
            client = Groq(api_key=self.api_key)

            with open(audio_path, "rb") as f:
                kwargs = {
                    "model": model,
                    "file": (audio_file.name, f, "audio/mpeg"),
                    "response_format": "json",
                }
                if language:
                    kwargs["language"] = language

                # Add prompt for better recognition of informal speech
                kwargs[
                    "prompt"
                ] = """Transcribe ALL sounds including whispered and informal speech.
Examples: psst, shh, hmm, uh, um, oh, wow, yeah, nope, hey, la la la.
Do NOT skip or filter out any sounds."""

                response = client.audio.transcriptions.create(**kwargs)

            # Post-process for common misrecognitions
            text = response.text
            text = self._post_process_transcript(text)

            return {
                "success": True,
                "text": text,
                "language": language or "auto",
                "provider": self.name,
                "model": model,
            }

        except Exception as e:
            logger.error(f"Groq Whisper STT error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _post_process_transcript(text: str) -> str:
        """Post-process transcript to fix common misrecognitions."""
        if not text:
            return text

        # Common corrections
        corrections = {
            "pust": "psst",
            "pust,": "psst,",
            "shush": "shh",
            "shhh": "shh",
            "ew,": "ooh,",
            "ew ": "ooh ",
            "erm": "um",
            "er,": "uh,",
        }

        for old, new in corrections.items():
            text = text.replace(old, new)

        return text

    async def text_to_speech(
        self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None
    ):
        """Groq doesn't provide TTS."""
        return {
            "success": False,
            "error": "Groq does not provide TTS. Use OpenAI, ElevenLabs, or edge-tts instead.",
        }

    async def stream_speech(
        self, text: str, voice_id: Optional[str] = None, chunk_size: int = 1024
    ):
        """Groq doesn't provide TTS streaming."""
        raise NotImplementedError("Groq does not provide TTS")
