"""
Local Voice Providers
=====================

Local-first voice providers using Whisper.cpp (STT) and Piper (TTS).
No external API calls - full privacy.
"""

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class LocalProvider:
    """Local voice provider using Whisper.cpp and Piper."""

    name = "local"

    def __init__(self):
        from ..config import get_config

        self.config = get_config()

    async def speech_to_text(
        self, audio_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert speech to text using local Whisper.cpp.

        Args:
            audio_path: Path to audio file
            language: Optional language code

        Returns:
            Dict with success, transcribed text, and metadata
        """
        # Try Python whisper package first
        try:
            return await self._whisper_python(audio_path, language)
        except ImportError:
            pass

        # Fall back to whisper.cpp binary
        if self.config.whisper_cpp_path and self.config.whisper_model_path:
            return await self._whisper_cpp(audio_path, language)

        return {
            "success": False,
            "error": "No local STT available. Install whisper package or whisper.cpp",
        }

    async def _whisper_python(
        self, audio_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use OpenAI whisper Python package (runs locally)."""
        import whisper

        audio_file = Path(audio_path)
        if not audio_file.exists():
            return {"success": False, "error": f"Audio file not found: {audio_path}"}

        try:
            # Load model (cached after first load)
            model = whisper.load_model("base")

            # Transcribe
            kwargs = {}
            if language:
                kwargs["language"] = language

            result = model.transcribe(str(audio_path), **kwargs)

            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", language or "auto"),
                "provider": f"{self.name}/whisper-python",
            }

        except Exception as e:
            logger.error(f"Local Whisper error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _whisper_cpp(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Use whisper.cpp binary for transcription."""
        audio_file = Path(audio_path)
        if not audio_file.exists():
            return {"success": False, "error": f"Audio file not found: {audio_path}"}

        try:
            # Build command
            cmd = [
                self.config.whisper_cpp_path,
                "-m",
                self.config.whisper_model_path,
                "-f",
                str(audio_path),
                "--no-timestamps",
            ]

            if language:
                cmd.extend(["-l", language])

            # Run whisper.cpp
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return {"success": False, "error": f"whisper.cpp error: {result.stderr}"}

            return {
                "success": True,
                "text": result.stdout.strip(),
                "language": language or "auto",
                "provider": f"{self.name}/whisper-cpp",
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "whisper.cpp timed out"}
        except Exception as e:
            logger.error(f"whisper.cpp error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def text_to_speech(
        self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert text to speech using Piper TTS.

        Args:
            text: Text to synthesize
            voice_id: Voice model path or name
            output_path: Optional path to save audio

        Returns:
            Dict with success, audio data, and metadata
        """
        # Try piper-tts Python package first
        try:
            return await self._piper_python(text, voice_id, output_path)
        except ImportError:
            pass

        # Fall back to piper binary
        if self.config.piper_path and self.config.piper_voice_path:
            return await self._piper_binary(text, voice_id, output_path)

        return {
            "success": False,
            "error": "No local TTS available. Install piper-tts or set PIPER_PATH",
        }

    async def _piper_python(
        self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use piper-tts Python package."""
        import wave

        from piper import PiperVoice

        try:
            # Use provided voice or default
            voice_path = voice_id or self.config.piper_voice_path
            if not voice_path:
                return {
                    "success": False,
                    "error": "No Piper voice model specified. Set PIPER_VOICE_PATH",
                }

            voice = PiperVoice.load(voice_path)

            # Generate audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)

                for audio_bytes in voice.synthesize_stream_raw(text):
                    wav_file.writeframes(audio_bytes)

            # Read the generated audio
            audio_data = Path(temp_path).read_bytes()

            result = {"success": True, "format": "wav", "provider": f"{self.name}/piper-python"}

            if output_path:
                Path(output_path).write_bytes(audio_data)
                result["audio_path"] = output_path
            else:
                result["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")

            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)

            return result

        except Exception as e:
            logger.error(f"Piper TTS error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _piper_binary(
        self, text: str, voice_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use piper binary for TTS."""
        try:
            voice_path = voice_id or self.config.piper_voice_path

            # Create temp output file if needed
            if output_path:
                out_file = output_path
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    out_file = f.name

            # Run piper
            cmd = [self.config.piper_path, "--model", voice_path, "--output_file", out_file]

            result = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return {"success": False, "error": f"Piper error: {result.stderr}"}

            audio_data = Path(out_file).read_bytes()

            result_dict = {
                "success": True,
                "format": "wav",
                "provider": f"{self.name}/piper-binary",
            }

            if output_path:
                result_dict["audio_path"] = output_path
            else:
                result_dict["audio_base64"] = base64.b64encode(audio_data).decode("utf-8")
                # Cleanup temp file
                Path(out_file).unlink(missing_ok=True)

            return result_dict

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Piper timed out"}
        except Exception as e:
            logger.error(f"Piper error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stream_speech(
        self, text: str, voice_id: Optional[str] = None, chunk_size: int = 1024
    ) -> AsyncIterator[bytes]:
        """
        Stream text-to-speech audio using Piper.

        Piper supports streaming naturally through synthesize_stream_raw.
        """
        try:
            from piper import PiperVoice

            voice_path = voice_id or self.config.piper_voice_path
            if not voice_path:
                logger.error("No Piper voice model specified")
                return

            voice = PiperVoice.load(voice_path)

            # Piper's synthesize_stream_raw yields audio chunks
            for audio_bytes in voice.synthesize_stream_raw(text):
                # Yield in requested chunk sizes
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i : i + chunk_size]

        except ImportError:
            # Fall back to non-streaming
            result = await self.text_to_speech(text, voice_id)
            if result.get("success") and result.get("audio_base64"):
                audio_data = base64.b64decode(result["audio_base64"])
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i : i + chunk_size]

        except Exception as e:
            logger.error(f"Piper stream error: {e}", exc_info=True)
