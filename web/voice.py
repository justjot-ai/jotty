"""
Voice Processing Module
========================

Provides voice-to-voice chat capabilities:
- Speech-to-Text: Deepgram (streaming, low latency)
- Text-to-Speech: edge-tts (Microsoft neural voices, free)

Usage:
    from .voice import VoiceProcessor

    processor = VoiceProcessor()

    # Convert speech to text
    text = await processor.speech_to_text(audio_bytes)

    # Convert text to speech
    audio = await processor.text_to_speech("Hello world")
"""

import os
import io
import logging
import asyncio
from typing import Optional, AsyncIterator, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Available voices for edge-tts (high quality neural voices)
VOICES = {
    "en-us-female": "en-US-AvaNeural",      # Natural, warm
    "en-us-male": "en-US-AndrewNeural",     # Professional
    "en-gb-female": "en-GB-SoniaNeural",    # British
    "en-gb-male": "en-GB-RyanNeural",       # British male
    "en-au-female": "en-AU-NatashaNeural",  # Australian
    "en-in-female": "en-IN-NeerjaNeural",   # Indian English
}

DEFAULT_VOICE = "en-US-AvaNeural"


@dataclass
class VoiceConfig:
    """Voice processing configuration."""
    voice: str = DEFAULT_VOICE
    rate: str = "+0%"      # Speech rate: -50% to +100%
    pitch: str = "+0Hz"    # Pitch adjustment
    volume: str = "+0%"    # Volume adjustment


class VoiceProcessor:
    """
    Handles voice-to-voice processing.

    Uses Deepgram for STT and edge-tts for TTS.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._deepgram_client = None

    @property
    def deepgram_client(self):
        """Lazy-load Deepgram client."""
        if self._deepgram_client is None:
            try:
                from deepgram import DeepgramClient

                api_key = os.environ.get("DEEPGRAM_API_KEY")
                if not api_key:
                    logger.warning("DEEPGRAM_API_KEY not set - STT will not work")
                    return None

                self._deepgram_client = DeepgramClient(api_key)
                logger.info("Deepgram client initialized")
            except ImportError:
                logger.warning("deepgram-sdk not installed. Run: pip install deepgram-sdk")
                return None
        return self._deepgram_client

    async def speech_to_text(self, audio_data: bytes, mime_type: str = "audio/webm") -> str:
        """
        Convert speech audio to text using Deepgram.

        Args:
            audio_data: Raw audio bytes
            mime_type: Audio MIME type (audio/webm, audio/wav, etc.)

        Returns:
            Transcribed text
        """
        client = self.deepgram_client
        if not client:
            # Fallback: try browser-based transcription hint
            logger.warning("Deepgram not available, returning empty transcription")
            return ""

        try:
            from deepgram import PrerecordedOptions

            options = PrerecordedOptions(
                model="nova-2",          # Best quality model
                language="en",
                smart_format=True,       # Punctuation, formatting
                punctuate=True,
                diarize=False,           # Single speaker
            )

            # Transcribe
            response = await asyncio.to_thread(
                lambda: client.listen.prerecorded.v("1").transcribe_file(
                    {"buffer": audio_data, "mimetype": mime_type},
                    options
                )
            )

            # Extract transcript
            transcript = ""
            if response.results and response.results.channels:
                for channel in response.results.channels:
                    for alternative in channel.alternatives:
                        transcript += alternative.transcript + " "

            transcript = transcript.strip()
            logger.info(f"STT result: {transcript[:50]}...")
            return transcript

        except Exception as e:
            logger.error(f"Deepgram STT failed: {e}")
            return ""

    async def text_to_speech(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Convert text to speech using edge-tts.

        Args:
            text: Text to speak
            voice: Voice ID (optional, uses default)

        Returns:
            MP3 audio bytes
        """
        if not text.strip():
            return b""

        try:
            import edge_tts

            voice_id = voice or self.config.voice

            # Create communicate object with voice settings
            communicate = edge_tts.Communicate(
                text,
                voice_id,
                rate=self.config.rate,
                pitch=self.config.pitch,
                volume=self.config.volume
            )

            # Collect audio chunks
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            audio_bytes = audio_data.getvalue()
            logger.info(f"TTS generated: {len(audio_bytes)} bytes for '{text[:30]}...'")
            return audio_bytes

        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return b""
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return b""

    async def text_to_speech_stream(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """
        Stream TTS audio chunks for lower latency.

        Yields audio chunks as they're generated.
        """
        if not text.strip():
            return

        try:
            import edge_tts

            voice_id = voice or self.config.voice
            communicate = edge_tts.Communicate(
                text,
                voice_id,
                rate=self.config.rate,
                pitch=self.config.pitch,
                volume=self.config.volume
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"TTS stream failed: {e}")

    async def process_voice_message(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        process_text_fn=None
    ) -> Tuple[str, str, bytes]:
        """
        Full voice-to-voice pipeline.

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            process_text_fn: Async function to process text (your LLM)

        Returns:
            Tuple of (user_text, response_text, response_audio)
        """
        # 1. Speech to text
        user_text = await self.speech_to_text(audio_data, mime_type)
        if not user_text:
            return "", "I couldn't understand that. Please try again.", b""

        # 2. Process with LLM
        if process_text_fn:
            response_text = await process_text_fn(user_text)
        else:
            response_text = f"You said: {user_text}"

        # 3. Text to speech
        response_audio = await self.text_to_speech(response_text)

        return user_text, response_text, response_audio

    @staticmethod
    def get_available_voices() -> dict:
        """Get list of available voices."""
        return VOICES.copy()


# Global processor instance
_processor: Optional[VoiceProcessor] = None


def get_voice_processor(config: Optional[VoiceConfig] = None) -> VoiceProcessor:
    """Get or create the global voice processor."""
    global _processor
    if _processor is None:
        _processor = VoiceProcessor(config)
    return _processor
