"""
Voice Processing Module
========================

Provides voice-to-voice chat capabilities:
- Speech-to-Text: Groq Whisper (primary), Deepgram (fallback)
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
import tempfile
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

# Whisper prompt to improve recognition of informal speech patterns
WHISPER_PROMPT = """This is a voice assistant conversation. The speaker may use:
- Informal sounds: psst, shh, hmm, uh, um, ah, oh, ooh, wow, huh
- Filler words: like, you know, basically, actually, literally
- Hesitations: uh, um, er, well, so
- Expressions: hey, hi, hello, bye, thanks, please, sorry, okay, yeah, yep, nope
- Onomatopoeia: la la la, tra la la, hmm hmm, boop, beep
"""

# Common misrecognitions to fix in post-processing
ONOMATOPOEIA_CORRECTIONS = {
    # Whispered sounds
    "pust": "psst",
    "pust,": "psst,",
    "shush": "shh",
    "shhh": "shh",
    # Hesitations
    "um,": "um,",
    "erm": "um",
    "er,": "uh,",
    # Common mishearings
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    # Musical sounds (preserve these)
    "la la la": "la la la",
    "tra la la": "tra la la",
    "hmm hmm": "hmm hmm",
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

    Uses Groq Whisper (primary) or Deepgram (fallback) for STT.
    Uses edge-tts for TTS.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._groq_client = None
        self._deepgram_client = None

    @staticmethod
    def _post_process_transcript(text: str) -> str:
        """
        Post-process transcript to fix common misrecognitions.

        Handles:
        - Onomatopoeia corrections (psst, shh, hmm)
        - Case normalization for sentence starts
        - Repeated word cleanup (stuttering artifacts)
        """
        if not text:
            return text

        result = text

        # Apply onomatopoeia corrections (case-insensitive)
        for wrong, correct in ONOMATOPOEIA_CORRECTIONS.items():
            # Replace at word boundaries
            import re
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            result = pattern.sub(correct, result)

        # Fix common Whisper artifacts
        # Remove excessive repeated words (stuttering over-correction)
        result = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', result, flags=re.IGNORECASE)

        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    @property
    def groq_client(self):
        """Lazy-load Groq client for Whisper STT."""
        if self._groq_client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not set - Groq Whisper STT unavailable")
                return None
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                logger.info("Groq Whisper client initialized")
            except ImportError:
                logger.warning("groq not installed. Run: pip install groq")
                return None
        return self._groq_client

    @property
    def deepgram_client(self):
        """Lazy-load Deepgram client (fallback)."""
        if self._deepgram_client is None:
            try:
                from deepgram import DeepgramClient
                api_key = os.environ.get("DEEPGRAM_API_KEY")
                if not api_key:
                    return None
                self._deepgram_client = DeepgramClient(api_key)
                logger.info("Deepgram client initialized (fallback)")
            except ImportError:
                return None
        return self._deepgram_client

    async def speech_to_text(self, audio_data: bytes, mime_type: str = "audio/webm") -> str:
        """
        Convert speech audio to text.

        Tries Groq Whisper first (fast, free tier), falls back to Deepgram.

        Args:
            audio_data: Raw audio bytes
            mime_type: Audio MIME type (audio/webm, audio/wav, etc.)

        Returns:
            Transcribed text
        """
        # Try Groq Whisper first (primary)
        transcript = await self._stt_groq_whisper(audio_data, mime_type)
        if transcript:
            return transcript

        # Fallback to Deepgram
        transcript = await self._stt_deepgram(audio_data, mime_type)
        if transcript:
            return transcript

        logger.warning("All STT providers failed")
        return ""

    async def _stt_groq_whisper(self, audio_data: bytes, mime_type: str) -> str:
        """Transcribe using Groq's Whisper API."""
        client = self.groq_client
        if not client:
            return ""

        try:
            # Determine file extension from mime type
            ext_map = {
                "audio/webm": ".webm",
                "audio/wav": ".wav",
                "audio/mp3": ".mp3",
                "audio/mpeg": ".mp3",
                "audio/ogg": ".ogg",
                "audio/flac": ".flac",
                "audio/m4a": ".m4a",
            }
            ext = ext_map.get(mime_type, ".webm")

            # Groq requires a file-like object with a name
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Transcribe with Groq Whisper
                # Use prompt to improve recognition of informal speech
                with open(temp_path, "rb") as audio_file:
                    response = await asyncio.to_thread(
                        lambda: client.audio.transcriptions.create(
                            file=(f"audio{ext}", audio_file),
                            model="whisper-large-v3",
                            language="en",
                            response_format="text",
                            prompt=WHISPER_PROMPT,  # Helps with onomatopoeia
                            temperature=0.0  # More deterministic
                        )
                    )

                transcript = response.strip() if isinstance(response, str) else str(response).strip()

                # Apply post-processing for onomatopoeia corrections
                transcript = self._post_process_transcript(transcript)

                logger.info(f"Groq Whisper STT: {transcript[:50]}...")
                return transcript
            finally:
                # Clean up temp file
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Groq Whisper STT failed: {e}")
            return ""

    async def _stt_deepgram(self, audio_data: bytes, mime_type: str) -> str:
        """Transcribe using Deepgram (fallback)."""
        client = self.deepgram_client
        if not client:
            return ""

        try:
            from deepgram import PrerecordedOptions

            options = PrerecordedOptions(
                model="nova-2",
                language="en",
                smart_format=True,
                punctuate=True,
                diarize=False,
            )

            response = await asyncio.to_thread(
                lambda: client.listen.prerecorded.v("1").transcribe_file(
                    {"buffer": audio_data, "mimetype": mime_type},
                    options
                )
            )

            transcript = ""
            if response.results and response.results.channels:
                for channel in response.results.channels:
                    for alternative in channel.alternatives:
                        transcript += alternative.transcript + " "

            transcript = transcript.strip()
            logger.info(f"Deepgram STT: {transcript[:50]}...")
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

    async def process_voice_message_streaming(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        process_text_fn=None
    ) -> AsyncIterator[Tuple[str, bytes]]:
        """
        Streaming voice-to-voice pipeline for lower latency.

        Yields audio chunks as soon as first sentence is ready,
        reducing perceived latency significantly.

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            process_text_fn: Async function to process text (your LLM)

        Yields:
            Tuples of (text_chunk, audio_chunk)
        """
        import re

        # 1. Speech to text
        user_text = await self.speech_to_text(audio_data, mime_type)
        if not user_text:
            error_audio = await self.text_to_speech("I couldn't understand that.")
            yield ("I couldn't understand that.", error_audio)
            return

        # 2. Process with LLM
        if process_text_fn:
            response_text = await process_text_fn(user_text)
        else:
            response_text = f"You said: {user_text}"

        # 3. Split into sentences and generate TTS for each
        # This allows streaming audio back sentence by sentence
        sentences = re.split(r'(?<=[.!?])\s+', response_text)

        for sentence in sentences:
            if sentence.strip():
                audio_chunk = await self.text_to_speech(sentence)
                yield (sentence, audio_chunk)

    async def process_voice_fast(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        process_text_fn=None,
        max_response_chars: int = 200
    ) -> Tuple[str, str, bytes]:
        """
        Optimized voice pipeline for minimum latency.

        Optimizations:
        - Limits response length for faster TTS
        - Uses faster speech rate
        - Truncates at sentence boundary

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            process_text_fn: Async function to process text
            max_response_chars: Max characters in response (default 200)

        Returns:
            Tuple of (user_text, response_text, response_audio)
        """
        import re

        # 1. Speech to text
        user_text = await self.speech_to_text(audio_data, mime_type)
        if not user_text:
            return "", "I couldn't understand that.", b""

        # 2. Process with LLM
        if process_text_fn:
            response_text = await process_text_fn(user_text)
        else:
            response_text = f"You said: {user_text}"

        # 3. Truncate at sentence boundary for faster TTS
        if len(response_text) > max_response_chars:
            # Find last sentence boundary before limit
            truncated = response_text[:max_response_chars]
            last_sentence_end = max(
                truncated.rfind('.'),
                truncated.rfind('!'),
                truncated.rfind('?')
            )
            if last_sentence_end > 50:
                response_text = truncated[:last_sentence_end + 1]
            else:
                response_text = truncated + "..."

        # 4. TTS with faster speech rate for quicker delivery
        fast_config = VoiceConfig(
            voice=self.config.voice,
            rate="+15%",  # 15% faster speech
            pitch=self.config.pitch,
            volume=self.config.volume
        )

        try:
            import edge_tts
            communicate = edge_tts.Communicate(
                response_text,
                fast_config.voice,
                rate=fast_config.rate,
                pitch=fast_config.pitch,
                volume=fast_config.volume
            )

            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            response_audio = audio_data.getvalue()
            logger.info(f"Fast TTS: {len(response_audio)} bytes for '{response_text[:30]}...'")

        except Exception as e:
            logger.error(f"Fast TTS failed: {e}")
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
