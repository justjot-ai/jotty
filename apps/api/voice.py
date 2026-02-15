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

import asyncio
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Set via environment variables
# =============================================================================
# LOCAL_WHISPER=1          - Use local faster-whisper instead of Groq API
# WHISPER_MODEL=base       - Model size: tiny, base, small, medium, large-v3
# SPECULATIVE_TTS=1        - Enable speculative TTS (start before LLM finishes)
# WEBSOCKET_VOICE=1        - Use WebSocket for audio streaming
# =============================================================================

# Available voices for edge-tts (high quality neural voices)
VOICES = {
    # English
    "en-us-female": "en-US-AvaNeural",  # Natural, warm
    "en-us-male": "en-US-AndrewNeural",  # Professional
    "en-gb-female": "en-GB-SoniaNeural",  # British
    "en-gb-male": "en-GB-RyanNeural",  # British male
    "en-au-female": "en-AU-NatashaNeural",  # Australian
    "en-in-female": "en-IN-NeerjaNeural",  # Indian English
    # Spanish
    "es-es-female": "es-ES-ElviraNeural",
    "es-mx-female": "es-MX-DaliaNeural",
    # French
    "fr-fr-female": "fr-FR-DeniseNeural",
    "fr-ca-female": "fr-CA-SylvieNeural",
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

# Whisper prompt to improve recognition of informal speech patterns
# IMPORTANT: Include examples of sounds to help Whisper recognize them
WHISPER_PROMPT = """Transcribe ALL sounds including whispered and informal speech.
Examples of sounds to preserve:
- Whispered: "psst", "shh", "shush"
- Thinking: "hmm", "uh", "um", "er"
- Exclamations: "oh", "ooh", "ah", "wow", "oops", "yay"
- Reactions: "huh", "meh", "ugh", "eww", "yuck"
- Agreement: "yeah", "yep", "uh-huh", "mm-hmm"
- Disagreement: "nah", "nope", "uh-uh"
- Greetings: "hey", "hi", "hello", "yo", "sup"
- Musical: "la la la", "tra la la", "do re mi", "hmm hmm"
- Fillers: "like", "you know", "basically", "I mean"
IMPORTANT: Do NOT skip or filter out any sounds. Transcribe everything spoken.
"""

# Common misrecognitions to fix in post-processing
ONOMATOPOEIA_CORRECTIONS = {
    # Whispered sounds
    "pust": "psst",
    "pust,": "psst,",
    "shush": "shh",
    "shhh": "shh",
    # Exclamations (common mishearings)
    "ew,": "ooh,",
    "ew ": "ooh ",
    "you,": "ooh,",  # Sometimes "ooh" → "you"
    # Hesitations
    "erm": "um",
    "er,": "uh,",
    "ah,": "ah,",
    # Common contractions/informal speech
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "coulda": "could have",
    "woulda": "would have",
    "shoulda": "should have",
    "oughta": "ought to",
    "outta": "out of",
    "lotsa": "lots of",
    "betcha": "bet you",
    "gotcha": "got you",
    "whatcha": "what are you",
    # Musical sounds (preserve these)
    "la la la": "la la la",
    "tra la la": "tra la la",
    "hmm hmm": "hmm hmm",
    # Greeting variations
    "hiya": "hi",
    "heya": "hey",
}

DEFAULT_VOICE = "en-US-AvaNeural"


@dataclass
class VoiceConfig:
    """Voice processing configuration."""

    voice: str = DEFAULT_VOICE
    rate: str = "+0%"  # Speech rate: -50% to +100%
    pitch: str = "+0Hz"  # Pitch adjustment
    volume: str = "+0%"  # Volume adjustment
    # New low-latency features (configurable)
    use_local_whisper: bool = field(
        default_factory=lambda: os.environ.get("LOCAL_WHISPER", "0") == "1"
    )
    whisper_model: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "base"))
    speculative_tts: bool = field(
        default_factory=lambda: os.environ.get("SPECULATIVE_TTS", "0") == "1"
    )
    websocket_voice: bool = field(
        default_factory=lambda: os.environ.get("WEBSOCKET_VOICE", "0") == "1"
    )


class VoiceProcessor:
    """
    Handles voice-to-voice processing.

    STT options (configurable):
    - Local faster-whisper (LOCAL_WHISPER=1) - lowest latency, runs on CPU/GPU
    - Groq Whisper API (default) - fast cloud API, free tier
    - Deepgram (fallback)

    TTS: edge-tts (Microsoft neural voices, free)

    Low-latency features:
    - Speculative TTS (SPECULATIVE_TTS=1) - start TTS before LLM finishes
    - WebSocket streaming (WEBSOCKET_VOICE=1) - bidirectional audio
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._groq_client = None
        self._deepgram_client = None
        self._local_whisper_model = None
        self._speculative_tts_tasks: List[asyncio.Task] = []

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
        result = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", result, flags=re.IGNORECASE)

        # Clean up multiple spaces
        result = re.sub(r"\s+", " ", result).strip()

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

    @property
    def local_whisper_model(self):
        """Lazy-load local faster-whisper model for lowest latency STT."""
        if self._local_whisper_model is None and self.config.use_local_whisper:
            try:
                from faster_whisper import WhisperModel

                model_size = self.config.whisper_model
                # Use GPU if available, otherwise CPU with int8 quantization
                device = "cuda" if self._check_cuda() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"

                logger.info(f"Loading local Whisper model: {model_size} on {device}")
                self._local_whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=os.path.expanduser("~/.cache/whisper"),
                )
                logger.info(f"Local Whisper model loaded: {model_size}")
            except ImportError:
                logger.warning("faster-whisper not installed. Run: pip install faster-whisper")
                return None
            except Exception as e:
                logger.error(f"Failed to load local Whisper: {e}")
                return None
        return self._local_whisper_model

    @staticmethod
    def _check_cuda():
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def speech_to_text(
        self, audio_data: bytes, mime_type: str = "audio/webm"
    ) -> Tuple[str, float]:
        """
        Convert speech audio to text.

        Priority (configurable):
        1. Local faster-whisper (if LOCAL_WHISPER=1) - lowest latency
        2. Groq Whisper API - fast cloud, free tier
        3. Deepgram - fallback

        Args:
            audio_data: Raw audio bytes
            mime_type: Audio MIME type (audio/webm, audio/wav, etc.)

        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        # Try local Whisper first if configured (lowest latency)
        if self.config.use_local_whisper:
            transcript, confidence = await self._stt_local_whisper(audio_data, mime_type)
            if transcript:
                return transcript, confidence

        # Try Groq Whisper (primary cloud option)
        transcript = await self._stt_groq_whisper(audio_data, mime_type)
        if transcript:
            return transcript, 0.95  # Groq doesn't return confidence, assume high

        # Fallback to Deepgram
        transcript = await self._stt_deepgram(audio_data, mime_type)
        if transcript:
            return transcript, 0.90

        logger.warning("All STT providers failed")
        return "", 0.0

    async def _stt_local_whisper(self, audio_data: bytes, mime_type: str) -> Tuple[str, float]:
        """
        Transcribe using local faster-whisper for lowest latency.

        Returns:
            Tuple of (transcript, confidence)
        """
        model = self.local_whisper_model
        if not model:
            return "", 0.0

        try:
            # Determine file extension
            ext_map = {
                "audio/webm": ".webm",
                "audio/wav": ".wav",
                "audio/mp3": ".mp3",
                "audio/mpeg": ".mp3",
                "audio/ogg": ".ogg",
            }
            ext = ext_map.get(mime_type, ".webm")

            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Run transcription in thread pool to avoid blocking
                def transcribe():
                    segments, info = model.transcribe(
                        temp_path,
                        language="en",
                        beam_size=5,
                        vad_filter=True,  # Filter out non-speech
                        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                        initial_prompt=WHISPER_PROMPT,
                    )
                    # Collect all segments
                    text_parts = []
                    total_prob = 0.0
                    count = 0
                    for segment in segments:
                        text_parts.append(segment.text)
                        total_prob += segment.avg_logprob
                        count += 1

                    transcript = " ".join(text_parts).strip()
                    # Convert log probability to confidence (0-1)
                    avg_logprob = total_prob / count if count > 0 else -1.0
                    confidence = min(1.0, max(0.0, 1.0 + avg_logprob))  # logprob is negative

                    return transcript, confidence

                transcript, confidence = await asyncio.to_thread(transcribe)

                # Post-process
                transcript = self._post_process_transcript(transcript)
                logger.info(f"Local Whisper STT ({confidence:.0%}): {transcript[:50]}...")
                return transcript, confidence

            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Local Whisper STT failed: {e}")
            return "", 0.0

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
                            temperature=0.0,  # More deterministic
                        )
                    )

                transcript = (
                    response.strip() if isinstance(response, str) else str(response).strip()
                )

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
                    {"buffer": audio_data, "mimetype": mime_type}, options
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

    @staticmethod
    def _preprocess_tts_text(text: str) -> str:
        """
        Preprocess text for TTS to handle edge cases.

        Fixes:
        - Multiple exclamation marks that can crash Edge-TTS
        - Very short utterances that may not generate audio
        - ALL CAPS text that Edge-TTS doesn't handle well
        """
        import re

        if not text:
            return text

        result = text

        # Fix multiple exclamation/question marks (Edge-TTS issue)
        result = re.sub(r"!+", ".", result)  # Replace ! with .
        result = re.sub(r"\?+", "?", result)  # Keep single ?

        # Convert ALL CAPS to title case (Edge-TTS fails on all caps)
        if result.isupper() and len(result) > 3:
            result = result.title()

        # Ensure very short text has enough content for TTS
        # Edge-TTS needs some minimum content to generate audio
        if len(result.strip()) < 3:
            result = f"{result} "  # Add trailing space

        # Remove any control characters that might break TTS
        result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

        return result

    async def text_to_speech(
        self, text: str, voice: Optional[str] = None, speed: Optional[float] = None
    ) -> bytes:
        """
        Convert text to speech using edge-tts.

        Args:
            text: Text to speak
            voice: Voice ID (optional, uses default)
            speed: Speech speed multiplier (0.5 to 2.0, default 1.0)

        Returns:
            MP3 audio bytes
        """
        if not text.strip():
            return b""

        # Preprocess text for TTS compatibility
        text = self._preprocess_tts_text(text)

        try:
            import edge_tts

            voice_id = voice or self.config.voice

            # Calculate rate from speed (edge-tts uses percentage like "+10%" or "-20%")
            if speed and speed != 1.0:
                rate_percent = int((speed - 1.0) * 100)
                rate = f"{rate_percent:+d}%"
            else:
                rate = self.config.rate

            # Create communicate object with voice settings
            communicate = edge_tts.Communicate(
                text, voice_id, rate=rate, pitch=self.config.pitch, volume=self.config.volume
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

    async def text_to_speech_stream(
        self, text: str, voice: Optional[str] = None
    ) -> AsyncIterator[bytes]:
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
                volume=self.config.volume,
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"TTS stream failed: {e}")

    async def process_voice_message(
        self, audio_data: bytes, mime_type: str = "audio/webm", process_text_fn=None
    ) -> Tuple[str, str, bytes, float]:
        """
        Full voice-to-voice pipeline.

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            process_text_fn: Async function to process text (your LLM)

        Returns:
            Tuple of (user_text, response_text, response_audio, confidence)
        """
        # 1. Speech to text
        user_text, confidence = await self.speech_to_text(audio_data, mime_type)
        if not user_text:
            return "", "I couldn't understand that. Please try again.", b"", 0.0

        # 2. Process with LLM
        if process_text_fn:
            response_text = await process_text_fn(user_text)
        else:
            response_text = f"You said: {user_text}"

        # 3. Text to speech
        response_audio = await self.text_to_speech(response_text)

        return user_text, response_text, response_audio, confidence

    async def process_voice_message_streaming(
        self, audio_data: bytes, mime_type: str = "audio/webm", process_text_fn=None
    ) -> AsyncIterator[Tuple[str, bytes, float]]:
        """
        Streaming voice-to-voice pipeline for lower latency.

        Yields audio chunks as soon as first sentence is ready,
        reducing perceived latency significantly.

        Args:
            audio_data: Input audio bytes
            mime_type: Audio MIME type
            process_text_fn: Async function to process text (your LLM)

        Yields:
            Tuples of (text_chunk, audio_chunk, confidence)
        """
        # 1. Speech to text
        user_text, confidence = await self.speech_to_text(audio_data, mime_type)
        if not user_text:
            error_audio = await self.text_to_speech("I couldn't understand that.")
            yield ("I couldn't understand that.", error_audio, 0.0)
            return

        # 2. Process with LLM
        if process_text_fn:
            response_text = await process_text_fn(user_text)
        else:
            response_text = f"You said: {user_text}"

        # 3. Split into sentences and generate TTS for each
        # This allows streaming audio back sentence by sentence
        sentences = re.split(r"(?<=[.!?])\s+", response_text)

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                audio_chunk = await self.text_to_speech(sentence)
                # Only include confidence on first chunk
                yield (sentence, audio_chunk, confidence if i == 0 else -1)

    # =========================================================================
    # SPECULATIVE TTS - Start generating audio before LLM finishes
    # =========================================================================

    async def speculative_tts_stream(
        self,
        text_generator: AsyncIterator[str],
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> AsyncIterator[Tuple[str, bytes]]:
        """
        Speculative TTS: Generate audio as text streams in.

        Instead of waiting for complete LLM response, starts TTS as soon
        as we have a complete sentence. This significantly reduces latency.

        Args:
            text_generator: Async iterator yielding text tokens
            voice: TTS voice to use
            speed: Speech speed

        Yields:
            Tuples of (sentence_text, audio_bytes)
        """
        buffer = ""
        sentence_endings = re.compile(r"([.!?])\s*")

        async for token in text_generator:
            buffer += token

            # Check if we have a complete sentence
            match = sentence_endings.search(buffer)
            if match:
                # Extract the complete sentence
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                buffer = buffer[end_pos:]

                if sentence:
                    # Start TTS immediately for this sentence
                    logger.debug(f"Speculative TTS: '{sentence[:30]}...'")
                    audio = await self.text_to_speech(sentence, voice, speed)
                    yield (sentence, audio)

        # Handle any remaining text
        if buffer.strip():
            audio = await self.text_to_speech(buffer.strip(), voice, speed)
            yield (buffer.strip(), audio)

    async def parallel_speculative_tts(
        self,
        text_generator: AsyncIterator[str],
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lookahead: int = 2,
    ) -> AsyncIterator[Tuple[str, bytes]]:
        """
        Advanced speculative TTS with parallel generation.

        Generates TTS for multiple sentences in parallel for even lower latency.
        Uses a lookahead buffer to start generating next sentences early.

        Args:
            text_generator: Async iterator yielding text tokens
            voice: TTS voice
            speed: Speech speed
            lookahead: Number of sentences to generate in parallel

        Yields:
            Tuples of (sentence_text, audio_bytes) in order
        """
        buffer = ""
        pending_sentences: List[str] = []
        tts_tasks: List[Tuple[str, asyncio.Task]] = []
        sentence_endings = re.compile(r"([.!?])\s*")

        async def generate_tts(text: str) -> bytes:
            return await self.text_to_speech(text, voice, speed)

        async for token in text_generator:
            buffer += token

            # Extract complete sentences
            while True:
                match = sentence_endings.search(buffer)
                if not match:
                    break
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                buffer = buffer[end_pos:]
                if sentence:
                    pending_sentences.append(sentence)

            # Start TTS tasks for pending sentences (up to lookahead)
            while pending_sentences and len(tts_tasks) < lookahead:
                sentence = pending_sentences.pop(0)
                task = asyncio.create_task(generate_tts(sentence))
                tts_tasks.append((sentence, task))

            # Yield completed tasks in order
            while tts_tasks and tts_tasks[0][1].done():
                sentence, task = tts_tasks.pop(0)
                try:
                    audio = await task
                    yield (sentence, audio)
                except Exception as e:
                    logger.error(f"TTS task failed: {e}")

        # Process remaining sentences
        for sentence in pending_sentences:
            task = asyncio.create_task(generate_tts(sentence))
            tts_tasks.append((sentence, task))

        # Handle remaining buffer
        if buffer.strip():
            task = asyncio.create_task(generate_tts(buffer.strip()))
            tts_tasks.append((buffer.strip(), task))

        # Yield all remaining tasks
        for sentence, task in tts_tasks:
            try:
                audio = await task
                yield (sentence, audio)
            except Exception as e:
                logger.error(f"TTS task failed: {e}")

    async def process_voice_fast(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        process_text_fn=None,
        max_response_chars: int = 200,
    ) -> Tuple[str, str, bytes, float]:
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
            Tuple of (user_text, response_text, response_audio, confidence)
        """
        # 1. Speech to text
        user_text, confidence = await self.speech_to_text(audio_data, mime_type)
        if not user_text:
            return "", "I couldn't understand that.", b"", 0.0

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
                truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?")
            )
            if last_sentence_end > 50:
                response_text = truncated[: last_sentence_end + 1]
            else:
                response_text = truncated + "..."

        # 4. TTS with faster speech rate for quicker delivery
        fast_config = VoiceConfig(
            voice=self.config.voice,
            rate="+15%",  # 15% faster speech
            pitch=self.config.pitch,
            volume=self.config.volume,
        )

        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                response_text,
                fast_config.voice,
                rate=fast_config.rate,
                pitch=fast_config.pitch,
                volume=fast_config.volume,
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

        return user_text, response_text, response_audio, confidence

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
