"""
Voice Modality - Voice Input/Output Handling
=============================================

Handles voice-based communication across all platforms.

## Responsibilities

- Convert speech to text (STT)
- Convert text to speech (TTS)
- Handle audio encoding/decoding
- Support multiple audio formats (mp3, ogg, wav, etc.)
"""

from .audio_processor import AudioProcessor
from .speech_to_text import SpeechToText, speech_to_text
from .text_to_speech import TextToSpeech, text_to_speech

__all__ = [
    "VoiceModality",
    "SpeechToText",
    "TextToSpeech",
    "AudioProcessor",
    "speech_to_text",
    "text_to_speech",
]


class VoiceModality:
    """
    Voice modality handler.

    Provides unified interface for voice input/output across platforms.
    """

    def __init__(self, platform: str = "generic"):
        """
        Initialize voice modality.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
        """
        self.platform = platform
        self.stt = SpeechToText(platform)
        self.tts = TextToSpeech(platform)
        self.processor = AudioProcessor(platform)

    def transcribe(self, audio_file: str, **kwargs) -> str:
        """Convert speech to text."""
        return self.stt.transcribe(audio_file, **kwargs)

    def synthesize(self, text: str, **kwargs) -> bytes:
        """Convert text to speech."""
        return self.tts.synthesize(text, **kwargs)

    def process_audio(self, audio_data: bytes, **kwargs) -> bytes:
        """Process audio data (normalize, convert format, etc.)."""
        return self.processor.process(audio_data, **kwargs)
