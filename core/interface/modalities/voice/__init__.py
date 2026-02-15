"""
Voice Modality - Unified Voice Input/Output
============================================

Multi-provider voice capabilities with intelligent auto-selection.

## STT Providers (priority order):
1. **Groq Whisper** (fast, free tier) - RECOMMENDED
2. **OpenAI Whisper** (reliable, paid)
3. **Local Whisper** (offline, privacy)

## TTS Providers (priority order):
1. **Edge TTS** (free, high quality) - RECOMMENDED
2. **OpenAI TTS** (paid, very high quality)
3. **ElevenLabs** (paid, ultra high quality)
4. **Local Piper** (offline, privacy)

## Usage

```python
from Jotty.core.interface.modalities.voice import speech_to_text, text_to_speech

# Speech to text (auto-selects best provider)
text = await speech_to_text("audio.mp3")

# Text to speech (auto-selects best provider)
audio = await text_to_speech("Hello world!")

# Specify provider
text = await speech_to_text("audio.mp3", provider="groq")
audio = await text_to_speech("Hello!", provider="edge")

# Streaming
from Jotty.core.interface.modalities.voice import TextToSpeech
tts = TextToSpeech(provider="edge")
async for chunk in tts.stream("Long text..."):
    # Process audio chunks
    pass
```
"""

from .audio_processor import AudioProcessor
from .config import VoiceConfig, get_config, set_config
from .providers import VoiceProviderBase, get_stt_provider, get_tts_provider
from .speech_to_text import SpeechToText, speech_to_text
from .text_to_speech import TextToSpeech, text_to_speech

__all__ = [
    # Main classes
    "VoiceModality",
    "SpeechToText",
    "TextToSpeech",
    "AudioProcessor",
    # Convenience functions
    "speech_to_text",
    "text_to_speech",
    # Configuration
    "VoiceConfig",
    "get_config",
    "set_config",
    # Providers
    "VoiceProviderBase",
    "get_stt_provider",
    "get_tts_provider",
]


class VoiceModality:
    """
    Voice modality handler with multi-provider support.

    Provides unified interface for voice input/output across platforms.
    """

    def __init__(
        self, platform: str = "generic", stt_provider: str = "auto", tts_provider: str = "auto"
    ):
        """
        Initialize voice modality.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
            stt_provider: STT provider (auto, groq, whisper, local)
            tts_provider: TTS provider (auto, edge, openai, elevenlabs, local)
        """
        self.platform = platform
        self.stt = SpeechToText(platform, stt_provider)
        self.tts = TextToSpeech(platform, tts_provider)
        self.processor = AudioProcessor(platform)

    async def transcribe(self, audio_file: str, **kwargs) -> str:
        """Convert speech to text."""
        result = await self.stt.transcribe(audio_file, **kwargs)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Unknown error"))
        return result["text"]

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Convert text to speech."""
        result = await self.tts.synthesize(text, **kwargs)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Unknown error"))

        # Return audio bytes
        if "audio_base64" in result:
            import base64

            return base64.b64decode(result["audio_base64"])
        elif "audio_path" in result:
            from pathlib import Path

            return Path(result["audio_path"]).read_bytes()
        else:
            raise RuntimeError("No audio data in TTS result")

    async def stream(self, text: str, **kwargs):
        """Stream text-to-speech audio."""
        async for chunk in self.tts.stream(text, **kwargs):
            yield chunk

    def process_audio(self, audio_data: bytes, **kwargs) -> bytes:
        """Process audio data (normalize, convert format, etc.)."""
        return self.processor.process(audio_data, **kwargs)
