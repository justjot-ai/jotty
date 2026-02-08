# Voice Skill

## Description

Multi-provider voice skill for speech-to-text (STT) and text-to-speech (TTS) capabilities.
Supports cloud providers (ElevenLabs, OpenAI Whisper) and local providers (Piper TTS, Whisper.cpp STT).

Inspired by OpenClaw's voice architecture with automatic provider selection.

## Features

- Text-to-speech (TTS) with multiple voice options
- Speech-to-text (STT) with high accuracy
- Streaming TTS for real-time audio output
- Automatic provider selection based on availability
- Local-first mode for privacy (no external API calls)

## Tools

### voice_to_text_tool

Convert audio file to text using speech recognition.

**Parameters:**
- `audio_path` (str, required): Path to audio file (WAV, MP3, OGG, etc.)
- `provider` (str, optional): Provider to use - "whisper", "local", or "auto" (default: "auto")
- `language` (str, optional): Language code (e.g., "en", "es", "fr"). Default: auto-detect

**Returns:**
- `success` (bool): Whether transcription succeeded
- `text` (str): Transcribed text
- `language` (str): Detected/used language
- `provider` (str): Provider used for transcription
- `error` (str, optional): Error message if failed

### text_to_voice_tool

Convert text to audio using speech synthesis.

**Parameters:**
- `text` (str, required): Text to synthesize
- `provider` (str, optional): Provider to use - "elevenlabs", "local", or "auto" (default: "auto")
- `voice_id` (str, optional): Voice ID or name. Default: provider's default voice
- `output_path` (str, optional): Output file path. If not provided, returns base64 audio

**Returns:**
- `success` (bool): Whether synthesis succeeded
- `audio_path` (str, optional): Path to output file if output_path provided
- `audio_base64` (str, optional): Base64 encoded audio if no output_path
- `format` (str): Audio format (mp3, wav, etc.)
- `provider` (str): Provider used for synthesis
- `error` (str, optional): Error message if failed

### stream_voice_tool

Stream text-to-speech audio for real-time playback.

**Parameters:**
- `text` (str, required): Text to synthesize
- `provider` (str, optional): Provider to use - "elevenlabs", "local", or "auto" (default: "auto")
- `voice_id` (str, optional): Voice ID or name. Default: provider's default voice
- `chunk_size` (int, optional): Size of audio chunks in bytes. Default: 1024

**Returns:**
- `success` (bool): Whether streaming started
- `stream_id` (str): Unique identifier for this stream
- `format` (str): Audio format
- `provider` (str): Provider used
- `error` (str, optional): Error message if failed

## Configuration

### Cloud Providers

Set these environment variables for cloud provider access:

```bash
# ElevenLabs TTS
ELEVENLABS_API_KEY=your_api_key

# OpenAI Whisper STT
OPENAI_API_KEY=your_api_key
```

### Local Providers

For local inference without API calls:

1. **Whisper.cpp STT**: Install whisper.cpp and set model path
   ```bash
   WHISPER_CPP_PATH=/path/to/whisper.cpp/main
   WHISPER_MODEL_PATH=/path/to/ggml-base.en.bin
   ```

2. **Piper TTS**: Install piper-tts
   ```bash
   pip install piper-tts
   # Or set custom path
   PIPER_PATH=/path/to/piper
   PIPER_VOICE_PATH=/path/to/en_US-lessac-medium.onnx
   ```

### Provider Selection

The skill automatically selects providers based on:
1. If `local_mode=True` in config: Use local providers only
2. If cloud API keys available: Use cloud providers
3. If local binaries available: Fall back to local providers

## Usage Examples

```python
from skills.voice.tools import (
    voice_to_text_tool,
    text_to_voice_tool,
    stream_voice_tool
)

# Speech-to-text
result = voice_to_text_tool({
    'audio_path': 'recording.wav',
    'provider': 'auto'
})
print(result['text'])

# Text-to-speech (save to file)
result = text_to_voice_tool({
    'text': 'Hello from Jotty!',
    'output_path': 'output.mp3'
})

# Text-to-speech (get base64)
result = text_to_voice_tool({
    'text': 'Hello from Jotty!',
    'provider': 'elevenlabs',
    'voice_id': 'Rachel'
})
audio_data = base64.b64decode(result['audio_base64'])

# Streaming TTS
result = stream_voice_tool({
    'text': 'This is a longer text that will be streamed...',
    'chunk_size': 2048
})
```

## Supported Formats

### Input (STT)
- WAV, MP3, OGG, FLAC, M4A, WEBM

### Output (TTS)
- MP3 (ElevenLabs default)
- WAV (Local/Piper default)
- OGG (optional)

## Voice IDs

### ElevenLabs Voices
- `Rachel` - Default female voice
- `Adam` - Default male voice
- `Josh` - Deep male voice
- `Bella` - Soft female voice
- Custom voice IDs from your ElevenLabs account

### Piper Voices
- `en_US-lessac-medium` - Default English voice
- Other Piper ONNX model files
