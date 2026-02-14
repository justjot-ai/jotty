# Voice Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`voice_to_text_tool`](#voice_to_text_tool) | Convert audio file to text using speech recognition. |
| [`text_to_voice_tool`](#text_to_voice_tool) | Convert text to audio using speech synthesis. |
| [`stream_voice_tool`](#stream_voice_tool) | Start streaming text-to-speech audio. |
| [`get_stream_chunk_tool`](#get_stream_chunk_tool) | Get next chunk from an active voice stream. |
| [`close_stream_tool`](#close_stream_tool) | Close an active voice stream. |
| [`list_voices_tool`](#list_voices_tool) | List available voices for TTS. |
| [`translate_audio_tool`](#translate_audio_tool) | Translate audio to English text using Whisper. |
| [`generate_speech_openai_tool`](#generate_speech_openai_tool) | Generate speech using OpenAI TTS with 6 voice options. |
| [`get_audio_info_tool`](#get_audio_info_tool) | Get metadata about an audio file. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`create_stream`](#create_stream) | No description available. |
| [`voice_to_text`](#voice_to_text) | Convert audio to text. |
| [`text_to_voice`](#text_to_voice) | Convert text to audio. |
| [`stream_voice`](#stream_voice) | Async generator for streaming TTS audio. |
| [`get_instance`](#get_instance) | No description available. |
| [`translate`](#translate) | Translate audio to English using Whisper. |
| [`generate_speech`](#generate_speech) | Generate speech using OpenAI TTS. |
| [`get_audio_info`](#get_audio_info) | Get audio file metadata. |

---

## `voice_to_text_tool`

Convert audio file to text using speech recognition.

**Parameters:**

- **audio_path** (`str, required`): Path to audio file
- **provider** (`str, optional`): "whisper", "local", or "auto"
- **language** (`str, optional`): Language code

**Returns:** Dictionary with success, text, language, provider

---

## `text_to_voice_tool`

Convert text to audio using speech synthesis.

**Parameters:**

- **text** (`str, required`): Text to synthesize
- **provider** (`str, optional`): "elevenlabs", "local", or "auto"
- **voice_id** (`str, optional`): Voice ID or name
- **output_path** (`str, optional`): Path to save audio

**Returns:** Dictionary with success, audio data/path, format, provider

---

## `stream_voice_tool`

Start streaming text-to-speech audio.

**Parameters:**

- **text** (`str, required`): Text to synthesize
- **provider** (`str, optional`): "elevenlabs", "local", or "auto"
- **voice_id** (`str, optional`): Voice ID or name
- **chunk_size** (`int, optional`): Audio chunk size

**Returns:** Dictionary with success, stream_id, format, provider

---

## `get_stream_chunk_tool`

Get next chunk from an active voice stream.

**Parameters:**

- **stream_id** (`str, required`): Stream ID from stream_voice_tool

**Returns:** Dictionary with success, chunk (base64), done flag

---

## `close_stream_tool`

Close an active voice stream.

**Parameters:**

- **stream_id** (`str, required`): Stream ID to close

**Returns:** Dictionary with success

---

## `list_voices_tool`

List available voices for TTS.

**Parameters:**

- **provider** (`str, optional`): "elevenlabs", "local", or "auto"

**Returns:** Dictionary with success, voices list, count

---

## `translate_audio_tool`

Translate audio to English text using Whisper.

**Parameters:**

- **audio_path** (`str, required`): Path to audio file (any language)
- **prompt** (`str, optional`): Context hint for translation

**Returns:** Dictionary with success, text, original_language, target_language

---

## `generate_speech_openai_tool`

Generate speech using OpenAI TTS with 6 voice options.

**Parameters:**

- **text** (`str, required`): Text to convert (max 4096 chars)
- **voice** (`str, optional`): alloy/echo/fable/onyx/nova/shimmer (default: nova)
- **speed** (`float, optional`): 0.25-4.0 (default: 1.0)
- **response_format** (`str, optional`): mp3/opus/aac/flac/wav (default: mp3)
- **output_path** (`str, optional`): Save path

**Returns:** Dictionary with success, audio_path, voice, duration_estimate

---

## `get_audio_info_tool`

Get metadata about an audio file.

**Parameters:**

- **audio_path** (`str, required`): Path to audio file

**Returns:** Dictionary with success, format, size, duration, sample_rate, channels

---

## `create_stream`

No description available.

---

## `voice_to_text`

Convert audio to text. Returns transcribed text or raises exception.

**Parameters:**

- **audio_path** (`str`)
- **provider** (`str`)
- **language** (`str`)

**Returns:** `str`

---

## `text_to_voice`

Convert text to audio. Returns audio bytes or raises exception.

**Parameters:**

- **text** (`str`)
- **provider** (`str`)
- **voice_id** (`str`)

**Returns:** `bytes`

---

## `stream_voice`

Async generator for streaming TTS audio.

**Parameters:**

- **text** (`str`)
- **provider** (`str`)
- **voice_id** (`str`)

---

## `get_instance`

No description available.

---

## `translate`

Translate audio to English using Whisper.

**Parameters:**

- **audio_path** (`str`)
- **prompt** (`str`)
- **response_format** (`str`)

**Returns:** `dict`

---

## `generate_speech`

Generate speech using OpenAI TTS.

**Parameters:**

- **text** (`str`)
- **voice** (`str`)
- **speed** (`float`)
- **response_format** (`str`)
- **output_path** (`str`)

**Returns:** `dict`

---

## `get_audio_info`

Get audio file metadata.

**Parameters:**

- **audio_path** (`str`)

**Returns:** `dict`
