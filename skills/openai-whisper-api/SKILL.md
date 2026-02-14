---
name: openai-whisper-api
description: "Transcribe audio recordings and voice memos to text. Translate speech from any language to English. Use for meeting recordings, interviews, podcasts, lectures, and video audio extraction. Supports mp3, wav, m4a, mp4 formats. Use when the user wants to transcribe, audio, speech."
---

# OpenAI Whisper API Skill

## Description
Transcribe audio recordings and voice memos to text. Translate speech from any language to English. Use for meeting recordings, interviews, podcasts, lectures, and video audio extraction. Supports mp3, wav, m4a, mp4 formats.


## Type
base


## Capabilities
- media
- transcribe

## Tools

### transcribe_audio_tool
Transcribe audio file to text.

**Parameters:**
- `file_path` (str, required): Path to audio file
- `language` (str, optional): ISO-639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'ja')
- `response_format` (str, optional): Output format - 'json', 'text', 'srt', 'vtt', 'verbose_json' (default: 'json')

**Returns:**
- `success` (bool): Whether transcription succeeded
- `text` (str): Transcribed text
- `file_path` (str): Source file path
- `response_format` (str): Format used
- For verbose_json: includes `language`, `duration`, `segments`

### translate_audio_tool
Translate audio to English (from any supported language).

**Parameters:**
- `file_path` (str, required): Path to audio file
- `response_format` (str, optional): Output format - 'json', 'text', 'srt', 'vtt', 'verbose_json' (default: 'json')

**Returns:**
- `success` (bool): Whether translation succeeded
- `text` (str): Translated English text
- `file_path` (str): Source file path
- `response_format` (str): Format used
- `target_language` (str): Always 'en'

## Supported Audio Formats
- mp3
- mp4
- mpeg
- mpga
- m4a
- wav
- webm

## Requirements
- `requests` library
- `OPENAI_API_KEY` environment variable

## Usage Examples

```python
# Transcribe audio file
result = transcribe_audio_tool({
    'file_path': '/path/to/audio.mp3',
    'language': 'en',
    'response_format': 'json'
})

# Transcribe with SRT subtitles output
result = transcribe_audio_tool({
    'file_path': '/path/to/video.mp4',
    'response_format': 'srt'
})

# Translate non-English audio to English
result = translate_audio_tool({
    'file_path': '/path/to/spanish_audio.mp3',
    'response_format': 'text'
})
```

## Notes
- Maximum file size: 25 MB (OpenAI limit)
- For longer audio, consider splitting into chunks
- Translation always outputs English text
- The `verbose_json` format includes word-level timestamps

## Triggers
- "transcribe"
- "audio"
- "speech"
- "recording"
- "voice"
- "openai whisper api"
- "speech to text"
- "audio transcription"
- "voice to text"

## Category
media-creation
