# Platforms, Modes, and Modalities

**Date:** 2026-02-15

## Three Distinct Concepts

Jotty has three architectural concepts that work together:

| Concept | What It Means | Examples | Location |
|---------|---------------|----------|----------|
| **Platforms** | WHERE users interact | WhatsApp, Telegram, CLI, Web | `apps/` |
| **Modes** | HOW execution happens | Chat, Workflow, Streaming | `core/modes/` |
| **Modalities** | WHAT medium is used | Text, Voice, Image | `core/interface/modalities/` |

## Industry Standards

These naming conventions match industry leaders:

**Platforms:**
- Google: Android, iOS, Web (platforms)
- Slack: Web, Desktop, Mobile (platforms)
- Discord: Desktop, Mobile, Browser (platforms)

**Modes:**
- OpenAI: Streaming mode, Batch mode
- Anthropic: Chat mode, Completion mode
- LangChain: Chain mode, Agent mode

**Modalities:**
- Google Assistant: Text, Voice, Visual (modalities)
- OpenAI: Text, Image, Audio (modalities)
- Anthropic: Text, Image (modalities)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  PLATFORMS (apps/)                                          │
│  ├── cli/          → Terminal platform                      │
│  ├── web/          → Web browser platform                   │
│  ├── telegram/     → Telegram platform                      │
│  └── whatsapp/     → WhatsApp platform                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓ Uses
┌────────────────────────┴────────────────────────────────────┐
│  SDK (sdk/)                                                 │
│  └── Handles all modalities and modes                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓ Calls
┌────────────────────────┴────────────────────────────────────┐
│  MODALITIES (core/interface/modalities/)                    │
│  ├── text/         → Text input/output handlers             │
│  └── voice/        → Voice input/output handlers            │
│      ├── speech_to_text.py (STT)                            │
│      ├── text_to_speech.py (TTS)                            │
│      └── audio_processor.py                                 │
└────────────────────────┬────────────────────────────────────┘
                         ↓ Uses
┌────────────────────────┴────────────────────────────────────┐
│  MODES (core/modes/)                                        │
│  ├── chat/         → Chat execution mode                    │
│  ├── workflow/     → Multi-step workflow mode               │
│  └── streaming/    → Streaming execution mode               │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Example

**Scenario:** User sends voice message via Telegram asking a question

```
1. PLATFORM: Telegram receives voice message
   ↓
2. MODALITY: Convert voice → text (speech_to_text)
   ↓
3. MODE: Process in chat mode (ChatExecutor)
   ↓
4. MODALITY: Convert text → voice (text_to_speech) [optional]
   ↓
5. PLATFORM: Send response back to Telegram
```

**Code:**
```python
# apps/telegram/bot.py (PLATFORM layer)
from Jotty import Jotty
from Jotty.core.interface.modalities.voice import speech_to_text, text_to_speech

async def handle_voice_message(update, context):
    # 1. Receive from platform
    voice_file = await update.message.voice.get_file()

    # 2. Modality: Voice → Text
    text = speech_to_text(voice_file.file_path, platform="telegram")

    # 3. Mode: Process in chat mode
    jotty = Jotty()
    response = jotty.chat(text)

    # 4. Modality: Text → Voice (optional)
    audio = text_to_speech(response, platform="telegram")

    # 5. Send back to platform
    await update.message.reply_voice(audio)
```

## Directory Structure

```
Jotty/
├── apps/                          # PLATFORMS
│   ├── cli/                       # Terminal platform
│   ├── web/                       # Browser platform
│   ├── telegram/                  # Telegram platform
│   └── whatsapp/                  # WhatsApp platform
│
├── sdk/                           # SDK Layer
│
├── core/
│   ├── interface/
│   │   ├── api/                   # API facades
│   │   ├── modalities/            # MODALITIES (NEW)
│   │   │   ├── text/              # Text handlers
│   │   │   │   ├── parser.py      # Parse text input
│   │   │   │   └── formatter.py   # Format text output
│   │   │   └── voice/             # Voice handlers
│   │   │       ├── speech_to_text.py   # STT (Whisper, Google, Azure)
│   │   │       ├── text_to_speech.py   # TTS (OpenAI, Google, ElevenLabs)
│   │   │       └── audio_processor.py  # Audio format conversion
│   │   └── use_cases/             # Use case wrappers
│   │
│   └── modes/                     # MODES
│       ├── chat/                  # Chat execution
│       ├── workflow/              # Workflow execution
│       └── streaming/             # Streaming execution
```

## API Reference

### Text Modality

```python
from Jotty.core.interface.modalities.text import (
    TextModality,
    parse_input,
    format_output
)

# Parse text from any platform
parsed = parse_input("Hello", platform="telegram")
# Returns: {'text': 'Hello', 'platform': 'telegram', 'modality': 'text'}

# Format text for specific platform (markdown, emoji, etc.)
formatted = format_output("**Bold** text", platform="whatsapp")
```

### Voice Modality

```python
from Jotty.core.interface.modalities.voice import (
    VoiceModality,
    speech_to_text,
    text_to_speech
)

# Speech-to-Text (STT)
text = speech_to_text(
    audio_file="voice.ogg",
    platform="telegram",
    provider="whisper",  # or 'google', 'azure'
    language="en"
)

# Text-to-Speech (TTS)
audio = text_to_speech(
    text="Hello, world!",
    platform="telegram",
    provider="openai",    # or 'google', 'azure', 'elevenlabs'
    voice="alloy"         # Provider-specific voice
)
```

### Supported Providers

**Speech-to-Text (STT):**
- OpenAI Whisper API (default)
- Google Speech-to-Text
- Azure Speech Services

**Text-to-Speech (TTS):**
- OpenAI TTS API (default)
- Google Text-to-Speech
- Azure Speech Services
- ElevenLabs

### Platform-Specific Features

**Audio Format Preferences:**
- Telegram: OGG Opus
- WhatsApp: OGG Opus
- CLI: MP3
- Web: MP3

**Text Formatting:**
- Telegram: Full markdown support
- WhatsApp: Limited markdown
- CLI: ANSI color codes
- Web: HTML/markdown

## Benefits

✅ **Clear Separation:** Platforms, modes, and modalities are independent
✅ **Reusable:** Voice modality works on ANY platform (Telegram, WhatsApp, CLI, Web)
✅ **Extensible:** Add new modalities (image, video) without changing platforms
✅ **Industry Standard:** Matches Google, OpenAI, Microsoft terminology
✅ **No Naming Collision:** Clear distinction between platform, mode, and modality

## Migration Notes

**No breaking changes** - This is purely additive:
- Existing `apps/` (platforms) unchanged
- Existing `core/modes/` unchanged
- New `core/interface/modalities/` added

**Documentation updated:**
- CLAUDE.md now clarifies terminology
- Architecture diagrams updated
- Usage examples added

## Future Modalities

Additional modalities can be added:
- `modalities/image/` - Image input/output (OCR, image generation)
- `modalities/video/` - Video input/output
- `modalities/file/` - File upload/download handling
- `modalities/location/` - Location data handling

All would work across ALL platforms using the same pattern.
