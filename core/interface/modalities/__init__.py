"""
Modalities - Input/Output Medium Handlers
==========================================

Handles different communication mediums (text, voice, etc.)
across all platforms (WhatsApp, Telegram, CLI, Web).

## Terminology

- **Platforms** (apps/): WHERE users interact (WhatsApp, Telegram, CLI, Web)
- **Modes** (core/modes/): HOW execution happens (Chat, Workflow, Streaming)
- **Modalities** (here): WHAT medium is used (Text, Voice, Image, Video)

## Usage

```python
from Jotty.core.interface.modalities.voice import speech_to_text, text_to_speech
from Jotty.core.interface.modalities.text import parse_input, format_output

# Convert voice to text (any platform)
text = speech_to_text(audio_file)

# Format output for text display
formatted = format_output(response, platform="telegram")
```
"""

from .text import TextModality, format_output, parse_input
from .voice import VoiceModality, speech_to_text, text_to_speech

__all__ = [
    "TextModality",
    "VoiceModality",
    "parse_input",
    "format_output",
    "speech_to_text",
    "text_to_speech",
]
