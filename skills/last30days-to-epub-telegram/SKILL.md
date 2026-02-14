---
name: sending-last30days-epub-telegram
description: "This composite skill combines: 1. **last30days-claude-cli**: Research topics from last 30 days 2. **document-converter**: Convert markdown to EPUB 3. **telegram-sender**: Send EPUB to Telegram. Use when the user wants to send to telegram, telegram message, notify via telegram."
---

# Last30Days → EPUB → Telegram Composite Skill

Research topics using last30days skill, generate EPUB, and send to Telegram.

## Description

This composite skill combines:
1. **last30days-claude-cli**: Research topics from last 30 days
2. **document-converter**: Convert markdown to EPUB
3. **telegram-sender**: Send EPUB to Telegram


## Type
composite

## Base Skills
- last30days-claude-cli
- document-converter
- telegram-sender

## Execution
sequential


## Capabilities
- research
- document
- communicate

## Usage

```python
from skills.last30days_to_epub_telegram.tools import last30days_to_epub_telegram_tool

result = await last30days_to_epub_telegram_tool({
    'topic': 'multi agent systems',
    'send_telegram': True,
    'telegram_chat_id': '810015653'
})
```

## Parameters

- `topic` (str, required): Research topic
- `deep` (bool, optional): Deep research mode (default: False)
- `quick` (bool, optional): Quick research mode (default: False)
- `title` (str, optional): Report title
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID
- `output_dir` (str, optional): Output directory

## Architecture

Uses composite skill framework for DRY workflow composition.
No code duplication - reuses existing skills.

## Workflow

```
Task Progress:
- [ ] Step 1: Research recent topics
- [ ] Step 2: Generate EPUB
- [ ] Step 3: Send to Telegram
```

**Step 1: Research recent topics**
Use last30days-claude-cli to research the topic from the past 30 days.

**Step 2: Generate EPUB**
Convert the research markdown into an EPUB ebook format.

**Step 3: Send to Telegram**
Deliver the EPUB file to the specified Telegram chat.

## Triggers
- "last30days to epub telegram"
- "send to telegram"
- "telegram message"
- "notify via telegram"
- "generate"

## Category
document-creation
