---
name: converting-v2v-to-pdf-telegram
description: "This composite skill combines: 1. V2V trending search 2. PDF generation 3. Telegram sending 4. reMarkable uploading. Use when the user wants to create pdf, generate pdf, convert to pdf."
---

# V2V to PDF + Telegram + reMarkable Skill

Complete workflow: Search V2V trending topics → Generate PDF → Send to Telegram and reMarkable.

## Description

This composite skill combines:
1. V2V trending search
2. PDF generation
3. Telegram sending
4. reMarkable uploading


## Type
composite

## Base Skills
- voice
- document-converter
- telegram-sender
- remarkable-sender

## Execution
sequential


## Capabilities
- media
- document
- communicate

## Usage

```python
from skills.v2v_to_pdf_telegram_remarkable.tools import v2v_to_pdf_and_send_tool

result = await v2v_to_pdf_and_send_tool({
    'query': 'multi agent systems',
    'send_telegram': True,
    'send_remarkable': True
})
```

## Parameters

- `query` (str, optional): Search query (default: 'trending topics')
- `title` (str, optional): Report title
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `send_remarkable` (bool, optional): Send to reMarkable (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID
- `remarkable_folder` (str, optional): reMarkable folder (default: '/')

## Workflow

```
Task Progress:
- [ ] Step 1: Search V2V trending
- [ ] Step 2: Generate PDF
- [ ] Step 3: Send to Telegram
- [ ] Step 4: Upload to reMarkable
```

**Step 1: Search V2V trending**
Find trending topics and discussions on V2V.ai.

**Step 2: Generate PDF**
Convert the research results into a formatted PDF document.

**Step 3: Send to Telegram**
Deliver the PDF to the specified Telegram chat.

**Step 4: Upload to reMarkable**
Upload the PDF to the reMarkable tablet for reading.

## Triggers
- "v2v to pdf telegram remarkable"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"
- "send to telegram"
- "telegram message"
- "notify via telegram"

## Category
document-creation
