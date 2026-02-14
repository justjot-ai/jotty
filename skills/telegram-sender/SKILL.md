---
name: telegram-sender
description: "This skill sends text messages and files (PDFs, images, etc.) to Telegram using the Telegram Bot API. Use when the user wants to send telegram, message on telegram, notify via telegram."
---

# Telegram Sender Skill

Send messages and files to Telegram channels/chats.

## Description

This skill sends text messages and files (PDFs, images, etc.) to Telegram using the Telegram Bot API.


## Type
base


## Capabilities
- communicate


## Triggers
- "send telegram"
- "message on telegram"
- "notify via telegram"
- "telegram notification"

## Category
workflow-automation

## Features

- Send text messages
- Send files (PDFs, images, documents)
- Support for multiple channels/chats
- HTML/Markdown formatting
- File uploads

## Usage

```python
from skills.telegram_sender.tools import send_telegram_message_tool, send_telegram_file_tool

# Send message
result = await send_telegram_message_tool({
    'message': 'Hello from Jotty!',
    'chat_id': '810015653',
    'parse_mode': 'HTML'
})

# Send file
result = await send_telegram_file_tool({
    'file_path': '/path/to/document.pdf',
    'chat_id': '810015653',
    'caption': 'Report PDF'
})
```

## Parameters

- `message` (str, required for messages): Message text
- `file_path` (str, required for files): Path to file
- `chat_id` (str, optional): Chat ID (defaults to TELEGRAM_CHAT_ID env var)
- `token` (str, optional): Bot token (defaults to TELEGRAM_TOKEN env var)
- `parse_mode` (str, optional): 'HTML' or 'Markdown'
- `caption` (str, optional): Caption for files

## Configuration

Set environment variables:
- `TELEGRAM_TOKEN`: Bot token
- `TELEGRAM_CHAT_ID`: Default chat ID
