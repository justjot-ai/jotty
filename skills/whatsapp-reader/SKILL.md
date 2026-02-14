# WhatsApp Reader Skill

Read messages from WhatsApp Web chats (personal WhatsApp via Baileys). Use with Jotty swarm to read channel/group messages and summarize learnings.

## Description

Reads messages from a WhatsApp chat or group using the existing WhatsApp Web session (same session as `/whatsapp login`). Supports listing chats and fetching messages by chat name or ID. Intended for summarizing learnings from a channel/group (e.g. #my-ai-ml).

## Type
base

## Capabilities
- read
- analyze

## Prerequisites

- WhatsApp Web session must be initialized in the same Jotty process (run `/whatsapp login` in CLI first, then use `/run` in the same session).
- Baileys bridge with store support (chats and messages are populated as you use WhatsApp).

## Tools

### read_whatsapp_chat_messages_tool

Read messages from a WhatsApp chat or group.

**Parameters:**
- `chat_name` (str, optional): Chat or group name to match (e.g. "my-ai-ml" or "#my-ai-ml"). Partial match supported.
- `chat_id` (str, optional): WhatsApp chat JID (e.g. "1234567890@s.whatsapp.net" or "xxx@g.us"). Use if you know the ID.
- `limit` (int, optional): Max messages to return (default 100, max 500).

**Returns:**
- `success` (bool): Whether the read succeeded
- `messages` (list): List of message objects with `body`, `timestamp`, `fromMe`, `id`
- `chat_id` (str): Resolved chat JID
- `error` (str, optional): Error message if success is False

### summarize_whatsapp_chat_learnings_tool

Read messages from a WhatsApp chat and produce a summary of key learnings (uses summarize skill).

**Parameters:**
- `chat_name` (str, required): Chat or group name (e.g. "my-ai-ml" or "#my-ai-ml")
- `limit` (int, optional): Max messages to include (default 200)
- `length` (str, optional): Summary length - 'short', 'medium', 'long' (default 'medium')
- `style` (str, optional): 'bullet', 'paragraph', 'numbered' (default 'bullet')

**Returns:**
- `success` (bool)
- `summary` (str): Text summary of learnings
- `message_count` (int): Number of messages summarized
- `error` (str, optional)

## Usage

From Jotty CLI (with WhatsApp already connected):

```
/whatsapp login
# ... scan QR, then in same session:
/run Read all messages from the WhatsApp chat #my-ai-ml and summarize the key learnings
```

Or call the skill directly from swarm/API:

- `read_whatsapp_chat_messages_tool({ "chat_name": "my-ai-ml", "limit": 200 })`
- `summarize_whatsapp_chat_learnings_tool({ "chat_name": "my-ai-ml", "length": "medium" })`

## Environment

Uses the global WhatsApp Web client set by the CLI when you run `/whatsapp login`. No extra env vars for reading.

## Triggers
- "whatsapp reader"
- "whatsapp message"
- "read whatsapp"
- "whatsapp chat"

## Category
communication
