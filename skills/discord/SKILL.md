---
name: discord
description: "This skill provides tools to send messages, read channels, create threads, and interact with Discord servers using the Discord API via requests. Use when the user wants to send discord, discord message, post to discord."
---

# Discord Skill

Interact with Discord servers and channels using the Discord API.

## Description

This skill provides tools to send messages, read channels, create threads, and interact with Discord servers using the Discord API via requests.


## Type
base


## Capabilities
- communicate


## Triggers
- "send discord"
- "discord message"
- "post to discord"
- "discord notification"

## Category
workflow-automation

## Features

- Send messages to channels
- List guild (server) channels
- Read recent messages from channels
- Add emoji reactions to messages
- Get user profile information
- Create threads from messages or in channels

## Usage

```python
from skills.discord.tools import (
    send_message_tool,
    list_channels_tool,
    read_messages_tool,
    add_reaction_tool,
    get_user_info_tool,
    create_thread_tool
)

# Send a message
result = send_message_tool({
    'channel_id': '1234567890123456789',
    'content': 'Hello from Jotty!'
})

# List channels in a guild
result = list_channels_tool({
    'guild_id': '1234567890123456789'
})

# Read messages
result = read_messages_tool({
    'channel_id': '1234567890123456789',
    'limit': 20
})

# Add reaction
result = add_reaction_tool({
    'channel_id': '1234567890123456789',
    'message_id': '9876543210987654321',
    'emoji': '\U0001F44D'  # thumbs up
})

# Get user info
result = get_user_info_tool({
    'user_id': '1234567890123456789'
})

# Create a thread from a message
result = create_thread_tool({
    'channel_id': '1234567890123456789',
    'message_id': '9876543210987654321',
    'name': 'Discussion Thread'
})

# Create a standalone thread
result = create_thread_tool({
    'channel_id': '1234567890123456789',
    'name': 'New Topic',
    'type': 11  # public thread
})
```

## Tools

### send_message_tool

Send a message to a Discord channel.

**Parameters:**
- `channel_id` (str, required): Channel ID to send message to
- `content` (str, required): Message content (up to 2000 characters)
- `token` (str, optional): Discord bot token
- `tts` (bool, optional): Whether this is a TTS message (default: False)
- `embed` (dict, optional): Embed object to include
- `embeds` (list, optional): Array of embed objects (max 10)
- `message_reference` (dict, optional): Message reference for replies

### list_channels_tool

List channels in a Discord guild (server).

**Parameters:**
- `guild_id` (str, required): Guild ID to list channels from
- `token` (str, optional): Discord bot token

**Returns:** List of channels with id, name, type, position, topic, etc.

### read_messages_tool

Read recent messages from a Discord channel.

**Parameters:**
- `channel_id` (str, required): Channel ID to read messages from
- `token` (str, optional): Discord bot token
- `limit` (int, optional): Number of messages (default: 50, max: 100)
- `before` (str, optional): Get messages before this message ID
- `after` (str, optional): Get messages after this message ID
- `around` (str, optional): Get messages around this message ID

### add_reaction_tool

Add an emoji reaction to a message.

**Parameters:**
- `channel_id` (str, required): Channel ID where the message is
- `message_id` (str, required): Message ID to react to
- `emoji` (str, required): Emoji (unicode emoji or custom format `name:id`)
- `token` (str, optional): Discord bot token

### get_user_info_tool

Get details about a Discord user.

**Parameters:**
- `user_id` (str, required): User ID to get info for
- `token` (str, optional): Discord bot token

**Returns:** User information including username, discriminator, avatar, etc.

### create_thread_tool

Create a thread in a Discord channel.

**Parameters:**
- `channel_id` (str, required): Channel ID to create thread in
- `name` (str, required): Thread name (1-100 characters)
- `token` (str, optional): Discord bot token
- `message_id` (str, optional): Message ID to start thread from
- `auto_archive_duration` (int, optional): Minutes until auto-archive (60, 1440, 4320, 10080)
- `type` (int, optional): Thread type - 10 (announcement), 11 (public), 12 (private)
- `invitable` (bool, optional): Whether non-moderators can add users (private threads only)

## Configuration

Set the Discord bot token via:

1. Environment variable: `DISCORD_BOT_TOKEN`
2. Config file: `~/.config/discord/token`

### Required Bot Permissions

Ensure your Discord bot has these permissions:
- `Send Messages` - Send messages to channels
- `Read Message History` - Read channel messages
- `Add Reactions` - Add reactions to messages
- `Create Public Threads` - Create public threads
- `Create Private Threads` - Create private threads
- `View Channels` - View/list channels

### Required OAuth2 Scopes

- `bot` - Bot user scope
- `applications.commands` - For slash commands (optional)

### Required Bot Intents

Enable these intents in the Discord Developer Portal:
- `Message Content Intent` - To read message content
- `Guild Members Intent` - To get member information (optional)

## API Reference

Base URL: https://discord.com/api/v10

This skill uses the Discord API with the following endpoints:
- `POST /channels/{channel.id}/messages` - Send messages
- `GET /guilds/{guild.id}/channels` - List guild channels
- `GET /channels/{channel.id}/messages` - Read messages
- `PUT /channels/{channel.id}/messages/{message.id}/reactions/{emoji}/@me` - Add reaction
- `GET /users/{user.id}` - Get user info
- `POST /channels/{channel.id}/messages/{message.id}/threads` - Create thread from message
- `POST /channels/{channel.id}/threads` - Create thread without message

## Channel Types

| Type ID | Type Name |
|---------|-----------|
| 0 | Text Channel |
| 2 | Voice Channel |
| 4 | Category |
| 5 | Announcement Channel |
| 10 | Announcement Thread |
| 11 | Public Thread |
| 12 | Private Thread |
| 13 | Stage Voice |
| 14 | Directory |
| 15 | Forum |
| 16 | Media |
