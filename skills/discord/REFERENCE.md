# Discord Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`send_message_tool`](#send_message_tool) | Send a message to a Discord channel. |
| [`list_channels_tool`](#list_channels_tool) | List channels in a Discord guild (server). |
| [`read_messages_tool`](#read_messages_tool) | Read recent messages from a Discord channel. |
| [`add_reaction_tool`](#add_reaction_tool) | Add an emoji reaction to a message. |
| [`get_user_info_tool`](#get_user_info_tool) | Get details about a Discord user. |
| [`create_thread_tool`](#create_thread_tool) | Create a thread in a Discord channel. |

---

## `send_message_tool`

Send a message to a Discord channel.

**Parameters:**

- **channel_id** (`str, required`): Channel ID to send message to
- **content** (`str, required`): Message content (up to 2000 characters)
- **token** (`str, optional`): Discord bot token
- **tts** (`bool, optional`): Whether this is a TTS message
- **embed** (`dict, optional`): Embed object to include
- **embeds** (`list, optional`): Array of embed objects (max 10)
- **message_reference** (`dict, optional`): Message reference for replies

**Returns:** Dictionary with success, id, channel_id, content, author, timestamp

---

## `list_channels_tool`

List channels in a Discord guild (server).

**Parameters:**

- **guild_id** (`str, required`): Guild ID to list channels from
- **token** (`str, optional`): Discord bot token

**Returns:** Dictionary with success, channels list, channel_count

---

## `read_messages_tool`

Read recent messages from a Discord channel.

**Parameters:**

- **channel_id** (`str, required`): Channel ID to read messages from
- **token** (`str, optional`): Discord bot token
- **limit** (`int, optional`): Number of messages (default: 50, max: 100) before/after/around (str, optional): Message ID for pagination

**Returns:** Dictionary with success, messages list, message_count

---

## `add_reaction_tool`

Add an emoji reaction to a message.

**Parameters:**

- **channel_id** (`str, required`): Channel ID
- **message_id** (`str, required`): Message ID to react to
- **emoji** (`str, required`): Emoji (unicode or custom format name:id)
- **token** (`str, optional`): Discord bot token

**Returns:** Dictionary with success, channel_id, message_id, emoji

---

## `get_user_info_tool`

Get details about a Discord user.

**Parameters:**

- **user_id** (`str, required`): User ID to get info for
- **token** (`str, optional`): Discord bot token

**Returns:** Dictionary with success, user object

---

## `create_thread_tool`

Create a thread in a Discord channel.

**Parameters:**

- **channel_id** (`str, required`): Channel ID
- **name** (`str, required`): Thread name (1-100 characters)
- **token** (`str, optional`): Discord bot token
- **message_id** (`str, optional`): Message ID to start thread from
- **auto_archive_duration** (`int, optional`): Minutes (60, 1440, 4320, 10080)
- **type** (`int, optional`): Thread type (10, 11, 12)
- **invitable** (`bool, optional`): Allow non-mods to add users

**Returns:** Dictionary with success, id, name, parent_id, owner_id, type
