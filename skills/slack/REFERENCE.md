# Slack Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`send_message_tool`](#send_message_tool) | Send a message to a Slack channel or DM. |
| [`list_channels_tool`](#list_channels_tool) | List Slack channels. |
| [`read_messages_tool`](#read_messages_tool) | Read recent messages from a Slack channel. |
| [`add_reaction_tool`](#add_reaction_tool) | Add an emoji reaction to a message. |
| [`upload_file_tool`](#upload_file_tool) | Upload a file to a Slack channel. |
| [`get_user_info_tool`](#get_user_info_tool) | Get details about a Slack user. |

---

## `send_message_tool`

Send a message to a Slack channel or DM.

**Parameters:**

- **channel** (`str, required`): Channel ID or name (e.g., '#general', 'C1234567890')
- **text** (`str, required`): Message text
- **token** (`str, optional`): Slack bot token
- **thread_ts** (`str, optional`): Thread timestamp to reply in thread
- **unfurl_links** (`bool, optional`): Enable unfurling of links (default: True)
- **unfurl_media** (`bool, optional`): Enable unfurling of media (default: True)

**Returns:** Dictionary with success, channel, ts, message

---

## `list_channels_tool`

List Slack channels.

**Parameters:**

- **token** (`str, optional`): Slack bot token
- **types** (`str, optional`): Comma-separated channel types (default: 'public_channel')
- **exclude_archived** (`bool, optional`): Exclude archived channels (default: True)
- **limit** (`int, optional`): Maximum number of channels (default: 100)
- **cursor** (`str, optional`): Pagination cursor

**Returns:** Dictionary with success, channels list, channel_count, response_metadata

---

## `read_messages_tool`

Read recent messages from a Slack channel.

**Parameters:**

- **channel** (`str, required`): Channel ID (e.g., 'C1234567890')
- **token** (`str, optional`): Slack bot token
- **limit** (`int, optional`): Number of messages (default: 10, max: 100)
- **oldest** (`str, optional`): Unix timestamp of oldest message
- **latest** (`str, optional`): Unix timestamp of latest message
- **inclusive** (`bool, optional`): Include messages with oldest/latest ts

**Returns:** Dictionary with success, messages list, message_count, has_more

---

## `add_reaction_tool`

Add an emoji reaction to a message.

**Parameters:**

- **channel** (`str, required`): Channel ID
- **timestamp** (`str, required`): Message timestamp (ts)
- **name** (`str, required`): Emoji name without colons (e.g., 'thumbsup')
- **token** (`str, optional`): Slack bot token

**Returns:** Dictionary with success, channel, timestamp, reaction

---

## `upload_file_tool`

Upload a file to a Slack channel.

**Parameters:**

- **file_path** (`str, required`): Path to file to upload
- **channels** (`str, required`): Comma-separated channel IDs
- **token** (`str, optional`): Slack bot token
- **title** (`str, optional`): Title of the file
- **initial_comment** (`str, optional`): Initial comment
- **thread_ts** (`str, optional`): Thread timestamp

**Returns:** Dictionary with success, file object

---

## `get_user_info_tool`

Get details about a Slack user.

**Parameters:**

- **user** (`str, required`): User ID (e.g., 'U1234567890')
- **token** (`str, optional`): Slack bot token
- **include_locale** (`bool, optional`): Include user's locale

**Returns:** Dictionary with success, user object
