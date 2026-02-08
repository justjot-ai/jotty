# Slack Skill

Interact with Slack channels and users using the Slack Web API.

## Description

This skill provides tools to send messages, read channels, upload files, and interact with Slack workspaces using the Slack Web API via requests.


## Type
base

## Features

- Send messages to channels and DMs
- List available channels
- Read recent messages from channels
- Add emoji reactions to messages
- Upload files to channels
- Get user profile information

## Usage

```python
from skills.slack.tools import (
    send_message_tool,
    list_channels_tool,
    read_messages_tool,
    add_reaction_tool,
    upload_file_tool,
    get_user_info_tool
)

# Send a message
result = send_message_tool({
    'channel': '#general',
    'text': 'Hello from Jotty!'
})

# List channels
result = list_channels_tool({
    'types': 'public_channel,private_channel',
    'limit': 50
})

# Read messages
result = read_messages_tool({
    'channel': 'C1234567890',
    'limit': 20
})

# Add reaction
result = add_reaction_tool({
    'channel': 'C1234567890',
    'timestamp': '1234567890.123456',
    'name': 'thumbsup'
})

# Upload file
result = upload_file_tool({
    'file_path': '/path/to/document.pdf',
    'channels': 'C1234567890',
    'title': 'Report PDF',
    'initial_comment': 'Here is the report'
})

# Get user info
result = get_user_info_tool({
    'user': 'U1234567890'
})
```

## Tools

### send_message_tool

Send a message to a Slack channel or DM.

**Parameters:**
- `channel` (str, required): Channel ID or name (e.g., '#general', 'C1234567890')
- `text` (str, required): Message text
- `token` (str, optional): Slack bot token
- `thread_ts` (str, optional): Thread timestamp to reply in thread
- `unfurl_links` (bool, optional): Enable unfurling of links (default: True)
- `unfurl_media` (bool, optional): Enable unfurling of media (default: True)

### list_channels_tool

List Slack channels.

**Parameters:**
- `token` (str, optional): Slack bot token
- `types` (str, optional): Comma-separated channel types (default: 'public_channel')
- `exclude_archived` (bool, optional): Exclude archived channels (default: True)
- `limit` (int, optional): Maximum channels to return (default: 100)
- `cursor` (str, optional): Pagination cursor

### read_messages_tool

Read recent messages from a Slack channel.

**Parameters:**
- `channel` (str, required): Channel ID
- `token` (str, optional): Slack bot token
- `limit` (int, optional): Number of messages (default: 10, max: 100)
- `oldest` (str, optional): Unix timestamp of oldest message
- `latest` (str, optional): Unix timestamp of latest message
- `inclusive` (bool, optional): Include oldest/latest messages (default: False)

### add_reaction_tool

Add an emoji reaction to a message.

**Parameters:**
- `channel` (str, required): Channel ID
- `timestamp` (str, required): Message timestamp (ts)
- `name` (str, required): Emoji name without colons (e.g., 'thumbsup', 'heart')
- `token` (str, optional): Slack bot token

### upload_file_tool

Upload a file to a Slack channel.

**Parameters:**
- `file_path` (str, required): Path to file
- `channels` (str, required): Comma-separated channel IDs
- `token` (str, optional): Slack bot token
- `title` (str, optional): Title of the file
- `initial_comment` (str, optional): Initial comment
- `thread_ts` (str, optional): Thread timestamp

### get_user_info_tool

Get details about a Slack user.

**Parameters:**
- `user` (str, required): User ID (e.g., 'U1234567890')
- `token` (str, optional): Slack bot token
- `include_locale` (bool, optional): Include user's locale (default: False)

## Configuration

Set the Slack bot token via:

1. Environment variable: `SLACK_BOT_TOKEN`
2. Config file: `~/.config/slack/token`

### Required Bot Token Scopes

Ensure your Slack app has these OAuth scopes:
- `channels:read` - List channels
- `channels:history` - Read channel messages
- `chat:write` - Send messages
- `reactions:write` - Add reactions
- `files:write` - Upload files
- `users:read` - Get user info
- `users:read.email` - Get user email (optional)

## API Reference

Base URL: https://slack.com/api/

This skill uses the Slack Web API with the following endpoints:
- `chat.postMessage` - Send messages
- `conversations.list` - List channels
- `conversations.history` - Read messages
- `reactions.add` - Add reactions
- `files.upload` - Upload files
- `users.info` - Get user details
