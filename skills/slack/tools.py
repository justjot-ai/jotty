"""
Slack Skill

Interact with Slack using the Slack Web API via requests.
Refactored to use Jotty core utilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any

# Use centralized utilities
from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.api_client import BaseAPIClient
from Jotty.core.utils.tool_helpers import (
    tool_response, tool_error, tool_wrapper
)

# Load environment variables
load_jotty_env()

logger = logging.getLogger(__name__)


class SlackAPIClient(BaseAPIClient):
    """Slack API client using base utilities."""

    BASE_URL = "https://slack.com/api"
    AUTH_PREFIX = "Bearer"
    TOKEN_ENV_VAR = "SLACK_BOT_TOKEN"
    TOKEN_CONFIG_PATH = ".config/slack/token"
    CONTENT_TYPE = "application/json; charset=utf-8"


def _get_client(params: Dict[str, Any]) -> tuple:
    """Get Slack client, returning (client, error) tuple."""
    client = SlackAPIClient(params.get('token'))
    if not client.token:
        return None, tool_error(
            'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
        )
    return client, None


@tool_wrapper(required_params=['channel', 'text'])
def send_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message to a Slack channel or DM.

    Args:
        params: Dictionary containing:
            - channel (str, required): Channel ID or name (e.g., '#general', 'C1234567890')
            - text (str, required): Message text
            - token (str, optional): Slack bot token
            - thread_ts (str, optional): Thread timestamp to reply in thread
            - unfurl_links (bool, optional): Enable unfurling of links (default: True)
            - unfurl_media (bool, optional): Enable unfurling of media (default: True)

    Returns:
        Dictionary with success, channel, ts, message
    """
    client, error = _get_client(params)
    if error:
        return error

    payload = {
        'channel': params['channel'],
        'text': params['text'],
        'unfurl_links': params.get('unfurl_links', True),
        'unfurl_media': params.get('unfurl_media', True)
    }

    if params.get('thread_ts'):
        payload['thread_ts'] = params['thread_ts']

    logger.info(f"Sending Slack message to channel {params['channel']}")
    result = client._make_request('chat.postMessage', json_data=payload)

    if result.get('success'):
        return tool_response(
            channel=result.get('channel'),
            ts=result.get('ts'),
            message=result.get('message')
        )

    return result


@tool_wrapper()
def list_channels_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List Slack channels.

    Args:
        params: Dictionary containing:
            - token (str, optional): Slack bot token
            - types (str, optional): Comma-separated channel types (default: 'public_channel')
            - exclude_archived (bool, optional): Exclude archived channels (default: True)
            - limit (int, optional): Maximum number of channels (default: 100)
            - cursor (str, optional): Pagination cursor

    Returns:
        Dictionary with success, channels list, channel_count, response_metadata
    """
    client, error = _get_client(params)
    if error:
        return error

    payload = {
        'types': params.get('types', 'public_channel'),
        'exclude_archived': params.get('exclude_archived', True),
        'limit': params.get('limit', 100)
    }

    if params.get('cursor'):
        payload['cursor'] = params['cursor']

    logger.info("Listing Slack channels")
    result = client._make_request('conversations.list', method="GET", json_data=payload)

    if result.get('success'):
        channels = result.get('channels', [])
        return tool_response(
            channels=[
                {
                    'id': ch.get('id'),
                    'name': ch.get('name'),
                    'is_private': ch.get('is_private', False),
                    'is_archived': ch.get('is_archived', False),
                    'num_members': ch.get('num_members', 0),
                    'topic': ch.get('topic', {}).get('value', ''),
                    'purpose': ch.get('purpose', {}).get('value', '')
                }
                for ch in channels
            ],
            channel_count=len(channels),
            response_metadata=result.get('response_metadata')
        )

    return result


@tool_wrapper(required_params=['channel'])
def read_messages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read recent messages from a Slack channel.

    Args:
        params: Dictionary containing:
            - channel (str, required): Channel ID (e.g., 'C1234567890')
            - token (str, optional): Slack bot token
            - limit (int, optional): Number of messages (default: 10, max: 100)
            - oldest (str, optional): Unix timestamp of oldest message
            - latest (str, optional): Unix timestamp of latest message
            - inclusive (bool, optional): Include messages with oldest/latest ts

    Returns:
        Dictionary with success, messages list, message_count, has_more
    """
    client, error = _get_client(params)
    if error:
        return error

    payload = {
        'channel': params['channel'],
        'limit': min(params.get('limit', 10), 100)
    }

    for key in ('oldest', 'latest', 'inclusive'):
        if params.get(key):
            payload[key] = params[key]

    logger.info(f"Reading messages from Slack channel {params['channel']}")
    result = client._make_request('conversations.history', method="GET", json_data=payload)

    if result.get('success'):
        messages = result.get('messages', [])
        return tool_response(
            messages=[
                {
                    'ts': msg.get('ts'),
                    'user': msg.get('user'),
                    'text': msg.get('text'),
                    'type': msg.get('type'),
                    'thread_ts': msg.get('thread_ts'),
                    'reply_count': msg.get('reply_count', 0),
                    'reactions': msg.get('reactions', [])
                }
                for msg in messages
            ],
            message_count=len(messages),
            has_more=result.get('has_more', False),
            response_metadata=result.get('response_metadata')
        )

    return result


@tool_wrapper(required_params=['channel', 'timestamp', 'name'])
def add_reaction_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an emoji reaction to a message.

    Args:
        params: Dictionary containing:
            - channel (str, required): Channel ID
            - timestamp (str, required): Message timestamp (ts)
            - name (str, required): Emoji name without colons (e.g., 'thumbsup')
            - token (str, optional): Slack bot token

    Returns:
        Dictionary with success, channel, timestamp, reaction
    """
    client, error = _get_client(params)
    if error:
        return error

    # Remove colons if provided
    emoji_name = params['name'].strip(':')

    payload = {
        'channel': params['channel'],
        'timestamp': params['timestamp'],
        'name': emoji_name
    }

    logger.info(f"Adding reaction :{emoji_name}: to message in channel {params['channel']}")
    result = client._make_request('reactions.add', json_data=payload)

    if result.get('success'):
        return tool_response(
            channel=params['channel'],
            timestamp=params['timestamp'],
            reaction=emoji_name
        )

    return result


@tool_wrapper(required_params=['file_path', 'channels'])
def upload_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload a file to a Slack channel.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to file to upload
            - channels (str, required): Comma-separated channel IDs
            - token (str, optional): Slack bot token
            - title (str, optional): Title of the file
            - initial_comment (str, optional): Initial comment
            - thread_ts (str, optional): Thread timestamp

    Returns:
        Dictionary with success, file object
    """
    file_path_obj = Path(params['file_path'])
    if not file_path_obj.exists():
        return tool_error(f'File not found: {params["file_path"]}')

    file_size = file_path_obj.stat().st_size
    if file_size > 1024 * 1024 * 1024:  # 1GB limit
        return tool_error(f'File too large: {file_size / 1024 / 1024:.2f}MB')

    client, error = _get_client(params)
    if error:
        return error

    with open(file_path_obj, 'rb') as f:
        files = {'file': (file_path_obj.name, f)}
        data = {'channels': params['channels']}

        for key in ('title', 'initial_comment', 'thread_ts'):
            if params.get(key):
                data[key] = params[key]

        logger.info(f"Uploading file {file_path_obj.name} to Slack channels {params['channels']}")
        result = client._make_request('files.upload', files=files, data=data)

    if result.get('success'):
        file_obj = result.get('file', {})
        return tool_response(
            file={
                'id': file_obj.get('id'),
                'name': file_obj.get('name'),
                'title': file_obj.get('title'),
                'mimetype': file_obj.get('mimetype'),
                'size': file_obj.get('size'),
                'url_private': file_obj.get('url_private'),
                'permalink': file_obj.get('permalink')
            }
        )

    return result


@tool_wrapper(required_params=['user'])
def get_user_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get details about a Slack user.

    Args:
        params: Dictionary containing:
            - user (str, required): User ID (e.g., 'U1234567890')
            - token (str, optional): Slack bot token
            - include_locale (bool, optional): Include user's locale

    Returns:
        Dictionary with success, user object
    """
    client, error = _get_client(params)
    if error:
        return error

    payload = {
        'user': params['user'],
        'include_locale': params.get('include_locale', False)
    }

    logger.info(f"Getting Slack user info for {params['user']}")
    result = client._make_request('users.info', method="GET", json_data=payload)

    if result.get('success'):
        user = result.get('user', {})
        profile = user.get('profile', {})
        return tool_response(
            user={
                'id': user.get('id'),
                'name': user.get('name'),
                'real_name': user.get('real_name'),
                'display_name': profile.get('display_name'),
                'email': profile.get('email'),
                'title': profile.get('title'),
                'phone': profile.get('phone'),
                'status_text': profile.get('status_text'),
                'status_emoji': profile.get('status_emoji'),
                'image_url': profile.get('image_192'),
                'is_admin': user.get('is_admin', False),
                'is_bot': user.get('is_bot', False),
                'tz': user.get('tz'),
                'tz_label': user.get('tz_label'),
                'locale': user.get('locale')
            }
        )

    return result


__all__ = [
    'send_message_tool',
    'list_channels_tool',
    'read_messages_tool',
    'add_reaction_tool',
    'upload_file_tool',
    'get_user_info_tool'
]
