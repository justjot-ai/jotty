"""
Slack Skill

Interact with Slack using the Slack Web API via requests.
"""
import os
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Try to load .env from Jotty root (parent of skills directory)
    current_file = Path(__file__).resolve()
    jotty_root = current_file.parent.parent.parent  # skills/slack -> skills -> Jotty
    env_file = jotty_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars
except ImportError:
    pass  # python-dotenv not available, fall back to os.getenv

logger = logging.getLogger(__name__)

SLACK_API_BASE = "https://slack.com/api/"


class SlackAPIClient:
    """Helper class for Slack API interactions."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or self._get_token()

    def _get_token(self) -> Optional[str]:
        """Get Slack token from environment or config file."""
        # Try environment variable first
        token = os.getenv('SLACK_BOT_TOKEN')
        if token:
            return token

        # Try config file
        config_path = Path.home() / ".config" / "slack" / "token"
        if config_path.exists():
            return config_path.read_text().strip()

        return None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=utf-8"
        }

    def _make_request(self, endpoint: str, method: str = "POST",
                      json_data: Optional[Dict] = None,
                      files: Optional[Dict] = None,
                      data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request to Slack API."""
        url = f"{SLACK_API_BASE}{endpoint}"

        try:
            if files:
                # For file uploads, use multipart form-data
                headers = {"Authorization": f"Bearer {self.token}"}
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            elif method == "GET":
                response = requests.get(url, headers=self._get_headers(), params=json_data, timeout=30)
            else:
                response = requests.post(url, headers=self._get_headers(), json=json_data, timeout=30)

            response.raise_for_status()
            result = response.json()

            if not result.get('ok'):
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown Slack API error'),
                    'response_metadata': result.get('response_metadata')
                }

            return {'success': True, **result}

        except requests.exceptions.RequestException as e:
            logger.error(f"Slack API request failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


def send_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message to a Slack channel or DM.

    Args:
        params: Dictionary containing:
            - channel (str, required): Channel ID or name (e.g., '#general', 'C1234567890')
            - text (str, required): Message text
            - token (str, optional): Slack bot token (defaults to SLACK_BOT_TOKEN env var)
            - thread_ts (str, optional): Thread timestamp to reply in thread
            - unfurl_links (bool, optional): Enable unfurling of links (default: True)
            - unfurl_media (bool, optional): Enable unfurling of media (default: True)

    Returns:
        Dictionary with:
            - success (bool): Whether message was sent
            - channel (str): Channel ID
            - ts (str): Message timestamp
            - message (dict): Message object
            - error (str, optional): Error message if failed
    """
    try:
        channel = params.get('channel')
        text = params.get('text')

        if not channel:
            return {'success': False, 'error': 'channel parameter is required'}

        if not text:
            return {'success': False, 'error': 'text parameter is required'}

        client = SlackAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
            }

        payload = {
            'channel': channel,
            'text': text,
            'unfurl_links': params.get('unfurl_links', True),
            'unfurl_media': params.get('unfurl_media', True)
        }

        if params.get('thread_ts'):
            payload['thread_ts'] = params['thread_ts']

        logger.info(f"Sending Slack message to channel {channel}")
        result = client._make_request('chat.postMessage', json_data=payload)

        if result.get('success'):
            return {
                'success': True,
                'channel': result.get('channel'),
                'ts': result.get('ts'),
                'message': result.get('message')
            }

        return result

    except Exception as e:
        logger.error(f"Slack send message error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to send Slack message: {str(e)}'}


def list_channels_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List Slack channels.

    Args:
        params: Dictionary containing:
            - token (str, optional): Slack bot token (defaults to SLACK_BOT_TOKEN env var)
            - types (str, optional): Comma-separated channel types: 'public_channel', 'private_channel', 'mpim', 'im' (default: 'public_channel')
            - exclude_archived (bool, optional): Exclude archived channels (default: True)
            - limit (int, optional): Maximum number of channels to return (default: 100)
            - cursor (str, optional): Pagination cursor

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - channels (list): List of channel objects
            - response_metadata (dict): Pagination metadata
            - error (str, optional): Error message if failed
    """
    try:
        client = SlackAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
            }

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
            return {
                'success': True,
                'channels': [
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
                'channel_count': len(channels),
                'response_metadata': result.get('response_metadata')
            }

        return result

    except Exception as e:
        logger.error(f"Slack list channels error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to list Slack channels: {str(e)}'}


def read_messages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read recent messages from a Slack channel.

    Args:
        params: Dictionary containing:
            - channel (str, required): Channel ID (e.g., 'C1234567890')
            - token (str, optional): Slack bot token (defaults to SLACK_BOT_TOKEN env var)
            - limit (int, optional): Number of messages to retrieve (default: 10, max: 100)
            - oldest (str, optional): Unix timestamp of oldest message to include
            - latest (str, optional): Unix timestamp of latest message to include
            - inclusive (bool, optional): Include messages with oldest/latest ts (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - messages (list): List of message objects
            - has_more (bool): Whether there are more messages
            - error (str, optional): Error message if failed
    """
    try:
        channel = params.get('channel')

        if not channel:
            return {'success': False, 'error': 'channel parameter is required'}

        client = SlackAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
            }

        payload = {
            'channel': channel,
            'limit': min(params.get('limit', 10), 100)
        }

        if params.get('oldest'):
            payload['oldest'] = params['oldest']
        if params.get('latest'):
            payload['latest'] = params['latest']
        if params.get('inclusive'):
            payload['inclusive'] = params['inclusive']

        logger.info(f"Reading messages from Slack channel {channel}")
        result = client._make_request('conversations.history', method="GET", json_data=payload)

        if result.get('success'):
            messages = result.get('messages', [])
            return {
                'success': True,
                'messages': [
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
                'message_count': len(messages),
                'has_more': result.get('has_more', False),
                'response_metadata': result.get('response_metadata')
            }

        return result

    except Exception as e:
        logger.error(f"Slack read messages error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to read Slack messages: {str(e)}'}


def add_reaction_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an emoji reaction to a message.

    Args:
        params: Dictionary containing:
            - channel (str, required): Channel ID where the message is
            - timestamp (str, required): Message timestamp (ts)
            - name (str, required): Emoji name without colons (e.g., 'thumbsup', 'heart')
            - token (str, optional): Slack bot token (defaults to SLACK_BOT_TOKEN env var)

    Returns:
        Dictionary with:
            - success (bool): Whether reaction was added
            - error (str, optional): Error message if failed
    """
    try:
        channel = params.get('channel')
        timestamp = params.get('timestamp')
        name = params.get('name')

        if not channel:
            return {'success': False, 'error': 'channel parameter is required'}

        if not timestamp:
            return {'success': False, 'error': 'timestamp parameter is required'}

        if not name:
            return {'success': False, 'error': 'name parameter is required (emoji name)'}

        # Remove colons if provided
        name = name.strip(':')

        client = SlackAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
            }

        payload = {
            'channel': channel,
            'timestamp': timestamp,
            'name': name
        }

        logger.info(f"Adding reaction :{name}: to message in channel {channel}")
        result = client._make_request('reactions.add', json_data=payload)

        if result.get('success'):
            return {
                'success': True,
                'channel': channel,
                'timestamp': timestamp,
                'reaction': name
            }

        return result

    except Exception as e:
        logger.error(f"Slack add reaction error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add Slack reaction: {str(e)}'}


def upload_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload a file to a Slack channel.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to file to upload
            - channels (str, required): Comma-separated channel IDs (e.g., 'C1234567890,C0987654321')
            - token (str, optional): Slack bot token (defaults to SLACK_BOT_TOKEN env var)
            - title (str, optional): Title of the file
            - initial_comment (str, optional): Initial comment to add to the file
            - thread_ts (str, optional): Thread timestamp to upload in thread

    Returns:
        Dictionary with:
            - success (bool): Whether file was uploaded
            - file (dict): Uploaded file object
            - error (str, optional): Error message if failed
    """
    try:
        file_path = params.get('file_path')
        channels = params.get('channels')

        if not file_path:
            return {'success': False, 'error': 'file_path parameter is required'}

        if not channels:
            return {'success': False, 'error': 'channels parameter is required'}

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {'success': False, 'error': f'File not found: {file_path}'}

        # Check file size (Slack has limits based on plan)
        file_size = file_path_obj.stat().st_size
        if file_size > 1024 * 1024 * 1024:  # 1GB limit
            return {'success': False, 'error': f'File too large: {file_size / 1024 / 1024:.2f}MB'}

        client = SlackAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
            }

        # Prepare file upload
        with open(file_path_obj, 'rb') as f:
            files = {
                'file': (file_path_obj.name, f)
            }

            data = {
                'channels': channels
            }

            if params.get('title'):
                data['title'] = params['title']
            if params.get('initial_comment'):
                data['initial_comment'] = params['initial_comment']
            if params.get('thread_ts'):
                data['thread_ts'] = params['thread_ts']

            logger.info(f"Uploading file {file_path_obj.name} to Slack channels {channels}")
            result = client._make_request('files.upload', files=files, data=data)

        if result.get('success'):
            file_obj = result.get('file', {})
            return {
                'success': True,
                'file': {
                    'id': file_obj.get('id'),
                    'name': file_obj.get('name'),
                    'title': file_obj.get('title'),
                    'mimetype': file_obj.get('mimetype'),
                    'size': file_obj.get('size'),
                    'url_private': file_obj.get('url_private'),
                    'permalink': file_obj.get('permalink')
                }
            }

        return result

    except Exception as e:
        logger.error(f"Slack upload file error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to upload Slack file: {str(e)}'}


def get_user_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get details about a Slack user.

    Args:
        params: Dictionary containing:
            - user (str, required): User ID (e.g., 'U1234567890')
            - token (str, optional): Slack bot token (defaults to SLACK_BOT_TOKEN env var)
            - include_locale (bool, optional): Include user's locale (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether request succeeded
            - user (dict): User profile information
            - error (str, optional): Error message if failed
    """
    try:
        user_id = params.get('user')

        if not user_id:
            return {'success': False, 'error': 'user parameter is required'}

        client = SlackAPIClient(params.get('token'))

        if not client.token:
            return {
                'success': False,
                'error': 'Slack token required. Set SLACK_BOT_TOKEN env var or provide token parameter'
            }

        payload = {
            'user': user_id,
            'include_locale': params.get('include_locale', False)
        }

        logger.info(f"Getting Slack user info for {user_id}")
        result = client._make_request('users.info', method="GET", json_data=payload)

        if result.get('success'):
            user = result.get('user', {})
            profile = user.get('profile', {})
            return {
                'success': True,
                'user': {
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
            }

        return result

    except Exception as e:
        logger.error(f"Slack get user info error: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to get Slack user info: {str(e)}'}


__all__ = [
    'send_message_tool',
    'list_channels_tool',
    'read_messages_tool',
    'add_reaction_tool',
    'upload_file_tool',
    'get_user_info_tool'
]
