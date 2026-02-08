"""
Telegram Sender Skill

Send messages and files to Telegram channels/chats using Telegram Bot API.
Refactored to use Jotty core utilities.
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any

# Use centralized utilities
from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.api_client import BaseAPIClient
from Jotty.core.utils.tool_helpers import (
    tool_response, tool_error, async_tool_wrapper
)

# Load environment variables
load_jotty_env()

logger = logging.getLogger(__name__)


class TelegramAPIClient(BaseAPIClient):
    """Telegram Bot API client using base utilities."""

    AUTH_PREFIX = ""  # Token is in URL, not header
    TOKEN_ENV_VAR = "TELEGRAM_TOKEN"
    TOKEN_CONFIG_PATH = ".config/telegram/token"
    CONTENT_TYPE = "application/json"

    def __init__(self, token: str = None, chat_id: str = None):
        super().__init__(token or os.getenv('TELEGRAM_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN'))
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.BASE_URL = f"https://api.telegram.org/bot{self.token}" if self.token else ""

    def _get_headers(self) -> Dict[str, str]:
        """Telegram doesn't use Authorization header."""
        return {"Content-Type": self.CONTENT_TYPE}

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle Telegram-specific response format."""
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                return {"success": True, **result.get('result', {})}
            else:
                return {
                    "success": False,
                    "error": f"Telegram API error: {result.get('description', 'Unknown error')}"
                }
        return super()._handle_response(response)


def _get_client(params: Dict[str, Any]) -> tuple:
    """Get Telegram client, returning (client, error) tuple."""
    client = TelegramAPIClient(params.get('token'), params.get('chat_id'))
    if not client.token:
        return None, tool_error(
            'Telegram token required. Set TELEGRAM_TOKEN env var or provide token parameter'
        )
    if not client.chat_id:
        return None, tool_error(
            'Chat ID required. Set TELEGRAM_CHAT_ID env var or provide chat_id parameter'
        )
    return client, None


@async_tool_wrapper(required_params=['message'])
async def send_telegram_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a text message to Telegram.

    Args:
        params: Dictionary containing:
            - message (str, required): Message text
            - chat_id (str, optional): Chat ID (defaults to TELEGRAM_CHAT_ID env var)
            - token (str, optional): Bot token (defaults to TELEGRAM_TOKEN env var)
            - parse_mode (str, optional): 'HTML' or 'Markdown'
            - disable_notification (bool, optional): Silent message

    Returns:
        Dictionary with success, message_id, chat_id
    """
    client, error = _get_client(params)
    if error:
        return error

    payload = {
        'chat_id': client.chat_id,
        'text': params['message'],
        'disable_notification': params.get('disable_notification', False)
    }

    if params.get('parse_mode'):
        payload['parse_mode'] = params['parse_mode']

    logger.info(f"Sending Telegram message to chat {client.chat_id}")
    result = client._make_request('sendMessage', json_data=payload)

    if result.get('success'):
        return tool_response(
            message_id=result.get('message_id'),
            chat_id=client.chat_id
        )

    return result


@async_tool_wrapper(required_params=['file_path'])
async def send_telegram_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a file to Telegram.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to file
            - chat_id (str, optional): Chat ID (defaults to TELEGRAM_CHAT_ID env var)
            - token (str, optional): Bot token (defaults to TELEGRAM_TOKEN env var)
            - caption (str, optional): File caption
            - parse_mode (str, optional): 'HTML' or 'Markdown' for caption

    Returns:
        Dictionary with success, message_id, chat_id, file_name, file_size
    """
    file_path_obj = Path(params['file_path'])
    if not file_path_obj.exists():
        return tool_error(f'File not found: {params["file_path"]}')

    file_size = file_path_obj.stat().st_size
    if file_size > 50 * 1024 * 1024:  # Telegram 50MB limit
        return tool_error(f'File too large: {file_size / 1024 / 1024:.2f}MB (max 50MB)')

    client, error = _get_client(params)
    if error:
        return error

    # Choose endpoint based on file type
    file_ext = file_path_obj.suffix.lower()
    if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        endpoint = 'sendPhoto'
        file_key = 'photo'
    else:
        endpoint = 'sendDocument'
        file_key = 'document'

    url = f"{client.BASE_URL}/{endpoint}"

    with open(file_path_obj, 'rb') as f:
        files = {file_key: (file_path_obj.name, f)}
        data = {'chat_id': client.chat_id}

        if params.get('caption'):
            data['caption'] = params['caption']
            if params.get('parse_mode'):
                data['parse_mode'] = params['parse_mode']

        logger.info(f"Sending file to Telegram: {file_path_obj.name}")
        response = requests.post(url, files=files, data=data, timeout=120)

    if response.status_code == 200:
        result = response.json()
        if result.get('ok'):
            return tool_response(
                message_id=result.get('result', {}).get('message_id'),
                chat_id=client.chat_id,
                file_name=file_path_obj.name,
                file_size=file_size
            )
        else:
            return tool_error(f"Telegram API error: {result.get('description', 'Unknown error')}")

    return tool_error(f"HTTP {response.status_code}: {response.text[:200]}")


__all__ = ['send_telegram_message_tool', 'send_telegram_file_tool']
