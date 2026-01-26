"""
Telegram Sender Skill

Send messages and files to Telegram channels/chats using Telegram Bot API.
"""
import os
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


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
        Dictionary with:
            - success (bool): Whether message was sent
            - message_id (int, optional): Telegram message ID
            - error (str, optional): Error message if failed
    """
    try:
        message = params.get('message')
        if not message:
            return {
                'success': False,
                'error': 'message parameter is required'
            }
        
        # Get credentials from params or environment
        token = params.get('token') or os.getenv('TELEGRAM_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = params.get('chat_id') or os.getenv('TELEGRAM_CHAT_ID')
        
        if not token:
            return {
                'success': False,
                'error': 'Telegram token required. Set TELEGRAM_TOKEN env var or provide token parameter'
            }
        
        if not chat_id:
            return {
                'success': False,
                'error': 'Chat ID required. Set TELEGRAM_CHAT_ID env var or provide chat_id parameter'
            }
        
        # Build API URL
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Prepare payload
        payload = {
            'chat_id': chat_id,
            'text': message,
            'disable_notification': params.get('disable_notification', False)
        }
        
        # Add parse mode if specified
        parse_mode = params.get('parse_mode')
        if parse_mode:
            payload['parse_mode'] = parse_mode
        
        # Send message
        logger.info(f"Sending Telegram message to chat {chat_id}")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                return {
                    'success': True,
                    'message_id': result.get('result', {}).get('message_id'),
                    'chat_id': chat_id
                }
            else:
                return {
                    'success': False,
                    'error': f"Telegram API error: {result.get('description', 'Unknown error')}"
                }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text[:200]}"
            }
            
    except Exception as e:
        logger.error(f"Telegram send error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to send Telegram message: {str(e)}'
        }


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
        Dictionary with:
            - success (bool): Whether file was sent
            - message_id (int, optional): Telegram message ID
            - error (str, optional): Error message if failed
    """
    try:
        file_path = params.get('file_path')
        if not file_path:
            return {
                'success': False,
                'error': 'file_path parameter is required'
            }
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {
                'success': False,
                'error': f'File not found: {file_path}'
            }
        
        # Get credentials
        token = params.get('token') or os.getenv('TELEGRAM_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = params.get('chat_id') or os.getenv('TELEGRAM_CHAT_ID')
        
        if not token:
            return {
                'success': False,
                'error': 'Telegram token required. Set TELEGRAM_TOKEN env var or provide token parameter'
            }
        
        if not chat_id:
            return {
                'success': False,
                'error': 'Chat ID required. Set TELEGRAM_CHAT_ID env var or provide chat_id parameter'
            }
        
        # Determine file type and API endpoint
        file_size = file_path_obj.stat().st_size
        file_ext = file_path_obj.suffix.lower()
        
        # Telegram has 50MB limit for files
        if file_size > 50 * 1024 * 1024:
            return {
                'success': False,
                'error': f'File too large: {file_size / 1024 / 1024:.2f}MB (max 50MB)'
            }
        
        # Choose API endpoint based on file type
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            endpoint = 'sendPhoto'
        elif file_ext == '.pdf':
            endpoint = 'sendDocument'
        else:
            endpoint = 'sendDocument'
        
        url = f"https://api.telegram.org/bot{token}/{endpoint}"
        
        # Prepare files and data
        with open(file_path_obj, 'rb') as f:
            files = {
                'document' if endpoint == 'sendDocument' else 'photo': (file_path_obj.name, f, 'application/octet-stream' if endpoint == 'sendDocument' else None)
            }
            
            data = {
                'chat_id': chat_id
            }
            
            caption = params.get('caption')
            if caption:
                data['caption'] = caption
                parse_mode = params.get('parse_mode')
                if parse_mode:
                    data['parse_mode'] = parse_mode
            
            # Send file
            logger.info(f"Sending file to Telegram: {file_path_obj.name}")
            response = requests.post(url, files=files, data=data, timeout=120)  # Longer timeout for large files
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                return {
                    'success': True,
                    'message_id': result.get('result', {}).get('message_id'),
                    'chat_id': chat_id,
                    'file_name': file_path_obj.name,
                    'file_size': file_size
                }
            else:
                return {
                    'success': False,
                    'error': f"Telegram API error: {result.get('description', 'Unknown error')}"
                }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text[:200]}"
            }
            
    except Exception as e:
        logger.error(f"Telegram file send error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Failed to send Telegram file: {str(e)}'
        }


__all__ = ['send_telegram_message_tool', 'send_telegram_file_tool']
