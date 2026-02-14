"""
Google Gemini Skill

Text generation, vision, and chat capabilities using Google Gemini API.
Supports multiple Gemini models including gemini-1.5-flash, gemini-1.5-pro, and gemini-2.0-flash.
"""
import os
import base64
import json
import logging
from typing import Dict, Any, List, Optional

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

logger = logging.getLogger(__name__)

# API Configuration

# Status emitter for progress updates
status = SkillStatus("gemini")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-1.5-flash"
SUPPORTED_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]


class GeminiAPIClient:
    """Client for interacting with Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API client.

        Args:
            api_key: Google API key. If not provided, reads from
                     GOOGLE_API_KEY or GEMINI_API_KEY environment variables.
        """
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")

    def _get_endpoint(self, model: str, action: str = "generateContent") -> str:
        """Get the API endpoint for a given model and action."""
        return f"{GEMINI_API_BASE}/{model}:{action}?key={self.api_key}"

    def _make_request(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Gemini API.

        Args:
            model: Model identifier
            payload: Request payload

        Returns:
            API response dictionary
        """
        import urllib.request
        import urllib.error

        endpoint = self._get_endpoint(model)
        headers = {"Content-Type": "application/json"}

        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(endpoint, data=data, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logger.error(f"Gemini API HTTP error: {e.code} - {error_body}")
            raise Exception(f"Gemini API error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            logger.error(f"Gemini API URL error: {e.reason}")
            raise Exception(f"Gemini API connection error: {e.reason}")

    def generate_content(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        image_data: Optional[str] = None,
        image_mime_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini API.

        Args:
            prompt: Text prompt
            model: Model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum output tokens
            image_data: Base64 encoded image data (for vision)
            image_mime_type: MIME type of the image

        Returns:
            API response with generated content
        """
        # Build content parts
        parts = []

        if image_data:
            parts.append({
                "inline_data": {
                    "mime_type": image_mime_type,
                    "data": image_data
                }
            })

        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature
            }
        }

        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        return self._make_request(model, payload)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Multi-turn chat conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            API response with generated content
        """
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Gemini uses 'user' and 'model' roles
            gemini_role = 'model' if role in ['assistant', 'model'] else 'user'

            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }

        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        return self._make_request(model, payload)

    @staticmethod
    def extract_text(response: Dict[str, Any]) -> str:
        """Extract text content from Gemini API response."""
        try:
            candidates = response.get('candidates', [])
            if not candidates:
                return ""

            content = candidates[0].get('content', {})
            parts = content.get('parts', [])

            text_parts = []
            for part in parts:
                if 'text' in part:
                    text_parts.append(part['text'])

            return ''.join(text_parts)
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting text from response: {e}")
            return ""


def _get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension."""
    extension = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(extension, 'image/jpeg')


def _read_image_as_base64(image_path: str) -> str:
    """Read an image file and return base64 encoded data."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


@tool_wrapper()
def generate_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text using Google Gemini API.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text generation prompt
            - model (str, optional): Gemini model - 'gemini-1.5-flash', 'gemini-1.5-pro',
                                     'gemini-2.0-flash' (default: 'gemini-1.5-flash')
            - temperature (float, optional): Sampling temperature 0.0-1.0 (default: 0.7)
            - max_tokens (int, optional): Maximum output tokens

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - model (str): Model used
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        prompt = params.get('prompt')
        if not prompt:
            return {
                'success': False,
                'error': 'prompt parameter is required'
            }

        model = params.get('model', DEFAULT_MODEL)
        if model not in SUPPORTED_MODELS:
            return {
                'success': False,
                'error': f"Invalid model '{model}'. Choose from: {SUPPORTED_MODELS}"
            }

        temperature = params.get('temperature', 0.7)
        max_tokens = params.get('max_tokens')

        # Validate temperature
        if not 0.0 <= temperature <= 1.0:
            return {
                'success': False,
                'error': 'temperature must be between 0.0 and 1.0'
            }

        client = GeminiAPIClient()
        response = client.generate_content(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        text = client.extract_text(response)

        if not text:
            return {
                'success': False,
                'error': 'No text generated',
                'raw_response': response
            }

        return {
            'success': True,
            'text': text,
            'model': model
        }

    except ValueError as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"Gemini text generation error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Text generation failed: {str(e)}'
        }


@tool_wrapper()
def generate_with_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text from an image and prompt using Gemini vision capabilities.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text prompt describing what to do with the image
            - image_path (str, required): Path to the image file
            - model (str, optional): Gemini model (default: 'gemini-1.5-flash')
            - temperature (float, optional): Sampling temperature 0.0-1.0 (default: 0.7)
            - max_tokens (int, optional): Maximum output tokens

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - model (str): Model used
            - image_path (str): Path to the processed image
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        prompt = params.get('prompt')
        if not prompt:
            return {
                'success': False,
                'error': 'prompt parameter is required'
            }

        image_path = params.get('image_path')
        if not image_path:
            return {
                'success': False,
                'error': 'image_path parameter is required'
            }

        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Image file not found: {image_path}'
            }

        model = params.get('model', DEFAULT_MODEL)
        if model not in SUPPORTED_MODELS:
            return {
                'success': False,
                'error': f"Invalid model '{model}'. Choose from: {SUPPORTED_MODELS}"
            }

        temperature = params.get('temperature', 0.7)
        max_tokens = params.get('max_tokens')

        # Read and encode image
        image_data = _read_image_as_base64(image_path)
        mime_type = _get_mime_type(image_path)

        client = GeminiAPIClient()
        response = client.generate_content(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            image_data=image_data,
            image_mime_type=mime_type
        )

        text = client.extract_text(response)

        if not text:
            return {
                'success': False,
                'error': 'No text generated from image',
                'raw_response': response
            }

        return {
            'success': True,
            'text': text,
            'model': model,
            'image_path': image_path
        }

    except ValueError as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"Gemini vision generation error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Vision generation failed: {str(e)}'
        }


@tool_wrapper()
def chat_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-turn chat conversation using Google Gemini API.

    Args:
        params: Dictionary containing:
            - messages (list, required): List of message dicts with 'role' and 'content'
                - role: 'user' or 'assistant'/'model'
                - content: Message text
            - model (str, optional): Gemini model (default: 'gemini-1.5-flash')
            - temperature (float, optional): Sampling temperature 0.0-1.0 (default: 0.7)
            - max_tokens (int, optional): Maximum output tokens

    Returns:
        Dictionary with:
            - success (bool): Whether chat succeeded
            - text (str): Assistant's response
            - model (str): Model used
            - message_count (int): Number of messages in conversation
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        messages = params.get('messages')
        if not messages:
            return {
                'success': False,
                'error': 'messages parameter is required'
            }

        if not isinstance(messages, list):
            return {
                'success': False,
                'error': 'messages must be a list of message dicts'
            }

        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return {
                    'success': False,
                    'error': f'Message at index {i} must be a dictionary'
                }
            if 'role' not in msg or 'content' not in msg:
                return {
                    'success': False,
                    'error': f'Message at index {i} must have "role" and "content" keys'
                }

        model = params.get('model', DEFAULT_MODEL)
        if model not in SUPPORTED_MODELS:
            return {
                'success': False,
                'error': f"Invalid model '{model}'. Choose from: {SUPPORTED_MODELS}"
            }

        temperature = params.get('temperature', 0.7)
        max_tokens = params.get('max_tokens')

        client = GeminiAPIClient()
        response = client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        text = client.extract_text(response)

        if not text:
            return {
                'success': False,
                'error': 'No response generated',
                'raw_response': response
            }

        return {
            'success': True,
            'text': text,
            'model': model,
            'message_count': len(messages)
        }

    except ValueError as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"Gemini chat error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Chat failed: {str(e)}'
        }


@tool_wrapper()
def list_models_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available Gemini models and their capabilities.

    Args:
        params: Dictionary (can be empty)

    Returns:
        Dictionary with:
            - success (bool): Always True
            - models (list): List of available models with details
    """
    status.set_callback(params.pop('_status_callback', None))

    models = [
        {
            "key": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "description": "Fast, efficient model for quick tasks. Good balance of speed and quality.",
            "supports_vision": True,
            "supports_chat": True,
            "context_window": "1M tokens"
        },
        {
            "key": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "description": "Advanced model for complex tasks. Best quality output.",
            "supports_vision": True,
            "supports_chat": True,
            "context_window": "2M tokens"
        },
        {
            "key": "gemini-2.0-flash",
            "name": "Gemini 2.0 Flash",
            "description": "Latest generation fast model with improved capabilities.",
            "supports_vision": True,
            "supports_chat": True,
            "context_window": "1M tokens"
        }
    ]

    return {
        "success": True,
        "models": models,
        "default_model": DEFAULT_MODEL
    }


__all__ = [
    'generate_text_tool',
    'generate_with_image_tool',
    'chat_tool',
    'list_models_tool',
    'GeminiAPIClient'
]
