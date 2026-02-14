"""
OpenAI Whisper API Skill

Speech-to-text transcription and translation using OpenAI's Whisper API.
"""
import os
import logging
from typing import Dict, Any

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

# OpenAI API endpoints

# Status emitter for progress updates
status = SkillStatus("openai-whisper-api")

TRANSCRIPTION_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
TRANSLATION_ENDPOINT = "https://api.openai.com/v1/audio/translations"


class WhisperAPIClient:
    """Client for OpenAI Whisper API operations."""

    def __init__(self):
        self.api_key = os.environ.get('OPENAI_API_KEY')

    def _validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate audio file exists and has supported format."""
        if not file_path:
            return {'valid': False, 'error': 'file_path parameter is required'}

        if not os.path.exists(file_path):
            return {'valid': False, 'error': f'File not found: {file_path}'}

        extension = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
        if extension not in SUPPORTED_FORMATS:
            return {
                'valid': False,
                'error': f'Unsupported format: {extension}. Supported: {", ".join(SUPPORTED_FORMATS)}'
            }

        return {'valid': True, 'extension': extension}

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {'Authorization': f'Bearer {self.api_key}'}

    def transcribe(self, file_path: str, language: str = None,
                   response_format: str = 'json') -> Dict[str, Any]:
        """
        Transcribe audio file to text using Whisper API.

        Args:
            file_path: Path to audio file
            language: Optional ISO-639-1 language code (e.g., 'en', 'es', 'fr')
            response_format: Output format - 'json', 'text', 'srt', 'vtt'

        Returns:
            Dictionary with transcription result or error
        """
        import requests

        if not self.api_key:
            return {'success': False, 'error': 'OPENAI_API_KEY environment variable not set'}

        validation = self._validate_file(file_path)
        if not validation.get('valid'):
            return {'success': False, 'error': validation.get('error')}

        valid_formats = {'json', 'text', 'srt', 'vtt', 'verbose_json'}
        if response_format not in valid_formats:
            return {'success': False, 'error': f'Invalid response_format. Valid: {", ".join(valid_formats)}'}

        try:
            with open(file_path, 'rb') as audio_file:
                files = {'file': (os.path.basename(file_path), audio_file)}
                data = {
                    'model': 'whisper-1',
                    'response_format': response_format
                }

                if language:
                    data['language'] = language

                response = requests.post(
                    TRANSCRIPTION_ENDPOINT,
                    headers=self._get_headers(),
                    files=files,
                    data=data,
                    timeout=300
                )

            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', response.text)
                return {'success': False, 'error': f'API error ({response.status_code}): {error_msg}'}

            if response_format == 'json':
                result = response.json()
                return {
                    'success': True,
                    'text': result.get('text', ''),
                    'file_path': file_path,
                    'response_format': response_format
                }
            elif response_format == 'verbose_json':
                result = response.json()
                return {
                    'success': True,
                    'text': result.get('text', ''),
                    'language': result.get('language'),
                    'duration': result.get('duration'),
                    'segments': result.get('segments', []),
                    'file_path': file_path,
                    'response_format': response_format
                }
            else:
                return {
                    'success': True,
                    'text': response.text,
                    'file_path': file_path,
                    'response_format': response_format
                }

        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timed out. Audio file may be too large.'}
        except requests.exceptions.RequestException as e:
            logger.error(f"Transcription request failed: {e}", exc_info=True)
            return {'success': False, 'error': f'Request failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return {'success': False, 'error': f'Transcription failed: {str(e)}'}

    def translate(self, file_path: str, response_format: str = 'json') -> Dict[str, Any]:
        """
        Translate audio to English using Whisper API.

        Args:
            file_path: Path to audio file
            response_format: Output format - 'json', 'text', 'srt', 'vtt'

        Returns:
            Dictionary with translation result or error
        """
        import requests

        if not self.api_key:
            return {'success': False, 'error': 'OPENAI_API_KEY environment variable not set'}

        validation = self._validate_file(file_path)
        if not validation.get('valid'):
            return {'success': False, 'error': validation.get('error')}

        valid_formats = {'json', 'text', 'srt', 'vtt', 'verbose_json'}
        if response_format not in valid_formats:
            return {'success': False, 'error': f'Invalid response_format. Valid: {", ".join(valid_formats)}'}

        try:
            with open(file_path, 'rb') as audio_file:
                files = {'file': (os.path.basename(file_path), audio_file)}
                data = {
                    'model': 'whisper-1',
                    'response_format': response_format
                }

                response = requests.post(
                    TRANSLATION_ENDPOINT,
                    headers=self._get_headers(),
                    files=files,
                    data=data,
                    timeout=300
                )

            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', response.text)
                return {'success': False, 'error': f'API error ({response.status_code}): {error_msg}'}

            if response_format == 'json':
                result = response.json()
                return {
                    'success': True,
                    'text': result.get('text', ''),
                    'file_path': file_path,
                    'response_format': response_format,
                    'target_language': 'en'
                }
            elif response_format == 'verbose_json':
                result = response.json()
                return {
                    'success': True,
                    'text': result.get('text', ''),
                    'language': result.get('language'),
                    'duration': result.get('duration'),
                    'segments': result.get('segments', []),
                    'file_path': file_path,
                    'response_format': response_format,
                    'target_language': 'en'
                }
            else:
                return {
                    'success': True,
                    'text': response.text,
                    'file_path': file_path,
                    'response_format': response_format,
                    'target_language': 'en'
                }

        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timed out. Audio file may be too large.'}
        except requests.exceptions.RequestException as e:
            logger.error(f"Translation request failed: {e}", exc_info=True)
            return {'success': False, 'error': f'Request failed: {str(e)}'}
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return {'success': False, 'error': f'Translation failed: {str(e)}'}


# Singleton client instance
_client = None


def _get_client() -> WhisperAPIClient:
    """Get or create WhisperAPIClient instance."""
    global _client
    if _client is None:
        _client = WhisperAPIClient()
    return _client


@tool_wrapper()
def transcribe_audio_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transcribe audio file to text using OpenAI Whisper API.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to audio file
            - language (str, optional): ISO-639-1 language code (e.g., 'en', 'es', 'fr')
            - response_format (str, optional): Output format - 'json', 'text', 'srt', 'vtt'
              (default: 'json')

    Returns:
        Dictionary with:
            - success (bool): Whether transcription succeeded
            - text (str): Transcribed text
            - file_path (str): Source file path
            - response_format (str): Format used
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    client = _get_client()

    file_path = params.get('file_path')
    language = params.get('language')
    response_format = params.get('response_format', 'json')

    logger.info(f"Transcribing audio: {file_path}")
    result = client.transcribe(file_path, language, response_format)

    if result.get('success'):
        logger.info(f"Transcription completed: {len(result.get('text', ''))} characters")
    else:
        logger.error(f"Transcription failed: {result.get('error')}")

    return result


@tool_wrapper()
def translate_audio_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate audio to English using OpenAI Whisper API.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to audio file
            - response_format (str, optional): Output format - 'json', 'text', 'srt', 'vtt'
              (default: 'json')

    Returns:
        Dictionary with:
            - success (bool): Whether translation succeeded
            - text (str): Translated English text
            - file_path (str): Source file path
            - response_format (str): Format used
            - target_language (str): Always 'en' for translations
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    client = _get_client()

    file_path = params.get('file_path')
    response_format = params.get('response_format', 'json')

    logger.info(f"Translating audio to English: {file_path}")
    result = client.translate(file_path, response_format)

    if result.get('success'):
        logger.info(f"Translation completed: {len(result.get('text', ''))} characters")
    else:
        logger.error(f"Translation failed: {result.get('error')}")

    return result
