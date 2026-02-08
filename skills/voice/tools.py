"""
Voice Skill Tools
=================

Multi-provider voice tools for STT and TTS.
Refactored to use Jotty core utilities.
"""

import uuid
import logging
from typing import Dict, Any

from Jotty.core.utils.env_loader import load_jotty_env
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.async_utils import run_sync

from Jotty.core.utils.skill_status import SkillStatus

load_jotty_env()

logger = logging.getLogger(__name__)

# Import providers with fallback for standalone loading
get_tts_provider = None
get_stt_provider = None

try:
    from .providers import get_tts_provider, get_stt_provider
except ImportError:
    try:
        # Fallback: load providers package directly when loaded standalone
        import importlib.util
        from pathlib import Path

        providers_init = Path(__file__).parent / 'providers' / '__init__.py'
        if providers_init.exists():
            spec = importlib.util.spec_from_file_location("voice_providers", providers_init)
            providers_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(providers_module)
            get_tts_provider = getattr(providers_module, 'get_tts_provider', None)
            get_stt_provider = getattr(providers_module, 'get_stt_provider', None)
    except Exception as e:
        logger.warning(f"Could not load voice providers: {e}")

# Store active streams for stream_voice_tool

# Status emitter for progress updates
status = SkillStatus("voice")

_active_streams: Dict[str, Any] = {}


@tool_wrapper(required_params=['audio_path'])
def voice_to_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert audio file to text using speech recognition.

    Args:
        params: Dictionary containing:
            - audio_path (str, required): Path to audio file
            - provider (str, optional): "whisper", "local", or "auto"
            - language (str, optional): Language code

    Returns:
        Dictionary with success, text, language, provider
    """
    status.set_callback(params.pop('_status_callback', None))
    provider_name = params.get('provider', 'auto')
    language = params.get('language')

    provider = get_stt_provider(provider_name)
    return run_sync(provider.speech_to_text(params['audio_path'], language))


@tool_wrapper(required_params=['text'])
def text_to_voice_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert text to audio using speech synthesis.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to synthesize
            - provider (str, optional): "elevenlabs", "local", or "auto"
            - voice_id (str, optional): Voice ID or name
            - output_path (str, optional): Path to save audio

    Returns:
        Dictionary with success, audio data/path, format, provider
    """
    status.set_callback(params.pop('_status_callback', None))
    provider_name = params.get('provider', 'auto')
    voice_id = params.get('voice_id')
    output_path = params.get('output_path')

    provider = get_tts_provider(provider_name)
    return run_sync(provider.text_to_speech(params['text'], voice_id, output_path))


@tool_wrapper(required_params=['text'])
def stream_voice_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start streaming text-to-speech audio.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to synthesize
            - provider (str, optional): "elevenlabs", "local", or "auto"
            - voice_id (str, optional): Voice ID or name
            - chunk_size (int, optional): Audio chunk size

    Returns:
        Dictionary with success, stream_id, format, provider
    """
    status.set_callback(params.pop('_status_callback', None))
    text = params['text']
    provider_name = params.get('provider', 'auto')
    voice_id = params.get('voice_id')
    chunk_size = params.get('chunk_size', 1024)

    provider = get_tts_provider(provider_name)
    stream_id = str(uuid.uuid4())

    async def create_stream():
        async for chunk in provider.stream_speech(text, voice_id, chunk_size):
            yield chunk

    _active_streams[stream_id] = {
        'generator': create_stream(),
        'provider': provider.name,
        'text': text,
        'voice_id': voice_id
    }

    return tool_response(
        stream_id=stream_id,
        format='mp3' if provider.name == 'elevenlabs' else 'wav',
        provider=provider.name
    )


@tool_wrapper(required_params=['stream_id'])
def get_stream_chunk_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get next chunk from an active voice stream.

    Args:
        params: Dictionary containing:
            - stream_id (str, required): Stream ID from stream_voice_tool

    Returns:
        Dictionary with success, chunk (base64), done flag
    """
    status.set_callback(params.pop('_status_callback', None))

    import base64
    import asyncio

    stream_id = params['stream_id']
    stream_info = _active_streams.get(stream_id)
    if not stream_info:
        return tool_error(f'Stream not found: {stream_id}')

    loop = asyncio.new_event_loop()
    try:
        chunk = loop.run_until_complete(stream_info['generator'].__anext__())
        return tool_response(
            chunk=base64.b64encode(chunk).decode('utf-8'),
            done=False
        )
    except StopAsyncIteration:
        del _active_streams[stream_id]
        return tool_response(chunk=None, done=True)
    finally:
        loop.close()


@tool_wrapper(required_params=['stream_id'])
def close_stream_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Close an active voice stream.

    Args:
        params: Dictionary containing:
            - stream_id (str, required): Stream ID to close

    Returns:
        Dictionary with success
    """
    status.set_callback(params.pop('_status_callback', None))

    stream_id = params['stream_id']
    if stream_id in _active_streams:
        del _active_streams[stream_id]
        return tool_response(message='Stream closed')
    return tool_error(f'Stream not found: {stream_id}')


@tool_wrapper()
def list_voices_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available voices for TTS.

    Args:
        params: Dictionary containing:
            - provider (str, optional): "elevenlabs", "local", or "auto"

    Returns:
        Dictionary with success, voices list, count
    """
    provider_name = params.get('provider', 'auto')
    voices = []

    if provider_name in ('auto', 'elevenlabs'):
        voices.extend([
            {'id': 'rachel', 'name': 'Rachel', 'provider': 'elevenlabs', 'gender': 'female'},
            {'id': 'adam', 'name': 'Adam', 'provider': 'elevenlabs', 'gender': 'male'},
            {'id': 'josh', 'name': 'Josh', 'provider': 'elevenlabs', 'gender': 'male'},
            {'id': 'bella', 'name': 'Bella', 'provider': 'elevenlabs', 'gender': 'female'},
            {'id': 'elli', 'name': 'Elli', 'provider': 'elevenlabs', 'gender': 'female'},
            {'id': 'sam', 'name': 'Sam', 'provider': 'elevenlabs', 'gender': 'male'},
        ])

    if provider_name in ('auto', 'whisper'):
        voices.extend([
            {'id': 'alloy', 'name': 'Alloy', 'provider': 'openai', 'gender': 'neutral'},
            {'id': 'echo', 'name': 'Echo', 'provider': 'openai', 'gender': 'male'},
            {'id': 'fable', 'name': 'Fable', 'provider': 'openai', 'gender': 'female'},
            {'id': 'onyx', 'name': 'Onyx', 'provider': 'openai', 'gender': 'male'},
            {'id': 'nova', 'name': 'Nova', 'provider': 'openai', 'gender': 'female'},
            {'id': 'shimmer', 'name': 'Shimmer', 'provider': 'openai', 'gender': 'female'},
        ])

    if provider_name in ('auto', 'local'):
        voices.append({
            'id': 'en_US-lessac-medium',
            'name': 'Lessac (English)',
            'provider': 'local/piper',
            'gender': 'female'
        })

    return tool_response(voices=voices, count=len(voices))


# Convenience functions for direct import
def voice_to_text(audio_path: str, provider: str = "auto", language: str = None) -> str:
    """Convert audio to text. Returns transcribed text or raises exception."""
    result = voice_to_text_tool({
        'audio_path': audio_path,
        'provider': provider,
        'language': language
    })
    if not result.get('success'):
        raise RuntimeError(result.get('error', 'Unknown error'))
    return result['text']


def text_to_voice(text: str, provider: str = "auto", voice_id: str = None) -> bytes:
    """Convert text to audio. Returns audio bytes or raises exception."""
    import base64
    result = text_to_voice_tool({
        'text': text,
        'provider': provider,
        'voice_id': voice_id
    })
    if not result.get('success'):
        raise RuntimeError(result.get('error', 'Unknown error'))
    return base64.b64decode(result['audio_base64'])


async def stream_voice(text: str, provider: str = "auto", voice_id: str = None):
    """Async generator for streaming TTS audio."""
    try:
        tts_provider = get_tts_provider(provider)
        async for chunk in tts_provider.stream_speech(text, voice_id):
            yield chunk
    except Exception as e:
        logger.error(f"stream_voice error: {e}", exc_info=True)
        raise


__all__ = [
    'voice_to_text_tool',
    'text_to_voice_tool',
    'stream_voice_tool',
    'get_stream_chunk_tool',
    'close_stream_tool',
    'list_voices_tool',
    # Convenience functions
    'voice_to_text',
    'text_to_voice',
    'stream_voice',
]
