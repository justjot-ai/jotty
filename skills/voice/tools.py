"""
Voice Skill Tools
=================

Multi-provider voice tools for STT and TTS.
Refactored to use Jotty core utilities.
"""

import uuid
import logging
from typing import Dict, Any

from Jotty.core.infrastructure.utils.env_loader import load_jotty_env
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.async_utils import run_sync

from Jotty.core.infrastructure.utils.skill_status import SkillStatus

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


# =========================================================================
# OpenAI Audio Tools (TTS, Translation, Info)
# =========================================================================

OPENAI_TTS_VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
OPENAI_VOICE_DESC = {
    'alloy': 'Neutral, balanced', 'echo': 'Warm, conversational',
    'fable': 'British, narrative', 'onyx': 'Deep, authoritative',
    'nova': 'Friendly, expressive (recommended)', 'shimmer': 'Clear, professional',
}
SUPPORTED_AUDIO_FORMATS = ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']


class OpenAIAudioClient:
    """OpenAI Audio API client for TTS and translation."""

    _instance = None

    def __init__(self):
        import os
        self._api_key = os.getenv("OPENAI_API_KEY")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_openai(self):
        try:
            from openai import OpenAI
            if not self._api_key:
                return None
            return OpenAI(api_key=self._api_key)
        except ImportError:
            return None

    def translate(self, audio_path: str, prompt: str = None,
                  response_format: str = "json") -> dict:
        """Translate audio to English using Whisper."""
        import os
        client = self._get_openai()
        if not client:
            return {"success": False, "error": "OpenAI not available. Check API key."}
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"Audio file not found: {audio_path}"}

        with open(audio_path, "rb") as f:
            kwargs = {"model": "whisper-1", "file": f, "response_format": response_format}
            if prompt:
                kwargs["prompt"] = prompt
            response = client.audio.translations.create(**kwargs)

        text = response.text if hasattr(response, 'text') else str(response)
        return {
            "success": True, "text": text,
            "original_language": "auto-detected", "target_language": "en"
        }

    def generate_speech(self, text: str, voice: str = "nova",
                        speed: float = 1.0, response_format: str = "mp3",
                        output_path: str = None) -> dict:
        """Generate speech using OpenAI TTS."""
        import os
        client = self._get_openai()
        if not client:
            return {"success": False, "error": "OpenAI not available. Check API key."}

        if voice not in OPENAI_TTS_VOICES:
            return {"success": False, "error": f"Invalid voice: {voice}. Available: {OPENAI_TTS_VOICES}"}
        if not 0.25 <= speed <= 4.0:
            return {"success": False, "error": "Speed must be 0.25-4.0"}
        if len(text) > 4096:
            return {"success": False, "error": f"Text too long: {len(text)} chars. Max 4096."}

        tts_model = os.getenv("TTS_MODEL", "tts-1")
        response = client.audio.speech.create(
            model=tts_model, voice=voice, input=text,
            speed=speed, response_format=response_format,
        )

        if not output_path:
            output_dir = os.path.expanduser("~/jotty/audio")
            os.makedirs(output_dir, exist_ok=True)
            ext = response_format if response_format != 'pcm' else 'raw'
            output_path = os.path.join(output_dir, f"tts_{voice}_{uuid.uuid4().hex[:8]}.{ext}")

        response.stream_to_file(output_path)
        word_count = len(text.split())
        duration_est = (word_count / 150) * 60 / speed

        return {
            "success": True, "audio_path": output_path, "voice": voice,
            "model": tts_model, "speed": speed, "format": response_format,
            "duration_estimate": round(duration_est, 1),
        }

    @staticmethod
    def get_audio_info(audio_path: str) -> dict:
        """Get audio file metadata."""
        import os
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"Audio not found: {audio_path}"}

        from pathlib import Path
        ext = Path(audio_path).suffix.lower().lstrip('.')
        size = os.path.getsize(audio_path)

        result = {
            "success": True, "path": audio_path, "format": ext,
            "size_bytes": size, "size_mb": round(size / (1024 * 1024), 2),
            "is_supported": ext in SUPPORTED_AUDIO_FORMATS,
        }

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            result.update({
                "duration": round(len(audio) / 1000, 2),
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
            })
        except ImportError:
            result["note"] = "Install pydub for detailed metadata"
        except Exception as e:
            result["metadata_error"] = str(e)

        return result


@tool_wrapper(required_params=['audio_path'])
def translate_audio_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate audio to English text using Whisper.

    Args:
        params: Dictionary containing:
            - audio_path (str, required): Path to audio file (any language)
            - prompt (str, optional): Context hint for translation

    Returns:
        Dictionary with success, text, original_language, target_language
    """
    status.set_callback(params.pop('_status_callback', None))
    status.emit("Translating", "Translating audio to English")
    return OpenAIAudioClient.get_instance().translate(
        params['audio_path'], params.get('prompt')
    )


@tool_wrapper(required_params=['text'])
def generate_speech_openai_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate speech using OpenAI TTS with 6 voice options.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to convert (max 4096 chars)
            - voice (str, optional): alloy/echo/fable/onyx/nova/shimmer (default: nova)
            - speed (float, optional): 0.25-4.0 (default: 1.0)
            - response_format (str, optional): mp3/opus/aac/flac/wav (default: mp3)
            - output_path (str, optional): Save path

    Returns:
        Dictionary with success, audio_path, voice, duration_estimate
    """
    status.set_callback(params.pop('_status_callback', None))
    status.emit("Generating", "Generating speech with OpenAI TTS")
    return OpenAIAudioClient.get_instance().generate_speech(
        params['text'],
        voice=params.get('voice', 'nova'),
        speed=params.get('speed', 1.0),
        response_format=params.get('response_format', 'mp3'),
        output_path=params.get('output_path'),
    )


@tool_wrapper(required_params=['audio_path'])
def get_audio_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata about an audio file.

    Args:
        params: Dictionary containing:
            - audio_path (str, required): Path to audio file

    Returns:
        Dictionary with success, format, size, duration, sample_rate, channels
    """
    status.set_callback(params.pop('_status_callback', None))
    return OpenAIAudioClient.get_audio_info(params['audio_path'])


__all__ = [
    'voice_to_text_tool',
    'text_to_voice_tool',
    'stream_voice_tool',
    'get_stream_chunk_tool',
    'close_stream_tool',
    'list_voices_tool',
    # OpenAI Audio tools
    'translate_audio_tool',
    'generate_speech_openai_tool',
    'get_audio_info_tool',
    # Convenience functions
    'voice_to_text',
    'text_to_voice',
    'stream_voice',
]
