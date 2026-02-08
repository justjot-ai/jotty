#!/usr/bin/env python3
"""
Provider API for JustJot.ai Integration
========================================

Exposes unified provider registry via HTTP API.
Allows JustJot.ai frontend to use Jotty's unified provider system.

This replaces direct Vercel AI SDK usage in JustJot.ai frontend.
"""
from flask import Blueprint, request, jsonify, Response, stream_with_context
from typing import Dict, Any, Optional
import json
import sys
import dspy

from ..foundation.unified_lm_provider import UnifiedLMProvider

provider_bp = Blueprint('provider', __name__)


@provider_bp.route('/api/provider/chat', methods=['POST'])
def chat():
    """
    Chat endpoint using unified provider registry.
    
    Request:
        {
            "provider": "opencode" | "openrouter" | "anthropic" | "claude-cli" | ...,
            "model": "model-name" (optional),
            "messages": [{"role": "user", "content": "..."}],
            "stream": true/false (optional, default: false)
        }
    
    Response (non-streaming):
        {
            "text": "...",
            "usage": {...}
        }
    
    Response (streaming):
        SSE format: data: {"type": "text-delta", "textDelta": "..."}
    """
    data = request.get_json() or {}
    provider = data.get('provider', 'opencode')  # Default to OpenCode (free)
    model = data.get('model')
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    
    if not messages:
        return jsonify({'error': 'Messages required'}), 400
    
    try:
        # Create LM instance using unified provider registry
        lm = UnifiedLMProvider.create_lm(provider, model=model)
        
        # Extract prompt from messages
        prompt = messages[-1].get('content', '') if messages else ''
        if not prompt:
            return jsonify({'error': 'Prompt required'}), 400
        
        if stream:
            # Streaming response
            def generate():
                try:
                    # Use DSPy forward method for streaming
                    # For streaming, we need to call the LM and stream the response
                    # Most DSPy LMs don't support streaming directly, so we'll get full response and stream it
                    result = lm.forward(prompt=prompt, messages=messages)
                    
                    # Extract text from result
                    if isinstance(result, dict) and 'choices' in result:
                        text = result['choices'][0]['message']['content']
                    elif isinstance(result, list) and len(result) > 0:
                        text = result[0] if isinstance(result[0], str) else str(result[0])
                    else:
                        text = str(result)
                    
                    # Stream character by character (simulate streaming)
                    for char in text:
                        yield f"data: {json.dumps({'type': 'text-delta', 'textDelta': char})}\n\n"
                    
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Non-streaming response
            result = lm.forward(prompt=prompt, messages=messages)
            
            # Extract text from result
            if isinstance(result, dict) and 'choices' in result:
                text = result['choices'][0]['message']['content']
                usage = result.get('usage', {})
            elif isinstance(result, list) and len(result) > 0:
                text = result[0] if isinstance(result[0], str) else str(result[0])
                usage = {}
            else:
                text = str(result)
                usage = {}
            
            return jsonify({
                'text': text,
                'usage': usage
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@provider_bp.route('/api/provider/list', methods=['GET'])
def list_providers():
    """
    List available providers (dynamic, checks actual availability).

    Response:
        {
            "providers": {...},
            "recommended": "provider_name",
            "total_available": 3
        }
    """
    try:
        result = UnifiedLMProvider.get_available_providers()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@provider_bp.route('/api/providers', methods=['GET'])
def get_providers():
    """
    Get all available providers and their configurations (Phase 3 API).

    This endpoint allows JustJot.ai to dynamically load provider configs
    from Jotty instead of maintaining duplicate configurations.

    Response:
        {
            "providers": {
                "anthropic": {"available": true, "models": [...], "default": "..."},
                ...
            },
            "recommended": "claude-cli",
            "total_available": 5
        }
    """
    try:
        result = UnifiedLMProvider.get_available_providers()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@provider_bp.route('/api/provider/configure', methods=['POST'])
def configure():
    """
    Configure default DSPy LM provider.

    Request:
        {
            "provider": "opencode",
            "model": "glm-4" (optional)
        }

    Response:
        {
            "success": true,
            "provider": "opencode",
            "model": "glm-4"
        }
    """
    data = request.get_json() or {}
    provider = data.get('provider')
    model = data.get('model')

    if not provider:
        return jsonify({'error': 'Provider required'}), 400

    try:
        lm = UnifiedLMProvider.configure_default_lm(provider=provider, model=model)
        return jsonify({
            'success': True,
            'provider': provider,
            'model': model or 'default'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# JustJot.ai Integration Endpoints
# These endpoints allow JustJot.ai to use Jotty as the single source of truth
# for LLM provider configuration.
# ============================================================================

# Model aliases for convenient shorthand names
MODEL_ALIASES = {
    'anthropic': {
        'sonnet': 'claude-sonnet-4-20250514',
        'opus': 'claude-opus-4-20250514',
        'haiku': 'claude-3-5-haiku-20241022',
    },
    'openai': {
        'gpt4': 'gpt-4-turbo',
        'gpt4o': 'gpt-4o',
        'gpt35': 'gpt-3.5-turbo',
    },
    'google': {
        'gemini-flash': 'gemini-2.0-flash-exp',
        'gemini-pro': 'gemini-pro',
    },
    'groq': {
        'llama-8b': 'llama-3.1-8b-instant',
        'llama-70b': 'llama-3.3-70b-versatile',
    },
    'openrouter': {
        'llama-free': 'meta-llama/llama-3.3-70b-instruct:free',
        'gemini-free': 'google/gemini-2.0-flash-exp:free',
    },
    'claude-cli': {
        'sonnet': 'sonnet',
        'opus': 'opus',
        'haiku': 'haiku',
    },
    'cursor-cli': {
        'sonnet': 'sonnet',
    },
}


@provider_bp.route('/api/providers', methods=['GET'])
def list_all_providers():
    """
    List all available LLM providers with full configuration.

    This is the main endpoint for JustJot.ai to query provider configs.
    Returns comprehensive provider information including:
    - Available providers
    - Model aliases for each provider
    - Current default provider (if configured)
    - Feature support flags

    Response:
        {
            "providers": [...],
            "model_aliases": {...},
            "current_provider": "anthropic" | null,
            "features": {
                "streaming": true,
                "tool_calling": true,
                "structured_output": true
            }
        }
    """
    import os

    providers = [
        {
            'id': 'anthropic',
            'name': 'Anthropic Claude',
            'free': False,
            'description': 'Claude models via Anthropic API',
            'models': ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-3-5-haiku-20241022'],
            'default_model': 'claude-sonnet-4-20250514',
            'available': bool(os.getenv('ANTHROPIC_API_KEY')),
            'features': {
                'streaming': True,
                'tool_calling': True,
                'structured_output': True,
                'vision': True,
            },
            'rate_limits': {
                'requests_per_minute': 60,
                'tokens_per_day': 1000000,
            }
        },
        {
            'id': 'openai',
            'name': 'OpenAI',
            'free': False,
            'description': 'GPT models via OpenAI API',
            'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            'default_model': 'gpt-4o',
            'available': bool(os.getenv('OPENAI_API_KEY')),
            'features': {
                'streaming': True,
                'tool_calling': True,
                'structured_output': True,
                'vision': True,
            },
            'rate_limits': {
                'requests_per_minute': 60,
                'tokens_per_day': 1000000,
            }
        },
        {
            'id': 'google',
            'name': 'Google Gemini',
            'free': False,
            'description': 'Gemini models via Google API',
            'models': ['gemini-2.0-flash-exp', 'gemini-pro'],
            'default_model': 'gemini-2.0-flash-exp',
            'available': bool(os.getenv('GOOGLE_API_KEY')),
            'features': {
                'streaming': True,
                'tool_calling': True,
                'structured_output': True,
                'vision': True,
            },
            'rate_limits': {
                'requests_per_minute': 60,
                'tokens_per_day': 1000000,
            }
        },
        {
            'id': 'groq',
            'name': 'Groq',
            'free': False,
            'description': 'Fast inference via Groq',
            'models': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'],
            'default_model': 'llama-3.1-8b-instant',
            'available': bool(os.getenv('GROQ_API_KEY')),
            'features': {
                'streaming': True,
                'tool_calling': True,
                'structured_output': False,
                'vision': False,
            },
            'rate_limits': {
                'requests_per_minute': 30,
                'tokens_per_day': 500000,
            }
        },
        {
            'id': 'openrouter',
            'name': 'OpenRouter',
            'free': True,
            'description': 'Multiple models via OpenRouter (free tier available)',
            'models': ['meta-llama/llama-3.3-70b-instruct:free', 'google/gemini-2.0-flash-exp:free'],
            'default_model': 'meta-llama/llama-3.3-70b-instruct:free',
            'available': bool(os.getenv('OPENROUTER_API_KEY')),
            'features': {
                'streaming': True,
                'tool_calling': False,
                'structured_output': False,
                'vision': False,
            },
            'rate_limits': {
                'requests_per_minute': 20,
                'tokens_per_day': 100000,
            }
        },
        {
            'id': 'opencode',
            'name': 'OpenCode',
            'free': True,
            'description': 'Free GLM model via OpenCode',
            'models': ['default', 'glm-4'],
            'default_model': 'default',
            'available': True,  # Always available (remote execution)
            'features': {
                'streaming': False,
                'tool_calling': False,
                'structured_output': False,
                'vision': False,
            },
            'rate_limits': {
                'requests_per_minute': 10,
                'tokens_per_day': 50000,
            }
        },
        {
            'id': 'claude-cli',
            'name': 'Claude CLI',
            'free': True,
            'description': 'Claude via CLI (uses OAuth credentials)',
            'models': ['sonnet', 'opus', 'haiku'],
            'default_model': 'sonnet',
            'available': _check_cli_available('claude'),
            'features': {
                'streaming': True,
                'tool_calling': True,
                'structured_output': True,
                'vision': True,
            },
            'rate_limits': {
                'requests_per_minute': 60,
                'tokens_per_day': None,  # Uses subscription
            }
        },
        {
            'id': 'cursor-cli',
            'name': 'Cursor CLI',
            'free': True,
            'description': 'Cursor via CLI',
            'models': ['sonnet'],
            'default_model': 'sonnet',
            'available': _check_cli_available('cursor-agent'),
            'features': {
                'streaming': True,
                'tool_calling': True,
                'structured_output': True,
                'vision': False,
            },
            'rate_limits': {
                'requests_per_minute': 60,
                'tokens_per_day': None,  # Uses subscription
            }
        },
    ]

    # Detect current provider
    current_provider = None
    try:
        import dspy
        if dspy.settings.lm:
            lm = dspy.settings.lm
            current_provider = getattr(lm, 'provider', None)
    except Exception:
        pass

    return jsonify({
        'providers': providers,
        'model_aliases': MODEL_ALIASES,
        'current_provider': current_provider,
        'features': {
            'streaming': True,
            'tool_calling': True,
            'structured_output': True,
        }
    })


@provider_bp.route('/api/providers/validate', methods=['POST'])
def validate_provider():
    """
    Validate a provider configuration.

    Request:
        {
            "provider": "anthropic",
            "api_key": "sk-..." (optional, uses env var if not provided)
        }

    Response:
        {
            "valid": true,
            "provider": "anthropic",
            "message": "Provider validated successfully",
            "models_available": ["claude-sonnet-4-20250514", ...]
        }
    """
    import os

    data = request.get_json() or {}
    provider = data.get('provider')
    api_key = data.get('api_key')

    if not provider:
        return jsonify({'valid': False, 'error': 'Provider required'}), 400

    provider = provider.lower()

    # Check for API key
    env_keys = {
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'groq': 'GROQ_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
    }

    if provider in env_keys:
        effective_key = api_key or os.getenv(env_keys[provider])
        if not effective_key:
            return jsonify({
                'valid': False,
                'provider': provider,
                'error': f'API key required. Set {env_keys[provider]} environment variable or provide api_key.'
            })

    # For CLI providers, check if binary exists
    if provider in ('claude-cli', 'cursor-cli'):
        cli_name = 'claude' if provider == 'claude-cli' else 'cursor-agent'
        if not _check_cli_available(cli_name):
            return jsonify({
                'valid': False,
                'provider': provider,
                'error': f'{cli_name} CLI not found. Install and authenticate first.'
            })

    # Try to create an LM instance to validate
    try:
        lm = UnifiedLMProvider.create_lm(
            provider,
            api_key=api_key,
            inject_context=False  # Don't wrap for validation
        )

        # Get available models for this provider
        models = MODEL_ALIASES.get(provider, {}).values() if provider in MODEL_ALIASES else []

        return jsonify({
            'valid': True,
            'provider': provider,
            'message': 'Provider validated successfully',
            'models_available': list(models)
        })

    except Exception as e:
        return jsonify({
            'valid': False,
            'provider': provider,
            'error': str(e)
        })


def _check_cli_available(cli_name: str) -> bool:
    """Check if a CLI tool is available."""
    import shutil
    return shutil.which(cli_name) is not None or \
           os.path.exists(f'/usr/local/bin/{cli_name}')
