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
                    result = lm.forward(prompt=prompt, messages=messages)
                    
                    # Extract text from result
                    if isinstance(result, dict) and 'choices' in result:
                        text = result['choices'][0]['message']['content']
                        # Stream character by character
                        for char in text:
                            yield f"data: {json.dumps({'type': 'text-delta', 'textDelta': char})}\n\n"
                    else:
                        # Fallback: return as single chunk
                        text = str(result)
                        yield f"data: {json.dumps({'type': 'text-delta', 'textDelta': text})}\n\n"
                    
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
    List available providers.
    
    Response:
        {
            "providers": [
                {"id": "opencode", "name": "OpenCode", "free": true, ...},
                {"id": "openrouter", "name": "OpenRouter", "free": false, ...},
                ...
            ]
        }
    """
    providers = [
        {
            'id': 'opencode',
            'name': 'OpenCode',
            'free': True,
            'description': 'Free GLM model via OpenCode',
            'models': ['default', 'glm-4']
        },
        {
            'id': 'openrouter',
            'name': 'OpenRouter',
            'free': True,
            'description': 'Multiple models via OpenRouter',
            'models': ['meta-llama/llama-3.3-70b-instruct:free', 'google/gemini-2.0-flash-exp:free']
        },
        {
            'id': 'anthropic',
            'name': 'Anthropic Claude',
            'free': False,
            'description': 'Claude models via Anthropic API',
            'models': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229']
        },
        {
            'id': 'claude-cli',
            'name': 'Claude CLI',
            'free': True,
            'description': 'Claude via CLI (uses OAuth)',
            'models': ['sonnet', 'opus', 'haiku']
        },
        {
            'id': 'cursor-cli',
            'name': 'Cursor CLI',
            'free': True,
            'description': 'Cursor via CLI',
            'models': ['sonnet']
        },
        {
            'id': 'openai',
            'name': 'OpenAI',
            'free': False,
            'description': 'GPT models via OpenAI API',
            'models': ['gpt-4o', 'gpt-4-turbo']
        },
        {
            'id': 'google',
            'name': 'Google Gemini',
            'free': False,
            'description': 'Gemini models via Google API',
            'models': ['gemini-2.0-flash-exp', 'gemini-pro']
        },
        {
            'id': 'groq',
            'name': 'Groq',
            'free': False,
            'description': 'Fast inference via Groq',
            'models': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile']
        },
    ]
    
    return jsonify({'providers': providers})


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
