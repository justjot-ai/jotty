"""
AI SDK Provider Adapter for DSPy
=================================

Bridges JustJot.ai AI SDK providers to DSPy's LM interface.
Allows Jotty agents to use all AI SDK providers (including CLI providers).

This enables:
- Jotty agents to use claude-cli, cursor-cli providers
- Jotty agents to use all API providers (anthropic, groq, openai, etc.)
- Unified provider system across the entire platform
- All agent capabilities accessible through Jotty
"""

import dspy
from dspy import BaseLM
from typing import Dict, Any, Optional, List
import requests
import json
import os
from urllib.parse import urljoin

from Jotty.core.foundation.config_defaults import (
    DEFAULT_MODEL_ALIAS, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS,
)


class AISDKProviderLM(BaseLM):
    """
    DSPy-compatible LM that uses JustJot.ai AI SDK providers via HTTP.
    
    Supports all AI SDK providers:
    - CLI providers: 'claude-cli', 'cursor-cli'
    - API providers: 'anthropic', 'groq', 'openai', 'google', 'openrouter'
    
    Usage:
        import dspy
        from Jotty.core.integration.ai_sdk_provider_adapter import AISDKProviderLM
        
        # Use CLI provider
        dspy.configure(lm=AISDKProviderLM(provider='cursor-cli', model='sonnet'))
        
        # Use API provider
        dspy.configure(lm=AISDKProviderLM(provider='anthropic', model='claude-3-5-sonnet'))
    """
    
    def __init__(
        self,
        provider: str,
        model: str = 'sonnet',
        base_url: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize AI SDK provider adapter
        
        Args:
            provider: Provider ID ('claude-cli', 'cursor-cli', 'anthropic', 'groq', etc.)
            model: Model ID (e.g., 'sonnet', 'claude-3-5-sonnet', 'llama-3.1-8b-instant')
            base_url: Base URL for JustJot.ai API (defaults to env or localhost:3000)
            api_key: Optional API key (for API providers, not needed for CLI providers)
        """
        super().__init__(model=f"{provider}/{model}", **kwargs)
        self.provider = provider
        self.model = model
        self.api_key = api_key
        from ..foundation.config_defaults import DEFAULTS as _DEFAULTS
        self.base_url = base_url or os.getenv('JUSTJOT_API_URL', _DEFAULTS.JUSTJOT_API_URL)
        self.history = []
        
        # Validate provider
        valid_providers = [
            'opencode',  # OpenCode (free GLM model)
            'claude-cli', 'cursor-cli',  # CLI providers
            'anthropic', 'groq', 'openai', 'google', 'openrouter'  # API providers
        ]
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Valid: {valid_providers}")
    
    def _call_ai_sdk(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """
        Call JustJot.ai AI SDK endpoint
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            
        Returns:
            Response dict with 'text' or streaming data
        """
        # OpenCode uses direct CLI execution endpoint to avoid circular dependency
        if self.provider == 'opencode':
            url = urljoin(self.base_url, '/api/ai/opencode')
            payload = {
                'prompt': messages[-1]['content'] if messages else '',
                'model': self.model or 'default',
                'stream': stream,
                'conversationHistory': messages[:-1] if len(messages) > 1 else []
            }
        else:
            # Other providers use unified execute endpoint
            url = urljoin(self.base_url, '/api/ai/execute')
            payload = {
                'mode': 'chat',
                'provider': self.provider,
                'model': self.model,
                'prompt': messages[-1]['content'] if messages else '',
                'context': {
                    'conversationHistory': messages
                },
                'stream': stream
            }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            response = requests.post(url, json=payload, headers=headers, stream=stream, timeout=LLM_TIMEOUT_SECONDS)
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                return self._handle_streaming_response(response)
            else:
                # Handle non-streaming response
                data = response.json()
                # OpenCode endpoint returns {content, usage}, execute endpoint returns {result: {text}, usage}
                if self.provider == 'opencode':
                    return {
                        'text': data.get('content', ''),
                        'usage': data.get('usage', {})
                    }
                else:
                    return {
                        'text': data.get('result', {}).get('text', '') or data.get('content', ''),
                        'usage': data.get('usage', {})
                    }
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"AI SDK provider error ({self.provider}): {e}")
    
    def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """Handle SSE streaming response"""
        text_parts = []
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data:'):
                    data_str = line_str[5:].strip()
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        # Handle both formats: {type: 'text-delta', data: {textDelta}} and {textDelta}
                        if data.get('type') == 'text-delta' and data.get('data', {}).get('textDelta'):
                            text_parts.append(data['data']['textDelta'])
                        elif 'textDelta' in data:
                            text_parts.append(data['textDelta'])
                    except json.JSONDecodeError:
                        pass
        
        return {
            'text': ''.join(text_parts),
            'usage': {}
        }
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        DSPy-compatible call interface
        
        Args:
            prompt: Single prompt string (optional)
            messages: List of message dicts (optional)
            **kwargs: Additional arguments
            
        Returns:
            List of response strings (DSPy format)
        """
        # Build messages list
        if messages is None:
            messages = []
            if prompt:
                messages = [{'role': 'user', 'content': prompt}]
        
        if not messages:
            raise ValueError("Either prompt or messages must be provided")
        
        # Ensure messages are in correct format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': str(msg.get('content', ''))
                })
            elif isinstance(msg, str):
                formatted_messages.append({'role': 'user', 'content': msg})
        
        # Call AI SDK
        result = self._call_ai_sdk(formatted_messages, stream=kwargs.get('stream', False))
        
        # Store in history
        self.history.append({
            'prompt': prompt or formatted_messages,
            'response': result['text'],
            'kwargs': kwargs
        })
        
        # Return in DSPy format (list of strings)
        return [result['text']]
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response (compatible with DSPy's expected interface)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Dict with 'choices' containing the response
        """
        result = self._call_ai_sdk(messages, stream=False)
        
        return {
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': result['text']
                    }
                }
            ],
            'usage': result.get('usage', {
                'total_tokens': len(result['text'].split())  # Rough estimate
            })
        }
    
    def inspect_history(self, n=1):
        """DSPy-compatible history inspection"""
        return self.history[-n:] if self.history else []


# Convenience function for easy configuration
def configure_dspy_with_ai_sdk_provider(
    provider: str,
    model: str = 'sonnet',
    base_url: str = None,
    api_key: str = None
):
    """
    Configure DSPy with an AI SDK provider
    
    Usage:
        from Jotty.core.integration.ai_sdk_provider_adapter import configure_dspy_with_ai_sdk_provider
        import dspy
        
        configure_dspy_with_ai_sdk_provider('cursor-cli', 'sonnet')
        # Now all DSPy agents use cursor-cli provider
    """
    lm = AISDKProviderLM(provider=provider, model=model, base_url=base_url, api_key=api_key)
    dspy.configure(lm=lm)
    return lm
