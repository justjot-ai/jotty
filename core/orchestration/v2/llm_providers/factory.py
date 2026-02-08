"""
Provider factory and auto-detection.
"""

import os
import logging
from typing import Optional

from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider, OpenRouterProvider, GroqProvider
from .google import GoogleProvider
from .adapter import JottyClaudeProviderAdapter

logger = logging.getLogger(__name__)


def create_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMProvider:
    """
    Create LLM provider instance.

    Args:
        provider: Provider name ('anthropic', 'openai', 'openrouter', 'groq', 'google')
        model: Model name (uses provider default if not specified)
        api_key: API key (uses environment variable if not specified)

    Returns:
        LLMProvider instance
    """
    provider = provider.lower()

    default_models = {
        'anthropic': 'claude-sonnet-4-20250514',
        'openai': 'gpt-4o',
        'openrouter': 'anthropic/claude-3.5-sonnet',
        'groq': 'llama-3.1-70b-versatile',
        'google': 'gemini-2.0-flash-exp'
    }

    model = model or default_models.get(provider, 'gpt-4o')

    if provider == 'anthropic':
        return AnthropicProvider(model=model, api_key=api_key)
    elif provider == 'openai':
        return OpenAIProvider(model=model, api_key=api_key)
    elif provider == 'openrouter':
        return OpenRouterProvider(model=model, api_key=api_key)
    elif provider == 'groq':
        return GroqProvider(model=model, api_key=api_key)
    elif provider == 'google':
        return GoogleProvider(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: anthropic, openai, openrouter, groq, google")


def auto_detect_provider() -> tuple:
    """
    Auto-detect available provider based on environment variables or CLI providers.

    Priority order:
    1. JottyClaudeProvider (CLI-based, most reliable)
    2. API key providers (anthropic, openai, openrouter, groq, google)

    Returns:
        Tuple of (provider_name, LLMProvider instance)
    """
    # 1. Try JottyClaudeProvider first (uses Claude CLI, most reliable)
    try:
        from ...foundation.jotty_claude_provider import JottyClaudeProvider, is_claude_available
        if is_claude_available():
            provider = JottyClaudeProvider(auto_start=True)
            # Return a wrapper that uses JottyClaudeProvider
            return 'jotty-claude', JottyClaudeProviderAdapter(provider)
    except Exception as e:
        logger.debug(f"JottyClaudeProvider not available: {e}")

    # 2. Check API key providers
    providers_to_check = [
        ('anthropic', 'ANTHROPIC_API_KEY'),
        ('openai', 'OPENAI_API_KEY'),
        ('openrouter', 'OPENROUTER_API_KEY'),
        ('groq', 'GROQ_API_KEY'),
        ('google', 'GOOGLE_API_KEY'),
    ]

    for provider_name, env_key in providers_to_check:
        if os.environ.get(env_key):
            try:
                return provider_name, create_provider(provider_name)
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")
                continue

    raise RuntimeError("No LLM provider available. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or install Claude CLI")
