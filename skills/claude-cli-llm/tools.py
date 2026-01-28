"""
Claude CLI LLM Skill

Text generation and summarization using the core LLM module.
Supports Claude CLI with automatic fallback to API providers.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def summarize_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize text using Claude.

    Args:
        params: Dictionary containing:
            - content (str, required): Text content to summarize
            - prompt (str, optional): Custom prompt (default: "Summarize the following:")
            - model (str, optional): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')
            - provider (str, optional): Provider to use (default: 'claude-cli')
            - fallback (bool, optional): Enable fallback to other providers (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Summarized text
            - error (str, optional): Error message if failed
    """
    try:
        from core.llm import generate

        content = params.get('content')
        if not content:
            return {'success': False, 'error': 'content parameter is required'}

        # Build prompt
        custom_prompt = params.get('prompt', 'Summarize the following content in a clear and concise way:')
        full_prompt = f"{custom_prompt}\n\n{content}"

        model = params.get('model', 'sonnet')
        # Auto-detect provider: prefer API if key available, otherwise CLI
        import os
        if os.getenv('ANTHROPIC_API_KEY'):
            default_provider = 'anthropic'  # Use API when key is available
        else:
            default_provider = 'claude-cli'  # Fallback to CLI
        provider = params.get('provider', default_provider)
        fallback = params.get('fallback', False)
        timeout = params.get('timeout', 120)

        # Call unified LLM
        response = generate(
            prompt=full_prompt,
            model=model,
            provider=provider,
            timeout=timeout,
            fallback=fallback
        )

        if not response.success:
            return {'success': False, 'error': response.error}

        return {
            'success': True,
            'summary': response.text,
            'model': response.model,
            'provider': response.provider
        }

    except Exception as e:
        logger.error(f"Summarize text error: {e}", exc_info=True)
        return {'success': False, 'error': f'Summarization failed: {str(e)}'}


def generate_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text using Claude.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text generation prompt
            - model (str, optional): Claude model (default: 'sonnet')
            - provider (str, optional): Provider to use (default: 'claude-cli')
            - timeout (int, optional): Timeout in seconds (default: 120)
            - max_tokens (int, optional): Maximum tokens (default: 4096)
            - fallback (bool, optional): Enable fallback to other providers (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - error (str, optional): Error message if failed
    """
    try:
        from core.llm import generate

        prompt = params.get('prompt')
        if not prompt:
            return {'success': False, 'error': 'prompt parameter is required'}

        model = params.get('model', 'sonnet')
        # Auto-detect provider: prefer API if key available, otherwise CLI
        import os
        if os.getenv('ANTHROPIC_API_KEY'):
            default_provider = 'anthropic'  # Use API when key is available
        else:
            default_provider = 'claude-cli'  # Fallback to CLI
        provider = params.get('provider', default_provider)
        timeout = params.get('timeout', 120)
        max_tokens = params.get('max_tokens', 4096)
        fallback = params.get('fallback', False)

        # Call unified LLM
        response = generate(
            prompt=prompt,
            model=model,
            provider=provider,
            timeout=timeout,
            max_tokens=max_tokens,
            fallback=fallback
        )

        if not response.success:
            return {'success': False, 'error': response.error}

        return {
            'success': True,
            'text': response.text,
            'model': response.model,
            'provider': response.provider
        }

    except Exception as e:
        logger.error(f"Generate text error: {e}", exc_info=True)
        return {'success': False, 'error': f'Text generation failed: {str(e)}'}


__all__ = ['summarize_text_tool', 'generate_text_tool']
