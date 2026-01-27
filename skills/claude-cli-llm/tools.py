"""
Claude CLI LLM Skill

Text generation and summarization using Claude CLI or Anthropic API.
Tries CLI first, falls back to API if CLI fails/times out.
"""
import subprocess
import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Model mapping for API
MODEL_MAP = {
    'haiku': 'claude-3-5-haiku-latest',
    'sonnet': 'claude-sonnet-4-20250514',
    'opus': 'claude-opus-4-20250514',
}


def _call_anthropic_api(prompt: str, model: str = 'sonnet', max_tokens: int = 4096) -> Dict[str, Any]:
    """
    Call Anthropic API directly.

    Args:
        prompt: User prompt
        model: Claude model (sonnet, opus, haiku)
        max_tokens: Maximum output tokens

    Returns:
        Dictionary with 'success', 'result', and optional 'error'
    """
    try:
        import anthropic

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return {'success': False, 'error': 'ANTHROPIC_API_KEY not set'}

        client = anthropic.Anthropic(api_key=api_key)

        model_id = MODEL_MAP.get(model, MODEL_MAP['sonnet'])

        message = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = message.content[0].text if message.content else ''

        return {
            'success': True,
            'result': result_text
        }

    except ImportError:
        return {'success': False, 'error': 'anthropic package not installed'}
    except Exception as e:
        logger.error(f"Anthropic API error: {e}", exc_info=True)
        return {'success': False, 'error': f'Anthropic API failed: {str(e)}'}


def _call_claude_cli(prompt: str, model: str = 'sonnet', timeout: int = 120, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
    Call Claude CLI with a prompt.

    Args:
        prompt: User prompt
        model: Claude model (sonnet, opus, haiku)
        timeout: Command timeout in seconds
        max_tokens: Maximum output tokens (if supported by CLI)

    Returns:
        Dictionary with 'success', 'result', and optional 'error'
    """
    try:
        # Build command - use stdin for prompt to handle long inputs
        cmd = [
            'claude',
            '--model', model,
            '-p',  # Non-interactive mode (short flag)
        ]

        # Handle OAuth tokens
        env = os.environ.copy()
        api_key = env.get('ANTHROPIC_API_KEY', '')
        if api_key.startswith('sk-ant-oat'):
            env.pop('ANTHROPIC_API_KEY', None)

        # Execute command with prompt via stdin
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout
        )

        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Claude CLI error: {result.stderr}'
            }

        # Return text output directly
        return {
            'success': True,
            'result': result.stdout.strip()
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Claude CLI timeout after {timeout} seconds'
        }
    except FileNotFoundError:
        return {
            'success': False,
            'error': 'Claude CLI not found. Install with: npm install -g @anthropic-ai/cli'
        }
    except Exception as e:
        logger.error(f"Claude CLI call error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Claude CLI call failed: {str(e)}'
        }


def summarize_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize text using Claude CLI.
    
    Args:
        params: Dictionary containing:
            - content (str, required): Text content to summarize
            - prompt (str, optional): Custom prompt (default: "Summarize the following:")
            - model (str, optional): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')
            - max_tokens (int, optional): Maximum tokens (not used, CLI handles it)
    
    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Summarized text
            - error (str, optional): Error message if failed
    """
    try:
        content = params.get('content')
        if not content:
            return {
                'success': False,
                'error': 'content parameter is required'
            }
        
        # Build prompt
        custom_prompt = params.get('prompt', 'Summarize the following content in a clear and concise way:')
        full_prompt = f"{custom_prompt}\n\n{content}"
        
        model = params.get('model', 'sonnet')
        
        # Call Claude CLI
        result = _call_claude_cli(full_prompt, model=model)
        
        if not result.get('success'):
            return result
        
        summary = result.get('result', '')
        
        return {
            'success': True,
            'summary': summary,
            'model': model
        }
    
    except Exception as e:
        logger.error(f"Summarize text error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Summarization failed: {str(e)}'
        }


def generate_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text using Claude CLI or API.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text generation prompt
            - model (str, optional): Claude model (default: 'sonnet')
            - max_tokens (int, optional): Maximum tokens (default: 4096)
            - timeout (int, optional): Timeout in seconds (default: 120)
            - use_api (bool, optional): Force API instead of CLI (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - error (str, optional): Error message if failed
    """
    try:
        prompt = params.get('prompt')
        if not prompt:
            return {
                'success': False,
                'error': 'prompt parameter is required'
            }

        model = params.get('model', 'sonnet')
        timeout = params.get('timeout', 120)
        max_tokens = params.get('max_tokens', 4096)
        use_api = params.get('use_api', False)

        # Use API if explicitly requested
        if use_api:
            result = _call_anthropic_api(prompt, model=model, max_tokens=max_tokens)
        else:
            # Use CLI with full timeout
            result = _call_claude_cli(prompt, model=model, timeout=timeout, max_tokens=max_tokens)

            # Fallback to API only if CLI fails and API key is available
            if not result.get('success') and os.environ.get('ANTHROPIC_API_KEY'):
                logger.info(f"CLI failed, falling back to API: {result.get('error')}")
                result = _call_anthropic_api(prompt, model=model, max_tokens=max_tokens)

        if not result.get('success'):
            return result

        generated_text = result.get('result', '')

        return {
            'success': True,
            'text': generated_text,
            'model': model
        }

    except Exception as e:
        logger.error(f"Generate text error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Text generation failed: {str(e)}'
        }


__all__ = ['summarize_text_tool', 'generate_text_tool']
