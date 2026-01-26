"""
Claude CLI LLM Skill

Text generation and summarization using Claude CLI.
Uses subprocess to call `claude` command-line tool.
"""
import subprocess
import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _call_claude_cli(prompt: str, model: str = 'sonnet', timeout: int = 120) -> Dict[str, Any]:
    """
    Call Claude CLI with a prompt.
    
    Args:
        prompt: User prompt
        model: Claude model (sonnet, opus, haiku)
        timeout: Command timeout in seconds
    
    Returns:
        Dictionary with 'success', 'result', and optional 'error'
    """
    try:
        # Build command
        cmd = [
            'claude',
            '--model', model,
            '--print',  # Non-interactive mode
            '--output-format', 'json',  # JSON output
        ]
        
        cmd.append(prompt)
        
        # Handle OAuth tokens (don't work with --print mode)
        env = os.environ.copy()
        api_key = env.get('ANTHROPIC_API_KEY', '')
        if api_key.startswith('sk-ant-oat'):
            env.pop('ANTHROPIC_API_KEY', None)
        
        # Execute command
        result = subprocess.run(
            cmd,
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
        
        # Parse JSON response
        try:
            response_data = json.loads(result.stdout.strip())
            # Extract result field
            response_text = response_data.get('result', result.stdout.strip())
            
            return {
                'success': True,
                'result': response_text
            }
        except json.JSONDecodeError:
            # Fallback to raw text
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
    Generate text using Claude CLI.
    
    Args:
        params: Dictionary containing:
            - prompt (str, required): Text generation prompt
            - model (str, optional): Claude model (default: 'sonnet')
            - max_tokens (int, optional): Maximum tokens (not used, CLI handles it)
            - timeout (int, optional): Timeout in seconds (default: 120)
    
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
        timeout = params.get('timeout', 120)  # Default 120s, can be overridden
        
        # Call Claude CLI
        result = _call_claude_cli(prompt, model=model, timeout=timeout)
        
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
