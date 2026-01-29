"""
Claude CLI LLM Skill

Text generation and summarization using DSPy LM.
Uses the configured DSPy language model for all operations.
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

    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Summarized text
            - error (str, optional): Error message if failed
    """
    try:
        import dspy

        content = params.get('content')
        if not content:
            return {'success': False, 'error': 'content parameter is required'}

        # Build prompt
        custom_prompt = params.get('prompt', 'Summarize the following content in a clear and concise way:')
        full_prompt = f"{custom_prompt}\n\n{content}"

        # Use DSPy's configured LM
        lm = dspy.settings.lm
        if not lm:
            return {'success': False, 'error': 'No LLM configured in DSPy'}

        # Direct LLM call
        response = lm(prompt=full_prompt)

        # Extract text from response
        if isinstance(response, list):
            text = response[0] if response else ""
        else:
            text = str(response)

        return {
            'success': True,
            'summary': text,
            'model': getattr(lm, 'model', 'unknown'),
            'provider': 'dspy'
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
            - max_tokens (int, optional): Maximum tokens (default: 4096)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - error (str, optional): Error message if failed
    """
    try:
        import dspy

        prompt = params.get('prompt')
        if not prompt:
            return {'success': False, 'error': 'prompt parameter is required'}

        # Use DSPy's configured LM
        lm = dspy.settings.lm
        if not lm:
            return {'success': False, 'error': 'No LLM configured in DSPy'}

        # Direct LLM call
        response = lm(prompt=prompt)

        # Extract text from response
        if isinstance(response, list):
            text = response[0] if response else ""
        else:
            text = str(response)

        return {
            'success': True,
            'text': text,
            'model': getattr(lm, 'model', 'unknown'),
            'provider': 'dspy'
        }

    except Exception as e:
        logger.error(f"Generate text error: {e}", exc_info=True)
        return {'success': False, 'error': f'Text generation failed: {str(e)}'}


__all__ = ['summarize_text_tool', 'generate_text_tool']
