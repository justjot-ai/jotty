#!/usr/bin/env python3
"""
Claude OpenAI-Compatible DSPy LM Provider
==========================================

Uses the claude-code-openai-wrapper as a local OpenAI-compatible endpoint.
This provides fast, reliable access to Claude CLI through DSPy's native OpenAI support.

Prerequisites:
- claude-code-openai-wrapper running on localhost:8765
- Start with: python -c "import uvicorn; from src.main import app; uvicorn.run(app, host='127.0.0.1', port=8765)"

Features:
- Uses DSPy's native OpenAI LM support
- Fast (~4-5s per request)
- No subprocess overhead
- Proper streaming support
"""

import os
import logging
import subprocess
import time
import requests
import dspy
from typing import Optional
from Jotty.core.foundation.exceptions import LLMError

logger = logging.getLogger(__name__)

# Default wrapper endpoint (centralized default, overridable via env var)
try:
    from .config_defaults import DEFAULTS as _DEFAULTS
    _PROXY_BASE = os.getenv("CLAUDE_OPENAI_PROXY_URL", _DEFAULTS.CLAUDE_OPENAI_PROXY_URL)
except ImportError:
    _PROXY_BASE = os.getenv("CLAUDE_OPENAI_PROXY_URL", "http://127.0.0.1:8765")
DEFAULT_WRAPPER_URL = f"{_PROXY_BASE}/v1"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def is_wrapper_running(base_url: str = DEFAULT_WRAPPER_URL) -> bool:
    """Check if the OpenAI wrapper is running."""
    try:
        health_url = base_url.replace("/v1", "/health")
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def start_wrapper_server(port: int = 8765, wait_seconds: int = 15) -> bool:
    """
    Start the claude-code-openai-wrapper server in background.

    Returns True if server is running (started or already running).
    """
    if is_wrapper_running(f"http://127.0.0.1:{port}/v1"):
        logger.info(" OpenAI wrapper already running")
        return True

    try:
        # Start server in background
        subprocess.Popen(
            ["python", "-c", f"""
import uvicorn
from src.main import app
uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')
"""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        # Wait for server to be ready
        for i in range(wait_seconds):
            if is_wrapper_running(f"http://127.0.0.1:{port}/v1"):
                logger.info(f" OpenAI wrapper started after {i+1}s")
                return True
            time.sleep(1)

        logger.warning(" OpenAI wrapper failed to start")
        return False

    except Exception as e:
        logger.error(f"Failed to start wrapper: {e}")
        return False


def create_claude_openai_lm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_WRAPPER_URL,
    auto_start: bool = True,
    **kwargs
) -> dspy.LM:
    """
    Create a DSPy LM using the local OpenAI-compatible Claude wrapper.

    Args:
        model: Model name (default: claude-sonnet-4-5-20250929)
        base_url: Wrapper API URL (default: http://127.0.0.1:8765/v1)
        auto_start: Auto-start wrapper if not running (default: True)
        **kwargs: Additional arguments for dspy.LM

    Returns:
        DSPy LM instance configured for Claude via OpenAI wrapper
    """
    # Auto-start wrapper if needed
    if auto_start and not is_wrapper_running(base_url):
        port = int(base_url.split(":")[-1].replace("/v1", ""))
        if not start_wrapper_server(port):
            raise LLMError(
                "OpenAI wrapper not running and failed to start. "
                "Start manually with: python -c \"import uvicorn; from src.main import app; "
                "uvicorn.run(app, host='127.0.0.1', port=8765)\""
            )

    # Create DSPy LM using OpenAI format
    # The wrapper doesn't need an API key, but DSPy requires one
    from Jotty.core.foundation.config_defaults import LLM_MAX_OUTPUT_TOKENS
    kwargs.setdefault('max_tokens', LLM_MAX_OUTPUT_TOKENS)
    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=base_url,
        api_key="not-needed",  # Wrapper doesn't require auth
        **kwargs
    )

    logger.info(f" Created Claude OpenAI LM: {model} @ {base_url}")
    return lm


def configure_claude_openai(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_WRAPPER_URL,
    auto_start: bool = True,
    **kwargs
) -> dspy.LM:
    """
    Configure DSPy with Claude via OpenAI-compatible wrapper.

    Usage:
        from core.foundation.claude_openai_lm import configure_claude_openai

        lm = configure_claude_openai()  # Auto-starts wrapper if needed
        # Now use DSPy normally
    """
    lm = create_claude_openai_lm(model, base_url, auto_start, **kwargs)
    dspy.configure(lm=lm)
    return lm
