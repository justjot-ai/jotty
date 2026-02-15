#!/usr/bin/env python3
"""
Jotty Claude Provider
=====================

Unified Claude CLI provider for Jotty SDK.
Auto-manages the OpenAI-compatible wrapper server for optimal performance.

Usage:
    from core.foundation.jotty_claude_provider import JottyClaudeProvider

    # Initialize (auto-starts wrapper if needed)
    provider = JottyClaudeProvider()

    # Get DSPy LM
    lm = provider.get_lm()
    dspy.configure(lm=lm)

    # Or use convenience function
    from core.foundation.jotty_claude_provider import configure_jotty_claude
    configure_jotty_claude()  # One-liner setup
"""

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import dspy
import requests

from Jotty.core.infrastructure.foundation.exceptions import LLMError

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_WORKSPACE = "/tmp/jotty_claude_workspace"
STARTUP_TIMEOUT = 15  # seconds


class JottyClaudeProvider:
    """
    Jotty's unified Claude provider.

    Manages the OpenAI-compatible wrapper server automatically.
    Falls back to direct CLI if wrapper unavailable.
    """

    _instance = None
    _server_process = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Singleton pattern - only one provider instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._server_process = None

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        model: str = DEFAULT_MODEL,
        workspace: str = DEFAULT_WORKSPACE,
        auto_start: bool = True,
    ) -> None:
        """
        Initialize Jotty Claude Provider.

        Args:
            host: Wrapper server host
            port: Wrapper server port
            model: Default Claude model
            workspace: Working directory for Claude
            auto_start: Auto-start wrapper server if not running
        """
        if self._initialized:
            return

        self.host = host
        self.port = port
        self.model = model
        self.workspace = Path(workspace)
        self.auto_start = auto_start
        self.base_url = f"http://{host}:{port}/v1"
        self._lm = None

        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Check if Claude CLI is available
        self.claude_path = shutil.which("claude")
        if not self.claude_path:
            raise LLMError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        # Auto-start wrapper if needed
        if auto_start:
            self._ensure_server_running()

        self._initialized = True
        logger.debug(f"JottyClaudeProvider initialized (port={port}, model={model})")

    def _is_server_running(self) -> bool:
        """Check if wrapper server is running."""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _start_server(self) -> bool:
        """Start the wrapper server."""
        if self._is_server_running():
            logger.info(" Wrapper server already running")
            return True

        logger.info(f" Starting Claude wrapper server on port {self.port}...")

        try:
            # Build the server command
            server_code = f"""
import os
os.environ['CLAUDE_CWD'] = '{self.workspace}'
import uvicorn
from src.main import app
uvicorn.run(app, host='{self.host}', port={self.port}, log_level='warning')
"""

            # Start server process (detached, survives Python exit)
            self._server_process = subprocess.Popen(
                [sys.executable, "-c", server_code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Note: We don't register atexit cleanup anymore
            # The server should persist for subsequent calls

            # Wait for server to be ready
            for i in range(STARTUP_TIMEOUT):
                if self._is_server_running():
                    logger.info(f" Wrapper server started in {i+1}s")
                    return True
                time.sleep(1)

            logger.error(" Wrapper server failed to start")
            return False

        except Exception as e:
            logger.error(f" Failed to start wrapper: {e}")
            return False

    def _stop_server(self) -> Any:
        """Stop the wrapper server."""
        if self._server_process:
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process = None
                logger.info(" Wrapper server stopped")
            except Exception:
                pass

    def _ensure_server_running(self) -> bool:
        """Ensure wrapper server is running, start if needed."""
        if self._is_server_running():
            return True
        return self._start_server()

    def get_lm(self, model: Optional[str] = None) -> dspy.LM:
        """
        Get DSPy LM instance.

        Uses OpenAI wrapper if available, falls back to direct CLI.

        Args:
            model: Model override (default: configured model)

        Returns:
            DSPy LM instance
        """
        model = model or self.model

        # Try OpenAI wrapper first (fastest)
        if self._is_server_running():
            if self._lm is None or getattr(self._lm, "_model", None) != model:
                from Jotty.core.infrastructure.foundation.config_defaults import (
                    LLM_MAX_OUTPUT_TOKENS,
                )

                self._lm = dspy.LM(
                    model=f"openai/{model}",
                    api_base=self.base_url,
                    api_key="not-needed",
                    max_tokens=LLM_MAX_OUTPUT_TOKENS,
                    # Enable JSON output mode for structured responses
                    response_format={"type": "json_object"},
                    # Longer timeout - Claude CLI calls can take 30-60s each
                    # When serialized, multiple calls queue up and need more time
                    timeout=300,  # 5 minutes to handle queued requests
                )
                self._lm._model = model
                logger.info(f" Using OpenAI wrapper LM: {model} (JSON mode, 300s timeout)")
            return self._lm

        # Fallback to direct CLI
        logger.warning(" Wrapper not available, using DirectClaudeCLI")
        try:
            from .direct_claude_cli_lm import DirectClaudeCLI

            return DirectClaudeCLI(model="sonnet")
        except ImportError:
            from ..integration.direct_claude_cli_lm import DirectClaudeCLI

            return DirectClaudeCLI(model="sonnet")

    def configure_dspy(self, model: Optional[str] = None) -> dspy.LM:
        """
        Configure DSPy with this provider's LM.

        Args:
            model: Model override

        Returns:
            Configured LM
        """
        lm = self.get_lm(model)
        dspy.configure(lm=lm)
        return lm

    @property
    def is_ready(self) -> bool:
        """Check if provider is ready."""
        return self._is_server_running() or self.claude_path is not None

    def status(self) -> dict:
        """Get provider status."""
        return {
            "initialized": self._initialized,
            "server_running": self._is_server_running(),
            "server_url": self.base_url,
            "model": self.model,
            "workspace": str(self.workspace),
            "claude_cli": self.claude_path,
        }


# Convenience functions


def configure_jotty_claude(
    model: str = DEFAULT_MODEL, auto_start: bool = True, **kwargs: Any
) -> dspy.LM:
    """
    One-liner to configure DSPy with Jotty's Claude provider.

    Usage:
        from core.foundation.jotty_claude_provider import configure_jotty_claude

        configure_jotty_claude()  # That's it!

        # Now use DSPy normally
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is 2+2?")
    """
    provider = JottyClaudeProvider(auto_start=auto_start, **kwargs)
    return provider.configure_dspy(model)


def get_jotty_claude_provider(**kwargs: Any) -> JottyClaudeProvider:
    """Get the singleton JottyClaudeProvider instance."""
    return JottyClaudeProvider(**kwargs)


def is_claude_available() -> bool:
    """Check if Claude CLI is available on this system."""
    return shutil.which("claude") is not None
