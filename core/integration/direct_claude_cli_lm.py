#!/usr/bin/env python3
"""
Direct Claude CLI DSPy LM Wrapper

This module provides a DSPy-compatible LM that calls the Claude CLI binary directly
via subprocess, bypassing the HTTP API (no authentication needed).
"""

import subprocess
import json
import logging
from typing import Any, Dict, List, Optional
import dspy

logger = logging.getLogger(__name__)


class DirectClaudeCLI(dspy.BaseLM):
    """DSPy-compatible LM that calls Claude CLI binary directly."""

    def __init__(self, model: str = "sonnet", **kwargs):
        """
        Initialize Direct Claude CLI wrapper.

        Args:
            model: Claude model (sonnet, opus, haiku)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(model=model, **kwargs)
        self.model = model
        self.kwargs = kwargs
        self.history: List[Dict[str, Any]] = []

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs) -> List[str]:
        """
        Call Claude CLI with prompt.

        Args:
            prompt: Text prompt (DSPy uses this)
            messages: Chat messages (alternative format)
            **kwargs: Additional arguments

        Returns:
            List of generated responses
        """
        # Extract prompt from either format
        if prompt:
            input_text = prompt
        elif messages:
            # Convert messages to text
            input_text = "\n\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            ])
        else:
            raise ValueError("Either prompt or messages must be provided")

        logger.info(f"Calling Claude CLI with model: {self.model}")
        logger.debug(f"Input: {input_text[:200]}...")

        try:
            # Call Claude CLI binary
            result = subprocess.run(
                ['claude', '--model', self.model, '--'],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout for complex tasks
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"Claude CLI error: {error_msg}")
                raise RuntimeError(f"Claude CLI failed: {error_msg}")

            response = result.stdout.strip()

            if not response:
                logger.warning("Empty response from Claude CLI")
                response = "Error: Empty response from Claude"

            logger.info(f"Response: {response[:200]}...")

            # Store in history
            self.history.append({
                'prompt': input_text,
                'response': response,
                'model': self.model
            })

            # DSPy expects list of responses
            return [response]

        except subprocess.TimeoutExpired:
            logger.error("Claude CLI timed out after 180 seconds")
            raise RuntimeError("Claude CLI timed out after 3 minutes")

        except FileNotFoundError:
            logger.error("Claude CLI binary not found. Install with: npm install -g @anthropic-ai/claude-code")
            raise RuntimeError("Claude CLI not installed")

        except Exception as e:
            logger.error(f"Unexpected error calling Claude CLI: {e}")
            raise

    def inspect_history(self, n: int = 1) -> Dict[str, Any]:
        """
        Get recent history for DSPy compatibility.

        Args:
            n: Number of recent calls to return

        Returns:
            Dictionary with recent history
        """
        recent = self.history[-n:] if self.history else []
        return {
            'history': recent,
            'model': self.model
        }
