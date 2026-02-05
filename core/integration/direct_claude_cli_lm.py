#!/usr/bin/env python3
"""
Direct Claude CLI DSPy LM Wrapper

This module provides a DSPy-compatible LM that calls the Claude CLI binary directly
via subprocess, bypassing the HTTP API (no authentication needed).

Features:
- Automatic retry with exponential backoff
- Handles API policy errors, timeouts, rate limits
- Tracks success/failure metrics
"""

import subprocess
import json
import logging
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
import dspy

logger = logging.getLogger(__name__)


def get_current_context() -> str:
    """
    Get current date/time context for LLM.
    This ensures the LLM knows the actual current date.
    """
    now = datetime.now()
    return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')}). Current time: {now.strftime('%H:%M:%S %Z')}."


class DirectClaudeCLI(dspy.BaseLM):
    """DSPy-compatible LM that calls Claude CLI binary directly with retry logic."""

    # Retry configuration
    MAX_RETRIES = 3
    BASE_DELAY = 2.0  # seconds
    MAX_DELAY = 30.0  # seconds

    # Retryable error patterns
    RETRYABLE_ERRORS = [
        "Usage Policy",
        "rate limit",
        "overloaded",
        "temporarily unavailable",
        "timeout",
        "connection",
    ]

    def __init__(self, model: str = "sonnet", max_retries: int = 3, **kwargs):
        """
        Initialize Direct Claude CLI wrapper.

        Args:
            model: Claude model (sonnet, opus, haiku)
            max_retries: Maximum retry attempts (default 3)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(model=model, **kwargs)
        self.model = model
        self.max_retries = max_retries
        self.kwargs = kwargs
        self.history: List[Dict[str, Any]] = []

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0

    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if error is retryable."""
        error_lower = error_msg.lower()
        return any(pattern.lower() in error_lower for pattern in self.RETRYABLE_ERRORS)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    def _single_call(self, input_text: str, timeout: int = 180) -> str:
        """Make a single call to Claude CLI."""
        # Use claude-sonnet-4 model which has better handling
        model = 'claude-sonnet-4-20250514' if self.model == 'sonnet' else self.model

        result = subprocess.run(
            ['claude', '--model', model, '--'],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"Claude CLI failed: {error_msg[:300]}")

        response = result.stdout.strip()
        if not response:
            raise RuntimeError("Empty response from Claude CLI")

        return response

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs) -> List[str]:
        """
        Call Claude CLI with prompt and automatic retry.

        Args:
            prompt: Text prompt (DSPy uses this)
            messages: Chat messages (alternative format)
            **kwargs: Additional arguments

        Returns:
            List of generated responses
        """
        self.total_calls += 1

        # Extract prompt from either format
        if prompt:
            input_text = prompt
        elif messages:
            input_text = "\n\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            ])
        else:
            raise ValueError("Either prompt or messages must be provided")

        logger.debug(f"Claude CLI call #{self.total_calls} with model: {self.model}")

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Adjust timeout based on attempt (longer timeout for retries)
                timeout = 180 + (attempt * 60)  # 180s, 240s, 300s, 360s

                response = self._single_call(input_text, timeout)

                # Success
                self.successful_calls += 1
                if attempt > 0:
                    logger.info(f"✅ Succeeded on retry {attempt}")

                # Store in history
                self.history.append({
                    'prompt': input_text[:500],
                    'response': response[:500],
                    'model': self.model,
                    'attempts': attempt + 1,
                    'timestamp': datetime.now().isoformat()
                })

                logger.debug(f"Response: {response[:150]}...")
                return [response]

            except subprocess.TimeoutExpired:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"⏱️ Attempt {attempt + 1}/{self.max_retries + 1}: {last_error}")

            except FileNotFoundError:
                logger.error("Claude CLI binary not found")
                self.failed_calls += 1
                raise RuntimeError("Claude CLI not installed")

            except RuntimeError as e:
                last_error = str(e)
                error_msg = str(e)

                # Check if retryable
                if not self._is_retryable_error(error_msg):
                    logger.error(f"❌ Non-retryable error: {error_msg[:200]}")
                    self.failed_calls += 1
                    raise

                logger.warning(f"⚠️ Attempt {attempt + 1}/{self.max_retries + 1}: {error_msg[:100]}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Attempt {attempt + 1}/{self.max_retries + 1}: {last_error[:100]}")

            # Should we retry?
            if attempt < self.max_retries:
                delay = self._calculate_delay(attempt)
                logger.info(f"🔄 Retrying in {delay:.1f}s...")
                self.retried_calls += 1
                time.sleep(delay)

        # All retries exhausted
        self.failed_calls += 1
        logger.error(f"❌ All {self.max_retries + 1} attempts failed. Last error: {last_error}")
        raise RuntimeError(f"Claude CLI failed after {self.max_retries + 1} attempts: {last_error}")

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

    def get_metrics(self) -> Dict[str, Any]:
        """Get call metrics for monitoring."""
        success_rate = (
            self.successful_calls / self.total_calls * 100
            if self.total_calls > 0 else 0
        )
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'retried_calls': self.retried_calls,
            'success_rate': f"{success_rate:.1f}%",
            'model': self.model
        }

    def reset_metrics(self):
        """Reset call metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
