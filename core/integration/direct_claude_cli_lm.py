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
from Jotty.core.foundation.exceptions import LLMError

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

    # Retry configuration - balanced for reliability
    MAX_RETRIES = 3  # 4 total attempts with adaptive timeouts
    BASE_DELAY = 0.5  # seconds
    MAX_DELAY = 5.0  # seconds

    # Retryable error patterns (transient issues only)
    RETRYABLE_ERRORS = [
        "rate limit",
        "overloaded",
        "temporarily unavailable",
        "timeout",
        "connection",
        "empty response",  # Transient - CLI may return empty on overload
    ]

    # Non-retryable errors (policy violations, etc.) - fail immediately
    NON_RETRYABLE_ERRORS = [
        "violation",
        "unable to respond",
        "policy",
        "inappropriate",
        "cannot assist",
        "refuse",
    ]

    # Model-specific timeout defaults (increased for complex DSPy prompts)
    MODEL_TIMEOUTS = {
        "haiku": 60,   # Haiku is fast but DSPy prompts are complex
        "sonnet": 120,  # Sonnet needs more time
        "opus": 180,   # Opus is slower but higher quality
    }

    def __init__(self, model: str = '', max_retries: int = 0, base_timeout: int = None, **kwargs: Any) -> None:
        """
        Initialize Direct Claude CLI wrapper.

        Args:
            model: Claude model (sonnet, opus, haiku)
            max_retries: Maximum retry attempts (default 3)
            base_timeout: Base timeout in seconds (auto-set based on model if None)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS, MAX_RETRIES
        model = model or DEFAULT_MODEL_ALIAS
        max_retries = max_retries or MAX_RETRIES

        super().__init__(model=model, **kwargs)
        self.model = model
        self.max_retries = max_retries
        # Auto-select timeout based on model if not specified
        self.base_timeout = base_timeout or self.MODEL_TIMEOUTS.get(model, 90)
        self.kwargs = kwargs
        self.history: List[Dict[str, Any]] = []

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0

        logger.debug(f"DirectClaudeCLI initialized: model={model}, base_timeout={self.base_timeout}s")

    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if error is retryable (not a policy violation).

        Default: retry unknown errors (most are transient).
        Only skip retry for explicit policy violations.
        """
        error_lower = error_msg.lower()

        # Check for non-retryable errors first (policy violations)
        if any(pattern.lower() in error_lower for pattern in self.NON_RETRYABLE_ERRORS):
            logger.error(f" Policy violation detected - will not retry")
            return False

        # Everything else is retryable (default to retry for unknown errors)
        # This includes explicit RETRYABLE_ERRORS and any unknown transient errors
        return True

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    # Model aliases that Claude CLI accepts directly
    # The CLI handles these natively: 'sonnet', 'haiku', 'opus'
    # Only map if we need specific versions
    MODEL_MAP = {
        # Claude CLI accepts aliases directly - no mapping needed for standard aliases
        # Only add mappings for specific version overrides if needed
    }

    def _single_call(self, input_text: str, timeout: int = 180) -> str:
        """Make a single call to Claude CLI."""
        # Use model directly (CLI accepts aliases like 'sonnet', 'haiku', 'opus')
        # Only use MODEL_MAP for specific version overrides
        model = self.MODEL_MAP.get(self.model, self.model)

        result = subprocess.run(
            ['claude', '--model', model, '--'],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise LLMError(f"Claude CLI failed: {error_msg[:300]}")

        response = result.stdout.strip()
        if not response:
            raise LLMError("Empty response from Claude CLI")

        return response

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs: Any) -> List[str]:
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
            # Better formatting for DSPy messages - extract system prompt and user content
            parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    # System prompts go first without role prefix
                    parts.insert(0, content)
                elif role == 'user':
                    parts.append(content)
                elif role == 'assistant':
                    parts.append(f"Assistant: {content}")
            input_text = "\n\n".join(parts)
        else:
            raise ValueError("Either prompt or messages must be provided")

        # Log prompt size for debugging
        if len(input_text) > 2000:
            logger.debug(f"Large prompt: {len(input_text)} chars")

        logger.debug(f"Claude CLI call #{self.total_calls} with model: {self.model}")

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Adaptive timeouts - use model-based base_timeout, increasing with retries
                # haiku: 45s, 68s, 90s, 113s (much faster)
                # sonnet: 90s, 135s, 180s, 225s
                # opus: 180s, 270s, 360s, 450s
                timeout = self.base_timeout + (attempt * int(self.base_timeout * 0.5))

                response = self._single_call(input_text, timeout)

                # Success
                self.successful_calls += 1
                if attempt > 0:
                    logger.info(f" Succeeded on retry {attempt}")

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
                logger.warning(f"⏱ Attempt {attempt + 1}/{self.max_retries + 1}: {last_error}")

            except FileNotFoundError as e:
                logger.error("Claude CLI binary not found")
                self.failed_calls += 1
                raise LLMError("Claude CLI not installed", original_error=e)

            except RuntimeError as e:
                last_error = str(e)
                error_msg = str(e)

                # Check if retryable
                if not self._is_retryable_error(error_msg):
                    logger.error(f" Non-retryable error: {error_msg[:200]}")
                    self.failed_calls += 1
                    raise

                logger.warning(f" Attempt {attempt + 1}/{self.max_retries + 1}: {error_msg[:100]}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f" Attempt {attempt + 1}/{self.max_retries + 1}: {last_error[:100]}")

            # Should we retry?
            if attempt < self.max_retries:
                delay = self._calculate_delay(attempt)
                logger.info(f" Retrying in {delay:.1f}s...")
                self.retried_calls += 1
                time.sleep(delay)

        # All retries exhausted
        self.failed_calls += 1
        logger.error(f" All {self.max_retries + 1} attempts failed. Last error: {last_error}")
        raise LLMError(f"Claude CLI failed after {self.max_retries + 1} attempts: {last_error}")

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

    def reset_metrics(self) -> None:
        """Reset call metrics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
