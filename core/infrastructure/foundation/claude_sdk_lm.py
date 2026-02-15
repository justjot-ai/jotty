#!/usr/bin/env python3
"""
Claude Agent SDK DSPy LM Provider
=================================

Uses the official claude-agent-sdk for Claude Code integration.
This is the most reliable and performant way to use Claude CLI.

Features:
- Uses official SDK (not subprocess calls)
- Async streaming support
- Proper session management
- Uses existing CLI credentials (no API key needed)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import dspy

from Jotty.core.infrastructure.foundation.exceptions import InputValidationError, LLMError

logger = logging.getLogger(__name__)


class ClaudeSDKLM(dspy.BaseLM):
    """
    DSPy-compatible LM using the official Claude Agent SDK.

    This is the recommended way to use Claude Code with DSPy.
    Uses existing `claude auth login` credentials.
    """

    def __init__(
        self, model: str = "", max_turns: int = 1, timeout: int = 0, **kwargs: Any
    ) -> None:
        """
        Initialize Claude SDK LM.

        Args:
            model: Claude model (sonnet, opus, haiku)
            max_turns: Max conversation turns (default 1 for single response)
            timeout: Timeout in seconds
            **kwargs: Additional arguments
        """
        from Jotty.core.infrastructure.foundation.config_defaults import (
            DEFAULT_MODEL_ALIAS,
            LLM_TIMEOUT_SECONDS,
        )

        model = model or DEFAULT_MODEL_ALIAS
        timeout = timeout or LLM_TIMEOUT_SECONDS

        super().__init__(model=f"claude-sdk/{model}", **kwargs)
        self.cli_model = model
        self.max_turns = max_turns
        self.timeout = timeout
        self.provider = "claude-sdk"
        self.history: List[Dict[str, Any]] = []

        # Verify SDK is available
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query

            self._query = query
            self._options_class = ClaudeAgentOptions
            logger.info(" Claude Agent SDK available")
        except ImportError as e:
            raise LLMError(
                "claude-agent-sdk not installed. " "Install with: pip install claude-agent-sdk",
                original_error=e,
            )

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs: Any) -> List[str]:
        """
        Synchronous call interface (required by DSPy).

        Runs the async SDK in an event loop.
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # In async context - use thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._async_call(prompt, messages, **kwargs))
                return future.result(timeout=self.timeout + 10)
        except RuntimeError:
            # Not in async context - safe to use asyncio.run
            return asyncio.run(self._async_call(prompt, messages, **kwargs))

    async def _async_call(
        self, prompt: str = None, messages: List[Dict] = None, **kwargs: Any
    ) -> List[str]:
        """
        Async implementation using Claude Agent SDK.
        """
        # Build the prompt text
        if prompt:
            input_text = prompt
        elif messages:
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)
                elif isinstance(msg, str):
                    parts.append(msg)
            input_text = "\n\n".join(parts)
        else:
            raise InputValidationError("Either prompt or messages must be provided")

        logger.debug(f"ClaudeSDK: Calling with model {self.cli_model}")
        logger.debug(f"ClaudeSDK: Input length: {len(input_text)} chars")

        try:
            # Build SDK options
            options = self._options_class(
                max_turns=self.max_turns,
                model=self.cli_model,
                # Disable tools for faster response (like the wrapper does)
                allowed_tools=[],
            )

            # Collect response from async generator
            response_text = ""
            async for message in self._query(prompt=input_text, options=options):
                # Handle different message types
                if hasattr(message, "type"):
                    if message.type == "assistant":
                        # Extract text content
                        if hasattr(message, "message") and hasattr(message.message, "content"):
                            for block in message.message.content:
                                if hasattr(block, "text"):
                                    response_text += block.text
                        elif hasattr(message, "content"):
                            response_text += str(message.content)
                    elif message.type == "text":
                        if hasattr(message, "text"):
                            response_text += message.text
                    elif message.type == "result":
                        if hasattr(message, "result"):
                            response_text = message.result
                            break
                elif isinstance(message, dict):
                    if message.get("type") == "assistant":
                        content = message.get("message", {}).get("content", [])
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_text += block.get("text", "")
                    elif message.get("type") == "result":
                        response_text = message.get("result", "")
                        break
                elif isinstance(message, str):
                    response_text += message

            if not response_text:
                logger.warning("ClaudeSDK: Empty response")
                response_text = "Error: Empty response from Claude SDK"

            logger.debug(f"ClaudeSDK: Response length: {len(response_text)} chars")

            # Store in history
            self.history.append(
                {
                    "prompt": input_text[:500],
                    "response": response_text[:500],
                    "model": self.cli_model,
                }
            )

            return [response_text]

        except asyncio.TimeoutError as e:
            raise LLMError(f"Claude SDK timed out after {self.timeout}s", original_error=e)
        except Exception as e:
            logger.error(f"ClaudeSDK error: {e}")
            raise

    def inspect_history(self, n: int = 1) -> List[Dict[str, Any]]:
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []


# Test if SDK is available
def is_sdk_available() -> bool:
    """Check if claude-agent-sdk is installed and working."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query

        return True
    except ImportError:
        return False


# Convenience function
def configure_claude_sdk(model: str = "", **kwargs: Any) -> ClaudeSDKLM:
    """
    Configure DSPy with ClaudeSDKLM.

    Usage:
        from core.foundation.claude_sdk_lm import configure_claude_sdk
        import dspy

        lm = configure_claude_sdk()
        dspy.configure(lm=lm)
    """
    lm = ClaudeSDKLM(model=model, **kwargs)
    dspy.configure(lm=lm)
    return lm
