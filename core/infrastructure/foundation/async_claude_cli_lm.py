#!/usr/bin/env python3
"""
Async Claude CLI DSPy LM Provider
=================================

Optimized for async contexts (like Orchestrator).
Based on patterns from claude-code-api.

Key features:
- Uses asyncio.create_subprocess_exec() for non-blocking calls
- Uses -p flag for prompt (more reliable than stdin)
- Uses --output-format stream-json for structured responses
- Properly integrates with DSPy's BaseLM interface
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from typing import Any, Dict, List

import dspy

from Jotty.core.infrastructure.foundation.exceptions import InputValidationError, LLMError

logger = logging.getLogger(__name__)


def get_current_context() -> str:
    """
    Get current date/time context for LLM.
    This ensures the LLM knows the actual current date.
    """
    now = datetime.now()
    return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')}). Current time: {now.strftime('%H:%M:%S')}."


class AsyncClaudeCLILM(dspy.BaseLM):
    """
    Async DSPy-compatible LM using Claude CLI.

    Optimized for async contexts with non-blocking subprocess calls.
    """

    def __init__(self, model: str = "", timeout: int = 0, **kwargs: Any) -> None:
        """
        Initialize Async Claude CLI LM.

        Args:
            model: Claude model (sonnet, opus, haiku)
            timeout: Timeout in seconds (default 120)
            **kwargs: Additional arguments
        """
        from Jotty.core.infrastructure.foundation.config_defaults import (
            DEFAULT_MODEL_ALIAS,
            LLM_TIMEOUT_SECONDS,
        )

        model = model or DEFAULT_MODEL_ALIAS
        timeout = timeout or LLM_TIMEOUT_SECONDS

        super().__init__(model=f"claude-cli/{model}", **kwargs)
        self.cli_model = model
        self.timeout = timeout
        self.provider = "claude-cli"
        self.history: List[Dict[str, Any]] = []

        # Find claude binary
        self.claude_path = shutil.which("claude")
        if not self.claude_path:
            raise LLMError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs: Any) -> List[str]:
        """
        Synchronous call interface (required by DSPy).

        Runs the async implementation in an event loop.
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # In async context - create a task
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
        Async implementation of the LLM call.
        """
        # Build the prompt text
        if prompt:
            input_text = prompt
        elif messages:
            # Extract from messages
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

        # Note: Date context is now injected centrally via ContextAwareLM wrapper
        # in unified_lm_provider.py - no need to add it here

        logger.debug(f"AsyncClaudeCLI: Calling with model {self.cli_model}")
        logger.debug(f"AsyncClaudeCLI: Input length: {len(input_text)} chars")

        # Build command (based on claude-code-api patterns)
        # Note: --output-format stream-json requires --verbose when using -p
        cmd = [
            self.claude_path,
            "-p",
            input_text,  # -p flag for prompt
            "--model",
            self.cli_model,
            "--output-format",
            "stream-json",  # JSON streaming format
            "--verbose",  # Required for stream-json with -p flag
        ]

        try:
            # Create async subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError as e:
                process.kill()
                await process.wait()
                raise LLMError(f"Claude CLI timed out after {self.timeout}s", original_error=e)

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise LLMError(f"Claude CLI error: {error_msg}")

            # Parse stream-json output
            response_text = self._parse_stream_json(stdout.decode())

            if not response_text:
                logger.warning("AsyncClaudeCLI: Empty response")
                response_text = "Error: Empty response from Claude CLI"

            logger.debug(f"AsyncClaudeCLI: Response length: {len(response_text)} chars")

            # Store in history
            self.history.append(
                {
                    "prompt": input_text[:500],
                    "response": response_text[:500],
                    "model": self.cli_model,
                }
            )

            return [response_text]

        except FileNotFoundError as e:
            raise LLMError("Claude CLI not found", original_error=e)
        except Exception as e:
            logger.error(f"AsyncClaudeCLI error: {e}")
            raise

    def _parse_stream_json(self, output: str) -> str:
        """
        Parse stream-json output format.

        Each line is a JSON object. We need to extract the actual content.
        """
        result_parts = []

        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)

                # Handle different message types
                msg_type = obj.get("type", "")

                if msg_type == "assistant":
                    # Main response content
                    content = obj.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            result_parts.append(block.get("text", ""))

                elif msg_type == "content_block_delta":
                    # Streaming delta
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        result_parts.append(delta.get("text", ""))

                elif msg_type == "result":
                    # Final result (from --output-format json)
                    if "result" in obj:
                        result_parts.append(obj["result"])

                elif msg_type == "text":
                    # Simple text output
                    result_parts.append(obj.get("text", ""))

            except json.JSONDecodeError:
                # Not JSON, use as-is
                result_parts.append(line)

        return "".join(result_parts) if result_parts else output.strip()

    def inspect_history(self, n: int = 1) -> List[Dict[str, Any]]:
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []


# Convenience function for DSPy configuration
def configure_async_claude_cli(model: str = "", **kwargs: Any) -> AsyncClaudeCLILM:
    """
    Configure DSPy with AsyncClaudeCLILM.

    Usage:
        from core.foundation.async_claude_cli_lm import configure_async_claude_cli
        import dspy

        lm = configure_async_claude_cli()
        dspy.configure(lm=lm)
    """
    lm = AsyncClaudeCLILM(model=model, **kwargs)
    dspy.configure(lm=lm)
    return lm
