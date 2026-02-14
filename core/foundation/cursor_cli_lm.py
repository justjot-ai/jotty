"""
Cursor CLI DSPy LM Provider
============================

Part of Jotty multi-agent framework.
Uses Cursor Agent CLI as a DSPy-compatible LM backend.
Properly forwards DSPy adapter-formatted messages (system, demos, input)
to Cursor CLI and returns raw LLM text for DSPy's adapter to parse.
"""

import subprocess
import json
import os
import dspy
from dspy import BaseLM
from typing import Dict, Any, Optional, List

from Jotty.core.foundation.config_defaults import LLM_TIMEOUT_SECONDS
from Jotty.core.foundation.exceptions import InputValidationError


class CursorCLILM(BaseLM):
    """DSPy-compatible LM using Cursor CLI.

    Properly handles DSPy's adapter-formatted messages by:
    - Extracting system messages and passing via --system-prompt
    - Flattening conversation history (few-shot demos) into the prompt
    - Returning raw LLM text for DSPy's adapter to parse with [[ ## ]] markers
    """

    def __init__(self, model: Any = 'sonnet-4', **kwargs: Any) -> None:
        super().__init__(model=f"cursor-cli/{model}", **kwargs)
        self.cli_model = model
        self.history = []

    def _extract_messages(self, messages: List) -> tuple:
        """
        Extract system prompt and user prompt from DSPy adapter-formatted messages.

        DSPy's ChatAdapter creates messages with system (format instructions),
        user/assistant pairs (few-shot demos), and the current user input.
        Cursor CLI takes a single prompt + optional --system-prompt, so we
        flatten demos into the user prompt to preserve few-shot context.

        Returns:
            (system_prompt, user_prompt) tuple
        """
        system_parts = []
        conversation_parts = []

        for msg in messages:
            if isinstance(msg, str):
                conversation_parts.append(msg)
                continue

            if not isinstance(msg, dict):
                continue

            role = msg.get('role', '')
            content = msg.get('content', '')

            if not content:
                continue

            if role == 'system':
                system_parts.append(content)
            elif role == 'assistant':
                conversation_parts.append(f"[Example Response]\n{content}")
            elif role == 'user':
                conversation_parts.append(content)

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n".join(conversation_parts) if conversation_parts else None

        return system_prompt, user_prompt

    def _extract_response(self, raw_output: str) -> str:
        """
        Extract raw LLM response text from Cursor CLI output.

        With --output-format json, CLI wraps the response in an envelope.
        Returns the raw LLM text for DSPy's adapter to parse.
        """
        raw_output = raw_output.strip()

        try:
            response_data = json.loads(raw_output)
            if isinstance(response_data, dict):
                # Try common envelope fields
                for key in ('result', 'response', 'message', 'content'):
                    if key in response_data and response_data[key]:
                        return response_data[key]
        except (json.JSONDecodeError, TypeError):
            pass

        return raw_output

    def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        """
        DSPy-compatible call interface.

        Properly handles DSPy adapter-formatted messages by:
        1. Extracting system message -> passed via --system-prompt flag
        2. Flattening demos + current input -> positional prompt arg
        3. Returning raw LLM text for DSPy's adapter to parse [[ ## ]] markers
        """
        if messages is None:
            messages = []
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        if not messages:
            raise InputValidationError("Either prompt or messages must be provided")

        system_prompt, user_prompt = self._extract_messages(messages)

        if not user_prompt:
            raise InputValidationError("No user message found in messages")

        cmd = [
            "cursor-agent",
            "--model", self.cli_model,
            "--print",
            "--output-format", "json",  # CLI envelope for reliable result extraction
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        cmd.append(user_prompt)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=kwargs.get('timeout', LLM_TIMEOUT_SECONDS)
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise RuntimeError(f"Cursor CLI error: {error_msg}")

        # Extract raw LLM response from CLI envelope
        response_text = self._extract_response(result.stdout)

        self.history.append({
            "prompt": user_prompt,
            "system": system_prompt,
            "response": response_text,
            "kwargs": kwargs
        })

        return [response_text]

    def inspect_history(self, n: Any = 1) -> Any:
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []
