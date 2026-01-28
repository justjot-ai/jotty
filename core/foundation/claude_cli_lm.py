"""
Claude CLI DSPy LM Provider
============================

Part of Jotty multi-agent framework.
Uses Claude CLI as a DSPy-compatible LM backend.
Properly forwards DSPy adapter-formatted messages (system, demos, input)
to Claude CLI and returns raw LLM text for DSPy's adapter to parse.
"""

import subprocess
import json
import os
import logging
import dspy
from dspy import BaseLM
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ClaudeCLILM(BaseLM):
    """DSPy-compatible LM using Claude CLI.

    Properly handles DSPy's adapter-formatted messages by:
    - Extracting system messages and passing via --system-prompt
    - Flattening conversation history (few-shot demos) into the prompt
    - Returning raw LLM text for DSPy's adapter to parse with [[ ## ]] markers
    """

    def __init__(self, model="sonnet", **kwargs):
        """
        Initialize Claude CLI LM.

        Args:
            model: Claude model (sonnet, opus, haiku)
            **kwargs: Additional arguments
        """
        # Drop enable_skills from kwargs for backwards compatibility
        kwargs.pop('enable_skills', None)
        super().__init__(model=f"claude-cli/{model}", **kwargs)
        self.cli_model = model
        self._verify_cli_available()
        self.provider = "claude-cli"
        self.history = []

    def _verify_cli_available(self):
        """Check if claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Claude CLI available: {result.stdout.strip()}")
            else:
                raise RuntimeError("Claude CLI not working")
        except FileNotFoundError:
            raise RuntimeError("Claude CLI not found")

    def _extract_messages(self, messages: List) -> tuple:
        """
        Extract system prompt and user prompt from DSPy adapter-formatted messages.

        DSPy's ChatAdapter creates messages in this structure:
        - system: signature docstring + field descriptions + [[ ## field ## ]] format markers
        - user/assistant pairs: few-shot demo exchanges
        - user (last): current input with field values and output instructions

        Claude CLI takes a single prompt string + optional --system-prompt,
        so we flatten demos into the user prompt to preserve few-shot context.

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
                # Few-shot demo response - preserve as example output
                conversation_parts.append(f"[Example Response]\n{content}")
            elif role == 'user':
                conversation_parts.append(content)

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n".join(conversation_parts) if conversation_parts else None

        return system_prompt, user_prompt

    def _extract_response(self, raw_output: str) -> str:
        """
        Extract raw LLM response text from Claude CLI output.

        With --output-format json, CLI wraps the response in an envelope:
        {"result": "<raw LLM text>", "session_id": "...", "cost_usd": ...}

        Returns the raw LLM text as-is for DSPy's adapter to parse
        (the adapter looks for [[ ## field_name ## ]] markers).
        """
        raw_output = raw_output.strip()

        try:
            envelope = json.loads(raw_output)
            if isinstance(envelope, dict) and 'result' in envelope:
                return envelope['result']
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: return raw output as-is
        return raw_output

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        DSPy-compatible call interface.

        Properly handles DSPy adapter-formatted messages by:
        1. Extracting system message -> passed via --system-prompt flag
        2. Flattening demos + current input -> positional prompt arg
        3. Returning raw LLM text for DSPy's adapter to parse [[ ## ]] markers

        DSPy's adapter (ChatAdapter) already embeds field descriptions and
        output format instructions in the system message. The LLM responds
        with [[ ## field_name ## ]] markers that DSPy parses back into fields.
        No custom JSON schema injection is needed.
        """
        # Build messages list
        if messages is None:
            messages = []
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        if not messages:
            raise ValueError("Either prompt or messages must be provided")

        # Extract system and user parts from DSPy-formatted messages
        system_prompt, user_prompt = self._extract_messages(messages)

        if not user_prompt:
            raise ValueError("No user message found in messages")

        # Build command
        cmd = [
            "claude",
            "--model", self.cli_model,
            "--print",
            "--output-format", "json",  # CLI envelope for reliable result extraction
        ]

        # Pass DSPy's system message (field descriptions, format instructions)
        # via --system-prompt so the LLM knows about [[ ## ]] output format
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        cmd.append(user_prompt)

        # Unset ANTHROPIC_API_KEY if it's an OAuth token (doesn't work with --print)
        env = os.environ.copy()
        api_key = env.get('ANTHROPIC_API_KEY', '')
        if api_key.startswith('sk-ant-oat'):
            env.pop('ANTHROPIC_API_KEY', None)

        timeout = kwargs.get('timeout', 60)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise TimeoutError(f"Claude CLI timed out: {error_msg}")
            raise RuntimeError(f"Claude CLI error: {error_msg}")

        # Extract raw LLM response from CLI JSON envelope
        # Return as-is — DSPy's adapter will parse the [[ ## ]] markers
        response_text = self._extract_response(result.stdout)

        # Store in history
        self.history.append({
            "prompt": user_prompt,
            "system": system_prompt,
            "response": response_text,
            "kwargs": kwargs
        })

        # Return in DSPy format (list of response strings)
        return [response_text]

    def inspect_history(self, n=1):
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []
