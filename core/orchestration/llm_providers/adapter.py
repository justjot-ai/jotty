from typing import Any
"""
JottyClaudeProvider adapter for ChatExecutor's provider interface.
"""

import logging

logger = logging.getLogger(__name__)


class JottyClaudeProviderAdapter:
    """Adapter to use JottyClaudeProvider with ChatExecutor's provider interface."""

    def __init__(self, jotty_provider: Any) -> None:
        self.jotty_provider = jotty_provider
        self._lm = None

    def _get_lm(self) -> Any:
        """Get or create DSPy LM from JottyClaudeProvider."""
        if self._lm is None:
            self._lm = self.jotty_provider.configure_dspy()
        return self._lm

    def call(self, messages: list, tools: list = None, **kwargs: Any) -> dict:
        """Make LLM call using JottyClaudeProvider."""
        # Convert messages to prompt for DSPy
        lm = self._get_lm()

        # Format messages for the LM
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.insert(0, f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)

        # Add tool instructions if provided (handles both Claude and OpenAI format)
        if tools:
            tool_desc_parts = []
            for t in tools:
                # Handle Claude format (name at top level) or OpenAI format (function.name)
                if 'function' in t:
                    name = t['function'].get('name', 'unknown')
                    desc = t['function'].get('description', '')
                else:
                    name = t.get('name', 'unknown')
                    desc = t.get('description', '')
                tool_desc_parts.append(f"- {name}: {desc}")
            tool_desc = "\n".join(tool_desc_parts)
            prompt += f"\n\nAvailable tools:\n{tool_desc}\n\nRespond with tool calls in JSON format if needed."

        # Call LM
        try:
            response = lm(prompt=prompt)
            if isinstance(response, list) and response:
                content = response[0] if isinstance(response[0], str) else str(response[0])
            else:
                content = str(response)

            return {
                'content': content,
                'tool_calls': [],  # JottyClaudeProvider doesn't support native tool calls yet
                'stop_reason': 'end_turn'
            }
        except Exception as e:
            logger.error(f"JottyClaudeProvider call failed: {e}")
            raise
