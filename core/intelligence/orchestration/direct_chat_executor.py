"""
Direct Chat Executor - Simple, Fast Chat
=========================================

Lightweight executor for simple queries that don't need:
- Task analysis
- Tool detection
- Multi-step planning
- Complex validation

Used for ValidationMode.DIRECT queries like:
- Greetings ("hi", "hello")
- Simple questions ("what is X?")
- Quick lookups ("list 3 examples")

Just makes a single LLM call and returns the response.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DirectChatResult:
    """Result from direct chat execution."""

    success: bool
    content: str
    error: Optional[str] = None
    tokens_used: int = 0
    model: str = ""


class DirectChatExecutor:
    """
    Simple executor for direct chat queries.

    No analysis, no tools, no complexity - just a single LLM call.
    """

    def __init__(self, model: str = "claude-haiku-3-5-20241022"):
        """
        Initialize direct chat executor.

        Args:
            model: LLM model to use (default: Haiku for speed/cost)
        """
        self.model = model
        self._provider = None

    async def execute(self, message: str) -> DirectChatResult:
        """
        Execute a simple chat query.

        Args:
            message: User message

        Returns:
            DirectChatResult with LLM response
        """
        try:
            # Get LLM provider
            provider = self._get_provider()

            # Simple system prompt
            system_prompt = (
                "You are a helpful AI assistant. Provide concise, accurate responses. "
                "For greetings, be friendly and brief. For questions, give clear answers."
            )

            # Make single LLM call
            response = await provider.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                model=self.model,
                max_tokens=500,  # Keep it short for simple queries
                temperature=0.7,
            )

            # Extract content
            content = response.get("content", "")
            if isinstance(content, list) and content:
                content = content[0].get("text", str(content))

            tokens = response.get("usage", {}).get("total_tokens", 0)

            logger.info(f"Direct chat completed: {len(content)} chars, {tokens} tokens")

            return DirectChatResult(
                success=True,
                content=content,
                tokens_used=tokens,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"Direct chat error: {e}", exc_info=True)
            return DirectChatResult(
                success=False,
                content="",
                error=str(e),
            )

    def _get_provider(self) -> Any:
        """Get LLM provider."""
        if self._provider is None:
            # Use the Jotty Claude provider
            from Jotty.core.infrastructure.foundation.jotty_claude_provider import (
                JottyClaudeProvider,
            )
            self._provider = JottyClaudeProvider()
        return self._provider
