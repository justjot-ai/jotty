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
from typing import Any, Optional

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
            # Get global LLM instance
            lm = self._get_provider()

            # Simple system prompt
            system_prompt = (
                "You are a helpful AI assistant. Provide concise, accurate responses. "
                "For greetings, be friendly and brief. For questions, give clear answers."
            )

            # Build prompt for DSPy
            prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"

            # Make single LLM call using DSPy format
            response = lm(prompt=prompt, temperature=0.7, max_tokens=500)

            # Extract content from DSPy response
            if isinstance(response, list) and response:
                content = response[0]
            elif isinstance(response, dict):
                content = response.get("choices", [{}])[0].get("text", str(response))
            else:
                content = str(response)

            # Estimate tokens (DSPy doesn't always return usage)
            tokens = len(prompt.split()) + len(content.split())

            logger.info(f"Direct chat completed: {len(content)} chars, ~{tokens} tokens")

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
        """Get global LLM instance (shared across all components)."""
        if self._provider is None:
            from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm

            self._provider = get_global_lm(provider="anthropic", model=self.model)
            logger.info(f"DirectChatExecutor: Using global LLM (model={self.model})")
        return self._provider
