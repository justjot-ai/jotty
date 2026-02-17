"""
Chat API

Simplified API for chat interactions.
Delegates to Orchestrator.chat() — the single entry point for conversational mode.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from Jotty.core.intelligence.orchestration import Orchestrator

logger = logging.getLogger(__name__)


class ChatAPI:
    """
    Simplified API for chat interactions.

    Thin facade over Orchestrator.chat().

    Usage:
        chat = ChatAPI(orchestrator)
        result = await chat.send(message="Hello", history=[...])

        # Streaming
        async for event in chat.stream(message="Hello"):
            print(event)
    """

    def __init__(
        self,
        conductor: Orchestrator,
        agent_id: Optional[str] = None,
        mode: str = "dynamic",
        auto_register_chat_assistant: bool = True,
        state_manager: Optional[Any] = None,
    ) -> None:
        """
        Initialize Chat API.

        Args:
            conductor: Jotty Orchestrator instance
            agent_id: Kept for backward compat (unused — Orchestrator handles routing)
            mode: Kept for backward compat
            auto_register_chat_assistant: Kept for backward compat
            state_manager: Kept for backward compat
        """
        self.orchestrator = conductor
        # Backward compat
        self.conductor = conductor

    async def send(
        self, message: str, history: Optional[List[Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Send a chat message.

        Args:
            message: User message
            history: Conversation history [{role, content}, ...]
            **kwargs: Additional arguments

        Returns:
            Chat response dictionary
        """
        result = await self.orchestrator.chat(message, history=history, **kwargs)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return {
            "success": getattr(result, "success", True),
            "output": getattr(result, "content", str(result)),
            "metadata": {},
            "execution_time": 0.0,
            "use_case_type": "chat",
        }

    async def stream(
        self, message: str, history: Optional[List[Any]] = None, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat response.

        Args:
            message: User message
            history: Conversation history
            **kwargs: Additional arguments

        Yields:
            Event dictionaries
        """
        async for event in self.orchestrator.chat(message, history=history, stream=True, **kwargs):
            yield event
