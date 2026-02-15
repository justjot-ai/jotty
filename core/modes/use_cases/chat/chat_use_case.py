"""
Chat Use Case

Main entry point for chat interactions.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..base import BaseUseCase, UseCaseConfig, UseCaseResult, UseCaseType
from .chat_context import ChatContext, ChatMessage
from .chat_executor import ChatExecutor
from .chat_orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)


class ChatUseCase(BaseUseCase):
    """
    Chat use case for conversational interactions.

    Usage:
        chat = ChatUseCase(conductor, agent_id="MyAgent")
        result = await chat.execute(message="Hello", history=[...])

        # Streaming
        async for event in chat.stream(message="Hello"):
            print(event)
    """

    def __init__(
        self,
        conductor: Any,
        agent_id: Optional[str] = None,
        mode: str = "dynamic",
        config: Optional[UseCaseConfig] = None,
        context: Optional[ChatContext] = None,
    ) -> None:
        """
        Initialize chat use case.

        Args:
            conductor: Jotty Conductor instance
            agent_id: Specific agent ID for single-agent chat (optional)
            mode: Orchestration mode ("static" or "dynamic")
            config: Use case configuration
            context: Chat context manager (optional)
        """
        super().__init__(conductor, config)

        # Create components
        self.orchestrator = ChatOrchestrator(conductor=conductor, agent_id=agent_id, mode=mode)
        self.executor = ChatExecutor(
            conductor=conductor, orchestrator=self.orchestrator, context=context
        )
        self.agent_id = agent_id
        self.mode = mode

    def _get_use_case_type(self) -> UseCaseType:
        """Return chat use case type."""
        return UseCaseType.CHAT

    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> UseCaseResult:
        """
        Execute chat interaction synchronously.

        Args:
            goal: User message
            context: Additional context
            history: Conversation history
            **kwargs: Additional arguments

        Returns:
            UseCaseResult with chat response
        """
        # DRY: Use base class error handling wrapper
        return await self._execute_with_error_handling(
            self.executor.execute, message=goal, history=history, context=context
        )

    def _extract_output(self, result: Dict[str, Any]) -> Any:
        """Extract chat message from result."""
        return result.get("message", "")

    def _extract_metadata(self, result: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Extract chat metadata."""
        metadata = {
            "agent": result.get("agent"),
            "execution_time": result.get("execution_time", execution_time),
        }
        metadata.update(result.get("metadata", {}))
        return metadata

    async def stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute chat interaction with streaming.

        Args:
            goal: User message
            context: Additional context
            history: Conversation history
            **kwargs: Additional arguments

        Yields:
            Event dictionaries
        """
        async for event in self.executor.stream(message=goal, history=history, context=context):
            yield event
