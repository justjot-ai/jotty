"""
Chat Use Case

Main entry point for chat interactions.
"""

from typing import Dict, Any, Optional, List, AsyncIterator
import logging

from ..base import BaseUseCase, UseCaseType, UseCaseResult, UseCaseConfig
from .chat_executor import ChatExecutor
from .chat_orchestrator import ChatOrchestrator
from .chat_context import ChatContext, ChatMessage

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
    
    def __init__(self, conductor: Any, agent_id: Optional[str] = None, mode: str = 'dynamic', config: Optional[UseCaseConfig] = None, context: Optional[ChatContext] = None) -> None:
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
        self.orchestrator = ChatOrchestrator(
            conductor=conductor,
            agent_id=agent_id,
            mode=mode
        )
        self.executor = ChatExecutor(
            conductor=conductor,
            orchestrator=self.orchestrator,
            context=context
        )
        self.agent_id = agent_id
        self.mode = mode
    
    def _get_use_case_type(self) -> UseCaseType:
        """Return chat use case type."""
        return UseCaseType.CHAT
    
    async def execute(self, goal: str, context: Optional[Dict[str, Any]] = None, history: Optional[List[ChatMessage]] = None, **kwargs: Any) -> UseCaseResult:
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
        import time
        start_time = time.time()
        
        try:
            result = await self.executor.execute(
                message=goal,
                history=history,
                context=context
            )
            
            execution_time = time.time() - start_time
            
            return self._create_result(
                success=result.get("success", False),
                output=result.get("message", ""),
                metadata={
                    "agent": result.get("agent"),
                    "execution_time": result.get("execution_time", execution_time),
                    **result.get("metadata", {})
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Chat execution failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            
            return self._create_result(
                success=False,
                output=f"Error: {str(e)}",
                metadata={"error": str(e)},
                execution_time=execution_time
            )
    
    async def stream(self, goal: str, context: Optional[Dict[str, Any]] = None, history: Optional[List[ChatMessage]] = None, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
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
        async for event in self.executor.stream(
            message=goal,
            history=history,
            context=context
        ):
            yield event
