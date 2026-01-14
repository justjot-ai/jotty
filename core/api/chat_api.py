"""
Chat API

Simplified API for chat interactions.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import logging

from ..use_cases.chat import ChatUseCase, ChatMessage
from ..orchestration.conductor import Conductor
from ..foundation.data_structures import JottyConfig

logger = logging.getLogger(__name__)


class ChatAPI:
    """
    Simplified API for chat interactions.
    
    Usage:
        chat = ChatAPI(conductor, agent_id="MyAgent")
        result = await chat.send(message="Hello", history=[...])
        
        # Streaming
        async for event in chat.stream(message="Hello"):
            print(event)
    """
    
    def __init__(
        self,
        conductor: Conductor,
        agent_id: Optional[str] = None,
        mode: str = "dynamic"
    ):
        """
        Initialize Chat API.
        
        Args:
            conductor: Jotty Conductor instance
            agent_id: Specific agent ID for single-agent chat (optional)
            mode: Orchestration mode ("static" or "dynamic")
        """
        self.conductor = conductor
        self.chat_use_case = ChatUseCase(
            conductor=conductor,
            agent_id=agent_id,
            mode=mode
        )
    
    async def send(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message.
        
        Args:
            message: User message
            history: Conversation history
            **kwargs: Additional arguments
            
        Returns:
            Chat response dictionary
        """
        result = await self.chat_use_case.execute(
            goal=message,
            history=history,
            **kwargs
        )
        return result.to_dict()
    
    async def stream(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        **kwargs
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
        async for event in self.chat_use_case.stream(
            goal=message,
            history=history,
            **kwargs
        ):
            yield event
