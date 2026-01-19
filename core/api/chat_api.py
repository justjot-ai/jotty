"""
Chat API

Simplified API for chat interactions.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import logging

from ..use_cases.chat import ChatUseCase, ChatMessage
from ..orchestration.conductor import Conductor
from ..foundation.data_structures import JottyConfig
from ..foundation.agent_config import AgentSpec
from ..agents.chat_assistant import create_chat_assistant

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
        mode: str = "dynamic",
        auto_register_chat_assistant: bool = True,
        state_manager: Optional[Any] = None
    ):
        """
        Initialize Chat API.

        Args:
            conductor: Jotty Conductor instance
            agent_id: Specific agent ID for single-agent chat (optional)
            mode: Orchestration mode ("static" or "dynamic")
            auto_register_chat_assistant: Auto-register ChatAssistant if agent_id="ChatAssistant" (default: True)
            state_manager: Optional state manager for ChatAssistant (for task queries)
        """
        self.conductor = conductor
        self.state_manager = state_manager

        # Auto-register ChatAssistant agent if requested and agent_id is "ChatAssistant"
        if auto_register_chat_assistant and agent_id == "ChatAssistant":
            self._ensure_chat_assistant_registered()

        self.chat_use_case = ChatUseCase(
            conductor=conductor,
            agent_id=agent_id,
            mode=mode
        )

    def _ensure_chat_assistant_registered(self):
        """
        Ensure ChatAssistant is registered with conductor.

        This makes Jotty "batteries included" - clients don't need to manually
        register the ChatAssistant agent.
        """
        # Check if ChatAssistant already exists
        if hasattr(self.conductor, 'actors') and isinstance(self.conductor.actors, dict):
            if "ChatAssistant" in self.conductor.actors:
                logger.debug("ChatAssistant already registered")
                return

        # Create and register ChatAssistant
        try:
            logger.info("🤖 Auto-registering ChatAssistant agent (world-class Jotty defaults)")
            chat_agent = create_chat_assistant(state_manager=self.state_manager)

            # Wrap in AgentSpec (Jotty's standard agent configuration)
            agent_spec = AgentSpec(
                name="ChatAssistant",
                agent=chat_agent,
                # Disable architect/auditor for chat agent (just execute, don't plan/validate)
                enable_architect=False,
                enable_auditor=False,
                # Chat assistant provides conversational capabilities
                capabilities=["chat", "task_queries", "system_status", "help"],
                # Mark as non-critical (optional agent)
                is_critical=False
            )

            # Register with conductor's actor registry
            if hasattr(self.conductor, 'actors') and isinstance(self.conductor.actors, dict):
                self.conductor.actors["ChatAssistant"] = agent_spec

                # Initialize local_memories for ChatAssistant (required by Conductor)
                if hasattr(self.conductor, 'local_memories') and isinstance(self.conductor.local_memories, dict):
                    try:
                        from ..memory.hierarchical_memory import HierarchicalMemory
                        self.conductor.local_memories["ChatAssistant"] = HierarchicalMemory(
                            config=self.conductor.config,
                            agent_name="ChatAssistant"
                        )
                        logger.debug("✅ ChatAssistant local_memories initialized with HierarchicalMemory")
                    except ImportError as e:
                        # Fallback to empty object with memories attribute
                        logger.debug(f"⚠️  HierarchicalMemory import failed ({e}), using empty fallback")
                        class EmptyMemory:
                            def __init__(self):
                                self.memories = {}
                        self.conductor.local_memories["ChatAssistant"] = EmptyMemory()
                        logger.debug("✅ ChatAssistant local_memories initialized with empty fallback")

                logger.info("✅ ChatAssistant registered successfully")
            else:
                logger.warning("⚠️  Conductor doesn't have 'actors' dict - ChatAssistant not registered")

        except Exception as e:
            logger.error(f"❌ Failed to auto-register ChatAssistant: {e}")
            # Don't fail - just log the error
            pass
    
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
