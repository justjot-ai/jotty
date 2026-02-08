"""
Unified API for Jotty

Single entry point for all use cases (chat, workflow).
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import logging

from ..foundation.agent_config import AgentConfig
from ..use_cases import ChatUseCase, WorkflowUseCase, UseCaseConfig
from ..orchestration import SwarmManager
from ..foundation.data_structures import JottyConfig

logger = logging.getLogger(__name__)


class JottyAPI:
    """
    Unified API for all Jotty use cases.
    
    Provides a single entry point for:
    - Chat interactions
    - Workflow execution
    - All execution modes (sync, async, streaming)
    
    Usage:
        # Create API
        api = JottyAPI(agents=[...], config=JottyConfig(...))
        
        # Chat
        result = await api.chat(message="Hello", history=[...])
        
        # Workflow
        result = await api.workflow(goal="...", context={...})
        
        # Streaming
        async for event in api.chat_stream(message="Hello"):
            print(event)
    """
    
    def __init__(
        self,
        agents: List[AgentConfig],
        config: Optional[JottyConfig] = None,
        conductor: Optional[SwarmManager] = None
    ):
        """
        Initialize Jotty API.
        
        Args:
            agents: List of agent configurations
            config: Jotty configuration
            conductor: Optional pre-configured conductor (if None, creates one)
        """
        if conductor is None:
            from ..jotty import create_swarm_manager
            self.conductor = create_swarm_manager(agents, config)
        else:
            self.conductor = conductor
        
        self.config = config or JottyConfig()
        self.agents = agents
        
        # Initialize use cases
        self._chat_use_case: Optional[ChatUseCase] = None
        self._workflow_use_case: Optional[WorkflowUseCase] = None
    
    @property
    def chat(self) -> ChatUseCase:
        """Get chat use case (lazy initialization)."""
        if self._chat_use_case is None:
            from ..use_cases.base import UseCaseType
            self._chat_use_case = ChatUseCase(
                conductor=self.conductor,
                config=UseCaseConfig(
                    use_case_type=UseCaseType.CHAT
                )
            )
        return self._chat_use_case
    
    @property
    def workflow(self) -> WorkflowUseCase:
        """Get workflow use case (lazy initialization)."""
        if self._workflow_use_case is None:
            from ..use_cases.base import UseCaseType
            self._workflow_use_case = WorkflowUseCase(
                conductor=self.conductor,
                config=UseCaseConfig(
                    use_case_type=UseCaseType.WORKFLOW
                )
            )
        return self._workflow_use_case
    
    async def chat_execute(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute chat interaction synchronously.
        
        Args:
            message: User message
            history: Conversation history
            agent_id: Specific agent ID (optional)
            **kwargs: Additional arguments
            
        Returns:
            Chat result dictionary
        """
        # Create chat use case with specific agent if provided
        if agent_id:
            chat = ChatUseCase(
                conductor=self.conductor,
                agent_id=agent_id,
                config=UseCaseConfig(
                    use_case_type=ChatUseCase._get_use_case_type(ChatUseCase)
                )
            )
        else:
            chat = self.chat
        
        result = await chat.execute(
            goal=message,
            history=history,
            **kwargs
        )
        
        return result.to_dict()
    
    async def chat_stream(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute chat interaction with streaming.
        
        Args:
            message: User message
            history: Conversation history
            agent_id: Specific agent ID (optional)
            **kwargs: Additional arguments
            
        Yields:
            Event dictionaries
        """
        # Create chat use case with specific agent if provided
        if agent_id:
            from ..use_cases.base import UseCaseType
            chat = ChatUseCase(
                conductor=self.conductor,
                agent_id=agent_id,
                config=UseCaseConfig(
                    use_case_type=UseCaseType.CHAT
                )
            )
        else:
            chat = self.chat
        
        async for event in chat.stream(
            goal=message,
            history=history,
            **kwargs
        ):
            yield event
    
    async def workflow_execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute workflow synchronously.
        
        Args:
            goal: Workflow goal
            context: Additional context
            mode: Orchestration mode ("static" or "dynamic")
            agent_order: Required for static mode
            **kwargs: Additional arguments
            
        Returns:
            Workflow result dictionary
        """
        # Create workflow use case with specific mode if provided
        if mode != "dynamic" or agent_order:
            from ..use_cases.base import UseCaseType
            workflow = WorkflowUseCase(
                conductor=self.conductor,
                mode=mode,
                agent_order=agent_order,
                config=UseCaseConfig(
                    use_case_type=UseCaseType.WORKFLOW
                )
            )
        else:
            workflow = self.workflow
        
        result = await workflow.execute(
            goal=goal,
            context=context,
            **kwargs
        )
        
        return result.to_dict()
    
    async def workflow_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute workflow with streaming.
        
        Args:
            goal: Workflow goal
            context: Additional context
            mode: Orchestration mode ("static" or "dynamic")
            agent_order: Required for static mode
            **kwargs: Additional arguments
            
        Yields:
            Event dictionaries
        """
        # Create workflow use case with specific mode if provided
        if mode != "dynamic" or agent_order:
            from ..use_cases.base import UseCaseType
            workflow = WorkflowUseCase(
                conductor=self.conductor,
                mode=mode,
                agent_order=agent_order,
                config=UseCaseConfig(
                    use_case_type=UseCaseType.WORKFLOW
                )
            )
        else:
            workflow = self.workflow
        
        async for event in workflow.stream(
            goal=goal,
            context=context,
            **kwargs
        ):
            yield event
    
    async def workflow_enqueue(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs
    ) -> str:
        """
        Enqueue workflow task for asynchronous execution.
        
        Args:
            goal: Workflow goal
            context: Additional context
            priority: Task priority (1-5)
            **kwargs: Additional arguments
            
        Returns:
            Task ID
        """
        return await self.workflow.enqueue(
            goal=goal,
            context=context,
            priority=priority,
            **kwargs
        )
