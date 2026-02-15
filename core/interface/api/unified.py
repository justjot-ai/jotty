"""
Unified API for Jotty

Single entry point for all use cases (chat, workflow).
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from Jotty.core.infrastructure.foundation.agent_config import AgentConfig
from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
from Jotty.core.intelligence.orchestration import Orchestrator
from Jotty.core.modes.use_cases import ChatUseCase, UseCaseConfig, WorkflowUseCase
from Jotty.core.modes.use_cases.base import BaseUseCase, UseCaseType

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
        api = JottyAPI(agents=[...], config=SwarmConfig(...))

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
        config: Optional[SwarmConfig] = None,
        conductor: Optional[Orchestrator] = None,
    ) -> None:
        """
        Initialize Jotty API.

        Args:
            agents: List of agent configurations
            config: Jotty configuration
            conductor: Optional pre-configured conductor (if None, creates one)
        """
        if conductor is None:
            from Jotty.core.jotty import create_swarm_manager

            self.conductor = create_swarm_manager(agents, config)
        else:
            self.conductor = conductor

        self.config = config or SwarmConfig()
        self.agents = agents

        # Initialize use cases
        self._chat_use_case: Optional[ChatUseCase] = None
        self._workflow_use_case: Optional[WorkflowUseCase] = None

    def _create_use_case(
        self, use_case_class: Type[BaseUseCase], use_case_type: UseCaseType, **kwargs: Any
    ) -> BaseUseCase:
        """
        DRY factory for creating use cases with optional overrides.

        Eliminates duplication in chat_execute, chat_stream, workflow_execute, workflow_stream.

        Args:
            use_case_class: ChatUseCase or WorkflowUseCase
            use_case_type: UseCaseType enum
            **kwargs: Overrides like agent_id, mode, agent_order

        Returns:
            Configured use case instance
        """
        return use_case_class(
            conductor=self.conductor, config=UseCaseConfig(use_case_type=use_case_type), **kwargs
        )

    @property
    def chat(self) -> ChatUseCase:
        """Get chat use case (lazy initialization)."""
        if self._chat_use_case is None:
            from Jotty.core.modes.use_cases.base import UseCaseType

            self._chat_use_case = ChatUseCase(
                conductor=self.conductor, config=UseCaseConfig(use_case_type=UseCaseType.CHAT)
            )
        return self._chat_use_case

    @property
    def workflow(self) -> WorkflowUseCase:
        """Get workflow use case (lazy initialization)."""
        if self._workflow_use_case is None:
            from Jotty.core.modes.use_cases.base import UseCaseType

            self._workflow_use_case = WorkflowUseCase(
                conductor=self.conductor, config=UseCaseConfig(use_case_type=UseCaseType.WORKFLOW)
            )
        return self._workflow_use_case

    async def chat_execute(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
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
        # DRY: Use factory if agent_id specified, otherwise use cached property
        chat = (
            self._create_use_case(ChatUseCase, UseCaseType.CHAT, agent_id=agent_id)
            if agent_id
            else self.chat
        )

        result = await chat.execute(goal=message, history=history, **kwargs)

        return result.to_dict()

    async def chat_stream(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
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
        # DRY: Use factory if agent_id specified
        chat = (
            self._create_use_case(ChatUseCase, UseCaseType.CHAT, agent_id=agent_id)
            if agent_id
            else self.chat
        )

        async for event in chat.stream(goal=message, history=history, **kwargs):
            yield event

    async def workflow_execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None,
        **kwargs: Any,
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
        # DRY: Use factory if mode/agent_order specified
        workflow = (
            self._create_use_case(
                WorkflowUseCase, UseCaseType.WORKFLOW, mode=mode, agent_order=agent_order
            )
            if (mode != "dynamic" or agent_order)
            else self.workflow
        )

        result = await workflow.execute(goal=goal, context=context, **kwargs)

        return result.to_dict()

    async def workflow_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None,
        **kwargs: Any,
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
        # DRY: Use factory if mode/agent_order specified
        workflow = (
            self._create_use_case(
                WorkflowUseCase, UseCaseType.WORKFLOW, mode=mode, agent_order=agent_order
            )
            if (mode != "dynamic" or agent_order)
            else self.workflow
        )

        async for event in workflow.stream(goal=goal, context=context, **kwargs):
            yield event

    async def workflow_enqueue(
        self, goal: str, context: Optional[Dict[str, Any]] = None, priority: int = 3, **kwargs: Any
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
        return await self.workflow.enqueue(goal=goal, context=context, priority=priority, **kwargs)
