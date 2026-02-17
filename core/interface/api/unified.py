"""
Unified API for Jotty

Thin facade over the Orchestrator's two-method API: run() and chat().
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from Jotty.core.infrastructure.foundation.agent_config import (
    AgentConfig,  # type: ignore[import-not-found, import]
)
from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
from Jotty.core.intelligence.orchestration import Orchestrator

logger = logging.getLogger(__name__)


class JottyAPI:
    """
    Unified API for all Jotty interactions.

    Wraps the Orchestrator's two-method public API:
    - run(goal)  → Do this thing (auto-routes, swarm, agent, pipeline)
    - chat(message) → Let's talk (conversational, tools, streaming)

    Usage:
        api = JottyAPI(agents=[...])

        # Chat
        result = await api.chat(message="Hello")

        # Run (auto-detect)
        result = await api.run(goal="Research AI trends")

        # Streaming
        async for event in api.chat(message="Hello", stream=True):
            print(event)
    """

    def __init__(
        self,
        agents: Optional[List[AgentConfig]] = None,
        config: Optional[SwarmConfig] = None,
        conductor: Optional[Orchestrator] = None,
    ) -> None:
        """
        Initialize Jotty API.

        Args:
            agents: List of agent configurations
            config: Jotty configuration
            conductor: Optional pre-configured Orchestrator
        """
        if conductor is not None:
            self.orchestrator = conductor
        elif agents is not None:
            from Jotty.core.jotty import create_swarm_manager  # type: ignore[import]

            self.orchestrator = create_swarm_manager(agents, config)
        else:
            self.orchestrator = Orchestrator(config=config)

        self.config = config or SwarmConfig()

        # Backward compat
        self.conductor = self.orchestrator

    async def run(
        self,
        goal: str,
        *,
        stream: bool = False,
        stages: Optional[List[Dict[str, Any]]] = None,
        swarm: Optional[Any] = None,
        agent: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a task. Delegates to Orchestrator.run().

        Args:
            goal: Task goal (natural language)
            stream: If True, returns AsyncIterator
            stages: Pipeline stage definitions
            swarm: Specific swarm template
            agent: Specific agent
            **kwargs: Additional arguments

        Returns:
            ExecutionResult (or AsyncIterator if stream=True)
        """
        return await self.orchestrator.run(
            goal, stream=stream, stages=stages, swarm=swarm, agent=agent, **kwargs
        )

    async def chat(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Conversational mode. Delegates to Orchestrator.chat().

        Args:
            message: User message
            history: Conversation history [{role, content}, ...]
            stream: If True, returns AsyncIterator
            **kwargs: Additional arguments (provider, model, etc.)

        Returns:
            LLMExecutionResult (or AsyncIterator if stream=True)
        """
        return await self.orchestrator.chat(message, history=history, stream=stream, **kwargs)

    # =========================================================================
    # Backward-compat aliases
    # =========================================================================

    async def chat_execute(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Backward-compat. Use chat() instead."""
        result = await self.orchestrator.chat(message, history=history, **kwargs)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return {
            "success": getattr(result, "success", True),
            "output": str(result),
            "metadata": {},
            "execution_time": 0.0,
            "use_case_type": "chat",
        }

    async def chat_stream(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Backward-compat. Use chat(stream=True) instead."""
        async for event in self.orchestrator.chat(message, history=history, stream=True, **kwargs):
            yield event

    async def workflow_execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Backward-compat. Use run() instead."""
        result = await self.orchestrator.run(goal, **kwargs)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return {
            "success": getattr(result, "success", True),
            "output": str(result),
            "metadata": {},
            "execution_time": 0.0,
            "use_case_type": "workflow",
        }

    async def workflow_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Backward-compat. Use run(stream=True) instead."""
        async for event in self.orchestrator.run(goal, stream=True, **kwargs):
            yield event

    async def workflow_enqueue(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs: Any,
    ) -> str:
        """Backward-compat. Use run() for synchronous execution."""
        result = await self.orchestrator.run(goal, **kwargs)
        return str(id(result))
