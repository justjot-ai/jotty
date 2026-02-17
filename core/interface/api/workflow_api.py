"""
Workflow API

Simplified API for workflow execution.
Delegates to Orchestrator.run() — workflows are just run() with optional stages.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from Jotty.core.intelligence.orchestration import Orchestrator

logger = logging.getLogger(__name__)


class WorkflowAPI:
    """
    Simplified API for workflow execution.

    Thin facade over Orchestrator.run(). Workflows are just run() calls —
    optionally with stages for pipeline mode.

    Usage:
        workflow = WorkflowAPI(orchestrator)
        result = await workflow.execute(goal="Research AI trends")

        # Streaming
        async for event in workflow.stream(goal="Research AI"):
            print(event)
    """

    def __init__(
        self,
        conductor: Orchestrator,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize Workflow API.

        Args:
            conductor: Jotty Orchestrator instance
            mode: Kept for backward compat
            agent_order: Kept for backward compat
        """
        self.orchestrator = conductor
        # Backward compat
        self.conductor = conductor

    async def execute(
        self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute workflow synchronously. Delegates to Orchestrator.run().

        Args:
            goal: Workflow goal
            context: Additional context
            **kwargs: Additional arguments

        Returns:
            Workflow result dictionary
        """
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

    async def stream(
        self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute workflow with streaming. Delegates to Orchestrator.run(stream=True).

        Args:
            goal: Workflow goal
            context: Additional context
            **kwargs: Additional arguments

        Yields:
            Event dictionaries
        """
        async for event in self.orchestrator.run(goal, stream=True, **kwargs):
            yield event

    async def enqueue(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs: Any,
    ) -> str:
        """Backward-compat. Use run() for synchronous execution."""
        result = await self.orchestrator.run(goal, **kwargs)
        return str(id(result))
