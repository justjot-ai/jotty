"""
Workflow API

Simplified API for workflow execution.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import logging

from Jotty.core.use_cases.workflow import WorkflowUseCase
from Jotty.core.orchestration import Orchestrator
from Jotty.core.foundation.data_structures import SwarmLearningConfig

logger = logging.getLogger(__name__)


class WorkflowAPI:
    """
    Simplified API for workflow execution.
    
    Usage:
        workflow = WorkflowAPI(conductor, mode="dynamic")
        result = await workflow.execute(goal="...", context={...})
        
        # Streaming
        async for event in workflow.stream(goal="..."):
            print(event)
        
        # Async
        task_id = await workflow.enqueue(goal="...", priority=5)
    """
    
    def __init__(self, conductor: Orchestrator, mode: str = 'dynamic', agent_order: Optional[List[str]] = None) -> None:
        """
        Initialize Workflow API.
        
        Args:
            conductor: Jotty Orchestrator instance
            mode: Orchestration mode ("static" or "dynamic")
            agent_order: Required for static mode
        """
        self.conductor = conductor
        self.workflow_use_case = WorkflowUseCase(
            conductor=conductor,
            mode=mode,
            agent_order=agent_order
        )
    
    async def execute(self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute workflow synchronously.
        
        Args:
            goal: Workflow goal
            context: Additional context
            **kwargs: Additional arguments
            
        Returns:
            Workflow result dictionary
        """
        result = await self.workflow_use_case.execute(
            goal=goal,
            context=context,
            **kwargs
        )
        return result.to_dict()
    
    async def stream(self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute workflow with streaming.
        
        Args:
            goal: Workflow goal
            context: Additional context
            **kwargs: Additional arguments
            
        Yields:
            Event dictionaries
        """
        async for event in self.workflow_use_case.stream(
            goal=goal,
            context=context,
            **kwargs
        ):
            yield event
    
    async def enqueue(self, goal: str, context: Optional[Dict[str, Any]] = None, priority: int = 3, **kwargs: Any) -> str:
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
        return await self.workflow_use_case.enqueue(
            goal=goal,
            context=context,
            priority=priority,
            **kwargs
        )
