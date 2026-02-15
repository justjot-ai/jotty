"""
Workflow Use Case

Main entry point for workflow interactions.
"""

from typing import Dict, Any, Optional, AsyncIterator
import logging
import time

from ..base import BaseUseCase, UseCaseType, UseCaseResult, UseCaseConfig
from .workflow_executor import WorkflowExecutor
from .workflow_orchestrator import WorkflowOrchestrator
from .workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class WorkflowUseCase(BaseUseCase):
    """
    Workflow use case for task-oriented multi-agent execution.
    
    Usage:
        workflow = WorkflowUseCase(conductor, mode="dynamic")
        result = await workflow.execute(goal="...", context={...})
        
        # Streaming
        async for event in workflow.stream(goal="..."):
            print(event)
        
        # Async execution
        task_id = await workflow.enqueue(goal="...", priority=5)
    """
    
    def __init__(self, conductor: Any, mode: str = 'dynamic', agent_order: Optional[list] = None, config: Optional[UseCaseConfig] = None, context: Optional[WorkflowContext] = None) -> None:
        """
        Initialize workflow use case.
        
        Args:
            conductor: Jotty Conductor instance
            mode: Orchestration mode ("static" or "dynamic")
            agent_order: Required for static mode - list of agent IDs
            config: Use case configuration
            context: Workflow context manager (optional)
        """
        super().__init__(conductor, config)
        
        # Create components
        self.orchestrator = WorkflowOrchestrator(
            conductor=conductor,
            mode=mode,
            agent_order=agent_order
        )
        self.executor = WorkflowExecutor(
            conductor=conductor,
            orchestrator=self.orchestrator,
            context=context
        )
        self.mode = mode
        self.agent_order = agent_order
    
    def _get_use_case_type(self) -> UseCaseType:
        """Return workflow use case type."""
        return UseCaseType.WORKFLOW
    
    async def execute(self, goal: str, context: Optional[Dict[str, Any]] = None, max_iterations: int = 100, **kwargs: Any) -> UseCaseResult:
        """
        Execute workflow synchronously.

        Args:
            goal: Workflow goal
            context: Additional context
            max_iterations: Maximum iterations
            **kwargs: Additional arguments

        Returns:
            UseCaseResult with workflow results
        """
        # DRY: Use base class error handling wrapper
        return await self._execute_with_error_handling(
            self.executor.execute,
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )

    def _extract_output(self, result: Dict[str, Any]) -> Any:
        """Extract workflow result."""
        return result.get("result")

    def _extract_metadata(self, result: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Extract workflow metadata."""
        return {
            "workflow_id": result.get("workflow_id"),
            "task_id": result.get("task_id"),
            "execution_time": result.get("execution_time", execution_time),
            "summary": result.get("summary", {})
        }

    def _error_output(self, error: Exception) -> Any:
        """Workflow errors return None instead of error string."""
        return None
    
    async def stream(self, goal: str, context: Optional[Dict[str, Any]] = None, max_iterations: int = 100, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute workflow with streaming.
        
        Args:
            goal: Workflow goal
            context: Additional context
            max_iterations: Maximum iterations
            **kwargs: Additional arguments
            
        Yields:
            Event dictionaries
        """
        async for event in self.executor.stream(
            goal=goal,
            context=context,
            max_iterations=max_iterations
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
        if not hasattr(self.conductor, 'enqueue_goal'):
            raise NotImplementedError(
                "Conductor does not support async execution. "
                "Ensure Conductor has a task_queue configured."
            )
        
        task_id = await self.conductor.enqueue_goal(
            goal=goal,
            priority=priority,
            **context or {},
            **kwargs
        )
        
        if task_id is None:
            raise RuntimeError("Failed to enqueue task")
        
        logger.info(f"Enqueued workflow task: {task_id}")
        return task_id
