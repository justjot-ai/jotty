"""
Workflow Executor

Handles execution of workflow tasks.
"""

from typing import Dict, Any, Optional, List, AsyncIterator
import logging
import time

from .workflow_orchestrator import WorkflowOrchestrator
from .workflow_context import WorkflowContext, WorkflowTask, TaskStatus

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Executes workflow tasks with agents.
    """
    
    def __init__(self, conductor: Any, orchestrator: WorkflowOrchestrator, context: Optional[WorkflowContext] = None) -> None:
        """
        Initialize workflow executor.
        
        Args:
            conductor: Jotty Conductor instance
            orchestrator: Workflow orchestrator for agent selection
            context: Workflow context manager
        """
        self.conductor = conductor
        self.orchestrator = orchestrator
        self.context = context or WorkflowContext()
    
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute workflow synchronously.
        
        Args:
            goal: Main workflow goal
            context: Additional context
            max_iterations: Maximum number of iterations
            
        Returns:
            Workflow result dictionary
        """
        start_time = time.time()
        
        # Create main task
        task_id = self.context.add_task(goal, metadata=context or {})
        
        try:
            # Execute workflow via conductor
            if hasattr(self.conductor, 'run'):
                result = await self.conductor.run(
                    goal=goal,
                    context=context,
                    max_iterations=max_iterations
                )
            else:
                # Fallback: execute tasks sequentially
                result = await self._execute_sequential(max_iterations)
            
            execution_time = time.time() - start_time
            
            # Update task status
            self.context.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                result=result
            )
            
            return {
                "success": True,
                "result": result,
                "workflow_id": self.context.workflow_id,
                "task_id": task_id,
                "execution_time": execution_time,
                "summary": self.context.get_summary()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            
            # Update task status
            self.context.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=str(e)
            )
            
            return {
                "success": False,
                "error": str(e),
                "workflow_id": self.context.workflow_id,
                "task_id": task_id,
                "execution_time": execution_time,
                "summary": self.context.get_summary()
            }
    
    async def _execute_sequential(self, max_iterations: int) -> Any:
        """Execute tasks sequentially (fallback)."""
        iteration = 0
        results = {}
        
        while iteration < max_iterations:
            ready_tasks = self.context.get_ready_tasks()
            
            if not ready_tasks:
                # No more tasks to execute
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                # Select agent
                agent_id = self.orchestrator.select_agent(task, {})
                
                # Prepare context
                agent_context = self.orchestrator.prepare_agent_context(
                    task, self.context, {}
                )
                
                # Execute agent
                try:
                    result = await self.conductor.run_actor(
                        actor_name=agent_id,
                        goal=task.goal,
                        context=agent_context
                    )
                    
                    # Update task
                    self.context.update_task_status(
                        task.id,
                        TaskStatus.COMPLETED,
                        result=result
                    )
                    
                    results[task.id] = result
                    
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    self.context.update_task_status(
                        task.id,
                        TaskStatus.FAILED,
                        error=str(e)
                    )
            
            iteration += 1
        
        # Return final result (last completed task or main task)
        main_task = self.context.get_task(list(self.context.tasks.keys())[0])
        if main_task and main_task.result is not None:
            return main_task.result
        
        return results
    
    async def stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute workflow with streaming.
        
        Args:
            goal: Main workflow goal
            context: Additional context
            max_iterations: Maximum number of iterations
            
        Yields:
            Event dictionaries
        """
        # Create main task
        task_id = self.context.add_task(goal, metadata=context or {})
        
        yield {
            "type": "workflow_started",
            "workflow_id": self.context.workflow_id,
            "task_id": task_id,
            "goal": goal,
            "timestamp": time.time()
        }
        
        try:
            # Stream workflow execution
            if hasattr(self.conductor, 'run_stream'):
                async for event in self.conductor.run_stream(
                    goal=goal,
                    context=context,
                    max_iterations=max_iterations
                ):
                    yield event
            else:
                # Fallback: execute and stream events
                ready_tasks = self.context.get_ready_tasks()
                
                for task in ready_tasks:
                    yield {
                        "type": "task_started",
                        "task_id": task.id,
                        "goal": task.goal,
                        "timestamp": time.time()
                    }
                    
                    # Select agent
                    agent_id = self.orchestrator.select_agent(task, {})
                    
                    yield {
                        "type": "agent_selected",
                        "agent": agent_id,
                        "task_id": task.id,
                        "timestamp": time.time()
                    }
                    
                    # Prepare context
                    agent_context = self.orchestrator.prepare_agent_context(
                        task, self.context, context or {}
                    )
                    
                    # Execute agent
                    try:
                        result = await self.conductor.run_actor(
                            actor_name=agent_id,
                            goal=task.goal,
                            context=agent_context
                        )
                        
                        self.context.update_task_status(
                            task.id,
                            TaskStatus.COMPLETED,
                            result=result
                        )
                        
                        yield {
                            "type": "task_completed",
                            "task_id": task.id,
                            "result": result,
                            "timestamp": time.time()
                        }
                        
                    except Exception as e:
                        self.context.update_task_status(
                            task.id,
                            TaskStatus.FAILED,
                            error=str(e)
                        )
                        
                        yield {
                            "type": "task_failed",
                            "task_id": task.id,
                            "error": str(e),
                            "timestamp": time.time()
                        }
            
            # Final summary
            yield {
                "type": "workflow_completed",
                "workflow_id": self.context.workflow_id,
                "summary": self.context.get_summary(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Workflow streaming failed: {e}", exc_info=True)
            yield {
                "type": "workflow_failed",
                "workflow_id": self.context.workflow_id,
                "error": str(e),
                "timestamp": time.time()
            }
