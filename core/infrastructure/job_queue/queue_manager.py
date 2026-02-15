"""
Task Queue Manager
High-level manager for processing tasks from queue using Orchestrator
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .task_queue import TaskQueue
from .task import Task
from ..orchestration import Orchestrator

logger = logging.getLogger(__name__)


class TaskQueueManager:
    """
    High-level task queue manager with agent orchestration
    Processes tasks from queue using Orchestrator
    """
    
    def __init__(self, conductor: Orchestrator, task_queue: TaskQueue, max_concurrent: int = 3, poll_interval: float = 1.0) -> None:
        """
        Initialize task queue manager
        
        Args:
            conductor: Orchestrator instance for agent orchestration
            task_queue: Task queue instance
            max_concurrent: Maximum concurrent task executions
            poll_interval: Polling interval in seconds
        """
        self.conductor = conductor
        self.task_queue = task_queue
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = asyncio.Event()
    
    async def start(self) -> Any:
        """Start processing tasks from queue"""
        logger.info(f" Starting TaskQueueManager (max_concurrent={self.max_concurrent})")
        
        while not self._stop_event.is_set():
            try:
                # Check concurrency limit
                running_count = len([t for t in self.running_tasks.values() if not t.done()])
                
                if running_count < self.max_concurrent:
                    # Try to get next task
                    task = await self.task_queue.dequeue()
                    
                    if task:
                        logger.info(f" Dequeued task: {task.task_id} ({task.title})")
                        self.running_tasks[task.task_id] = asyncio.create_task(
                            self._process_task(task)
                        )
                
                # Clean up completed tasks
                self.running_tasks = {
                    tid: t for tid, t in self.running_tasks.items()
                    if not t.done()
                }
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f" Error in task queue manager loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
    
    async def stop(self) -> Any:
        """Stop processing tasks"""
        logger.info(" Stopping TaskQueueManager...")
        self._stop_event.set()
        
        # Wait for running tasks to complete (with timeout)
        if self.running_tasks:
            logger.info(f"⏳ Waiting for {len(self.running_tasks)} running tasks to complete...")
            await asyncio.wait(
                list(self.running_tasks.values()),
                timeout=60.0,
                return_when=asyncio.ALL_COMPLETED
            )
        
        logger.info(" TaskQueueManager stopped")
    
    async def _process_task(self, task: Task) -> Any:
        """Process a single task"""
        task_id = task.task_id
        
        try:
            # Update status to in_progress
            await self.task_queue.update_status(
                task_id,
                status='in_progress',
                started_at=datetime.now(),
            )
            
            logger.info(f" Processing task: {task_id} - {task.title}")
            
            # Extract goal from task payload
            goal = task.description or task.title
            
            # Prepare context from task metadata
            context = {
                'task_id': task_id,
                'category': task.category,
                'priority': task.priority,
                **task.metadata,
            }
            
            # Execute via Orchestrator (using LangGraph if enabled)
            result = await self.conductor.run(
                goal=goal,
                max_iterations=100,
                **context
            )
            
            # Update status to completed
            await self.task_queue.update_status(
                task_id,
                status='completed',
                completed_at=datetime.now(),
                metadata={
                    'result': result.to_dict() if hasattr(result, 'to_dict') else str(result),
                    'final_output': result.final_output if hasattr(result, 'final_output') else str(result),
                }
            )
            
            logger.info(f" Task completed: {task_id}")
            
        except Exception as e:
            logger.error(f" Task failed: {task_id} - {e}", exc_info=True)
            
            # Check retry logic
            if task.retry_count < task.max_retries:
                await self.task_queue.update_status(
                    task_id,
                    status='retrying',
                    error=str(e),
                    retry_count=task.retry_count + 1,
                )
                logger.info(f" Task will be retried: {task_id} (attempt {task.retry_count + 1}/{task.max_retries})")
            else:
                await self.task_queue.update_status(
                    task_id,
                    status='failed',
                    error=str(e),
                    completed_at=datetime.now(),
                )
                logger.error(f" Task failed permanently: {task_id}")
