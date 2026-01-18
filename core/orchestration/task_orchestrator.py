"""
Task Orchestrator - Generic task lifecycle management
Reusable across projects
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..queue.task_queue import TaskQueue
from ..queue.task import Task, TaskStatus
from .agent_spawner import AgentSpawner
from .deployment_hook import DeploymentHook

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """
    Generic task orchestrator for autonomous execution.
    
    Manages task lifecycle: pending → in_progress → completed/failed
    Spawns agents, monitors execution, and triggers deployments.
    
    This is a reusable component that works with any project's agent spawner
    and deployment hook implementations.
    """
    
    def __init__(
        self,
        queue: TaskQueue,
        agent_spawner: AgentSpawner,
        deployment_hook: Optional[DeploymentHook] = None,
        max_concurrent: int = 1,
        poll_interval: int = 30,
        heartbeat_interval: int = 60,
    ):
        """
        Initialize task orchestrator.
        
        Args:
            queue: Task queue for task storage
            agent_spawner: Agent spawner implementation (project-specific)
            deployment_hook: Optional deployment hook (project-specific)
            max_concurrent: Maximum concurrent tasks per agent type
            poll_interval: Seconds between polling for new tasks
            heartbeat_interval: Seconds between health checks
        """
        self.queue = queue
        self.agent_spawner = agent_spawner
        self.deployment_hook = deployment_hook
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        
        # Track running tasks: {task_id: {'pid': int, 'log_file': str, 'started_at': datetime}}
        self._running_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Control flag
        self._running = False
        self._stop_event = asyncio.Event()
    
    async def start(self):
        """Start orchestrator main loop."""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        self._stop_event.clear()
        logger.info("Starting Task Orchestrator")
        logger.info(f"Max concurrent: {self.max_concurrent}, Poll interval: {self.poll_interval}s")
        
        try:
            await self._orchestrate_loop()
        except asyncio.CancelledError:
            logger.info("Orchestrator stopped")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            raise
        finally:
            self._running = False
    
    async def stop(self):
        """Stop orchestrator."""
        logger.info("Stopping Task Orchestrator")
        self._running = False
        self._stop_event.set()
    
    async def _orchestrate_loop(self):
        """Main orchestration loop."""
        while self._running and not self._stop_event.is_set():
            try:
                # 1. Monitor running tasks
                await self._monitor_tasks()
                
                # 2. Spawn new tasks if under limit
                await self._spawn_pending_tasks()
                
                # 3. Sleep before next iteration
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
    
    async def _spawn_pending_tasks(self):
        """Spawn pending tasks if under concurrency limit."""
        # Get running count per agent type
        running_by_agent = await self._get_running_count_by_agent()
        
        # Spawn tasks for each agent type
        for agent_type in ['claude', 'cursor', 'opencode']:
            running_count = running_by_agent.get(agent_type, 0)
            
            if running_count < self.max_concurrent:
                # Get next pending task for this agent type
                # Try with agent_type filter first, then without
                task = await self.queue.dequeue(filters={
                    'status': TaskStatus.PENDING,
                    'agent_type': agent_type,
                })
                
                # If no task found with agent_type filter, try without (for tasks without agent_type set)
                if not task:
                    task = await self.queue.dequeue(filters={
                        'status': TaskStatus.PENDING,
                    })
                    # If task found but has no agent_type, use default
                    if task and not task.agent_type:
                        task.agent_type = agent_type
                
                if task:
                    logger.info(f"Spawning task {task.task_id} ({task.title}) with {agent_type}")
                    await self._spawn_task(task)
    
    async def _spawn_task(self, task: Task):
        """Spawn task using agent spawner."""
        try:
            logger.info(f"Spawning task: {task.task_id} ({task.title})")
            
            # Get agent type
            agent_type = await self.agent_spawner.get_agent_type(task)
            
            # Check credentials
            if not await self.agent_spawner.check_credentials(agent_type):
                logger.warning(f"Credentials not available for {agent_type}, skipping task {task.task_id}")
                await self.queue.update_status(
                    task.task_id,
                    TaskStatus.FAILED.value,
                    error=f"Credentials not available for {agent_type}"
                )
                return
            
            # Spawn agent (project-specific implementation)
            pid, log_file = await self.agent_spawner.spawn(task)

            # Update status to in_progress with PID and log_file
            await self.queue.update_status(
                task.task_id,
                TaskStatus.IN_PROGRESS.value,
                pid=pid,
                log_file=log_file,
                agent_type=agent_type
            )

            # Track running task
            self._running_tasks[task.task_id] = {
                'pid': pid,
                'log_file': log_file,
                'started_at': datetime.now(),
                'agent_type': agent_type,
            }

            # Start background monitor
            asyncio.create_task(self._monitor_task_completion(task, pid, log_file))

            logger.info(f"Task {task.task_id} spawned (PID: {pid}, log: {log_file})")
            
        except Exception as e:
            logger.error(f"Error spawning task {task.task_id}: {e}", exc_info=True)
            await self.queue.update_status(
                task.task_id,
                TaskStatus.FAILED.value,
                error=str(e)
            )
    
    async def _monitor_task_completion(self, task: Task, pid: int, log_file: str):
        """Monitor task completion in background."""
        try:
            logger.info(f"Monitoring task {task.task_id} (PID: {pid})")
            
            # Wait for process to exit
            await self._wait_for_process(pid)
            
            logger.info(f"Process {pid} exited for task {task.task_id}")
            
            # Check log file for completion markers
            success = await self._check_completion(log_file)
            
            # Update status
            if success:
                logger.info(f"Task {task.task_id} completed successfully")
                await self.queue.update_status(task.task_id, TaskStatus.COMPLETED.value)
                
                # Trigger deployment if hook provided
                if self.deployment_hook and await self.deployment_hook.should_deploy(task):
                    logger.info(f"Triggering deployment for task {task.task_id}")
                    deploy_success = await self.deployment_hook.trigger(task)
                    if deploy_success:
                        logger.info(f"Deployment successful for task {task.task_id}")
                    else:
                        logger.warning(f"Deployment failed for task {task.task_id}")
            else:
                logger.warning(f"Task {task.task_id} failed")
                await self.queue.update_status(
                    task.task_id,
                    TaskStatus.FAILED.value,
                    error="Task execution failed (check log file)"
                )
            
            # Remove from running tasks
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
                
        except Exception as e:
            logger.error(f"Error monitoring task {task.task_id}: {e}", exc_info=True)
            await self.queue.update_status(
                task.task_id,
                TaskStatus.FAILED.value,
                error=f"Monitoring error: {str(e)}"
            )
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
    
    async def _wait_for_process(self, pid: int):
        """Wait for process to exit."""
        # Check if process exists
        while True:
            try:
                # Try to send signal 0 (doesn't kill, just checks if process exists)
                os.kill(pid, 0)
                await asyncio.sleep(10)  # Check every 10 seconds
            except ProcessLookupError:
                # Process doesn't exist (exited)
                break
            except OSError:
                # Process doesn't exist or permission denied
                break
    
    async def _check_completion(self, log_file: str) -> bool:
        """
        Check log file for completion markers.
        
        Returns:
            True if task completed successfully, False otherwise
        """
        if not os.path.exists(log_file):
            logger.warning(f"Log file not found: {log_file}")
            return False
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for success markers
            success_markers = [
                "All tasks completed!",
                "Task completed successfully",
                "Merged.*to main",
                "Pushed to origin",
            ]
            
            import re
            for marker in success_markers:
                if re.search(marker, content, re.IGNORECASE):
                    return True
            
            # Check for failure markers (only if no success markers found)
            failure_markers = ["error", "Error", "ERROR", "failed", "Failed", "FAILED"]
            for marker in failure_markers:
                if marker in content:
                    # Check last 20 lines for errors
                    lines = content.split('\n')
                    last_lines = '\n'.join(lines[-20:])
                    if marker in last_lines:
                        return False
            
            # Default to success if no clear failure markers
            return True
            
        except Exception as e:
            logger.error(f"Error checking completion for {log_file}: {e}")
            return False
    
    async def _monitor_tasks(self):
        """Monitor running tasks for health."""
        # Validate all running task PIDs
        tasks_to_remove = []
        
        for task_id, task_info in self._running_tasks.items():
            pid = task_info['pid']
            
            try:
                # Check if process exists
                os.kill(pid, 0)
            except ProcessLookupError:
                # Process doesn't exist (exited without notification)
                logger.warning(f"Process {pid} for task {task_id} no longer exists")
                tasks_to_remove.append(task_id)
            except OSError:
                # Process doesn't exist or permission denied
                logger.warning(f"Process {pid} for task {task_id} invalid")
                tasks_to_remove.append(task_id)
        
        # Remove invalid tasks
        for task_id in tasks_to_remove:
            logger.info(f"Removing invalid task {task_id} from running tasks")
            await self.queue.update_status(
                task_id,
                TaskStatus.FAILED.value,
                error="Process exited unexpectedly"
            )
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
    
    async def _get_running_count_by_agent(self) -> Dict[str, int]:
        """Get running task count per agent type."""
        counts = {'claude': 0, 'cursor': 0, 'opencode': 0}
        
        for task_info in self._running_tasks.values():
            agent_type = task_info.get('agent_type', 'claude')
            if agent_type in counts:
                counts[agent_type] += 1
        
        return counts
