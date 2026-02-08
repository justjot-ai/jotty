"""
Supervisor StateManager Adapter
Wraps supervisor's StateManager to implement TaskQueue interface
Enables backward compatibility - supervisor can use Jotty's TaskQueue OR continue using StateManager
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from .task_queue import TaskQueue
from .task import Task


class SupervisorStateManagerAdapter(TaskQueue):
    """
    Adapter that wraps supervisor's StateManager to implement TaskQueue interface
    This allows supervisor to optionally use Jotty's TaskQueue without migration pain
    """
    
    def __init__(self, state_manager):
        """
        Initialize adapter with supervisor's StateManager
        
        Args:
            state_manager: Instance of supervisor's StateManager class
        """
        self.state_manager = state_manager
    
    async def enqueue(self, task: Task) -> str:
        """Add task to queue"""
        # Convert Task to supervisor format and create via StateManager
        task_id = await self.create_task(
            title=task.title,
            description=task.description,
            priority=task.priority,
            category=task.category,
            context_files=task.context_files,
            status=task.status,
            agent_type=task.agent_type,
        )
        
        # Update any additional fields
        if task_id:
            await self.update_task_metadata(
                task_id,
                title=task.title,
                description=task.description,
                priority=task.priority,
                category=task.category,
                context_files=task.context_files,
                agent_type=task.agent_type,
            )
        
        return task_id
    
    async def dequeue(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Task]:
        """Get next pending task"""
        filters = filters or {}
        agent_type = filters.get('agent_type')
        
        # Use supervisor's get_next_pending_task
        task_dict = self.state_manager.get_next_pending_task(agent_type=agent_type)
        
        if task_dict:
            # Get full task details
            full_task = self.state_manager.get_task_by_task_id(task_dict['task_id'])
            if full_task:
                return Task.from_dict(full_task)
        
        return None
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        task_dict = self.state_manager.get_task_by_task_id(task_id)
        if task_dict:
            return Task.from_dict(task_dict)
        return None
    
    async def update_status(
        self,
        task_id: str,
        status: str,
        pid: Optional[int] = None,
        error: Optional[str] = None,
        log_file: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update task status"""
        return self.state_manager.update_task_status(
            task_id=task_id,
            status=status,
            pid=pid,
            error=error,
            log_file=log_file,
            agent_type=agent_type,
        )
    
    async def heartbeat(self, task_id: str) -> bool:
        """Update task heartbeat"""
        self.state_manager.heartbeat(task_id)
        return True
    
    async def get_running_count(self) -> int:
        """Get count of running tasks"""
        stats = self.state_manager.get_stats()
        return stats.get('active_pids', 0)
    
    async def get_running_count_by_agent(self, agent_type: str) -> int:
        """Get count of running tasks for specific agent type"""
        return self.state_manager.get_running_count_by_agent(agent_type)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return self.state_manager.get_stats()
    
    async def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get all tasks with given status"""
        task_dicts = self.state_manager.get_tasks_by_status(status)
        return [Task.from_dict(t) for t in task_dicts]
    
    async def get_running_tasks(self) -> List[Task]:
        """Get all running tasks"""
        task_dicts = self.state_manager.get_running_tasks()
        return [Task.from_dict(t) for t in task_dicts]
    
    async def update_task_priority(self, task_id: str, priority: int) -> bool:
        """Update task priority"""
        return self.state_manager.update_task_priority(task_id, priority)
    
    async def update_task_metadata(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        category: Optional[str] = None,
        context_files: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update task metadata"""
        return self.state_manager.update_task_metadata(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            category=category,
            context_files=context_files,
            agent_type=agent_type,
        )
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        return self.state_manager.delete_task(task_id)
    
    async def create_task(
        self,
        title: str,
        description: str = "",
        priority: int = 3,
        category: str = "",
        context_files: Optional[str] = None,
        status: str = "backlog",
        agent_type: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """Create a new task"""
        return self.state_manager.create_task(
            title=title,
            description=description,
            priority=priority,
            category=category,
            context_files=context_files,
            status=status,
            agent_type=agent_type,
            **kwargs
        )
    
    async def reset_task_to_backlog(self, task_id: str) -> bool:
        """Reset failed task back to backlog"""
        return self.state_manager.reset_task_to_backlog(task_id)
    
    async def validate_pids(self) -> int:
        """Clean up stale PIDs"""
        return self.state_manager.validate_pids()
    
    async def export_to_json(self) -> Dict[str, Any]:
        """Export state to JSON"""
        return self.state_manager.export_to_json()
