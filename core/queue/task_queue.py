"""
Task Queue Abstract Interface
Defines the contract for task queue implementations
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from .task import Task


class TaskQueue(ABC):
    """
    Abstract task queue interface
    Compatible with supervisor's StateManager API
    """
    
    @abstractmethod
    async def enqueue(self, task: Task) -> str:
        """
        Add task to queue
        Returns task_id
        """
        pass
    
    @abstractmethod
    async def dequeue(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Task]:
        """
        Get next task from queue
        Filters can include: agent_type, status, priority_min, priority_max
        """
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        pass
    
    @abstractmethod
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
        """
        Update task status
        Compatible with supervisor's update_task_status signature
        """
        pass
    
    @abstractmethod
    async def heartbeat(self, task_id: str) -> bool:
        """Update task heartbeat timestamp"""
        pass
    
    @abstractmethod
    async def get_running_count(self) -> int:
        """Get count of running tasks (status='in_progress' and pid IS NOT NULL)"""
        pass
    
    @abstractmethod
    async def get_running_count_by_agent(self, agent_type: str) -> int:
        """Get count of running tasks for specific agent type"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics
        Returns dict with: pending, in_progress, completed, failed, active_pids, pids, by_agent
        """
        pass
    
    @abstractmethod
    async def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get all tasks with given status"""
        pass
    
    @abstractmethod
    async def get_running_tasks(self) -> List[Task]:
        """Get all running tasks"""
        pass

    @abstractmethod
    async def get_by_filename(self, filename: str) -> Optional[Task]:
        """Get task by filename (legacy support for supervisor)"""
        pass

    @abstractmethod
    async def update_task_priority(self, task_id: str, priority: int) -> bool:
        """Update task priority (1-5)"""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        pass
    
    @abstractmethod
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
        """
        Create a new task
        Returns task_id
        Compatible with supervisor's create_task signature
        """
        pass
    
    @abstractmethod
    async def reset_task_to_backlog(self, task_id: str) -> bool:
        """Reset failed task back to backlog for retry"""
        pass
    
    @abstractmethod
    async def validate_pids(self) -> int:
        """
        Clean up stale PIDs (processes that no longer exist)
        Returns count of cleaned tasks
        """
        pass
    
    @abstractmethod
    async def export_to_json(self) -> Dict[str, Any]:
        """
        Export state to JSON format
        Compatible with supervisor's export_to_json format
        """
        pass
