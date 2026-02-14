"""
In-Memory Task Queue Implementation
For testing and development
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import defaultdict

from .task_queue import TaskQueue
from .task import Task


class MemoryTaskQueue(TaskQueue):
    """
    In-memory task queue for testing/development
    Preserves all supervisor functionality
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
    
    async def enqueue(self, task: Task) -> str:
        """Add task to queue"""
        async with self._lock:
            self.tasks[task.task_id] = task
            return task.task_id
    
    async def dequeue(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Task]:
        """Get next pending task"""
        filters = filters or {}
        agent_type = filters.get('agent_type')
        
        async with self._lock:
            pending_tasks = [
                t for t in self.tasks.values()
                if t.status == 'pending'
                and (not agent_type or (t.agent_type or 'claude') == agent_type)
            ]
            
            if not pending_tasks:
                return None
            
            # Sort by priority (ascending) then task_id
            pending_tasks.sort(key=lambda t: (t.priority, t.task_id))
            return pending_tasks[0]
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        async with self._lock:
            return self.tasks.get(task_id)
    
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
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = status
            
            if pid is not None:
                task.pid = pid
            if error is not None:
                task.error_message = error
            if log_file is not None:
                task.log_file = log_file
            if agent_type is not None:
                task.agent_type = agent_type
            
            if status == 'in_progress':
                task.started_at = datetime.now()
                task.last_heartbeat = datetime.now()
            elif status == 'completed':
                task.completed_at = datetime.now()
                task.pid = None
            elif status == 'failed':
                task.completed_at = datetime.now()
                task.pid = None
            
            return True
    
    async def heartbeat(self, task_id: str) -> bool:
        """Update task heartbeat"""
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id].last_heartbeat = datetime.now()
            return True
    
    async def get_running_count(self) -> int:
        """Get count of running tasks"""
        async with self._lock:
            return sum(1 for t in self.tasks.values() if t.pid is not None)
    
    async def get_running_count_by_agent(self, agent_type: str) -> int:
        """Get count of running tasks for specific agent type"""
        async with self._lock:
            return sum(
                1 for t in self.tasks.values()
                if t.pid is not None and (t.agent_type or 'claude') == agent_type
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            stats = {
                'pending': sum(1 for t in self.tasks.values() if t.status == 'pending'),
                'in_progress': sum(1 for t in self.tasks.values() if t.status == 'in_progress'),
                'completed': sum(1 for t in self.tasks.values() if t.status == 'completed'),
                'failed': sum(1 for t in self.tasks.values() if t.status == 'failed'),
                'active_pids': sum(1 for t in self.tasks.values() if t.pid is not None),
            }
            
            # Get active PIDs
            stats['pids'] = [t.pid for t in self.tasks.values() if t.pid is not None]
            
            # Get per-agent stats
            by_agent = defaultdict(int)
            for t in self.tasks.values():
                if t.pid is not None:
                    agent = t.agent_type or 'claude'
                    by_agent[agent] += 1
            stats['by_agent'] = dict(by_agent)
            
            return stats
    
    async def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get all tasks with given status"""
        async with self._lock:
            tasks = [t for t in self.tasks.values() if t.status == status]
            tasks.sort(key=lambda t: (t.priority, t.created_at or datetime.min))
            return tasks
    
    async def get_running_tasks(self) -> List[Task]:
        """Get all running tasks"""
        async with self._lock:
            tasks = [t for t in self.tasks.values() if t.status == 'in_progress']
            tasks.sort(key=lambda t: t.started_at or datetime.min)
            return tasks
    

    async def get_by_filename(self, filename: str) -> Optional[Task]:
        """Get task by filename (legacy support for supervisor)"""
        async with self._lock:
            for task in self.tasks.values():
                if task.filename == filename:
                    return task
            return None

    async def update_task_priority(self, task_id: str, priority: int) -> bool:
        """Update task priority"""
        if priority < 1 or priority > 5:
            return False
        
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id].priority = priority
            return True
    
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
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if title is not None:
                task.title = title
            if description is not None:
                task.description = description
            if priority is not None:
                if priority < 1 or priority > 5:
                    return False
                task.priority = priority
            if category is not None:
                task.category = category
            if context_files is not None:
                task.context_files = context_files
            if agent_type is not None:
                if agent_type not in ['claude', 'cursor', 'opencode']:
                    return False
                task.agent_type = agent_type
            
            return True
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False
    
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
        if agent_type is None:
            agent_type = 'claude'
        elif agent_type not in ['claude', 'cursor', 'opencode']:
            agent_type = 'claude'
        
        # Generate task_id: TASK-YYYYMMDD-XXXXX
        now = datetime.now()
        date_part = now.strftime("%Y%m%d")
        
        async with self._lock:
            # Find next available number for today
            existing_ids = [
                t.task_id for t in self.tasks.values()
                if t.task_id.startswith(f"TASK-{date_part}-")
            ]
            
            if existing_ids:
                last_num = max(int(tid.split('-')[-1]) for tid in existing_ids)
                next_num = last_num + 1
            else:
                next_num = 1
            
            task_id = f"TASK-{date_part}-{next_num:05d}"
            
            task = Task(
                task_id=task_id,
                title=title,
                description=description,
                category=category,
                priority=priority,
                status=status,
                context_files=context_files,
                created_by='AI' if kwargs.get('suggested_by') else 'user',
                agent_type=agent_type,
            )
            
            self.tasks[task_id] = task
            return task_id
    
    async def reset_task_to_backlog(self, task_id: str) -> bool:
        """Reset failed task back to backlog"""
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = 'backlog'
            task.error_message = None
            task.pid = None
            task.started_at = None
            task.completed_at = None
            task.last_heartbeat = None
            task.retry_count += 1
            
            return True
    
    async def validate_pids(self) -> int:
        """Clean up stale PIDs (no-op for memory queue)"""
        # In-memory queue doesn't track actual processes
        return 0
    
    async def export_to_json(self) -> Dict[str, Any]:
        """Export state to JSON"""
        async with self._lock:
            tasks = list(self.tasks.values())
            
            task_details = {}
            for task in tasks:
                task_details[task.task_id] = task.to_dict()
            
            state = {
                'version': '2.0-memory-jotty',
                'total_tasks': len(tasks),
                'completed_tasks': sum(1 for t in tasks if t.status == 'completed'),
                'failed_tasks': sum(1 for t in tasks if t.status == 'failed'),
                'suggested_tasks': [t.task_id for t in tasks if t.status == 'suggested'],
                'backlog_tasks': [t.task_id for t in tasks if t.status == 'backlog'],
                'pending_tasks': [t.task_id for t in tasks if t.status == 'pending'],
                'in_progress_tasks': [t.task_id for t in tasks if t.status == 'in_progress'],
                'completed_task_files': [t.task_id for t in tasks if t.status == 'completed'],
                'failed_task_files': [t.task_id for t in tasks if t.status == 'failed'],
                'task_pids': {t.task_id: t.pid for t in tasks if t.pid},
                'task_status': {t.task_id: t.status for t in tasks},
                'task_details': task_details,
                'last_validated': datetime.now().isoformat()
            }
            
            return state
