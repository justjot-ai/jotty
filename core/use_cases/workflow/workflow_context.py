"""
Workflow Context Management

Handles task context and state for workflow use cases.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class WorkflowTask:
    """A workflow task."""
    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class WorkflowContext:
    """
    Manages workflow context and task state.
    """
    
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        max_tasks: int = 100
    ):
        """
        Initialize workflow context.
        
        Args:
            workflow_id: Unique workflow identifier
            max_tasks: Maximum number of tasks to track
        """
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.max_tasks = max_tasks
        self.tasks: Dict[str, WorkflowTask] = {}
        self.execution_order: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_task(
        self,
        goal: str,
        task_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a task to the workflow.
        
        Args:
            goal: Task goal
            task_id: Optional task ID (auto-generated if not provided)
            dependencies: List of task IDs this task depends on
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{len(self.tasks)}_{int(time.time())}"
        
        task = WorkflowTask(
            id=task_id,
            goal=goal,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        
        # Trim if needed
        if len(self.tasks) > self.max_tasks:
            # Remove oldest completed tasks
            completed_tasks = [
                (tid, t) for tid, t in self.tasks.items()
                if t.status == TaskStatus.COMPLETED
            ]
            completed_tasks.sort(key=lambda x: x[1].completed_at or 0)
            
            for tid, _ in completed_tasks[:len(self.tasks) - self.max_tasks]:
                del self.tasks[tid]
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[WorkflowTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Any = None,
        error: Optional[str] = None
    ):
        """Update task status."""
        task = self.tasks.get(task_id)
        if task:
            task.status = status
            if result is not None:
                task.result = result
            if error:
                task.error = error
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = time.time()
    
    def get_ready_tasks(self) -> List[WorkflowTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.tasks.get(dep_id, WorkflowTask(
                        id=dep_id,
                        goal="",
                        status=TaskStatus.COMPLETED
                    )).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if deps_completed:
                    ready.append(task)
        
        return ready
    
    def get_execution_order(self) -> List[str]:
        """Get recommended execution order based on dependencies."""
        # Topological sort
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(task_id: str):
            if task_id in temp_visited:
                # Circular dependency detected
                logger.warning(f"Circular dependency detected involving task: {task_id}")
                return
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            task = self.tasks.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            order.append(task_id)
        
        for task_id in self.tasks.keys():
            if task_id not in visited:
                visit(task_id)
        
        return order
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            for task in self.tasks.values()
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for task in self.tasks.values()
                if task.status == status
            )
        
        return {
            "workflow_id": self.workflow_id,
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "is_complete": self.is_complete(),
            "execution_order": self.get_execution_order()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "max_tasks": self.max_tasks,
            "tasks": {
                tid: {
                    "id": task.id,
                    "goal": task.goal,
                    "status": task.status.value,
                    "dependencies": task.dependencies,
                    "result": str(task.result) if task.result else None,
                    "error": task.error,
                    "metadata": task.metadata,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at
                }
                for tid, task in self.tasks.items()
            },
            "execution_order": self.execution_order,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """Create from dictionary."""
        context = cls(
            workflow_id=data.get("workflow_id"),
            max_tasks=data.get("max_tasks", 100)
        )
        
        for tid, task_data in data.get("tasks", {}).items():
            task = WorkflowTask(
                id=task_data["id"],
                goal=task_data["goal"],
                status=TaskStatus(task_data["status"]),
                dependencies=task_data.get("dependencies", []),
                result=task_data.get("result"),
                error=task_data.get("error"),
                metadata=task_data.get("metadata", {}),
                created_at=task_data.get("created_at", time.time()),
                completed_at=task_data.get("completed_at")
            )
            context.tasks[tid] = task
        
        context.execution_order = data.get("execution_order", [])
        context.metadata = data.get("metadata", {})
        
        return context
