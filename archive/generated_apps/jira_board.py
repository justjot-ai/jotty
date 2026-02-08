from enum import Enum
from typing import Optional

class Status(Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"

class Task:
    def __init__(
        self,
        id: int,
        title: str,
        status: Status = Status.TODO,
        priority: int = 3,
        assignee: Optional[str] = None
    ):
        self.id = id
        self.title = title
        self.status = status
        self.priority = self._validate_priority(priority)
        self.assignee = assignee
    
    def _validate_priority(self, priority: int) -> int:
        if not 1 <= priority <= 5:
            raise ValueError("Priority must be between 1 and 5")
        return priority
    
    def __repr__(self):
        return f"Task(id={self.id}, title='{self.title}', status={self.status.value}, priority={self.priority}, assignee={self.assignee})"

def create_task(
    id: int,
    title: str,
    status: Status = Status.TODO,
    priority: int = 3,
    assignee: Optional[str] = None
) -> Task:
    return Task(id, title, status, priority, assignee)

def move_task(task: Task, new_status: Status) -> Task:
    task.status = new_status
    return task