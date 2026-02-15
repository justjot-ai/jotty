"""
Task Data Model
Compatible with supervisor's task structure
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from ..foundation.config_defaults import MAX_RETRIES

# REFACTORING PHASE 1.2: Import TaskStatus from canonical location


class TaskPriority(Enum):
    """Task priority enum - matches supervisor's priority system"""

    CRITICAL = 1  # Highest priority
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5  # Lowest priority


@dataclass
class Task:
    """
    Task data model - compatible with supervisor's task structure
    Preserves all supervisor fields for backward compatibility
    """

    # Identity
    task_id: str
    title: str
    description: str = ""

    # Categorization
    category: str = ""
    priority: int = 3  # 1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW, 5=OPTIONAL
    tags: Optional[str] = None  # JSON array string

    # Status & Progress
    status: str = (
        "backlog"  # suggested, backlog, pending, in_progress, completed, failed, blocked, cancelled
    )
    progress_percent: int = 0

    # Execution metadata
    pid: Optional[int] = None  # Process ID for running tasks
    worktree_path: Optional[str] = None
    git_branch: Optional[str] = None
    log_file: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Dependencies
    depends_on: Optional[str] = None  # JSON array string
    blocks: Optional[str] = None  # JSON array string

    # Metadata
    created_by: str = "system"  # 'user', 'system', 'ai'
    assigned_to: Optional[str] = None
    agent_type: Optional[str] = "claude"  # 'claude', 'cursor', 'opencode'

    # Legacy/compatibility fields
    context_files: Optional[str] = None  # JSON array string
    task_content: Optional[str] = None  # Full markdown content (legacy)
    filename: Optional[str] = None  # Original task filename (legacy)

    # Best-in-class references
    reference_apps: Optional[str] = None  # JSON array string
    estimated_effort: Optional[str] = None  # '2 hours', '1 day', '1 week'

    # Additional fields for Jotty integration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize defaults"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.status is None:
            self.status = "backlog"
        if self.agent_type is None:
            self.agent_type = "claude"

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary (compatible with supervisor format)"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "tags": self.tags,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "pid": self.pid,
            "worktree_path": self.worktree_path,
            "git_branch": self.git_branch,
            "log_file": self.log_file,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "created_by": self.created_by,
            "assigned_to": self.assigned_to,
            "agent_type": self.agent_type or "claude",
            "context_files": self.context_files,
            "task_content": self.task_content,
            "filename": self.filename,
            "reference_apps": self.reference_apps,
            "estimated_effort": self.estimated_effort,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary (compatible with supervisor format)"""
        # Parse datetime strings
        created_at = None
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            else:
                created_at = data["created_at"]

        started_at = None
        if data.get("started_at"):
            if isinstance(data["started_at"], str):
                started_at = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
            else:
                started_at = data["started_at"]

        completed_at = None
        if data.get("completed_at"):
            if isinstance(data["completed_at"], str):
                completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))
            else:
                completed_at = data["completed_at"]

        last_heartbeat = None
        if data.get("last_heartbeat"):
            if isinstance(data["last_heartbeat"], str):
                last_heartbeat = datetime.fromisoformat(
                    data["last_heartbeat"].replace("Z", "+00:00")
                )
            else:
                last_heartbeat = data["last_heartbeat"]

        return cls(
            task_id=data["task_id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            priority=data.get("priority", 3),
            tags=data.get("tags"),
            status=data.get("status", "backlog"),
            progress_percent=data.get("progress_percent", 0),
            pid=data.get("pid"),
            worktree_path=data.get("worktree_path"),
            git_branch=data.get("git_branch"),
            log_file=data.get("log_file"),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            last_heartbeat=last_heartbeat,
            estimated_hours=data.get("estimated_hours"),
            actual_hours=data.get("actual_hours"),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", MAX_RETRIES),
            depends_on=data.get("depends_on"),
            blocks=data.get("blocks"),
            created_by=data.get("created_by", "system"),
            assigned_to=data.get("assigned_to"),
            agent_type=data.get("agent_type", "claude"),
            context_files=data.get("context_files"),
            task_content=data.get("task_content"),
            filename=data.get("filename"),
            reference_apps=data.get("reference_apps"),
            estimated_effort=data.get("estimated_effort"),
            metadata=data.get("metadata", {}),
        )
