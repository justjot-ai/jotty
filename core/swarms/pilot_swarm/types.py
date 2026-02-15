"""Pilot Swarm - Type definitions.

Defines the data structures for autonomous goal-completion:
- SubtaskType: categories of work the Pilot can perform
- Subtask: a single unit of work in the execution plan
- PilotConfig: swarm configuration
- PilotResult: execution result with artifacts
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..base_swarm import SwarmResult
from ..swarm_types import SwarmConfig


class SubtaskType(enum.Enum):
    """Types of subtasks the Pilot can execute."""
    SEARCH = "search"
    CODE = "code"
    TERMINAL = "terminal"
    CREATE_SKILL = "create_skill"
    DELEGATE = "delegate"
    ANALYZE = "analyze"
    BROWSE = "browse"


class SubtaskStatus(enum.Enum):
    """Execution status of a subtask."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A single subtask in the Pilot's execution plan."""
    id: str
    type: SubtaskType
    description: str
    tool_hint: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PilotConfig(SwarmConfig):
    """Configuration for PilotSwarm."""
    max_subtasks: int = 10
    max_retries: int = 2
    max_concurrent: int = 3
    allow_terminal: bool = True
    allow_file_write: bool = True
    allow_delegation: bool = True
    send_telegram: bool = False
    working_directory: str = ""
    llm_model: str = "haiku"
    use_fast_predict: bool = True
    llm_timeout: int = 0

    def __post_init__(self) -> None:
        self.name = "PilotSwarm"
        self.domain = "pilot"
        if self.llm_timeout <= 0:
            from Jotty.core.foundation.config_defaults import LLM_TIMEOUT_SECONDS
            self.llm_timeout = LLM_TIMEOUT_SECONDS


@dataclass
class PilotResult(SwarmResult):
    """Result from PilotSwarm."""
    goal: str = ""
    subtasks_completed: int = 0
    subtasks_total: int = 0
    artifacts: List[str] = field(default_factory=list)
    skills_created: List[str] = field(default_factory=list)
    delegated_to: List[str] = field(default_factory=list)
    retry_count: int = 0


# Available swarms for delegation
AVAILABLE_SWARMS = [
    "research — financial research with data + charts + PDF",
    "coding — production-quality code with tests + docs",
    "testing — comprehensive test suites (unit, integration, e2e)",
    "data_analysis — EDA, statistics, ML, visualizations",
    "review — code/security/performance review",
    "devops — infrastructure, CI/CD, containers, cloud",
    "idea_writer — articles, reports, blogs, whitepapers",
    "olympiad_learning — educational content (STEM, basics to competition)",
    "perspective_learning — multi-perspective learning (6 angles, 4 languages)",
]


__all__ = [
    'SubtaskType', 'SubtaskStatus', 'Subtask',
    'PilotConfig', 'PilotResult', 'AVAILABLE_SWARMS',
]
