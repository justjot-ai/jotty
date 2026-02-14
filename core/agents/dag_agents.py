"""
DAG Agents - Task Breakdown and Actor Assignment for Jotty Swarm

This module re-exports from the decomposed submodules:
- dag_types: Enums, signatures, data structures
- task_breakdown_agent: TaskBreakdownAgent
- todo_creator_agent: TodoCreatorAgent
"""

from .dag_types import (                         # noqa: F401
    DAGAgentMixin,
    SwarmResources,
    TaskType,
    ExecutableDAG,
    Actor,
    ExtractTasksSignature,
    IdentifyDependenciesSignature,
    OptimizeWorkflowSignature,
    ActorAssignmentSignature,
    DAGValidationSignature,
    OptimizeDAGSignature,
)

from .task_breakdown_agent import TaskBreakdownAgent   # noqa: F401
from .todo_creator_agent import TodoCreatorAgent       # noqa: F401

from ..foundation.data_structures import SwarmConfig


def get_swarm_resources(config: Optional[SwarmConfig] = None) -> SwarmResources:
    """Get or create shared SwarmResources."""
    return SwarmResources(config or SwarmConfig())


def create_task_breakdown_agent(config: Optional[SwarmConfig] = None) -> TaskBreakdownAgent:
    """Create a TaskBreakdownAgent with shared resources."""
    return TaskBreakdownAgent(resources=get_swarm_resources(config))


def create_todo_creator_agent(config: Optional[SwarmConfig] = None, resources: Optional[SwarmResources] = None) -> TodoCreatorAgent:
    """Create a TodoCreatorAgent with shared resources."""
    return TodoCreatorAgent(resources=resources or get_swarm_resources(config))


def reset_swarm_resources() -> None:
    """Reset shared resources (for testing)."""
    SwarmResources._instance = None
