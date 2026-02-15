"""
Stage Configuration for CUSTOM Coordination Pattern
===================================================

Declarative multi-stage workflows for complex swarm coordination.

Used with CoordinationPattern.CUSTOM to define explicit stage
dependencies and execution order.

Example:
    STAGES = [
        StageConfig("research", ["_researcher"], description="Gather information"),
        StageConfig("analyze", ["_analyst"], needs=["research"]),
        StageConfig("write", ["_writer"], needs=["analyze"]),
    ]

Author: Jotty Team
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageConfig:
    """Configuration for a single stage in a multi-stage workflow."""

    name: str  # Stage identifier (e.g., "research", "analyze")
    agents: List[str]  # Agent attribute names to execute (e.g., ["_researcher"])
    needs: List[str] = field(default_factory=list)  # Stage dependencies (must complete first)
    description: str = ""  # Human-readable description
    parallel: bool = False  # Execute agents in this stage in parallel
    optional: bool = False  # Stage can fail without blocking dependent stages
    timeout: float = 0.0  # Optional per-stage timeout override
    retry: int = 0  # Number of retries on failure

    # Stage output handling
    output_key: Optional[str] = None  # Key to store output in shared context
    pass_to_next: bool = True  # Pass output to next stage

    # Advanced
    condition: Optional[Dict[str, Any]] = None  # Optional condition to run stage
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.name:
            raise ValueError("Stage name is required")
        if not self.agents:
            raise ValueError(f"Stage {self.name}: at least one agent required")
        if self.name in self.needs:
            raise ValueError(f"Stage {self.name}: cannot depend on itself")


@dataclass
class StageResult:
    """Result from executing a single stage."""

    stage_name: str
    success: bool
    outputs: Dict[str, Any]  # agent_attr -> output
    execution_time: float = 0.0
    error: Optional[str] = None
    skipped: bool = False  # Stage was skipped due to condition/optional
    retry_count: int = 0


def validate_stages(stages: List[StageConfig]) -> None:
    """
    Validate stage configuration for cycles and missing dependencies.

    Raises:
        ValueError: If configuration is invalid
    """
    if not stages:
        raise ValueError("At least one stage required")

    stage_names = {s.name for s in stages}

    # Check for missing dependencies
    for stage in stages:
        for dep in stage.needs:
            if dep not in stage_names:
                raise ValueError(f"Stage {stage.name}: dependency '{dep}' not found in stages")

    # Check for cycles using DFS
    def has_cycle(name: str, visited: set, stack: set) -> bool:
        visited.add(name)
        stack.add(name)

        # Find stage by name
        stage = next((s for s in stages if s.name == name), None)
        if not stage:
            return False

        for dep in stage.needs:
            if dep not in visited:
                if has_cycle(dep, visited, stack):
                    return True
            elif dep in stack:
                return True

        stack.remove(name)
        return False

    visited = set()
    for stage in stages:
        if stage.name not in visited:
            if has_cycle(stage.name, visited, set()):
                raise ValueError(f"Cycle detected in stage dependencies involving {stage.name}")


def topological_sort(stages: List[StageConfig]) -> List[StageConfig]:
    """
    Sort stages in execution order using topological sort.

    Returns:
        Stages in order where all dependencies execute before dependents
    """
    # Build adjacency list
    adj = {s.name: [] for s in stages}
    in_degree = {s.name: 0 for s in stages}

    for stage in stages:
        for dep in stage.needs:
            adj[dep].append(stage.name)
            in_degree[stage.name] += 1

    # Kahn's algorithm
    queue = [name for name, deg in in_degree.items() if deg == 0]
    sorted_names = []

    while queue:
        # Sort by original order to maintain stability
        queue.sort(key=lambda n: next(i for i, s in enumerate(stages) if s.name == n))
        name = queue.pop(0)
        sorted_names.append(name)

        for neighbor in adj[name]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Map back to StageConfig objects
    stage_map = {s.name: s for s in stages}
    return [stage_map[name] for name in sorted_names]


__all__ = ["StageConfig", "StageResult", "validate_stages", "topological_sort"]
