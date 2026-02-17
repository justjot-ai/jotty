"""
Swarm Base Module
=================

Architecture:
    Skills → Agents → Swarm (agents + coordination + learning)

A swarm IS a coordinated group of agents. No separate "Team" layer.

Provides:
- AgentCoordinator: Declarative agent composition with coordination patterns
  (backward compat: TeamCoordinator)
- SwarmTemplate: Template for domain-specific swarms with learning
- CoordinationPattern: How agents work together (pipeline, parallel, etc.)
- MergeStrategy: How to combine parallel results

Usage:
    from Jotty.core.intelligence.orchestration.swarms.base import (
        SwarmTemplate, AgentCoordinator, CoordinationPattern
    )

    class MySwarm(SwarmTemplate):
        AGENTS = AgentCoordinator.define(
            (AgentA, "AgentA"),
            (AgentB, "AgentB"),
            pattern=CoordinationPattern.PIPELINE,
        )

        async def _execute_domain(self, task: str, **kwargs):
            result = await self.coordinate(task=task)
            return SwarmResult(output=result.merged_output)
"""

from .swarm_template import PhaseExecutor, SwarmTemplate, _safe_join, _safe_num, _split_field
from .team_coordinator import (
    AgentSpec,
    CoordinationPattern,
    MergeStrategy,
    TeamCoordinator,
    TeamResult,
)

# New names (Team IS Swarm — no separate layer)
AgentCoordinator = TeamCoordinator  # Preferred name
CoordinationResult = TeamResult  # Preferred name

__all__ = [
    # New preferred names
    "AgentCoordinator",
    "CoordinationResult",
    # Backward compat names
    "TeamCoordinator",
    "TeamResult",
    # Shared
    "AgentSpec",
    "CoordinationPattern",
    "MergeStrategy",
    # Swarm base
    "SwarmTemplate",
    "PhaseExecutor",
    # Defensive utilities
    "_split_field",
    "_safe_join",
    "_safe_num",
]
