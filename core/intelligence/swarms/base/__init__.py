"""
Swarm Base Module
=================

Architecture:
    Agent (skills, LLM) → Team (coordination) → Swarm (learning)

Provides:
- TeamCoordinator: Declarative agent composition with coordination patterns
- SwarmTemplate: Template for domain-specific swarms with learning
- CoordinationPattern: How agents work together (pipeline, parallel, etc.)
- MergeStrategy: How to combine parallel results

Usage:
    # Simple swarm (manual coordination)
    from Jotty.core.execution.base import SwarmTemplate, TeamCoordinator

    class MySwarm(SwarmTemplate):
        AGENT_TEAM = TeamCoordinator.define(
            (AgentA, "AgentA"),
            (AgentB, "AgentB"),
        )

        async def _execute_domain(self, task: str, **kwargs):
            result_a = await self._agent_a.execute(task)
            result_b = await self._agent_b.execute(result_a)
            return result_b

    # Team-coordinated swarm (automatic orchestration)
    from Jotty.core.execution.base import CoordinationPattern, MergeStrategy

    class ReviewSwarm(SwarmTemplate):
        AGENT_TEAM = TeamCoordinator.define(
            (SecurityReviewer, "Security"),
            (PerformanceReviewer, "Performance"),
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.CONCAT,
        )

        async def _execute_domain(self, code: str, **kwargs):
            team_result = await self.execute_team(task=code)
            return ReviewResult(findings=team_result.merged_output)
"""

from .swarm_template import PhaseExecutor, SwarmTemplate, _safe_join, _safe_num, _split_field
from .team_coordinator import (
    AgentSpec,
    CoordinationPattern,
    MergeStrategy,
    TeamCoordinator,
    TeamResult,
)

__all__ = [
    # Team composition
    "TeamCoordinator",
    "AgentSpec",
    "TeamResult",
    # Coordination patterns
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
