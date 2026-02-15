"""
Swarm Base Module
=================

Architecture:
    Agent (skills, LLM) → Team (coordination) → Swarm (learning)

Provides:
- AgentTeam: Declarative agent composition with coordination patterns
- DomainSwarm: Template for domain-specific swarms with learning
- CoordinationPattern: How agents work together (pipeline, parallel, etc.)
- MergeStrategy: How to combine parallel results

Usage:
    # Simple swarm (manual coordination)
    from Jotty.core.intelligence.swarms.base import DomainSwarm, AgentTeam

    class MySwarm(DomainSwarm):
        AGENT_TEAM = AgentTeam.define(
            (AgentA, "AgentA"),
            (AgentB, "AgentB"),
        )

        async def _execute_domain(self, task: str, **kwargs):
            result_a = await self._agent_a.execute(task)
            result_b = await self._agent_b.execute(result_a)
            return result_b

    # Team-coordinated swarm (automatic orchestration)
    from Jotty.core.intelligence.swarms.base import CoordinationPattern, MergeStrategy

    class ReviewSwarm(DomainSwarm):
        AGENT_TEAM = AgentTeam.define(
            (SecurityReviewer, "Security"),
            (PerformanceReviewer, "Performance"),
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.CONCAT,
        )

        async def _execute_domain(self, code: str, **kwargs):
            team_result = await self.execute_team(task=code)
            return ReviewResult(findings=team_result.merged_output)
"""

from .agent_team import (
    AgentTeam,
    AgentSpec,
    TeamResult,
    CoordinationPattern,
    MergeStrategy,
)
from .domain_swarm import DomainSwarm, PhaseExecutor, _split_field, _safe_join, _safe_num

__all__ = [
    # Team composition
    'AgentTeam',
    'AgentSpec',
    'TeamResult',
    # Coordination patterns
    'CoordinationPattern',
    'MergeStrategy',
    # Swarm base
    'DomainSwarm',
    'PhaseExecutor',
    # Defensive utilities
    '_split_field',
    '_safe_join',
    '_safe_num',
]
