"""
Swarm Template Base — re-exported from canonical location.

The canonical implementation lives in:
    swarms/base/swarm_template.py

Architecture: Skills → Agents → Swarm (agents + coordination + learning)
No separate "Team" layer — a Swarm IS a coordinated team of agents.

For swarm templates:
    from Jotty.core.intelligence.orchestration.swarms.base import SwarmTemplate
"""

from ..base import AgentCoordinator, CoordinationResult
from ..base.swarm_template import PhaseExecutor, SwarmTemplate

__all__ = ["SwarmTemplate", "PhaseExecutor", "AgentCoordinator", "CoordinationResult"]
