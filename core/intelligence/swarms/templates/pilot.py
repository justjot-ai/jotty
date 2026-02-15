"""Pilot Template"""

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.team_coordinator import TeamCoordinator
from ..swarm_learning import SwarmBaseConfig, SwarmLearning, SwarmResult


class PilotTemplate(SwarmLearning):
    AGENT_TEAM = TeamCoordinator.define(pattern=CoordinationPattern.AUTO)
    TEMPLATE_NAME = "pilot"

    def __init__(self, config: SwarmBaseConfig = None) -> None:
        super().__init__(config or SwarmBaseConfig(name="Pilot", domain="pilot"))

    async def execute(self, **kwargs) -> SwarmResult:
        pass


PilotSwarm = PilotTemplate
__all__ = ["PilotTemplate", "PilotSwarm"]
