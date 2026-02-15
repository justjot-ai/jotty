"""Fundamental Template"""

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.team_coordinator import TeamCoordinator
from ..swarm_learning import SwarmBaseConfig, SwarmLearning, SwarmResult


class FundamentalTemplate(SwarmLearning):
    AGENT_TEAM = TeamCoordinator.define(pattern=CoordinationPattern.AUTO)
    TEMPLATE_NAME = "fundamental"

    def __init__(self, config: SwarmBaseConfig = None):
        super().__init__(config or SwarmBaseConfig(name="Fundamental", domain="fundamental"))

    async def execute(self, **kwargs) -> SwarmResult:
        pass


FundamentalSwarm = FundamentalTemplate
__all__ = ["FundamentalTemplate", "FundamentalSwarm"]
