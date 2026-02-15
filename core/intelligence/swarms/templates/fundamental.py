"""Fundamental Template"""

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.agent_team import AgentTeam
from ..base_swarm import BaseSwarm, SwarmBaseConfig, SwarmResult


class FundamentalTemplate(BaseSwarm):
    AGENT_TEAM = AgentTeam.define(pattern=CoordinationPattern.AUTO)
    TEMPLATE_NAME = "fundamental"

    def __init__(self, config: SwarmBaseConfig = None):
        super().__init__(config or SwarmBaseConfig(name="Fundamental", domain="fundamental"))

    async def execute(self, **kwargs) -> SwarmResult:
        pass


FundamentalSwarm = FundamentalTemplate
__all__ = ["FundamentalTemplate", "FundamentalSwarm"]
