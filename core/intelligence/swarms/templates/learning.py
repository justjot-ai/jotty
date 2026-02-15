"""Learning Template"""

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.agent_team import AgentTeam
from ..base_swarm import BaseSwarm, SwarmBaseConfig, SwarmResult


class LearningTemplate(BaseSwarm):
    AGENT_TEAM = AgentTeam.define(pattern=CoordinationPattern.AUTO)
    TEMPLATE_NAME = "learning"

    def __init__(self, config: SwarmBaseConfig = None):
        super().__init__(config or SwarmBaseConfig(name="Learning", domain="education"))

    async def execute(self, **kwargs) -> SwarmResult:
        pass


LearningSwarm = LearningTemplate
__all__ = ["LearningTemplate", "LearningSwarm"]
