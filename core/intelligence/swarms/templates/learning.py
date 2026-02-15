"""Learning Template"""

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.team_coordinator import TeamCoordinator
from ..swarm_learning import SwarmBaseConfig, SwarmLearning, SwarmResult


class LearningTemplate(SwarmLearning):
    AGENT_TEAM = TeamCoordinator.define(pattern=CoordinationPattern.AUTO)
    TEMPLATE_NAME = "learning"

    def __init__(self, config: SwarmBaseConfig = None) -> None:
        super().__init__(config or SwarmBaseConfig(name="Learning", domain="education"))

    async def execute(self, **kwargs) -> SwarmResult:
        pass


LearningSwarm = LearningTemplate
__all__ = ["LearningTemplate", "LearningSwarm"]
