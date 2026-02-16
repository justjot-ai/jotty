"""Sequential Team Pattern Template"""

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..._base.swarm_learning import SwarmBaseConfig, SwarmLearning, SwarmResult
from ...base.team_coordinator import TeamCoordinator


class SequentialTemplate(SwarmLearning):
    """Sequential team pattern - agents work in pipeline order."""

    AGENT_TEAM = TeamCoordinator.define(pattern=CoordinationPattern.SEQUENTIAL)
    TEMPLATE_NAME = "sequential_team"

    def __init__(self, config: SwarmBaseConfig = None) -> None:
        super().__init__(config or SwarmBaseConfig(name="SequentialTeam", domain="pipeline"))

    async def execute(self, **kwargs) -> SwarmResult:
        pass


SequentialTeam = SequentialTemplate
__all__ = ["SequentialTemplate", "SequentialTeam"]
