"""
PerspectiveLearning Template - Learning-based workflow

Demonstrates learning template pattern.
"""

from typing import Any, Optional

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.swarm_template import SwarmTemplate
from ..base.team_coordinator import TeamCoordinator

# Import from existing swarm
try:
    from ..perspective_learning_swarm import PerspectiveLearningSwarm

    Config = getattr(PerspectiveLearningSwarm, "__config__", None)
    Result = getattr(PerspectiveLearningSwarm, "__result__", None)
except (ImportError, AttributeError):
    from .._base.swarm_learning import SwarmBaseConfig as Config
    from .._base.swarm_learning import SwarmResult as Result

PerspectiveLearningConfig = Config or type("Config", (), {})
PerspectiveLearningResult = Result or type("Result", (), {})


class PerspectiveLearningTemplate(SwarmTemplate):
    """
    PerspectiveLearning template - delegates to existing swarm.

    This is a thin wrapper around the existing perspective_learning_swarm.
    """

    # Use SEQUENTIAL pattern for simple delegation
    AGENT_TEAM = TeamCoordinator.define(
        pattern=CoordinationPattern.SEQUENTIAL,
    )

    TASK_TYPE = "perspective_learning"

    def __init__(self, config: Optional[PerspectiveLearningConfig] = None) -> None:
        """Initialize perspective_learning template."""
        super().__init__(config or PerspectiveLearningConfig())
        self._swarm = None

    async def _execute_domain(self, query: str, **kwargs: Any) -> PerspectiveLearningResult:
        """
        Execute using existing swarm (called by SwarmTemplate.execute()).

        Args:
            query: Task description
            **kwargs: Additional arguments

        Returns:
            PerspectiveLearningResult with execution results
        """
        # Lazy-load existing swarm
        if self._swarm is None:
            try:
                from ..perspective_learning_swarm import PerspectiveLearningSwarm

                self._swarm = PerspectiveLearningSwarm(self.config)
            except ImportError:
                return PerspectiveLearningResult(
                    success=False,
                    output={"error": "perspective_learning_swarm not available"},
                    execution_time=0.0,
                )

        # Delegate to existing swarm
        try:
            result = await self._swarm.execute(query, **kwargs)
            return result
        except Exception as e:
            return PerspectiveLearningResult(
                success=False,
                output={"error": str(e)},
                execution_time=0.0,
            )


# Backward compatibility
PerspectiveLearningSwarm = PerspectiveLearningTemplate

__all__ = ["PerspectiveLearningTemplate", "PerspectiveLearningSwarm"]
