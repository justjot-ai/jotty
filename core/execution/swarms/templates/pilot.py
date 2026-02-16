"""
Pilot Template - Learning-based workflow

Demonstrates learning template pattern.
"""

from typing import Any, Optional

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base.swarm_template import SwarmTemplate
from ..base.team_coordinator import TeamCoordinator

# Import from existing swarm
try:
    from ..pilot_swarm import PilotSwarm

    Config = getattr(PilotSwarm, "__config__", None)
    Result = getattr(PilotSwarm, "__result__", None)
except (ImportError, AttributeError):
    from .._base.swarm_learning import SwarmBaseConfig as Config
    from .._base.swarm_learning import SwarmResult as Result

PilotConfig = Config or type("Config", (), {})
PilotResult = Result or type("Result", (), {})


class PilotTemplate(SwarmTemplate):
    """
    Pilot template - delegates to existing swarm.

    This is a thin wrapper around the existing pilot_swarm.
    """

    # Use SEQUENTIAL pattern for simple delegation
    AGENT_TEAM = TeamCoordinator.define(
        pattern=CoordinationPattern.SEQUENTIAL,
    )

    TASK_TYPE = "pilot"

    def __init__(self, config: Optional[PilotConfig] = None) -> None:
        """Initialize pilot template."""
        super().__init__(config or PilotConfig())
        self._swarm = None

    async def _execute_domain(self, query: str, **kwargs: Any) -> PilotResult:
        """
        Execute using existing swarm (called by SwarmTemplate.execute()).

        Args:
            query: Task description
            **kwargs: Additional arguments

        Returns:
            PilotResult with execution results
        """
        # Lazy-load existing swarm
        if self._swarm is None:
            try:
                from ..pilot_swarm import PilotSwarm

                self._swarm = PilotSwarm(self.config)
            except ImportError:
                return PilotResult(
                    success=False,
                    output={"error": "pilot_swarm not available"},
                    execution_time=0.0,
                )

        # Delegate to existing swarm
        try:
            result = await self._swarm.execute(query, **kwargs)
            return result
        except Exception as e:
            return PilotResult(
                success=False,
                output={"error": str(e)},
                execution_time=0.0,
            )


# Backward compatibility
PilotSwarm = PilotTemplate

__all__ = ["PilotTemplate", "PilotSwarm"]
