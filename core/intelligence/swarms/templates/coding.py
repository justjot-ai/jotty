"""
Coding Template - Complex Multi-Stage Workflow
===============================================

Example of complex template using CUSTOM pattern with STAGES.
Shows declarative multi-stage workflow with dependencies.

Author: Jotty Team
Date: February 2026
"""

from typing import Any

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base_swarm import BaseSwarm, SwarmBaseConfig, SwarmResult
from ..stage_config import StageConfig

# Import existing agents (will be defined/imported from original swarm)
try:
    from ..coding_swarm import (
        ArchitectAgent,
        CodingConfig,
        CodingResult,
        DeveloperAgent,
        TestWriterAgent,
    )
except ImportError:
    # Fallback
    from Jotty.core.modes.agent.base import BaseSwarmAgent as ArchitectAgent
    from Jotty.core.modes.agent.base import BaseSwarmAgent as DeveloperAgent
    from Jotty.core.modes.agent.base import BaseSwarmAgent as TestWriterAgent

    CodingConfig = SwarmBaseConfig
    CodingResult = SwarmResult

from ..base.agent_team import AgentTeam


class CodingTemplate(BaseSwarm):
    """
    Coding swarm template - CUSTOM pattern with STAGES.

    Multi-stage workflow:
    1. Design (Architect)
    2. Implement (Developer) - needs Design
    3. Test (TestWriter) - needs Implement

    Inherits ALL learning layers from BaseSwarm automatically.

    Usage:
        swarm = CodingTemplate()
        result = await swarm.execute(requirements="Build a REST API")
    """

    # Agent team definition
    AGENT_TEAM = AgentTeam.define(
        (ArchitectAgent, "Architect", "_architect"),
        (DeveloperAgent, "Developer", "_developer"),
        (TestWriterAgent, "TestWriter", "_test_writer"),
        pattern=CoordinationPattern.CUSTOM,  # Multi-stage with dependencies
        # STAGES defined below
    )

    # Multi-stage workflow configuration
    STAGES = [
        StageConfig(
            name="design",
            agents=["_architect"],
            description="Design system architecture",
            output_key="architecture",
        ),
        StageConfig(
            name="implement",
            agents=["_developer"],
            needs=["design"],  # Waits for design stage
            description="Implement code based on architecture",
            output_key="code",
        ),
        StageConfig(
            name="test",
            agents=["_test_writer"],
            needs=["implement"],  # Waits for implement stage
            description="Write comprehensive tests",
            output_key="tests",
        ),
    ]

    # Template metadata
    TEMPLATE_NAME = "coding"
    TEMPLATE_VERSION = "2.0.0"
    RESULT_CLASS = CodingResult

    def __init__(self, config: SwarmBaseConfig = None):
        """Initialize coding template."""
        super().__init__(config or CodingConfig())
        # Wire STAGES into agent team
        self.AGENT_TEAM.stages = self.STAGES

    async def execute(self, requirements: str = None, **kwargs: Any) -> SwarmResult:
        """
        Execute coding workflow.

        Args:
            requirements: What to build
            **kwargs: Additional arguments

        Returns:
            CodingResult with architecture, code, and tests
        """
        # Initialize agents
        self._init_agents()

        # Pre-execution learning
        await self._pre_execute_learning()

        # Execute team with CUSTOM pattern (uses STAGES)
        # STAGES are automatically executed in topological order
        team_result = await self.execute_team(task={"requirements": requirements}, context=kwargs)

        # Build result from stage outputs
        result = self.RESULT_CLASS(
            success=team_result.success,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                "architecture": team_result.outputs.get("_architect", ""),
                "code": team_result.outputs.get("_developer", ""),
                "tests": team_result.outputs.get("_test_writer", ""),
                "metadata": team_result.metadata,
            },
            execution_time=0.0,
        )

        # Post-execution learning
        await self._post_execute_learning(
            success=result.success,
            execution_time=0.0,
            tools_used=["architecture_design", "code_generation", "test_generation"],
            task_type="coding",
            output_data={"stages_completed": team_result.metadata.get("stages", [])},
            input_data={"requirements_length": len(requirements or "")},
        )

        return result


# Backward compatibility
CodingSwarm = CodingTemplate

__all__ = ["CodingTemplate", "CodingSwarm"]
