"""
Coding Template - Multi-Stage Software Development

Demonstrates CUSTOM pattern with STAGES for complex coding workflows.

Architecture:
    Stage 1: Design (Architect) - Creates system architecture
    Stage 2: Implement (Developer) - Generates code based on design
    Stage 3: Test (TestWriter) - Creates comprehensive tests

Example:
    from Jotty.core.intelligence.orchestration.swarms.templates.coding import CodingTemplate

    template = CodingTemplate()
    result = await template.execute("Build a REST API for user management")
"""

from typing import Any, Optional

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from .._base.stage_config import StageConfig
from ..base.swarm_template import SwarmTemplate
from ..base.team_coordinator import TeamCoordinator

# Import agents from coding_swarm
try:
    from ..coding_swarm import (
        ArchitectAgent,
        DeveloperAgent,
        TestWriterAgent,
    )
    from ..coding_swarm.types import CodingConfig, CodingResult
except ImportError:
    # Fallback
    from Jotty.core.intelligence.reasoning.agents.swarm_agent import SwarmLearningAgent as BaseAgent

    ArchitectAgent = BaseAgent
    DeveloperAgent = BaseAgent
    TestWriterAgent = BaseAgent

    from .._base.swarm_learning import SwarmBaseConfig as CodingConfig  # type: ignore[assignment]
    from .._base.swarm_learning import SwarmResult as CodingResult  # type: ignore[assignment]


class CodingTemplate(SwarmTemplate):
    """
    Coding swarm template with CUSTOM pattern and STAGES.

    Multi-stage workflow:
    1. Design (Architect) - System architecture
    2. Implement (Developer) - Code generation
    3. Test (TestWriter) - Test suite creation
    """

    # Multi-stage workflow definition
    STAGES = [
        StageConfig(
            name="design",
            agents=["_architect"],
            parallel=False,
            output_key="architecture",
        ),
        StageConfig(
            name="implement",
            agents=["_developer"],
            needs=["design"],  # Waits for design
            parallel=False,
            output_key="code",
        ),
        StageConfig(
            name="test",
            agents=["_test_writer"],
            needs=["implement"],  # Waits for code
            parallel=False,
            output_key="tests",
        ),
    ]

    # Agent team definition
    AGENT_TEAM = TeamCoordinator.define(
        (ArchitectAgent, "Architect", "_architect"),
        (DeveloperAgent, "Developer", "_developer"),
        (TestWriterAgent, "TestWriter", "_test_writer"),
        pattern=CoordinationPattern.CUSTOM,
        stages=STAGES,
    )

    TASK_TYPE = "coding"
    DEFAULT_TOOLS = ["design", "code_generation", "test_generation"]

    def __init__(self, config: Optional[CodingConfig] = None) -> None:
        """Initialize coding template."""
        super().__init__(config or CodingConfig())
        self._initialized = False

    async def _execute_domain(self, query: str, **kwargs: Any) -> CodingResult:
        """
        Execute coding workflow (called by SwarmTemplate.execute()).

        Args:
            query: What to build (requirements)
            **kwargs: Additional arguments (language, style, framework, etc.)

        Returns:
            CodingResult with architecture, code, and tests
        """
        # Prepare context for stage execution
        context = {
            "query": query,
            "requirements": query,
            "language": kwargs.get("language", "Python"),
            "style": kwargs.get("style", "clean"),
            "framework": kwargs.get("framework", "pytest"),
            "config": self.config,
        }

        # Execute using CUSTOM pattern with STAGES
        result = await self.execute_team(
            task=f"coding: {query}",
            context=context,
            tools_used=self.DEFAULT_TOOLS,
        )

        # Convert team result to CodingResult
        return self._build_result(result, context)

    def _build_result(self, team_result: Any, context: dict) -> CodingResult:
        """Build CodingResult from team execution."""
        outputs = team_result.outputs if hasattr(team_result, "outputs") else {}

        # Extract outputs from each stage
        architecture = outputs.get("_architect", {}) or {}
        code = outputs.get("_developer", {}) or {}
        tests = outputs.get("_test_writer", {}) or {}

        return CodingResult(
            success=team_result.success if hasattr(team_result, "success") else True,
            swarm_name="CodingTemplate",
            domain="coding",
            output={
                "architecture": architecture.get("architecture", ""),
                "components": architecture.get("components", []),
                "code": code.get("code", ""),
                "filename": code.get("filename", ""),
                "tests": tests.get("tests", ""),
                "test_framework": tests.get("framework", "pytest"),
                "metadata": team_result.metadata if hasattr(team_result, "metadata") else {},
            },
            execution_time=getattr(team_result, "execution_time", 0.0),
        )


# Backward compatibility
CodingSwarm = CodingTemplate

__all__ = ["CodingTemplate", "CodingSwarm"]
