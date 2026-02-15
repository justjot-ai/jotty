"""
ML Template - Iterative Improvement Pattern
============================================

Example of template using ITERATIVE pattern.
Keeps improving model until quality threshold met.

Author: Jotty Team
Date: February 2026
"""

from typing import Any

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from ..base_swarm import BaseSwarm, SwarmBaseConfig, SwarmResult

# Import existing agents (will be defined/imported from SwarmML)
try:
    from ...orchestration.templates.swarm_ml import (
        DataPreprocessorAgent,
        FeatureEngineerAgent,
        MLConfig,
        MLResult,
        ModelTrainerAgent,
    )
except ImportError:
    # Fallback
    from Jotty.core.modes.agent.base import BaseSwarmAgent as DataPreprocessorAgent
    from Jotty.core.modes.agent.base import BaseSwarmAgent as FeatureEngineerAgent
    from Jotty.core.modes.agent.base import BaseSwarmAgent as ModelTrainerAgent

    MLConfig = SwarmBaseConfig
    MLResult = SwarmResult

from ..base.agent_team import AgentTeam


class MLTemplate(BaseSwarm):
    """
    ML swarm template - ITERATIVE pattern.

    Iterative improvement loop:
    1. Preprocess data + engineer features + train model
    2. Evaluate model quality
    3. If quality < 0.85: improve and repeat (max 5 iterations)
    4. Return best model across all iterations

    Inherits ALL learning layers from BaseSwarm automatically.

    Usage:
        swarm = MLTemplate()
        result = await swarm.execute(
            data="data.csv",
            target="price",
            quality_threshold=0.85
        )
    """

    # Agent team definition - ITERATIVE pattern
    AGENT_TEAM = AgentTeam.define(
        (DataPreprocessorAgent, "DataPreprocessor"),
        (FeatureEngineerAgent, "FeatureEngineer"),
        (ModelTrainerAgent, "ModelTrainer"),
        pattern=CoordinationPattern.ITERATIVE,  # Feedback loop until quality met
        quality_threshold=0.85,  # Stop when model score >= 0.85
        max_iterations=5,  # Maximum improvement rounds
    )

    # Template metadata
    TEMPLATE_NAME = "ml"
    TEMPLATE_VERSION = "2.0.0"
    RESULT_CLASS = MLResult

    def __init__(self, config: SwarmBaseConfig = None):
        """Initialize ML template."""
        super().__init__(config or MLConfig())

    async def execute(
        self,
        data: str = None,
        target: str = None,
        quality_threshold: float = 0.85,
        **kwargs: Any,
    ) -> SwarmResult:
        """
        Execute ML training with iterative improvement.

        Args:
            data: Path to training data
            target: Target column name
            quality_threshold: Quality score to achieve (0-1)
            **kwargs: Additional arguments

        Returns:
            MLResult with best model across all iterations
        """
        # Initialize agents
        self._init_agents()

        # Override quality threshold if specified
        if quality_threshold:
            self.AGENT_TEAM.quality_threshold = quality_threshold

        # Pre-execution learning
        await self._pre_execute_learning()

        # Execute team with ITERATIVE pattern
        # Automatically loops until quality_threshold met or max_iterations reached
        team_result = await self.execute_team(task={"data": data, "target": target}, context=kwargs)

        # Build result
        result = self.RESULT_CLASS(
            success=team_result.success,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={
                "best_model": team_result.merged_output,  # Best iteration
                "iterations": team_result.metadata.get("iterations", 1),
                "best_quality": team_result.metadata.get("best_quality", 0.0),
                "converged": team_result.metadata.get("converged", False),
            },
            execution_time=0.0,
        )

        # Post-execution learning
        await self._post_execute_learning(
            success=result.success,
            execution_time=0.0,
            tools_used=["data_preprocessing", "feature_engineering", "model_training"],
            task_type="ml_training",
            output_data={
                "final_quality": team_result.metadata.get("best_quality", 0.0),
                "iterations_used": team_result.metadata.get("iterations", 1),
            },
            input_data={"target": target, "quality_threshold": quality_threshold},
        )

        return result


# Backward compatibility
SwarmML = MLTemplate

__all__ = ["MLTemplate", "SwarmML"]
