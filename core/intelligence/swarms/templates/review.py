"""
Review Template - Simple Parallel Review Swarm
===============================================

Example of simple template using PARALLEL pattern.
All 8 learning layers come from SwarmLearning automatically.

Author: Jotty Team
Date: February 2026
"""

from typing import Any

from Jotty.core.infrastructure.foundation.types.execution_types import (
    CoordinationPattern,
    MergeStrategy,
    SynthesisStrategy,
)

from ..swarm_learning import SwarmBaseConfig, SwarmLearning, SwarmResult

# Import existing agents (will be defined/imported from original swarm)
# For now, using placeholder - in real implementation, import from review_swarm
try:
    from ..review_swarm import (
        CodeReviewer,
        PerformanceAnalyzer,
        ReviewConfig,
        ReviewResult,
        SecurityScanner,
    )
except ImportError:
    # Fallback if not available yet
    from Jotty.core.modes.agent.base import SwarmLearningAgent as CodeReviewer
    from Jotty.core.modes.agent.base import SwarmLearningAgent as PerformanceAnalyzer
    from Jotty.core.modes.agent.base import SwarmLearningAgent as SecurityScanner

    ReviewConfig = SwarmBaseConfig
    ReviewResult = SwarmResult

from ..base.team_coordinator import TeamCoordinator


class ReviewTemplate(SwarmLearning):
    """
    Code review swarm template - PARALLEL pattern.

    Template (not base class) - inherits ALL learning from SwarmLearning:
    1. ✅ Memory (5 levels: EPISODIC, SEMANTIC, PROCEDURAL, META, CAUSAL)
    2. ✅ TD-Lambda reinforcement learning
    3. ✅ Swarm Intelligence meta-learning
    4. ✅ Gold Standard evaluation
    5. ✅ Improvement Agents (Expert, Reviewer, Planner, Actor, Auditor, Learner)
    6. ✅ Pattern Learning
    7. ✅ Transfer Learning
    8. ✅ Adaptive Components

    Usage:
        swarm = ReviewTemplate()
        result = await swarm.execute(code="def foo(): pass", language="python")
    """

    # Agent team definition - PARALLEL execution
    AGENT_TEAM = TeamCoordinator.define(
        (CodeReviewer, "CodeReviewer"),
        (SecurityScanner, "SecurityScanner"),
        (PerformanceAnalyzer, "PerformanceAnalyzer"),
        pattern=CoordinationPattern.PARALLEL,  # All reviewers run concurrently
        merge_strategy=MergeStrategy.COMBINE,  # Combine all outputs
        synthesis_strategy=SynthesisStrategy.SYNTHESIZE,  # LLM synthesizes final report
    )

    # Template metadata
    TEMPLATE_NAME = "review"
    TEMPLATE_VERSION = "2.0.0"
    RESULT_CLASS = ReviewResult

    def __init__(self, config: SwarmBaseConfig = None):
        """Initialize review template."""
        super().__init__(config or ReviewConfig())

    async def execute(
        self, code: str = None, language: str = "python", **kwargs: Any
    ) -> SwarmResult:
        """
        Execute code review.

        Args:
            code: Code to review
            language: Programming language
            **kwargs: Additional arguments

        Returns:
            ReviewResult with synthesis of all reviewer outputs
        """
        # Initialize agents
        self._init_agents()

        # Pre-execution learning
        await self._pre_execute_learning()

        # Execute team with PARALLEL pattern
        # SwarmLearning automatically uses synthesis_strategy=SYNTHESIZE
        # to intelligently combine all reviewer outputs
        team_result = await self.execute_team(
            task={"code": code, "language": language}, context=kwargs
        )

        # Build result
        result = self.RESULT_CLASS(
            success=team_result.success,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output=team_result.merged_output,  # Synthesized review
            execution_time=0.0,  # Would track in real implementation
        )

        # Post-execution learning
        await self._post_execute_learning(
            success=result.success,
            execution_time=0.0,
            tools_used=["code_review", "security_scan", "performance_analysis"],
            task_type="code_review",
            output_data={"synthesized_review": str(team_result.merged_output)[:500]},
            input_data={"code_length": len(code or ""), "language": language},
        )

        return result


# Backward compatibility
ReviewSwarm = ReviewTemplate

__all__ = ["ReviewTemplate", "ReviewSwarm"]
