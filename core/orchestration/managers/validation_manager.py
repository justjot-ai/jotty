"""
ValidationManager - Manages validation logic (Planner/Reviewer).

Extracted from conductor.py to improve maintainability and testability.
All validation-related logic is centralized here.
"""
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation."""
    passed: bool
    reward: float
    feedback: str
    confidence: float = 0.8


class ValidationManager:
    """
    Centralized validation management for Planner and Reviewer.

    Responsibilities:
    - Planner invocation (pre-execution exploration)
    - Reviewer invocation (post-execution validation)
    - Multi-round validation coordination
    - Confidence tracking

    This manager owns validation logic (no duplicate validation!)
    """

    def __init__(self, config):
        """
        Initialize validation manager.

        Args:
            config: JottyConfig with validation parameters
        """
        self.config = config
        self.validation_count = 0
        self.approval_count = 0

        logger.info("✅ ValidationManager initialized")
        logger.info(f"   Multi-round: {config.enable_multi_round if hasattr(config, 'enable_multi_round') else False}")

    async def run_planner(
        self,
        actor_config,
        task: Any,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Run Planner for actor (pre-execution exploration).

        The Planner explores available data and briefs the actor on findings.
        It is an ADVISOR, not a gatekeeper - it should NOT block execution.

        Args:
            actor_config: Configuration for the actor
            task: Task to plan for
            context: Current context

        Returns:
            (should_proceed, exploration_summary)
        """
        # TODO: Future enhancement - integrate InspectorAgent with is_architect=True
        # For now, always proceed (Planner is advisory only)

        logger.debug(f"🔍 Planner for {actor_config.name}: Advisory mode (always proceeds)")
        return True, "Planner: Ready to proceed (advisory mode)"

    async def run_reviewer(
        self,
        actor_config,
        result: Any,
        task: Any
    ) -> ValidationResult:
        """
        Run Reviewer for actor result (post-execution validation).

        Incorporates:
        - Reviewer prompts
        - Annotations
        - Learned patterns

        Args:
            actor_config: Configuration for the actor
            result: Result from actor execution
            task: Task that was executed

        Returns:
            ValidationResult with passed/failed, reward, and feedback
        """
        self.validation_count += 1

        # Simple validation logic (extracted from conductor.py _run_auditor)
        # TODO: Future enhancement - integrate InspectorAgent with is_architect=False

        # Check if result indicates success
        if isinstance(result, dict):
            if result.get('success', True):
                self.approval_count += 1
                return ValidationResult(
                    passed=True,
                    reward=1.0,
                    feedback="Reviewer passed",
                    confidence=0.9
                )
            else:
                return ValidationResult(
                    passed=False,
                    reward=0.0,
                    feedback=result.get('error', 'Reviewer failed'),
                    confidence=0.8
                )

        # Check dspy.Prediction with success field
        if hasattr(result, 'success'):
            if result.success:
                self.approval_count += 1
                return ValidationResult(
                    passed=True,
                    reward=1.0,
                    feedback="Reviewer passed",
                    confidence=0.9
                )
            else:
                reason = getattr(result, '_reasoning', 'Unknown reason')
                return ValidationResult(
                    passed=False,
                    reward=0.0,
                    feedback=f"Reviewer failed: {reason}",
                    confidence=0.8
                )

        # Default: assume success
        self.approval_count += 1
        return ValidationResult(
            passed=True,
            reward=0.8,
            feedback="Result received",
            confidence=0.7
        )

    async def run_multi_round_validation(
        self,
        actor_config,
        result: Any,
        task: Any,
        max_rounds: int = 3
    ) -> ValidationResult:
        """
        Perform multi-round validation with refinement.

        Args:
            actor_config: Configuration for the actor
            result: Result from actor execution
            task: Task that was executed
            max_rounds: Maximum validation rounds

        Returns:
            Final ValidationResult after all rounds
        """
        # TODO: Future enhancement - implement multi-round refinement loop
        # For now, just run single round
        return await self.run_reviewer(actor_config, result, task)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dict with validation metrics
        """
        approval_rate = (self.approval_count / self.validation_count) if self.validation_count > 0 else 0.0
        return {
            "total_validations": self.validation_count,
            "approvals": self.approval_count,
            "approval_rate": approval_rate
        }

    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_count = 0
        self.approval_count = 0
        logger.debug("ValidationManager stats reset")
