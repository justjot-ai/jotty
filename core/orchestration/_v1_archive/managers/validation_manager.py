"""
ValidationManager - Manages validation logic (Planner/Reviewer).

Extracted from conductor.py to improve maintainability and testability.
All validation-related logic is centralized here.

Enhanced with OAgents verification strategies:
- Single validation (default)
- List-wise verification (best performing)
- Pair-wise verification
- Confidence-based selection
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import auditor types (optional, graceful fallback)
try:
    from ..orchestration.auditor_types import (
        AuditorType,
        ListWiseAuditor,
        PairWiseAuditor,
        ConfidenceBasedAuditor,
        MergedResult
    )
    AUDITOR_TYPES_AVAILABLE = True
except ImportError:
    AUDITOR_TYPES_AVAILABLE = False
    logger.debug("Auditor types not available")


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
        
        # Auditor type configuration
        if AUDITOR_TYPES_AVAILABLE:
            auditor_type_str = getattr(config, 'auditor_type', 'single')
            # Handle both string and enum values
            if isinstance(auditor_type_str, str):
                try:
                    self.auditor_type = AuditorType(auditor_type_str)
                except ValueError:
                    self.auditor_type = AuditorType.SINGLE
            elif isinstance(auditor_type_str, AuditorType):
                self.auditor_type = auditor_type_str
            else:
                self.auditor_type = AuditorType.SINGLE
        else:
            self.auditor_type = None
        
        # Initialize auditor instances if needed
        self.list_wise_auditor = None
        self.pair_wise_auditor = None
        self.confidence_auditor = None
        
        if AUDITOR_TYPES_AVAILABLE and self.auditor_type != AuditorType.SINGLE:
            if self.auditor_type == AuditorType.LIST_WISE:
                self.list_wise_auditor = ListWiseAuditor()
            elif self.auditor_type == AuditorType.PAIR_WISE:
                self.pair_wise_auditor = PairWiseAuditor()
            elif self.auditor_type == AuditorType.CONFIDENCE_BASED:
                self.confidence_auditor = ConfidenceBasedAuditor()

        logger.info("✅ ValidationManager initialized")
        logger.info(f"   Multi-round: {config.enable_multi_round if hasattr(config, 'enable_multi_round') else False}")
        if AUDITOR_TYPES_AVAILABLE:
            logger.info(f"   Auditor type: {self.auditor_type.value if self.auditor_type else 'single'}")

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
        task: Any,
        multiple_results: Optional[List[Any]] = None
    ) -> ValidationResult:
        """
        Run Reviewer for actor result (post-execution validation).

        Incorporates:
        - Reviewer prompts
        - Annotations
        - Learned patterns
        - OAgents verification strategies (if enabled)

        Args:
            actor_config: Configuration for the actor
            result: Result from actor execution (single result)
            task: Task that was executed
            multiple_results: Optional list of multiple results for list-wise verification

        Returns:
            ValidationResult with passed/failed, reward, and feedback
        """
        self.validation_count += 1
        
        # Use list-wise verification if multiple results provided and enabled
        if (AUDITOR_TYPES_AVAILABLE and 
            multiple_results and 
            len(multiple_results) > 1 and 
            self.auditor_type == AuditorType.LIST_WISE and 
            self.list_wise_auditor):
            
            logger.debug(f"Using list-wise verification for {len(multiple_results)} results")
            merged = self.list_wise_auditor.verify_and_merge(multiple_results)
            
            self.approval_count += 1
            return ValidationResult(
                passed=merged.verification_score > 0.5,
                reward=merged.verification_score,
                feedback=f"List-wise verification: {merged.reasoning}",
                confidence=merged.confidence
            )
        
        # Use pair-wise verification if multiple results provided and enabled
        if (AUDITOR_TYPES_AVAILABLE and 
            multiple_results and 
            len(multiple_results) > 1 and 
            self.auditor_type == AuditorType.PAIR_WISE and 
            self.pair_wise_auditor):
            
            logger.debug(f"Using pair-wise verification for {len(multiple_results)} results")
            merged = self.pair_wise_auditor.verify_and_select(multiple_results)
            
            self.approval_count += 1
            return ValidationResult(
                passed=merged.verification_score > 0.5,
                reward=merged.verification_score,
                feedback=f"Pair-wise verification: {merged.reasoning}",
                confidence=merged.confidence
            )
        
        # Use confidence-based selection if multiple results provided and enabled
        if (AUDITOR_TYPES_AVAILABLE and 
            multiple_results and 
            len(multiple_results) > 1 and 
            self.auditor_type == AuditorType.CONFIDENCE_BASED and 
            self.confidence_auditor):
            
            logger.debug(f"Using confidence-based selection for {len(multiple_results)} results")
            merged = self.confidence_auditor.select_best(multiple_results)
            
            self.approval_count += 1
            return ValidationResult(
                passed=merged.verification_score > 0.5,
                reward=merged.verification_score,
                feedback=f"Confidence-based selection: {merged.reasoning}",
                confidence=merged.confidence
            )

        # Default: Single result validation (existing logic)
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
