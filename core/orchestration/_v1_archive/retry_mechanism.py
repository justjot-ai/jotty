"""
Retry Mechanism - Deduplicated Validation Retry Logic
=======================================================

Eliminates 200+ lines of duplicated retry logic between Architect and Auditor phases.

JOTTY Framework Enhancement - Fix #3
"""

import logging
from typing import Dict, Any, List, Callable, Awaitable, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_retries: int
    confidence_divisor: float = 4.0  # Divisor for confidence updates
    feedback_field_name: str = "_feedback"  # Field name for injecting feedback
    phase_name: str = "validation"  # Name for logging (e.g., "ARCHITECT", "AUDITOR")


@dataclass
class RetryResult:
    """Result from retry mechanism."""
    success: bool  # Whether validation passed
    retry_count: int  # Number of retries performed
    final_confidence: float  # Final confidence value
    trajectory_entries: List[Dict[str, Any]]  # Trajectory entries for each retry


class RetryMechanism:
    """
    Generic retry mechanism for validation loops.

    Eliminates duplication between Architect and Auditor retry logic.

    Usage:
        retry_mechanism = RetryMechanism(
            config=RetryConfig(max_retries=3, phase_name="ARCHITECT"),
            initial_confidence=0.7
        )

        result = await retry_mechanism.retry_until_valid(
            should_proceed_fn=lambda results: all(r.should_proceed for r in results),
            validate_fn=validator.validate,
            build_feedback_fn=self._build_architect_feedback,
            kwargs=kwargs,
            validate_inputs=architect_inputs,
            goal=goal
        )
    """

    def __init__(self, config: RetryConfig, initial_confidence: float):
        """
        Initialize retry mechanism.

        Args:
            config: Retry configuration
            initial_confidence: Starting confidence value
        """
        self.config = config
        self.confidence = initial_confidence
        self.trajectory = []

    async def retry_until_valid(
        self,
        should_proceed_fn: Callable[[List[Any]], bool],
        validate_fn: Callable[..., Awaitable[Tuple[List[Any], bool]]],
        build_feedback_fn: Callable[[List[Any]], str],
        kwargs: Dict[str, Any],
        validate_inputs: Dict[str, Any],
        goal: str,
        is_architect: bool = False
    ) -> RetryResult:
        """
        Execute retry loop until validation passes or max retries reached.

        Args:
            should_proceed_fn: Function to check if results indicate success
            validate_fn: Async function to call validator
            build_feedback_fn: Function to build feedback string from results
            kwargs: Keyword arguments to modify with feedback
            validate_inputs: Inputs to pass to validator
            goal: Goal string for validation
            is_architect: Whether this is architect phase (vs auditor)

        Returns:
            RetryResult with success status and metadata
        """
        retry_count = 0
        proceed = should_proceed_fn([])  # Initial check
        validation_results = []

        # Run initial validation
        validation_results, proceed = await validate_fn(
            goal=goal,
            inputs=validate_inputs,
            trajectory=[],
            is_architect=is_architect
        )

        # Retry loop
        while not proceed and retry_count < self.config.max_retries:
            retry_count += 1

            # Build detailed feedback
            feedback = build_feedback_fn(validation_results)

            logger.info(
                f"🔄 [{self.config.phase_name} RETRY] "
                f"Retry {retry_count}/{self.config.max_retries}"
            )
            logger.info(
                f"🔄 [{self.config.phase_name} RETRY] "
                f"Confidence (moving avg): {self.confidence:.3f}"
            )
            logger.info(
                f"🔄 [{self.config.phase_name} RETRY] "
                f"Feedback:\n{feedback}"
            )

            # Inject feedback into kwargs
            kwargs[self.config.feedback_field_name] = feedback
            kwargs['_retry_count'] = retry_count
            kwargs['_confidence'] = self.confidence

            if is_architect:
                kwargs['_instruction'] = (
                    "Previous attempt was blocked. "
                    "Please address the feedback and try again with improved reasoning."
                )

            # Update validation inputs with retry context
            validate_inputs['retry_feedback'] = feedback
            validate_inputs['retry_count'] = retry_count
            validate_inputs['actor_confidence'] = self.confidence
            if is_architect:
                validate_inputs['_instruction'] = kwargs.get('_instruction', '')

            # Re-run validation
            validation_results, proceed = await validate_fn(
                goal=goal,
                inputs=validate_inputs,
                trajectory=[],
                is_architect=is_architect
            )

            # Update confidence (moving average)
            if validation_results:
                current_score = sum(
                    getattr(r, 'confidence', 0.5) for r in validation_results
                ) / len(validation_results)
                confidence_delta = current_score / self.config.confidence_divisor
                self.confidence = self.confidence + confidence_delta
                logger.info(
                    f"🔄 [{self.config.phase_name} RETRY] "
                    f"Updated confidence: {self.confidence:.3f} "
                    f"(added {confidence_delta:.3f})"
                )

            logger.info(
                f"🔄 [{self.config.phase_name} RETRY] "
                f"Result: proceed={proceed}"
            )

            # Record in trajectory
            self.trajectory.append({
                'step': f'{self.config.phase_name.lower()}_retry',
                'retry_count': retry_count,
                'confidence': self.confidence,
                'proceed': proceed,
                'feedback': feedback,
                'validation_confidences': [
                    getattr(r, 'confidence', None) for r in validation_results
                ]
            })

        return RetryResult(
            success=proceed,
            retry_count=retry_count,
            final_confidence=self.confidence,
            trajectory_entries=self.trajectory
        )


def build_architect_feedback(results: List[Any]) -> str:
    """
    Build feedback string from Architect validation results.

    Args:
        results: List of validation results from Architect

    Returns:
        Formatted feedback string
    """
    feedback_parts = []
    for i, result in enumerate(results):
        feedback_parts.append(
            f"Validation Agent {i+1} ({getattr(result, 'agent_name', 'Unknown')}):\n"
            f"  - Blocked: {not getattr(result, 'should_proceed', True)}\n"
            f"  - Confidence: {getattr(result, 'confidence', 'N/A')}\n"
            f"  - Reasoning: {getattr(result, 'reasoning', 'N/A')}\n"
        )
    return "\n".join(feedback_parts)


def build_auditor_feedback(results: List[Any]) -> str:
    """
    Build feedback string from Auditor validation results.

    Args:
        results: List of validation results from Auditor

    Returns:
        Formatted feedback string
    """
    feedback_parts = []
    for result in results:
        feedback_part = (
            f"Validation Agent ({getattr(result, 'agent_name', 'Unknown')}):\n"
            f"  - Valid: {getattr(result, 'is_valid', False)}\n"
            f"  - Confidence: {getattr(result, 'confidence', 'N/A')}\n"
        )
        if hasattr(result, 'reasoning') and result.reasoning:
            feedback_part += f"  - Reasoning: {result.reasoning}\n"
        if hasattr(result, 'why_invalid') and result.why_invalid:
            feedback_part += f"  - Why Invalid: {result.why_invalid}\n"
        if hasattr(result, 'suggested_fixes') and result.suggested_fixes:
            feedback_part += f"  - Suggested Fixes: {', '.join(result.suggested_fixes)}\n"
        feedback_parts.append(feedback_part)
    return "\n".join(feedback_parts)


def update_confidence(
    current_confidence: float,
    validation_results: List[Any],
    confidence_divisor: float = 4.0
) -> float:
    """
    Update confidence using moving average formula.

    Formula: new_confidence = old_confidence + (current_score / divisor)

    Args:
        current_confidence: Current confidence value
        validation_results: List of validation results
        confidence_divisor: Divisor for confidence delta (default 4.0)

    Returns:
        Updated confidence value
    """
    if not validation_results:
        return current_confidence

    current_score = sum(
        getattr(r, 'confidence', 0.5) for r in validation_results
    ) / len(validation_results)

    confidence_delta = current_score / confidence_divisor
    new_confidence = current_confidence + confidence_delta

    logger.info(
        f"🔄 [CONFIDENCE UPDATE] "
        f"Updated: {new_confidence:.3f} "
        f"(was {current_confidence:.3f}, added {confidence_delta:.3f})"
    )

    return new_confidence


def create_retry_trajectory_entry(
    phase_name: str,
    retry_count: int,
    confidence: float,
    success: bool,
    feedback: str,
    validation_results: List[Any]
) -> Dict[str, Any]:
    """
    Create standardized trajectory entry for retry attempt.

    Args:
        phase_name: Phase name (e.g., "architect", "auditor")
        retry_count: Current retry count
        confidence: Current confidence value
        success: Whether validation passed
        feedback: Feedback string
        validation_results: List of validation results

    Returns:
        Dictionary for trajectory entry
    """
    return {
        'step': f'{phase_name}_retry',
        'retry_count': retry_count,
        'confidence': confidence,
        'success': success,
        'feedback': feedback,
        'validation_confidences': [
            getattr(r, 'confidence', None) for r in validation_results
        ]
    }
