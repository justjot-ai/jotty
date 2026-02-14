"""
Validator Agent - Safety Gate for Execution Pipeline
=====================================================

Specialized agent that validates inputs/outputs against safety constraints.

Does NOT execute tasks - only validates.
Runs as a gate before/after execution to ensure:
- Inputs are safe (no malicious content, rate limits OK)
- Outputs are compliant (no PII, quality meets threshold)
- Costs are within budget

Acts as the "immune system" for the execution pipeline.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .validators import (
    SafetyConstraint,
    ValidationResult,
    ValidationReport
)

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    Specialized agent for pre/post execution validation.

    ARCHITECTURE:
    ┌──────────────────────────────────┐
    │  PRE-EXECUTION VALIDATORS        │
    │  - Cost budget check             │
    │  - Input safety scan             │
    │  - Rate limit check              │
    └────────────┬─────────────────────┘
                 │ PASS → Execute task
                 ▼
    ┌──────────────────────────────────┐
    │      TASK EXECUTION              │
    └────────────┬─────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────┐
    │  POST-EXECUTION VALIDATORS       │
    │  - PII detection                 │
    │  - Quality threshold             │
    │  - Output safety scan            │
    └────────────┬─────────────────────┘
                 │ PASS → Return result
                 ▼
    ┌──────────────────────────────────┐
    │      RETURN TO USER              │
    └──────────────────────────────────┘

    EXAMPLE USAGE:
    >>> validator = ValidatorAgent(constraints=[
    ...     PIIConstraint(),
    ...     CostBudgetConstraint(max_cost_usd=0.50),
    ...     QualityThresholdConstraint(min_quality=0.75)
    ... ])
    >>>
    >>> # Before execution
    >>> pre_report = validator.validate_pre_execution({
    ...     'user_input': 'Research AI trends',
    ...     'cost_usd': 0.0
    ... })
    >>> if not pre_report.passed:
    ...     print(f"BLOCKED: {pre_report.blocking_failures[0].message}")
    >>>
    >>> # After execution
    >>> post_report = validator.validate_post_execution({
    ...     'output': 'AI trends include...',
    ...     'quality_score': 0.85
    ... })
    >>> if not post_report.passed:
    ...     print(f"BLOCKED: {post_report.blocking_failures[0].message}")
    """

    # Pre-execution constraints (run before task starts)
    PRE_EXECUTION_CONSTRAINTS = [
        'cost_budget',
        'rate_limit',
        'malicious_input'
    ]

    # Post-execution constraints (run after task completes)
    POST_EXECUTION_CONSTRAINTS = [
        'pii_detection',
        'quality_threshold'
    ]

    def __init__(self, constraints: List[SafetyConstraint]):
        """
        Initialize validator with safety constraints.

        Args:
            constraints: List of SafetyConstraint objects to enforce
        """
        self.constraints = {c.name: c for c in constraints}
        self.validation_history: List[ValidationReport] = []
        self.max_history = 1000  # Keep last 1000 validations

        logger.info(
            f" ValidatorAgent initialized with {len(constraints)} constraints"
        )

    def validate_pre_execution(self, context: Dict[str, Any]) -> ValidationReport:
        """
        Run pre-execution validators.

        Checks:
        - Cost budget (do we have budget left?)
        - Rate limits (are we calling APIs too fast?)
        - Malicious input (is the user trying to attack us?)

        Args:
            context: Dict with 'user_input', 'cost_usd', etc.

        Returns:
            ValidationReport with pass/fail + blocking failures + warnings
        """
        results = []

        # Run only pre-execution constraints
        for constraint_name in self.PRE_EXECUTION_CONSTRAINTS:
            if constraint_name in self.constraints:
                constraint = self.constraints[constraint_name]

                if not constraint.enabled:
                    continue

                try:
                    result = constraint.validate(context)
                    results.append(result)

                    # Log blocking failures immediately
                    if not result.passed and result.severity == 'blocking':
                        logger.error(
                            f"🚨 PRE-EXECUTION BLOCKED: {constraint_name} - {result.message}"
                        )

                except Exception as e:
                    logger.exception(f"Validation error in {constraint_name}: {e}")
                    # Continue with other validators even if one fails
                    results.append(ValidationResult(
                        passed=False,
                        constraint=constraint_name,
                        message=f"Validation error: {str(e)}",
                        severity='blocking'
                    ))

        return self._generate_report(results, stage='pre_execution', context=context)

    def validate_post_execution(self, context: Dict[str, Any]) -> ValidationReport:
        """
        Run post-execution validators.

        Checks:
        - PII detection (does output contain sensitive data?)
        - Quality threshold (is output good enough?)

        Args:
            context: Dict with 'output', 'quality_score', etc.

        Returns:
            ValidationReport with pass/fail + blocking failures + warnings
        """
        results = []

        # Run only post-execution constraints
        for constraint_name in self.POST_EXECUTION_CONSTRAINTS:
            if constraint_name in self.constraints:
                constraint = self.constraints[constraint_name]

                if not constraint.enabled:
                    continue

                try:
                    result = constraint.validate(context)
                    results.append(result)

                    # Log blocking failures immediately
                    if not result.passed and result.severity == 'blocking':
                        logger.error(
                            f"🚨 POST-EXECUTION BLOCKED: {constraint_name} - {result.message}"
                        )

                except Exception as e:
                    logger.exception(f"Validation error in {constraint_name}: {e}")
                    results.append(ValidationResult(
                        passed=False,
                        constraint=constraint_name,
                        message=f"Validation error: {str(e)}",
                        severity='blocking'
                    ))

        return self._generate_report(results, stage='post_execution', context=context)

    def _generate_report(
        self,
        results: List[ValidationResult],
        stage: str,
        context: Dict[str, Any]
    ) -> ValidationReport:
        """
        Generate comprehensive validation report.

        Args:
            results: List of ValidationResult objects
            stage: 'pre_execution' or 'post_execution'
            context: Validation context (for metadata)

        Returns:
            ValidationReport with aggregated results
        """
        # Categorize results
        blocking_failures = [
            r for r in results
            if not r.passed and r.severity == 'blocking'
        ]

        warnings = [
            r for r in results
            if not r.passed and r.severity == 'warning'
        ]

        # Overall pass = no blocking failures
        passed = len(blocking_failures) == 0

        report = ValidationReport(
            stage=stage,
            passed=passed,
            blocking_failures=blocking_failures,
            warnings=warnings,
            total_checks=len(results),
            timestamp=datetime.now().isoformat(),
            metadata={
                'constraint_results': {r.constraint: r.passed for r in results}
            }
        )

        # Store in history (limit size)
        self.validation_history.append(report)
        if len(self.validation_history) > self.max_history:
            self.validation_history.pop(0)

        # Log summary
        if not passed:
            logger.error(
                f"🚨 {stage.upper()} FAILED: "
                f"{len(blocking_failures)} blocking failures, {len(warnings)} warnings"
            )
            for failure in blocking_failures:
                logger.error(f"   - {failure.constraint}: {failure.message}")
        else:
            if warnings:
                logger.warning(
                    f"⚠️  {stage.upper()} PASSED with {len(warnings)} warnings"
                )
            else:
                logger.debug(f"✅ {stage.upper()} PASSED ({len(results)} checks)")

        return report

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about validation history.

        Returns:
            Dict with pass rate, common failures, etc.
        """
        if not self.validation_history:
            return {
                'total_validations': 0,
                'pass_rate': 0.0,
                'common_failures': []
            }

        total = len(self.validation_history)
        passed_count = sum(1 for r in self.validation_history if r.passed)

        # Count failure types
        failure_counts = {}
        for report in self.validation_history:
            for failure in report.blocking_failures:
                failure_counts[failure.constraint] = \
                    failure_counts.get(failure.constraint, 0) + 1

        # Sort by frequency
        common_failures = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5

        return {
            'total_validations': total,
            'passed': passed_count,
            'failed': total - passed_count,
            'pass_rate': passed_count / total,
            'common_failures': [
                {'constraint': name, 'count': count}
                for name, count in common_failures
            ]
        }

    def enable_constraint(self, constraint_name: str):
        """Enable a specific constraint."""
        if constraint_name in self.constraints:
            self.constraints[constraint_name].enabled = True
            logger.info(f"Enabled constraint: {constraint_name}")

    def disable_constraint(self, constraint_name: str):
        """Disable a specific constraint."""
        if constraint_name in self.constraints:
            self.constraints[constraint_name].enabled = False
            logger.info(f"Disabled constraint: {constraint_name}")

    def get_enabled_constraints(self) -> List[str]:
        """Get list of enabled constraint names."""
        return [
            name for name, c in self.constraints.items()
            if c.enabled
        ]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['ValidatorAgent']
