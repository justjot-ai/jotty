"""
ValidationCapability - Mixin for Domain-Specific Validation

Adds validation capabilities to any agent, swarm, or workflow:
- Domain-specific output validation
- Syntax checking
- Semantic validation
- Quality metrics

Usage:
    from Jotty.core.execution.base import BaseAgent
    from Jotty.core.execution.capabilities import ValidationCapability

    class MermaidAgent(BaseAgent, ValidationCapability):
        def __init__(self, config=None):
            BaseAgent.__init__(self, config)
            ValidationCapability.__init__(self, domain="mermaid")

        async def _execute_impl(self, task: str, **kwargs):
            result = await self._generate_diagram(task)

            # Validate output
            validation = await self.validate(result)
            if not validation["valid"]:
                # Fix and retry
                result = await self._fix_errors(result, validation["errors"])

            return result

        async def _validate_impl(self, output: Any, **kwargs) -> Dict[str, Any]:
            # Domain-specific validation logic
            errors = []
            if not output.startswith("graph"):
                errors.append("Mermaid diagram must start with 'graph'")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "score": 1.0 if len(errors) == 0 else 0.5
            }
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationCapability:
    """
    Mixin that adds domain-specific validation to any execution pattern.

    Provides:
    - Standard validation interface
    - Error collection and reporting
    - Quality scoring
    - Validation history tracking

    Subclasses implement _validate_impl() for domain-specific logic.
    """

    def __init__(
        self,
        domain: str,
        strict_mode: bool = False,
        quality_threshold: float = 0.7,
    ) -> None:
        """
        Initialize validation capability.

        Args:
            domain: Domain name (e.g., "mermaid", "latex", "python")
            strict_mode: If True, fail on any validation error
            quality_threshold: Minimum quality score (0.0-1.0) to pass
        """
        self.validation_domain = domain
        self.strict_mode = strict_mode
        self.quality_threshold = quality_threshold

        # Validation statistics
        self._validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "average_score": 0.0,
        }

        # Validation history (last 100)
        self._validation_history: List[Dict[str, Any]] = []
        self._max_history = 100

        logger.info(
            f"ValidationCapability initialized for domain '{domain}' "
            f"(strict={strict_mode}, threshold={quality_threshold})"
        )

    async def validate(
        self,
        output: Any,
        expected: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate output against domain criteria.

        Args:
            output: Output to validate
            expected: Expected output for comparison (optional)
            context: Additional validation context
            **kwargs: Domain-specific validation parameters

        Returns:
            Validation result:
            {
                "valid": bool,
                "score": float (0.0-1.0),
                "errors": List[str],
                "warnings": List[str],
                "metadata": Dict[str, Any]
            }
        """
        self._validation_stats["total_validations"] += 1

        try:
            # Call domain-specific validation
            result = await self._validate_impl(
                output=output, expected=expected, context=context or {}, **kwargs
            )

            # Ensure result has required fields
            result.setdefault("valid", True)
            result.setdefault("score", 1.0)
            result.setdefault("errors", [])
            result.setdefault("warnings", [])
            result.setdefault("metadata", {})

            # Apply quality threshold
            if result["score"] < self.quality_threshold:
                result["valid"] = False
                result["errors"].append(
                    f"Quality score {result['score']:.2f} below threshold {self.quality_threshold}"
                )

            # Apply strict mode
            if self.strict_mode and result["errors"]:
                result["valid"] = False

            # Update statistics
            if result["valid"]:
                self._validation_stats["passed"] += 1
            else:
                self._validation_stats["failed"] += 1

            # Update average score
            total = self._validation_stats["total_validations"]
            old_avg = self._validation_stats["average_score"]
            self._validation_stats["average_score"] = (
                old_avg * (total - 1) + result["score"]
            ) / total

            # Add to history
            self._add_to_history(result)

            return result

        except Exception as e:
            logger.error(f"Validation error in {self.validation_domain}: {e}")
            self._validation_stats["failed"] += 1

            error_result = {
                "valid": False,
                "score": 0.0,
                "errors": [f"Validation exception: {str(e)}"],
                "warnings": [],
                "metadata": {"exception": str(e)},
            }

            self._add_to_history(error_result)
            return error_result

    async def _validate_impl(
        self,
        output: Any,
        expected: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Domain-specific validation logic.

        Subclasses should override this method to implement
        domain-specific validation (syntax checking, semantic
        validation, quality metrics, etc.).

        Args:
            output: Output to validate
            expected: Expected output (optional)
            context: Additional context
            **kwargs: Domain-specific parameters

        Returns:
            Validation result dict
        """
        # Default implementation: always valid
        return {
            "valid": True,
            "score": 1.0,
            "errors": [],
            "warnings": ["No domain-specific validation implemented"],
            "metadata": {},
        }

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._validation_stats.copy()

    def get_validation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get validation history.

        Args:
            limit: Maximum number of results (default: all)

        Returns:
            List of validation results (newest first)
        """
        history = list(reversed(self._validation_history))
        if limit:
            return history[:limit]
        return history

    def clear_history(self) -> None:
        """Clear validation history."""
        self._validation_history.clear()
        logger.debug(f"Cleared validation history for {self.validation_domain}")

    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """Add validation result to history."""
        self._validation_history.append(result)

        # Keep history size limited
        if len(self._validation_history) > self._max_history:
            self._validation_history.pop(0)


class SyntaxValidator(ValidationCapability):
    """
    Specialized validator for syntax checking.

    Can be used standalone or as part of a validation pipeline.
    """

    def __init__(self, domain: str, syntax_checker: Optional[Callable] = None, **kwargs):
        """
        Initialize syntax validator.

        Args:
            domain: Domain name
            syntax_checker: Function that checks syntax (returns errors list)
            **kwargs: Additional ValidationCapability parameters
        """
        super().__init__(domain=domain, **kwargs)
        self.syntax_checker = syntax_checker

    async def _validate_impl(self, output: Any, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Validate syntax."""
        if not self.syntax_checker:
            return {
                "valid": True,
                "score": 1.0,
                "errors": [],
                "warnings": ["No syntax checker configured"],
            }

        try:
            errors = self.syntax_checker(output)  # type: ignore[misc]

            return {
                "valid": len(errors) == 0,
                "score": 1.0 if len(errors) == 0 else 0.0,
                "errors": errors,
                "warnings": [],
                "metadata": {"syntax_check": "completed"},
            }

        except Exception as e:
            return {
                "valid": False,
                "score": 0.0,
                "errors": [f"Syntax check failed: {str(e)}"],
                "warnings": [],
                "metadata": {"exception": str(e)},
            }
