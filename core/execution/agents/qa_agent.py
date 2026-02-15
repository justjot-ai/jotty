"""
QAAgent - Domain Agent for Test Strategy Generation

Migrated from: core/intelligence/reasoning/experts/qa_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for quality assurance and testing strategies.
Evaluates test coverage, test cases, performance testing, and quality metrics.
"""

import logging
from typing import Any, Dict, List, Optional

from Jotty.core.execution.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.execution.capabilities import LearningCapability, ValidationCapability

logger = logging.getLogger(__name__)

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class QAAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for test strategy generation with learning and validation.

    Features:
    - Test coverage (unit, integration, e2e)
    - Test cases and scenarios
    - Performance and load testing
    - Quality metrics
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize QAAgent.

        Args:
            config: Agent runtime configuration
            enable_learning: Enable gold standard learning
            strict_validation: Enable strict validation mode
        """
        # Initialize BaseAgent
        BaseAgent.__init__(
            self,
            config
            or AgentRuntimeConfig(
                name="QAAgent",
                system_prompt="You are an expert in test strategies, test cases, and quality metrics.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="quality_assurance",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="quality_assurance",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_qa,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None

        logger.info(
            "QAAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute test strategy generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (backend_architecture, previous_feedback, etc.)

        Returns:
            Generated test strategy
        """
        backend_architecture = kwargs.get("backend_architecture", "")
        previous_feedback = kwargs.get("previous_feedback", "")

        # Generate test strategy using DSPy
        strategy = await self._generate_with_dspy(
            task=task,
            backend_architecture=backend_architecture,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        # Validate strategy
        validation = await self.validate(strategy, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated strategy failed validation: {validation['errors']}")

        return strategy

    async def _generate_with_dspy(
        self, task: str, backend_architecture: str, previous_feedback: str, **kwargs
    ) -> str:
        """
        Generate test strategy using DSPy.

        Args:
            task: Task description
            backend_architecture: Backend architecture from backend developer
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Test strategy specification
        """
        # Lazy-load DSPy agent
        if self._dspy_agent is None:
            self._dspy_agent = self._create_dspy_agent()

        try:
            # Call DSPy agent
            result = self._dspy_agent(
                backend_architecture=backend_architecture,
                previous_feedback=previous_feedback,
            )

            output = result.test_strategy if hasattr(result, "test_strategy") else str(result)
            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            return "Test strategy generation failed."

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for test strategy generation."""
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for QAAgent. Install with: pip install dspy-ai")

        class TestStrategyGenerator(dspy.Signature):
            """Generate comprehensive test strategy with test cases."""

            backend_architecture: str = dspy.InputField(
                desc="Backend architecture from backend developer"
            )
            previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
            test_strategy: str = dspy.OutputField(
                desc="Test strategy with unit tests, integration tests, e2e tests, test cases"
            )

        return dspy.ChainOfThought(TestStrategyGenerator)

    async def _validate_qa(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate test strategy (domain validator for LearningCapability).

        Args:
            output: Generated strategy
            expected: Expected strategy (gold standard)
            task: Task description
            context: Additional context

        Returns:
            Validation result dict
        """
        output_str = str(output).lower()

        errors = []
        warnings = []
        score = 0.0
        issues = []

        # Check for key components
        has_unit_tests = "unit test" in output_str or "jest" in output_str or "vitest" in output_str
        has_integration = "integration test" in output_str or "api test" in output_str
        has_e2e = (
            "e2e" in output_str
            or "end-to-end" in output_str
            or "playwright" in output_str
            or "cypress" in output_str
        )
        has_test_cases = (
            "test case" in output_str or "scenario" in output_str or "given" in output_str
        )
        has_coverage = (
            "coverage" in output_str or "metric" in output_str or "threshold" in output_str
        )

        # Scoring
        if has_unit_tests:
            score += 0.25
        else:
            issues.append("Missing unit test strategy")

        if has_integration:
            score += 0.2
        else:
            issues.append("Missing integration test approach")

        if has_e2e:
            score += 0.2
        else:
            issues.append("Missing e2e test strategy")

        if has_test_cases:
            score += 0.2
        else:
            issues.append("Missing specific test cases/scenarios")

        if has_coverage:
            score += 0.15
        else:
            issues.append("Missing coverage metrics/thresholds")

        # Length check
        if len(output) < 700:
            score *= 0.8
            warnings.append("Test strategy too brief (< 700 chars)")

        # Determine validity
        valid = score >= 0.5

        if issues:
            if score < 0.7:
                errors.extend(issues)
            else:
                warnings.extend(issues)

        return {
            "valid": valid,
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "metadata": {
                "has_unit_tests": has_unit_tests,
                "has_integration": has_integration,
                "has_e2e": has_e2e,
                "has_test_cases": has_test_cases,
                "has_coverage": has_coverage,
                "issues": issues,
            },
        }

    async def _validate_impl(
        self,
        output: Any,
        expected: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Domain-specific validation (ValidationCapability interface).

        Args:
            output: Output to validate
            expected: Expected output
            context: Additional context
            **kwargs: Additional parameters

        Returns:
            Validation result
        """
        return await self._validate_qa(
            output=output,
            expected=expected,
            task=context.get("task", "") if context else "",
            context=context or {},
        )

    # =========================================================================
    # TRAINING AND VALIDATION DATA
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for test strategy."""
        return []

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for test strategy."""
        return []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_test_strategy(
        self, backend_architecture: str = "", previous_feedback: str = "", **kwargs: Any
    ) -> str:
        """
        Generate test strategy.

        Args:
            backend_architecture: Backend architecture from backend developer
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Test strategy specification as string
        """
        result = await self.execute(
            task="Generate test strategy",
            backend_architecture=backend_architecture,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        if result.success:
            return result.output
        else:
            raise ValueError(f"Test strategy generation failed: {result.error}")
