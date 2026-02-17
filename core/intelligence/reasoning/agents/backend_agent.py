"""
BackendAgent - Domain Agent for Backend Architecture Generation

Migrated from: core/intelligence/reasoning/experts/backend_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for backend development and API design.
Evaluates backend architecture for API design, data models, authentication, and performance.
"""

import logging
from typing import Any, Dict, List, Optional

from Jotty.core.intelligence.reasoning.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.intelligence.reasoning.capabilities import (  # type: ignore[import-not-found, import]
    LearningCapability,
    ValidationCapability,
)

logger = logging.getLogger(__name__)

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class BackendAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for backend architecture generation with learning and validation.

    Features:
    - API design evaluation (REST/GraphQL)
    - Data models and database schema
    - Authentication and authorization
    - Performance and scalability assessment
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize BackendAgent.

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
                name="BackendAgent",
                system_prompt="You are an expert in API design, database schema, and backend architecture.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="backend_development",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="backend_development",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_backend,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None

        logger.info(
            "BackendAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute backend architecture generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (frontend_architecture, previous_feedback, etc.)

        Returns:
            Generated backend architecture
        """
        frontend_architecture = kwargs.get("frontend_architecture", "")
        previous_feedback = kwargs.get("previous_feedback", "")

        # Generate architecture using DSPy
        architecture = await self._generate_with_dspy(
            task=task,
            frontend_architecture=frontend_architecture,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        # Validate architecture
        validation = await self.validate(architecture, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated architecture failed validation: {validation['errors']}")

        return architecture

    async def _generate_with_dspy(
        self, task: str, frontend_architecture: str, previous_feedback: str, **kwargs
    ) -> str:
        """
        Generate backend architecture using DSPy.

        Args:
            task: Task description
            frontend_architecture: Frontend architecture from frontend developer
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Backend architecture specification
        """
        # Lazy-load DSPy agent
        if self._dspy_agent is None:
            self._dspy_agent = self._create_dspy_agent()

        try:
            # Call DSPy agent
            result = self._dspy_agent(  # type: ignore[misc]
                frontend_architecture=frontend_architecture,
                previous_feedback=previous_feedback,
            )

            output = result.architecture if hasattr(result, "architecture") else str(result)
            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            return "Backend architecture generation failed."

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for backend architecture generation."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for BackendAgent. Install with: pip install dspy-ai"
            )

        class BackendArchitectureGenerator(dspy.Signature):
            """Generate backend architecture with API endpoints and data models."""

            frontend_architecture: str = dspy.InputField(
                desc="Frontend architecture from frontend developer"
            )
            previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
            architecture: str = dspy.OutputField(
                desc="Backend architecture with API endpoints, data models, auth, database schema"
            )

        return dspy.ChainOfThought(BackendArchitectureGenerator)

    async def _validate_backend(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate backend architecture (domain validator for LearningCapability).

        Args:
            output: Generated architecture
            expected: Expected architecture (gold standard)
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
        has_api = (
            "api" in output_str
            or "endpoint" in output_str
            or "rest" in output_str
            or "graphql" in output_str
        )
        has_data_models = "model" in output_str or "schema" in output_str or "entity" in output_str
        has_database = (
            "database" in output_str
            or "postgres" in output_str
            or "mongodb" in output_str
            or "mysql" in output_str
        )
        has_auth = (
            "auth" in output_str
            or "jwt" in output_str
            or "session" in output_str
            or "oauth" in output_str
        )
        has_validation = (
            "validation" in output_str or "middleware" in output_str or "error" in output_str
        )

        # Scoring
        if has_api:
            score += 0.25
        else:
            issues.append("Missing API endpoint definitions")

        if has_data_models:
            score += 0.25
        else:
            issues.append("Missing data models/schema")

        if has_database:
            score += 0.2
        else:
            issues.append("Missing database design")

        if has_auth:
            score += 0.15
        else:
            issues.append("Missing authentication/authorization")

        if has_validation:
            score += 0.15
        else:
            issues.append("Missing validation/error handling")

        # Length check
        if len(output) < 800:
            score *= 0.8
            warnings.append("Architecture spec too brief (< 800 chars)")

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
                "has_api": has_api,
                "has_data_models": has_data_models,
                "has_database": has_database,
                "has_auth": has_auth,
                "has_validation": has_validation,
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
        return await self._validate_backend(
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
        """Get default training cases for backend architecture."""
        return []

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for backend architecture."""
        return []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_backend_architecture(
        self, frontend_architecture: str = "", previous_feedback: str = "", **kwargs: Any
    ) -> str:
        """
        Generate backend architecture.

        Args:
            frontend_architecture: Frontend architecture from frontend developer
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Backend architecture specification as string
        """
        result = await self.execute(
            task="Generate backend architecture",
            frontend_architecture=frontend_architecture,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        if result.success:
            return result.output  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Backend architecture generation failed: {result.error}")
