"""
FrontendAgent - Domain Agent for Frontend Architecture Generation

Migrated from: core/intelligence/reasoning/experts/frontend_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for frontend development and React architecture.
Evaluates frontend architecture for component design, state management, and best practices.
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


class FrontendAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for frontend architecture generation with learning and validation.

    Features:
    - Component architecture (React)
    - State management approach
    - API integration patterns
    - Performance and best practices
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize FrontendAgent.

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
                name="FrontendAgent",
                system_prompt="You are an expert in React architecture, state management, and frontend best practices.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="frontend_development",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="frontend_development",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_frontend,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None

        logger.info(
            "FrontendAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute frontend architecture generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (design, previous_feedback, etc.)

        Returns:
            Generated frontend architecture
        """
        design = kwargs.get("design", "")
        previous_feedback = kwargs.get("previous_feedback", "")

        # Generate architecture using DSPy
        architecture = await self._generate_with_dspy(
            task=task, design=design, previous_feedback=previous_feedback, **kwargs
        )

        # Validate architecture
        validation = await self.validate(architecture, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated architecture failed validation: {validation['errors']}")

        return architecture

    async def _generate_with_dspy(
        self, task: str, design: str, previous_feedback: str, **kwargs
    ) -> str:
        """
        Generate frontend architecture using DSPy.

        Args:
            task: Task description
            design: UI/UX design from designer
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Frontend architecture specification
        """
        # Lazy-load DSPy agent
        if self._dspy_agent is None:
            self._dspy_agent = self._create_dspy_agent()

        try:
            # Call DSPy agent
            result = self._dspy_agent(
                design=design,
                previous_feedback=previous_feedback,
            )

            output = result.architecture if hasattr(result, "architecture") else str(result)
            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            return "Frontend architecture generation failed."

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for frontend architecture generation."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for FrontendAgent. Install with: pip install dspy-ai"
            )

        class FrontendArchitectureGenerator(dspy.Signature):
            """Generate frontend architecture with React components and state management."""

            design: str = dspy.InputField(desc="UI/UX design from designer")
            previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
            architecture: str = dspy.OutputField(
                desc="Frontend architecture with React components, state management, API integration"
            )

        return dspy.ChainOfThought(FrontendArchitectureGenerator)

    async def _validate_frontend(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate frontend architecture (domain validator for LearningCapability).

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
        has_components = "component" in output_str or "react" in output_str
        has_state_mgmt = (
            "state" in output_str
            or "redux" in output_str
            or "context" in output_str
            or "zustand" in output_str
        )
        has_hooks = "hook" in output_str or "usestate" in output_str or "useeffect" in output_str
        has_api = "api" in output_str or "fetch" in output_str or "axios" in output_str
        has_props = "props" in output_str or "interface" in output_str or "type" in output_str

        # Scoring
        if has_components:
            score += 0.25
        else:
            issues.append("Missing React component structure")

        if has_state_mgmt:
            score += 0.2
        else:
            issues.append("Missing state management approach")

        if has_hooks:
            score += 0.2
        else:
            issues.append("Missing React hooks usage")

        if has_api:
            score += 0.2
        else:
            issues.append("Missing API integration patterns")

        if has_props:
            score += 0.15
        else:
            issues.append("Missing TypeScript types/interfaces")

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
                "has_components": has_components,
                "has_state_mgmt": has_state_mgmt,
                "has_hooks": has_hooks,
                "has_api": has_api,
                "has_props": has_props,
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
        return await self._validate_frontend(
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
        """Get default training cases for frontend architecture."""
        return []

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for frontend architecture."""
        return []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_frontend_architecture(
        self, design: str = "", previous_feedback: str = "", **kwargs: Any
    ) -> str:
        """
        Generate frontend architecture.

        Args:
            design: UI/UX design from designer
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Frontend architecture specification as string
        """
        result = await self.execute(
            task="Generate frontend architecture",
            design=design,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        if result.success:
            return result.output
        else:
            raise ValueError(f"Frontend architecture generation failed: {result.error}")
