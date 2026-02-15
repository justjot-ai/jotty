"""
DesignerAgent - Domain Agent for UI/UX Design Generation

Migrated from: core/intelligence/reasoning/experts/designer_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for UI/UX design and visual systems.
Evaluates design for visual hierarchy, component structure, accessibility, and consistency.
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


class DesignerAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for UI/UX design generation with learning and validation.

    Features:
    - Visual hierarchy and layout
    - Component structure
    - Accessibility and usability
    - Design system consistency
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize DesignerAgent.

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
                name="DesignerAgent",
                system_prompt="You are an expert in UI/UX design, wireframes, and design systems.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="ui_ux_design",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="ui_ux_design",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_design,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None

        logger.info(
            "DesignerAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute UI/UX design generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (ux_research, previous_feedback, etc.)

        Returns:
            Generated design specification
        """
        ux_research = kwargs.get("ux_research", "")
        previous_feedback = kwargs.get("previous_feedback", "")

        # Generate design using DSPy
        design = await self._generate_with_dspy(
            task=task, ux_research=ux_research, previous_feedback=previous_feedback, **kwargs
        )

        # Validate design
        validation = await self.validate(design, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated design failed validation: {validation['errors']}")

        return design

    async def _generate_with_dspy(
        self, task: str, ux_research: str, previous_feedback: str, **kwargs
    ) -> str:
        """
        Generate UI/UX design using DSPy.

        Args:
            task: Task description
            ux_research: UX research from researcher
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Design specification
        """
        # Lazy-load DSPy agent
        if self._dspy_agent is None:
            self._dspy_agent = self._create_dspy_agent()

        try:
            # Call DSPy agent
            result = self._dspy_agent(
                ux_research=ux_research,
                previous_feedback=previous_feedback,
            )

            output = result.design if hasattr(result, "design") else str(result)
            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            return "UI/UX design generation failed."

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for design generation."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for DesignerAgent. Install with: pip install dspy-ai"
            )

        class DesignGenerator(dspy.Signature):
            """Generate UI/UX design with wireframes and component structure."""

            ux_research: str = dspy.InputField(desc="UX research from researcher")
            previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
            design: str = dspy.OutputField(
                desc="UI/UX design with Mermaid wireframes, component hierarchy, design tokens"
            )

        return dspy.ChainOfThought(DesignGenerator)

    async def _validate_design(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate UI/UX design (domain validator for LearningCapability).

        Args:
            output: Generated design
            expected: Expected design (gold standard)
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
        has_wireframes = (
            "wireframe" in output_str or "mermaid" in output_str or "layout" in output_str
        )
        has_components = "component" in output_str or "element" in output_str
        has_hierarchy = (
            "hierarchy" in output_str or "structure" in output_str or "navigation" in output_str
        )
        has_accessibility = (
            "accessibility" in output_str or "a11y" in output_str or "aria" in output_str
        )
        has_responsive = (
            "responsive" in output_str or "mobile" in output_str or "breakpoint" in output_str
        )

        # Scoring
        if has_wireframes:
            score += 0.25
        else:
            issues.append("Missing wireframes/layout diagrams")

        if has_components:
            score += 0.2
        else:
            issues.append("Missing component structure")

        if has_hierarchy:
            score += 0.2
        else:
            issues.append("Missing visual hierarchy/navigation")

        if has_accessibility:
            score += 0.2
        else:
            issues.append("Missing accessibility considerations")

        if has_responsive:
            score += 0.15
        else:
            issues.append("Missing responsive design approach")

        # Length check
        if len(output) < 700:
            score *= 0.8
            warnings.append("Design spec too brief (< 700 chars)")

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
                "has_wireframes": has_wireframes,
                "has_components": has_components,
                "has_hierarchy": has_hierarchy,
                "has_accessibility": has_accessibility,
                "has_responsive": has_responsive,
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
        return await self._validate_design(
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
        """Get default training cases for UI/UX design."""
        return []

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for UI/UX design."""
        return []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_design(
        self, ux_research: str = "", previous_feedback: str = "", **kwargs: Any
    ) -> str:
        """
        Generate UI/UX design.

        Args:
            ux_research: UX research from researcher
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            Design specification as string
        """
        result = await self.execute(
            task="Generate UI/UX design",
            ux_research=ux_research,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        if result.success:
            return result.output
        else:
            raise ValueError(f"Design generation failed: {result.error}")
