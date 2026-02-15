"""
UXResearcherAgent - Domain Agent for UX Research Generation

Migrated from: core/intelligence/reasoning/experts/ux_researcher_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for UX research and user understanding.
Evaluates user personas, pain points, user journey mapping, and research methodology.
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


class UXResearcherAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for UX research generation with learning and validation.

    Features:
    - User personas and demographics
    - Pain points and needs
    - User journey mapping
    - Research methodology
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize UXResearcherAgent.

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
                name="UXResearcherAgent",
                system_prompt="You are an expert in user research, personas, and journey mapping.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="ux_research",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="ux_research",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_ux_research,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None

        logger.info(
            "UXResearcherAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute UX research generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (requirements, previous_feedback, etc.)

        Returns:
            Generated UX research
        """
        requirements = kwargs.get("requirements", "")
        previous_feedback = kwargs.get("previous_feedback", "")

        # Generate research using DSPy
        research = await self._generate_with_dspy(
            task=task, requirements=requirements, previous_feedback=previous_feedback, **kwargs
        )

        # Validate research
        validation = await self.validate(research, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated research failed validation: {validation['errors']}")

        return research

    async def _generate_with_dspy(
        self, task: str, requirements: str, previous_feedback: str, **kwargs
    ) -> str:
        """
        Generate UX research using DSPy.

        Args:
            task: Task description
            requirements: Product requirements from PM
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            UX research specification
        """
        # Lazy-load DSPy agent
        if self._dspy_agent is None:
            self._dspy_agent = self._create_dspy_agent()

        try:
            # Call DSPy agent
            result = self._dspy_agent(
                requirements=requirements,
                previous_feedback=previous_feedback,
            )

            output = result.research if hasattr(result, "research") else str(result)
            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            return "UX research generation failed."

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for UX research generation."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for UXResearcherAgent. Install with: pip install dspy-ai"
            )

        class UXResearchGenerator(dspy.Signature):
            """Generate UX research with personas and user journeys."""

            requirements: str = dspy.InputField(desc="Product requirements from PM")
            previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
            research: str = dspy.OutputField(
                desc="User research with personas, pain points, journey maps"
            )

        return dspy.ChainOfThought(UXResearchGenerator)

    async def _validate_ux_research(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate UX research (domain validator for LearningCapability).

        Args:
            output: Generated research
            expected: Expected research (gold standard)
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
        has_personas = "persona" in output_str or "user profile" in output_str
        has_pain_points = (
            "pain point" in output_str or "frustration" in output_str or "challenge" in output_str
        )
        has_journey = "journey" in output_str or "workflow" in output_str or "flow" in output_str
        has_goals = "goal" in output_str or "objective" in output_str or "need" in output_str

        # Scoring
        if has_personas:
            score += 0.3
        else:
            issues.append("Missing user personas")

        if has_pain_points:
            score += 0.25
        else:
            issues.append("Missing pain points analysis")

        if has_journey:
            score += 0.25
        else:
            issues.append("Missing user journey/workflow")

        if has_goals:
            score += 0.2
        else:
            issues.append("Missing user goals/needs")

        # Length check
        if len(output) < 600:
            score *= 0.8
            warnings.append("Research too brief (< 600 chars)")

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
                "has_personas": has_personas,
                "has_pain_points": has_pain_points,
                "has_journey": has_journey,
                "has_goals": has_goals,
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
        return await self._validate_ux_research(
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
        """Get default training cases for UX research."""
        return []

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for UX research."""
        return []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_ux_research(
        self, requirements: str = "", previous_feedback: str = "", **kwargs: Any
    ) -> str:
        """
        Generate UX research.

        Args:
            requirements: Product requirements from PM
            previous_feedback: Feedback from previous iterations
            **kwargs: Additional context

        Returns:
            UX research specification as string
        """
        result = await self.execute(
            task="Generate UX research",
            requirements=requirements,
            previous_feedback=previous_feedback,
            **kwargs,
        )

        if result.success:
            return result.output
        else:
            raise ValueError(f"UX research generation failed: {result.error}")
