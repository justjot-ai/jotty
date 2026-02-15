"""
LaTeXAgent - Domain Agent for Math LaTeX Expression Generation

Migrated from: core/intelligence/reasoning/experts/math_latex_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for generating perfect Math LaTeX expressions.
Uses capability mixins for learning and validation.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from Jotty.core.execution.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.execution.capabilities import LearningCapability, ValidationCapability

logger = logging.getLogger(__name__)

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class LaTeXAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for Math LaTeX expression generation with learning and validation.

    Features:
    - Gold standard learning (via LearningCapability)
    - Domain-specific validation (via ValidationCapability)
    - DSPy-based generation
    - LaTeX syntax validation
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = True,
    ):
        """
        Initialize LaTeXAgent.

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
                name="LaTeXAgent",
                system_prompt="You are an expert at creating Math LaTeX expressions.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="math_latex",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="math_latex",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_latex,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None
        self._dspy_teacher = None

        logger.info(
            "LaTeXAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute Math LaTeX expression generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (description, expression_type, etc.)

        Returns:
            Generated Math LaTeX expression
        """
        description = kwargs.get("description", task)
        expression_type = kwargs.get("expression_type", "display")

        # Generate expression using DSPy
        expression = await self._generate_with_dspy(
            task=task, description=description, expression_type=expression_type, **kwargs
        )

        # Validate expression
        validation = await self.validate(expression, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated expression failed validation: {validation['errors']}")

            # Try to fix errors if possible
            if strict_validation := kwargs.get("strict", self.strict_mode):
                # In strict mode, raise error
                raise ValueError(f"LaTeX validation failed: {validation['errors']}")

        # Learn from gold standards if enabled
        if hasattr(self, "learn_from_gold_standards"):
            expression = await self.learn_from_gold_standards(task, expression, **kwargs)

        return expression

    async def _generate_with_dspy(
        self, task: str, description: str, expression_type: str, **kwargs
    ) -> str:
        """
        Generate Math LaTeX expression using DSPy.

        Args:
            task: Task description
            description: Mathematical description
            expression_type: Type of expression (inline, display, etc.)
            **kwargs: Additional context

        Returns:
            Math LaTeX expression
        """
        # Lazy-load DSPy agent
        if self._dspy_agent is None:
            self._dspy_agent = self._create_dspy_agent()

        # Get learned improvements if available
        learned_improvements = ""
        if hasattr(self, "learned_improvements"):
            improvements = self.learned_improvements[:3]  # Top 3
            if improvements:
                learned_improvements = "\n".join(
                    [
                        f"- {imp.get('pattern', '')}: {imp.get('improvement', '')}"
                        for imp in improvements
                    ]
                )

        try:
            # Call DSPy agent
            result = self._dspy_agent(
                task=task,
                description=description,
                expression_type=expression_type,
                learned_improvements=learned_improvements,
            )

            output = result.output if hasattr(result, "output") else str(result)

            # Clean output
            output = output.strip()

            return output

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            # Fallback to simple template
            return self._fallback_expression(description, expression_type)

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for LaTeX generation."""
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for LaTeXAgent. Install with: pip install dspy-ai")

        class MathLaTeXGenerationSignature(dspy.Signature):
            """Generate a Math LaTeX expression from a description.

            You are an expert at creating Math LaTeX expressions. Generate valid, correct
            LaTeX syntax based on the mathematical description provided.

            CRITICAL RULES:
            1. Use proper LaTeX math delimiters: $...$ for inline, $$...$$ for display
            2. Use correct LaTeX commands: \\frac, \\sqrt, \\sum, \\int, etc.
            3. Escape special characters properly: \\{ \\} \\_ \\^
            4. Use proper spacing: \\, \\; \\quad \\qquad
            5. For display math, use $$...$$ or \\[...\\]

            Supported LaTeX elements:
            - Fractions: \\frac{numerator}{denominator}
            - Roots: \\sqrt{x}, \\sqrt[n]{x}
            - Sums/Integrals: \\sum, \\int, \\prod
            - Greek letters: \\alpha, \\beta, \\gamma, etc.
            - Operators: \\times, \\div, \\pm, \\mp
            - Relations: \\leq, \\geq, \\neq, \\approx
            - Sets: \\in, \\subset, \\cup, \\cap
            - Functions: \\sin, \\cos, \\log, \\ln, \\exp

            Example format:
            $$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$
            """

            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Mathematical description")
            expression_type: str = dspy.InputField(desc="Type: inline, display, equation, formula")
            learned_improvements: str = dspy.InputField(
                desc="Previously learned patterns and improvements", default=""
            )

            output: str = dspy.OutputField(
                desc="Complete Math LaTeX expression. Use $...$ for inline or $$...$$ for display math."
            )

        return dspy.ChainOfThought(MathLaTeXGenerationSignature)

    async def _validate_latex(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate Math LaTeX expression (domain validator for LearningCapability).

        Args:
            output: Generated expression
            expected: Expected expression (gold standard)
            task: Task description
            context: Additional context

        Returns:
            Validation result dict
        """
        output_str = str(output).strip()
        expected_str = str(expected).strip() if expected else ""

        errors = []
        warnings = []
        score = 1.0

        # Check for delimiters
        has_inline = "$" in output_str and "$$" not in output_str
        has_display = "$$" in output_str or "\\[" in output_str
        has_delimiters = has_inline or has_display

        if not has_delimiters:
            errors.append("Missing LaTeX math delimiters ($ or $$)")
            score -= 0.3

        # Check for basic LaTeX commands
        latex_commands = re.findall(r"\\[a-zA-Z]+", output_str)
        if not latex_commands and len(output_str) > 10:
            warnings.append("No LaTeX commands found")
            score -= 0.1

        # Check for balanced braces
        open_braces = output_str.count("{")
        close_braces = output_str.count("}")
        if open_braces != close_braces:
            errors.append("Unbalanced braces")
            score -= 0.2

        # Check if matches expected (if provided)
        matches_expected = False
        if expected_str:
            similarity = self._calculate_similarity(output_str, expected_str)
            matches_expected = similarity > 0.9
            if not matches_expected:
                warnings.append("Output doesn't match gold standard exactly")
                score = min(score, similarity)

        # Check for required elements
        required_elements = context.get("required_elements", [])
        found_elements = []
        if required_elements:
            for element in required_elements:
                if element.lower() in output_str.lower():
                    found_elements.append(element)

            element_coverage = len(found_elements) / len(required_elements)
            if element_coverage < 1.0:
                warnings.append(
                    f"Missing required elements: {set(required_elements) - set(found_elements)}"
                )
                score *= 0.7 + 0.3 * element_coverage

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return {
            "valid": len(errors) == 0,
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "matches_expected": matches_expected,
            "metadata": {
                "has_delimiters": has_delimiters,
                "latex_commands": latex_commands,
                "element_coverage": (
                    len(found_elements) / len(required_elements) if required_elements else 1.0
                ),
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
        # Use the domain validator
        return await self._validate_latex(
            output=output,
            expected=expected,
            task=context.get("task", "") if context else "",
            context=context or {},
        )

    def _calculate_similarity(self, output: str, gold_standard: str) -> float:
        """Calculate similarity between output and gold standard."""
        # Normalize both strings
        output_norm = self._normalize_latex(output)
        gold_norm = self._normalize_latex(gold_standard)

        if output_norm == gold_norm:
            return 1.0

        # Simple character-based similarity
        matches = sum(1 for a, b in zip(output_norm, gold_norm) if a == b)
        max_len = max(len(output_norm), len(gold_norm))

        if max_len == 0:
            return 0.0

        similarity = matches / max_len

        # Boost score if key LaTeX commands match
        output_commands = set(re.findall(r"\\[a-zA-Z]+", output_norm))
        gold_commands = set(re.findall(r"\\[a-zA-Z]+", gold_norm))

        if gold_commands:
            command_match = len(output_commands & gold_commands) / len(gold_commands)
            similarity = similarity * 0.6 + command_match * 0.4

        return similarity

    def _normalize_latex(self, latex: str) -> str:
        """Normalize LaTeX for comparison."""
        # Remove whitespace
        latex = re.sub(r"\s+", "", latex)
        # Normalize delimiters
        latex = latex.replace("\\[", "$$").replace("\\]", "$$")
        latex = latex.replace("\\(", "$").replace("\\)", "$")
        return latex

    def _fallback_expression(self, description: str, expression_type: str) -> str:
        """Generate simple fallback expression if DSPy fails."""
        if expression_type == "inline":
            return "$x$"
        else:
            return "$$x = y$$"

    # =========================================================================
    # TRAINING AND VALIDATION DATA
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for Math LaTeX."""
        return [
            {
                "input": "Generate quadratic formula",
                "output": "$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$",
                "context": {
                    "description": "Quadratic formula for solving ax² + bx + c = 0",
                    "expression_type": "display",
                    "required_elements": ["frac", "sqrt", "pm"],
                },
            },
            {
                "input": "Generate Pythagorean theorem",
                "output": "$$a^2 + b^2 = c^2$$",
                "context": {
                    "description": "Pythagorean theorem: a² + b² = c²",
                    "expression_type": "display",
                    "required_elements": ["^", "="],
                },
            },
            {
                "input": "Generate Euler's identity",
                "output": "$$e^{i\\pi} + 1 = 0$$",
                "context": {
                    "description": "Euler's identity: e^(iπ) + 1 = 0",
                    "expression_type": "display",
                    "required_elements": ["e", "pi", "="],
                },
            },
            {
                "input": "Generate integral formula",
                "output": "$$\\int_a^b f(x) \\, dx$$",
                "context": {
                    "description": "Definite integral from a to b",
                    "expression_type": "display",
                    "required_elements": ["int", "dx"],
                },
            },
            {
                "input": "Generate sum formula",
                "output": "$$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$",
                "context": {
                    "description": "Sum from i=1 to n",
                    "expression_type": "display",
                    "required_elements": ["sum", "="],
                },
            },
        ]

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for Math LaTeX."""
        return [
            {
                "input": "Generate simple fraction",
                "output": "$\\frac{1}{2}$",
                "context": {"description": "One half", "expression_type": "inline"},
            },
            {
                "input": "Generate square root",
                "output": "$\\sqrt{x}$",
                "context": {"description": "Square root of x", "expression_type": "inline"},
            },
        ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_math_latex(
        self, description: str, expression_type: str = "display", **kwargs: Any
    ) -> str:
        """
        Generate a Math LaTeX expression.

        Args:
            description: Description of the mathematical expression
            expression_type: Type of expression (inline, display, equation, formula)
            **kwargs: Additional context

        Returns:
            Math LaTeX expression as string
        """
        result = await self.execute(
            task=f"Generate {expression_type} math expression",
            description=description,
            expression_type=expression_type,
            **kwargs,
        )

        if result.success:
            return result.output
        else:
            raise ValueError(f"LaTeX generation failed: {result.error}")
