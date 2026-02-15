"""
Math LaTeX Expert Agent

Expert agent for generating perfect Math LaTeX expressions.
Uses OptimizationPipeline internally to learn from mistakes.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class MathLaTeXExpertAgent(BaseExpert):
    """
    Expert agent for Math LaTeX expression generation.

    This agent is pre-trained to always generate valid Math LaTeX expressions.
    It uses OptimizationPipeline internally to learn from mistakes.
    """

    # =========================================================================
    # REQUIRED PROPERTIES (BaseExpert interface)
    # =========================================================================

    @property
    def domain(self) -> str:
        """Return domain name for this expert."""
        return "math_latex"

    @property
    def description(self) -> str:
        """Return description for this expert."""
        return "Expert agent for generating perfect Math LaTeX expressions"

    # =========================================================================
    # DOMAIN-SPECIFIC AGENT CREATION (BaseExpert interface)
    # =========================================================================

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create the Math LaTeX generation agent using DSPy."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for expert agents. Install with: pip install dspy-ai"
            )

        # Base signature
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

            task: str = dspy.InputField(
                desc="Task description (e.g., 'Generate quadratic formula')"
            )
            description: str = dspy.InputField(
                desc="Mathematical description of the expression to generate"
            )
            expression_type: str = dspy.InputField(desc="Type: inline, display, equation, formula")
            learned_improvements: str = dspy.InputField(
                desc="Previously learned patterns and improvements (optional)", default=""
            )

            output: str = dspy.OutputField(
                desc="Complete Math LaTeX expression. Use $...$ for inline or $$...$$ for display math."
            )

        # Inject improvements into signature if provided
        signature_class = MathLaTeXGenerationSignature
        if improvements:
            try:
                from .dspy_improvements import inject_improvements_into_signature

                signature_class = inject_improvements_into_signature(
                    MathLaTeXGenerationSignature, improvements
                )
            except ImportError:
                logger.warning("Could not inject improvements into signature")

        # Use ChainOfThought for better reasoning
        agent = dspy.ChainOfThought(signature_class)
        return agent

    def _create_domain_teacher(self) -> Any:
        """Create the Math LaTeX teacher agent using DSPy."""
        if not DSPY_AVAILABLE:
            return None

        class MathLaTeXTeacherSignature(dspy.Signature):
            """Provide the correct Math LaTeX expression.

            You are a teacher correcting a student's Math LaTeX expression.
            Return ONLY the correct LaTeX code, no explanations.
            """

            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Mathematical description")
            student_output: str = dspy.InputField(desc="Student's incorrect LaTeX expression")
            gold_standard: str = dspy.InputField(desc="Correct LaTeX expression")
            evaluation_result: str = dspy.InputField(desc="What was wrong with student output")

            output: str = dspy.OutputField(
                desc="Correct Math LaTeX expression. Use $...$ for inline or $$...$$ for display."
            )

        teacher = dspy.ChainOfThought(MathLaTeXTeacherSignature)
        return teacher

    # =========================================================================
    # DOMAIN-SPECIFIC EVALUATION (BaseExpert interface)
    # =========================================================================

    async def _evaluate_domain(
        self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate Math LaTeX expression.

        Args:
            output: Generated LaTeX expression
            gold_standard: Correct LaTeX expression
            task: Task description
            context: Context dictionary

        Returns:
            Evaluation result dictionary
        """
        from .domain_validators import get_validator
        from .math_latex_renderer import validate_math_latex_syntax

        output_str = str(output).strip()

        # Get domain validator
        validator = get_validator("math_latex")
        expected_type = context.get("expression_type", "display")

        # Validate using domain validator
        is_valid, error_msg, metadata = validator.validate(
            output=output_str, expected_type=expected_type, context=context
        )

        # Also validate via renderer (with fallback)
        renderer_valid, renderer_error, renderer_metadata = validate_math_latex_syntax(
            output_str, use_renderer=True
        )

        # Combine validation results
        # If renderer fails but structure validation passed, use structure result
        if not renderer_valid:
            # Check if structure validation passed
            structure_valid = metadata.get("type_match", False) and not error_msg
            if structure_valid and "QuickLaTeX error: -1" in renderer_error:
                # QuickLaTeX API issue, but structure is valid - accept it
                logger.debug("QuickLaTeX API issue, but structure validation passed")
                is_valid = True
                error_msg = ""  # Clear error since structure is valid
            else:
                is_valid = False
                if error_msg:
                    error_msg += f"; Renderer: {renderer_error}"
                else:
                    error_msg = f"Renderer: {renderer_error}"

        # Calculate score
        score = 0.0
        if is_valid:
            # Check similarity to gold standard
            if gold_standard:
                similarity = self._calculate_similarity(output_str, gold_standard)
                score = similarity
            else:
                score = 1.0  # Valid but no gold standard to compare
        else:
            score = 0.0

        # Check for required elements
        required_elements = context.get("required_elements", [])
        found_elements = []
        if required_elements:
            for element in required_elements:
                if element.lower() in output_str.lower():
                    found_elements.append(element)

        element_coverage = (
            len(found_elements) / len(required_elements) if required_elements else 1.0
        )

        # Adjust score based on element coverage
        if is_valid and required_elements:
            score = score * 0.7 + element_coverage * 0.3

        return {
            "score": score,
            "status": "CORRECT" if score >= 0.9 else "FAIL",
            "is_valid": is_valid,
            "error": error_msg,
            "metadata": {
                **metadata,
                **renderer_metadata,
                "element_coverage": element_coverage,
                "found_elements": found_elements,
                "required_elements": required_elements,
            },
        }

    # =========================================================================
    # HELPER METHODS (domain-specific)
    # =========================================================================

    def _calculate_similarity(self, output: str, gold_standard: str) -> float:
        """Calculate similarity between output and gold standard."""
        # Normalize both strings
        output_norm = self._normalize_latex(output)
        gold_norm = self._normalize_latex(gold_standard)

        if output_norm == gold_norm:
            return 1.0

        # Simple character-based similarity
        # Count matching characters
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

    # =========================================================================
    # TRAINING AND VALIDATION DATA (BaseExpert interface)
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for Math LaTeX."""
        return [
            {
                "task": "Generate quadratic formula",
                "context": {
                    "description": "Quadratic formula for solving ax² + bx + c = 0",
                    "expression_type": "display",
                    "required_elements": ["frac", "sqrt", "pm"],
                },
                "gold_standard": "$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$",
            },
            {
                "task": "Generate Pythagorean theorem",
                "context": {
                    "description": "Pythagorean theorem: a² + b² = c²",
                    "expression_type": "display",
                    "required_elements": ["^", "="],
                },
                "gold_standard": "$$a^2 + b^2 = c^2$$",
            },
            {
                "task": "Generate Euler's identity",
                "context": {
                    "description": "Euler's identity: e^(iπ) + 1 = 0",
                    "expression_type": "display",
                    "required_elements": ["e", "pi", "="],
                },
                "gold_standard": "$$e^{i\\pi} + 1 = 0$$",
            },
            {
                "task": "Generate integral formula",
                "context": {
                    "description": "Definite integral from a to b",
                    "expression_type": "display",
                    "required_elements": ["int", "dx"],
                },
                "gold_standard": "$$\\int_a^b f(x) \\, dx$$",
            },
            {
                "task": "Generate sum formula",
                "context": {
                    "description": "Sum from i=1 to n",
                    "expression_type": "display",
                    "required_elements": ["sum", "="],
                },
                "gold_standard": "$$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$",
            },
        ]

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for Math LaTeX."""
        return [
            {
                "task": "Generate simple fraction",
                "context": {"description": "One half", "expression_type": "inline"},
                "gold_standard": "$\\frac{1}{2}$",
            },
            {
                "task": "Generate square root",
                "context": {"description": "Square root of x", "expression_type": "inline"},
                "gold_standard": "$\\sqrt{x}$",
            },
        ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_math_latex(
        self, description: str, expression_type: str = "display", **kwargs: Any
    ) -> str:
        """
        Generate a Math LaTeX expression using the trained expert agent.

        Args:
            description: Description of the mathematical expression
            expression_type: Type of expression (inline, display, equation, formula)
            **kwargs: Additional context

        Returns:
            Math LaTeX expression as string
        """
        task = f"Generate {expression_type} math expression"
        context = {"description": description, "expression_type": expression_type, **kwargs}

        output = await self.generate(task=task, context=context)
        return str(output)
