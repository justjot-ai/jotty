"""
PlantUMLAgent - Domain Agent for PlantUML Diagram Generation

Migrated from: core/intelligence/reasoning/experts/plantuml_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for generating perfect PlantUML diagrams.
Uses capability mixins for learning and validation.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from Jotty.core.execution.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.execution.capabilities import (  # type: ignore[import-not-found, import]
    LearningCapability,
    ValidationCapability,
)

logger = logging.getLogger(__name__)

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class PlantUMLAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for PlantUML diagram generation with learning and validation.

    Features:
    - Gold standard learning (via LearningCapability)
    - Domain-specific validation (via ValidationCapability)
    - DSPy-based generation
    - Optimization pipeline integration
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = True,
    ):
        """
        Initialize PlantUMLAgent.

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
                name="PlantUMLAgent",
                system_prompt="You are an expert at creating PlantUML diagrams.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="plantuml",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="plantuml",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_plantuml,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None
        self._dspy_teacher = None

        logger.info(
            "PlantUMLAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute PlantUML diagram generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (description, diagram_type, etc.)

        Returns:
            Generated PlantUML diagram code
        """
        description = kwargs.get("description", task)
        diagram_type = kwargs.get("diagram_type", "sequence")

        # Generate diagram using DSPy
        diagram = await self._generate_with_dspy(
            task=task, description=description, diagram_type=diagram_type, **kwargs
        )

        # Validate diagram
        validation = await self.validate(diagram, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated diagram failed validation: {validation['errors']}")

            # Try to fix errors if possible
            if strict_validation := kwargs.get("strict", self.strict_mode):
                # In strict mode, raise error
                raise ValueError(f"PlantUML validation failed: {validation['errors']}")

        # Learn from gold standards if enabled
        if hasattr(self, "learn_from_gold_standards"):
            diagram = await self.learn_from_gold_standards(task, diagram, **kwargs)

        return diagram

    async def _generate_with_dspy(
        self, task: str, description: str, diagram_type: str, **kwargs
    ) -> str:
        """
        Generate PlantUML diagram using DSPy.

        Args:
            task: Task description
            description: Diagram description
            diagram_type: Type of diagram
            **kwargs: Additional context

        Returns:
            PlantUML diagram code
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
            result = self._dspy_agent(  # type: ignore[misc]
                task=task,
                description=description,
                diagram_type=diagram_type,
                learned_improvements=learned_improvements,
            )

            output = result.output if hasattr(result, "output") else str(result)

            # Clean output (remove markdown fences if present)
            output = output.strip()
            if output.startswith("```plantuml"):
                output = output[11:]  # Remove ```plantuml
            if output.startswith("```"):
                output = output[3:]  # Remove ```
            if output.endswith("```"):
                output = output[:-3]  # Remove trailing ```

            # Ensure @startuml/@enduml tags
            output_lower = output.lower()
            if "@startuml" not in output_lower:
                output = "@startuml\n" + output
            if "@enduml" not in output_lower:
                output = output + "\n@enduml"

            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            # Fallback to simple template
            return self._fallback_diagram(description, diagram_type)

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for PlantUML generation."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for PlantUMLAgent. Install with: pip install dspy-ai"
            )

        class PlantUMLGenerationSignature(dspy.Signature):
            """Generate a PlantUML diagram from a description.

            You are an expert at creating PlantUML diagrams. Generate valid, correct
            PlantUML syntax based on the description provided.

            CRITICAL RULES:
            1. ALL PlantUML diagrams MUST start with @startuml and end with @enduml
            2. Do NOT include markdown code fences (```plantuml)
            3. Return ONLY the PlantUML code between @startuml and @enduml

            Supported diagram types:
            - sequence: Sequence diagrams
            - class: Class diagrams
            - activity: Activity diagrams
            - component: Component diagrams
            - state: State diagrams
            - usecase: Use case diagrams
            - deployment: Deployment diagrams

            Example format:
            @startuml
            [diagram content here]
            @enduml
            """

            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Description of the diagram")
            diagram_type: str = dspy.InputField(
                desc="Type of diagram: sequence, class, activity, component, state, usecase, deployment"
            )
            learned_improvements: str = dspy.InputField(
                desc="Previously learned patterns and improvements", default=""
            )

            output: str = dspy.OutputField(
                desc="Complete PlantUML diagram code. MUST include @startuml at the start and @enduml at the end."
            )

        return dspy.ChainOfThought(PlantUMLGenerationSignature)

    async def _validate_plantuml(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate PlantUML diagram (domain validator for LearningCapability).

        Args:
            output: Generated diagram
            expected: Expected diagram (gold standard)
            task: Task description
            context: Additional context

        Returns:
            Validation result dict
        """
        output_str = str(output).strip()
        expected_str = str(expected).strip() if expected else ""

        # Remove markdown code fences if present
        output_str = re.sub(r"^```plantuml\s*\n?", "", output_str, flags=re.MULTILINE)
        output_str = re.sub(r"^```\s*$", "", output_str, flags=re.MULTILINE)

        errors = []
        warnings = []
        score = 1.0

        # Check for @startuml/@enduml tags (case insensitive)
        output_lower = output_str.lower()
        has_start_tag = "@startuml" in output_lower
        has_end_tag = "@enduml" in output_lower

        if not has_start_tag:
            errors.append("Missing @startuml tag")
            score -= 0.3

        if not has_end_tag:
            errors.append("Missing @enduml tag")
            score -= 0.3

        # Check for basic content
        if len(output_str) < 20:
            errors.append("Diagram content too short")
            score -= 0.2

        # Check if matches expected (if provided)
        matches_expected = False
        if expected_str:
            # Normalize for comparison
            output_normalized = re.sub(r"\s+", "", output_str.lower())
            expected_normalized = re.sub(r"\s+", "", expected_str.lower())
            matches_expected = output_normalized == expected_normalized
            if not matches_expected:
                warnings.append("Output doesn't match gold standard exactly")
                score -= 0.1

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return {
            "valid": len(errors) == 0,
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "matches_expected": matches_expected,
            "metadata": {
                "has_start_tag": has_start_tag,
                "has_end_tag": has_end_tag,
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
        return await self._validate_plantuml(
            output=output,
            expected=expected,
            task=context.get("task", "") if context else "",
            context=context or {},
        )

    def _fallback_diagram(self, description: str, diagram_type: str) -> str:
        """Generate simple fallback diagram if DSPy fails."""
        if diagram_type == "sequence":
            return """@startuml
User -> System: Request
System --> User: Response
@enduml"""
        elif diagram_type == "class":
            return """@startuml
class Entity {
    +field: string
    +method()
}
@enduml"""
        elif diagram_type == "activity":
            return """@startuml
start
:Process;
stop
@enduml"""
        else:
            return f"""@startuml
participant A as "Component A"
participant B as "Component B"
A -> B: {description[:40]}
@enduml"""

    # =========================================================================
    # TRAINING AND VALIDATION DATA
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for PlantUML."""
        return [
            {
                "input": "Generate simple sequence diagram",
                "output": """@startuml
User -> System: Request
System --> User: Response
@enduml""",
                "context": {
                    "description": "User and System interaction",
                    "diagram_type": "sequence",
                },
            },
            {
                "input": "Generate class diagram",
                "output": """@startuml
class Animal {
    +name: string
    +speak()
}
class Dog {
    +breed: string
}
Animal <|-- Dog
@enduml""",
                "context": {"description": "Basic class structure", "diagram_type": "class"},
            },
            {
                "input": "Generate activity diagram",
                "output": """@startuml
start
:Process;
stop
@enduml""",
                "context": {"description": "Simple process flow", "diagram_type": "activity"},
            },
        ]

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for PlantUML."""
        return [
            {
                "input": "Generate sequence diagram",
                "output": """@startuml
Client -> Server: Request
Server --> Client: Response
@enduml""",
                "context": {"description": "Client-Server interaction", "diagram_type": "sequence"},
            },
            {
                "input": "Generate state diagram",
                "output": """@startuml
[*] --> State1
State1 --> State2
State2 --> [*]
@enduml""",
                "context": {"description": "State transitions", "diagram_type": "state"},
            },
        ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_plantuml(
        self, description: str, diagram_type: str = "sequence", **kwargs: Any
    ) -> str:
        """
        Generate a PlantUML diagram.

        Args:
            description: Description of the diagram to generate
            diagram_type: Type of diagram (sequence, class, activity, etc.)
            **kwargs: Additional context

        Returns:
            PlantUML diagram code as string
        """
        result = await self.execute(
            task=f"Generate {diagram_type} diagram",
            description=description,
            diagram_type=diagram_type,
            **kwargs,
        )

        if result.success:
            return result.output  # type: ignore[no-any-return]
        else:
            raise ValueError(f"PlantUML generation failed: {result.error}")
