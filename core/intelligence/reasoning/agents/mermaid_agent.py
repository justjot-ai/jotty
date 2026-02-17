"""
MermaidAgent - Domain Agent for Mermaid Diagram Generation

Migrated from: core/intelligence/reasoning/experts/mermaid_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for generating perfect Mermaid diagrams.
Uses capability mixins for learning and validation.
"""

import logging
from typing import Any, Dict, List, Optional

from Jotty.core.intelligence.reasoning.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.intelligence.reasoning.capabilities import (  # type: ignore[import-not-found, import]
    LearningCapability,
    ValidationCapability,
)

logger = logging.getLogger(__name__)


class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for Mermaid diagram generation with learning and validation.

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
        Initialize MermaidAgent.

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
                name="MermaidAgent", system_prompt="You are an expert at creating Mermaid diagrams."
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain="mermaid",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain="mermaid",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_mermaid,
            )

        # DSPy agent (lazy-loaded)
        self._dspy_agent = None
        self._dspy_teacher = None

        logger.info(
            "MermaidAgent initialized (learning={}, strict={})".format(
                enable_learning, strict_validation
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute Mermaid diagram generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (description, diagram_type, etc.)

        Returns:
            Generated Mermaid diagram code
        """
        description = kwargs.get("description", task)
        diagram_type = kwargs.get("diagram_type", "flowchart")

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
                raise ValueError(f"Mermaid validation failed: {validation['errors']}")

        # Learn from gold standards if enabled
        if hasattr(self, "learn_from_gold_standards"):
            diagram = await self.learn_from_gold_standards(task, diagram, **kwargs)

        return diagram

    async def _generate_with_dspy(
        self, task: str, description: str, diagram_type: str, **kwargs
    ) -> str:
        """
        Generate Mermaid diagram using DSPy.

        Args:
            task: Task description
            description: Diagram description
            diagram_type: Type of diagram
            **kwargs: Additional context

        Returns:
            Mermaid diagram code
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
            if output.startswith("```mermaid"):
                output = output[10:]  # Remove ```mermaid
            if output.startswith("```"):
                output = output[3:]  # Remove ```
            if output.endswith("```"):
                output = output[:-3]  # Remove trailing ```

            return output.strip()

        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            # Fallback to simple template
            return self._fallback_diagram(description, diagram_type)

    def _create_dspy_agent(self) -> Any:
        """Create DSPy agent for Mermaid generation."""
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required for MermaidAgent. Install with: pip install dspy-ai"
            )

        class MermaidGenerationSignature(dspy.Signature):
            """Generate a Mermaid diagram from a description.

            You are an expert at creating Mermaid diagrams. Generate valid, correct
            Mermaid syntax based on the description provided.

            Supported diagram types:
            - graph/flowchart: Flowcharts and decision trees
            - sequenceDiagram: Sequence diagrams with participants
            - classDiagram: Class diagrams with relationships
            - stateDiagram-v2: State diagrams
            - gantt: Gantt charts
            - gitgraph: Git graphs
            - pie: Pie charts
            - erDiagram: Entity-relationship diagrams
            - journey: User journey diagrams
            """

            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Description of the diagram")
            diagram_type: str = dspy.InputField(desc="Type of diagram")
            learned_improvements: str = dspy.InputField(
                desc="Previously learned patterns", default=""
            )

            output: str = dspy.OutputField(
                desc="Complete Mermaid diagram code. Must be valid Mermaid syntax."
            )

        return dspy.ChainOfThought(MermaidGenerationSignature)

    async def _validate_mermaid(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate Mermaid diagram (domain validator for LearningCapability).

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

        errors = []
        warnings = []
        score = 1.0

        # Check for valid diagram type
        valid_starts = [
            "graph",
            "flowchart",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram-v2",
            "gantt",
            "gitGraph",
            "pie",
            "erDiagram",
            "journey",
        ]

        has_valid_start = any(output_str.startswith(start) for start in valid_starts)
        if not has_valid_start:
            errors.append(f"Diagram must start with valid type: {', '.join(valid_starts)}")
            score -= 0.5

        # Check for basic structure
        if "graph" in output_str.lower() or "flowchart" in output_str.lower():
            # Flowchart validation
            if "[" not in output_str and "{" not in output_str:
                errors.append("Flowchart missing node definitions")
                score -= 0.2

            if "-->" not in output_str and "---" not in output_str:
                errors.append("Flowchart missing connections")
                score -= 0.2

        # Check if matches expected (if provided)
        matches_expected = False
        if expected_str:
            matches_expected = output_str == expected_str
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
                "has_valid_start": has_valid_start,
                "diagram_type": self._detect_diagram_type(output_str),
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
        return await self._validate_mermaid(
            output=output,
            expected=expected,
            task=context.get("task", "") if context else "",
            context=context or {},
        )

    def _detect_diagram_type(self, diagram: str) -> str:
        """Detect diagram type from code."""
        diagram_lower = diagram.lower()

        if diagram.startswith("sequenceDiagram"):
            return "sequence"
        elif diagram.startswith("classDiagram"):
            return "class"
        elif diagram.startswith("stateDiagram"):
            return "state"
        elif diagram.startswith("gantt"):
            return "gantt"
        elif diagram.startswith("gitGraph"):
            return "git"
        elif diagram.startswith("pie"):
            return "pie"
        elif diagram.startswith("erDiagram"):
            return "er"
        elif diagram.startswith("journey"):
            return "journey"
        elif "graph" in diagram_lower or "flowchart" in diagram_lower:
            return "flowchart"
        else:
            return "unknown"

    def _fallback_diagram(self, description: str, diagram_type: str) -> str:
        """Generate simple fallback diagram if DSPy fails."""
        if diagram_type == "flowchart" or diagram_type == "graph":
            return f"""graph TD
    A[Start: {description[:30]}]
    B[Process]
    C[End]
    A --> B
    B --> C"""
        elif diagram_type == "sequence":
            return """sequenceDiagram
    participant A as User
    participant B as System
    A->>B: Request
    B-->>A: Response"""
        else:
            return f"""graph TD
    A[{description[:40]}]"""

    # =========================================================================
    # TRAINING AND VALIDATION DATA
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for Mermaid."""
        return [
            {
                "input": "Generate simple flowchart",
                "output": "graph TD\n    A[Start]\n    B[End]\n    A --> B",
                "context": {"description": "Start to End flow", "diagram_type": "flowchart"},
            },
            {
                "input": "Generate decision flowchart",
                "output": """graph TD
    A[User Login]
    B{Valid?}
    C[Show Dashboard]
    D[Show Error]
    A --> B
    B -->|Yes| C
    B -->|No| D""",
                "context": {
                    "description": "User login with validation",
                    "diagram_type": "flowchart",
                },
            },
            {
                "input": "Generate sequence diagram",
                "output": """sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: Request
    S-->>C: Response""",
                "context": {"description": "Client-Server interaction", "diagram_type": "sequence"},
            },
            {
                "input": "Generate class diagram",
                "output": """classDiagram
    class Animal {
        +name: string
        +speak()
    }
    class Dog {
        +breed: string
    }
    Animal <|-- Dog""",
                "context": {"description": "Basic class structure", "diagram_type": "class"},
            },
        ]

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for Mermaid."""
        return [
            {
                "input": "Generate workflow diagram",
                "output": """graph TD
    A[Start]
    B[Process]
    C[End]
    A --> B
    B --> C""",
                "context": {"description": "Process workflow", "diagram_type": "flowchart"},
            },
            {
                "input": "Generate state diagram",
                "output": """stateDiagram-v2
    [*] --> State1
    State1 --> State2
    State2 --> [*]""",
                "context": {"description": "State transitions", "diagram_type": "state"},
            },
        ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_mermaid(
        self, description: str, diagram_type: str = "flowchart", **kwargs: Any
    ) -> str:
        """
        Generate a Mermaid diagram.

        Args:
            description: Description of the diagram to generate
            diagram_type: Type of diagram (flowchart, sequence, class, etc.)
            **kwargs: Additional context

        Returns:
            Mermaid diagram code as string
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
            raise ValueError(f"Mermaid generation failed: {result.error}")
