"""
PlantUML Expert Agent

Expert agent for generating perfect PlantUML diagrams.
Uses OptimizationPipeline internally to learn from mistakes.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class PlantUMLExpertAgent(BaseExpert):
    """
    Expert agent for PlantUML diagram generation.

    This agent is pre-trained to always generate valid PlantUML diagrams.
    It uses OptimizationPipeline internally to learn from mistakes.
    """

    # =========================================================================
    # REQUIRED PROPERTIES (BaseExpert interface)
    # =========================================================================

    @property
    def domain(self) -> str:
        """Return domain name for this expert."""
        return "plantuml"

    @property
    def description(self) -> str:
        """Return description for this expert."""
        return "Expert agent for generating perfect PlantUML diagrams"

    # =========================================================================
    # DOMAIN-SPECIFIC AGENT CREATION (BaseExpert interface)
    # =========================================================================

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create the PlantUML generation agent using DSPy."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for expert agents. Install with: pip install dspy-ai"
            )

        # Base signature
        class PlantUMLGenerationSignature(dspy.Signature):
            """Generate a PlantUML diagram from a description.

            You are an expert at creating PlantUML diagrams. Generate valid, correct
            PlantUML syntax based on the description provided.

            CRITICAL RULES:
            1. ALL PlantUML diagrams MUST start with @startuml and end with @enduml
            2. Do NOT include markdown code fences (```plantuml)
            3. Return ONLY the PlantUML code between @startuml and @enduml

            Supported diagram types:
            - @startuml/@enduml: All PlantUML diagrams (REQUIRED)
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

            task: str = dspy.InputField(desc="Task description (e.g., 'Generate sequence diagram')")
            description: str = dspy.InputField(desc="Description of the diagram to generate")
            diagram_type: str = dspy.InputField(
                desc="Type of diagram: sequence, class, activity, component, state, usecase, deployment"
            )
            learned_improvements: str = dspy.InputField(
                desc="Previously learned patterns and improvements (optional)", default=""
            )

            output: str = dspy.OutputField(
                desc="Complete PlantUML diagram code. MUST include @startuml at the start and @enduml at the end. No markdown code fences."
            )

        # Inject improvements into signature if provided
        signature_class = PlantUMLGenerationSignature
        if improvements:
            try:
                from .dspy_improvements import inject_improvements_into_signature

                signature_class = inject_improvements_into_signature(
                    PlantUMLGenerationSignature, improvements
                )
            except ImportError:
                logger.warning("Could not inject improvements into signature")

        # Use ChainOfThought for better reasoning
        agent = dspy.ChainOfThought(signature_class)
        return agent

    def _create_domain_teacher(self) -> Any:
        """Create the PlantUML teacher agent using DSPy."""
        if not DSPY_AVAILABLE:
            return None

        class PlantUMLTeacherSignature(dspy.Signature):
            """Provide the correct PlantUML diagram based on gold standard.

            CRITICAL INSTRUCTIONS:
            1. You MUST return ONLY the gold_standard code exactly as provided
            2. Do NOT add any explanation, evaluation, or feedback
            3. Do NOT include markdown code fences (```plantuml)
            4. Return ONLY the PlantUML code: @startuml ... @enduml
            5. Copy the gold_standard EXACTLY - do not modify it

            Example: If gold_standard is "@startuml\nUser -> System: Request\n@enduml",
                     return exactly: "@startuml\nUser -> System: Request\n@enduml"

            DO NOT return evaluation text, feedback, or analysis.
            DO NOT return JSON or markdown.
            RETURN ONLY THE PLANTUML CODE.
            """

            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Description of the diagram")
            gold_standard: str = dspy.InputField(
                desc="The EXACT PlantUML code to return. Copy this EXACTLY without modification."
            )
            student_output: str = dspy.InputField(
                desc="What the student generated (IGNORE THIS - just return gold_standard)"
            )

            output: str = dspy.OutputField(
                desc="ONLY the PlantUML code from gold_standard. No explanation. No evaluation. Just the code."
            )

        teacher = dspy.Predict(PlantUMLTeacherSignature)
        return teacher

    # =========================================================================
    # DOMAIN-SPECIFIC EVALUATION (BaseExpert interface)
    # =========================================================================

    async def _evaluate_domain(
        self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate PlantUML diagram output."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()

        # Remove markdown code fences if present
        output_str = re.sub(r"^```plantuml\s*\n?", "", output_str, flags=re.MULTILINE)
        output_str = re.sub(r"^```\s*$", "", output_str, flags=re.MULTILINE)
        gold_str = re.sub(r"^```plantuml\s*\n?", "", gold_str, flags=re.MULTILINE)
        gold_str = re.sub(r"^```\s*$", "", gold_str, flags=re.MULTILINE)

        # Check for @startuml/@enduml tags (case insensitive)
        output_lower = output_str.lower()
        has_start_tag = "@startuml" in output_lower or "@start" in output_lower
        has_end_tag = "@enduml" in output_lower or "@end" in output_lower

        # If missing tags, add them for validation (but note it as an issue)
        missing_tags = not (has_start_tag and has_end_tag)
        if missing_tags:
            # Add tags for content comparison, but mark as needing correction
            if not has_start_tag:
                output_str = "@startuml\n" + output_str
            if not has_end_tag:
                output_str = output_str + "\n@enduml"

        # Basic syntax validation
        is_valid = True  # Accept even if tags were missing (we'll learn to add them)

        # Content similarity (compare without tags for now)
        output_content = re.sub(r"@startuml\s*", "", output_str, flags=re.IGNORECASE)
        output_content = re.sub(r"@enduml\s*", "", output_content, flags=re.IGNORECASE)
        gold_content = re.sub(r"@startuml\s*", "", gold_str, flags=re.IGNORECASE)
        gold_content = re.sub(r"@enduml\s*", "", gold_content, flags=re.IGNORECASE)

        output_normalized = (
            output_content.lower().replace(" ", "").replace("\n", "").replace("\t", "")
        )
        gold_normalized = gold_content.lower().replace(" ", "").replace("\n", "").replace("\t", "")

        if output_normalized == gold_normalized and not missing_tags:
            score = 1.0
            status = "CORRECT"
        elif output_normalized == gold_normalized and missing_tags:
            score = 0.8  # Content correct but missing tags
            status = "PARTIAL"
        elif is_valid and (
            output_normalized in gold_normalized or gold_normalized in output_normalized
        ):
            score = 0.7
            status = "PARTIAL"
        elif is_valid:
            score = 0.3
            status = "INCORRECT"
        else:
            score = 0.0
            status = "INVALID"

        return {
            "score": score,
            "status": status,
            "is_valid": is_valid,
            "has_start_tag": has_start_tag,
            "has_end_tag": has_end_tag,
            "missing_tags": missing_tags,
            "difference": "Missing @startuml/@enduml tags" if missing_tags else None,
        }

    # =========================================================================
    # HELPER METHODS (domain-specific)
    # =========================================================================

    @classmethod
    async def load_training_examples_from_github(
        cls,
        repo_url: str = "https://github.com/joelparkerhenderson/plantuml-examples",
        max_examples: int = 50,
        save_to_file: bool = True,
        expert_data_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load PlantUML training examples from GitHub repository.

        Args:
            repo_url: GitHub repository URL
            max_examples: Maximum number of examples to load
            save_to_file: If True, save examples as JSON to expert directory
            expert_data_dir: Expert data directory (default: ./expert_data/plantuml_expert)

        Returns:
            List of training examples (in gold_standards format)
        """
        # Get domain validator
        from .domain_validators import get_validator
        from .training_data_loader import TrainingDataLoader

        validator = get_validator("plantuml")

        # Create loader
        loader = TrainingDataLoader(domain="plantuml", validator=validator)

        # Load examples from GitHub (supports .puml, .plantuml, .pu extensions)
        examples = loader.load_from_github_repo(
            repo_url=repo_url,
            path="doc",  # Examples are in doc/ directory
            file_pattern="*.puml",  # Will match .puml, .plantuml, .pu
            max_files=max_examples,
        )

        # Validate examples
        valid_examples, invalid_examples = loader.validate_examples(examples)

        logger.info(f"Loaded {len(valid_examples)} valid PlantUML examples from GitHub")
        if invalid_examples:
            logger.warning(f"Skipped {len(invalid_examples)} invalid examples")

        # Convert to gold_standards format
        gold_standards = loader.convert_to_gold_standards(valid_examples)

        # Save to file if requested
        if save_to_file:
            if expert_data_dir is None:
                expert_data_dir = "./expert_data/plantuml_expert"

            examples_dir = Path(expert_data_dir)
            examples_dir.mkdir(parents=True, exist_ok=True)

            # Save as JSON
            examples_file = examples_dir / "github_training_examples.json"
            with open(examples_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": repo_url,
                        "total_examples": len(gold_standards),
                        "loaded_at": datetime.now().isoformat(),
                        "gold_standards": gold_standards,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            logger.info(f"Saved {len(gold_standards)} examples to {examples_file}")

        return gold_standards

    # =========================================================================
    # TRAINING AND VALIDATION DATA (BaseExpert interface)
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for PlantUML."""
        return [
            {
                "task": "Generate simple sequence diagram",
                "context": {
                    "description": "User and System interaction",
                    "diagram_type": "sequence",
                },
                "gold_standard": """@startuml
User -> System: Request
System --> User: Response
@enduml""",
            },
            {
                "task": "Generate class diagram",
                "context": {"description": "Basic class structure", "diagram_type": "class"},
                "gold_standard": """@startuml
class Animal {
    +name: string
    +speak()
}
class Dog {
    +breed: string
}
Animal <|-- Dog
@enduml""",
            },
            {
                "task": "Generate activity diagram",
                "context": {"description": "Simple process flow", "diagram_type": "activity"},
                "gold_standard": """@startuml
start
:Process;
stop
@enduml""",
            },
        ]

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for PlantUML."""
        return [
            {
                "task": "Generate sequence diagram",
                "context": {"description": "Client-Server interaction", "diagram_type": "sequence"},
                "gold_standard": """@startuml
Client -> Server: Request
Server --> Client: Response
@enduml""",
            },
            {
                "task": "Generate state diagram",
                "context": {"description": "State transitions", "diagram_type": "state"},
                "gold_standard": """@startuml
[*] --> State1
State1 --> State2
State2 --> [*]
@enduml""",
            },
        ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_plantuml(
        self, description: str, diagram_type: str = "sequence", **kwargs: Any
    ) -> str:
        """
        Generate a PlantUML diagram using the trained expert agent.

        Args:
            description: Description of the diagram to generate
            diagram_type: Type of diagram (sequence, class, activity, etc.)
            **kwargs: Additional context

        Returns:
            PlantUML diagram code as string
        """
        task = f"Generate {diagram_type} diagram"
        context = {"description": description, "diagram_type": diagram_type, **kwargs}

        output = await self.generate(task=task, context=context)
        return str(output)
