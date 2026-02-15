"""
PipelineAgent - Domain Agent for CI/CD Pipeline Diagram Generation

Migrated from: core/intelligence/reasoning/experts/pipeline_expert.py
Pattern: BaseAgent + LearningCapability + ValidationCapability

A specialized domain agent for generating perfect CI/CD pipeline diagrams.
Supports Mermaid, PlantUML, and other formats.
"""

import logging
from typing import Any, Dict, List, Optional

from Jotty.core.execution.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.execution.capabilities import LearningCapability, ValidationCapability

logger = logging.getLogger(__name__)


class PipelineAgent(BaseAgent, LearningCapability, ValidationCapability):
    """
    Domain agent for CI/CD pipeline diagram generation with learning and validation.

    Features:
    - Gold standard learning (via LearningCapability)
    - Domain-specific validation (via ValidationCapability)
    - Multi-format support (Mermaid, PlantUML)
    - Pipeline optimization
    """

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        enable_learning: bool = True,
        strict_validation: bool = True,
        output_format: str = "mermaid",
    ):
        """
        Initialize PipelineAgent.

        Args:
            config: Agent runtime configuration
            enable_learning: Enable gold standard learning
            strict_validation: Enable strict validation mode
            output_format: Output format (mermaid, plantuml, etc.)
        """
        self.output_format = output_format

        # Initialize BaseAgent
        BaseAgent.__init__(
            self,
            config
            or AgentRuntimeConfig(
                name="PipelineAgent",
                system_prompt=f"You are an expert at creating {output_format} pipeline diagrams.",
            ),
        )

        # Initialize ValidationCapability
        ValidationCapability.__init__(
            self,
            domain=f"pipeline_{output_format}",
            strict_mode=strict_validation,
            quality_threshold=0.7,
        )

        # Initialize LearningCapability
        if enable_learning:
            LearningCapability.__init__(
                self,
                domain=f"pipeline_{output_format}",
                gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                domain_validator=self._validate_pipeline,
            )

        # Simple agent (no DSPy needed for pipelines)
        self._agent = None

        logger.info(
            "PipelineAgent initialized (learning={}, strict={}, format={})".format(
                enable_learning, strict_validation, output_format
            )
        )

    async def _execute_impl(self, task: str, **kwargs) -> Any:
        """
        Execute pipeline diagram generation.

        Args:
            task: Task description
            **kwargs: Additional parameters (stages, description, etc.)

        Returns:
            Generated pipeline diagram code
        """
        description = kwargs.get("description", task)
        stages = kwargs.get("stages", ["Build", "Test", "Deploy"])

        # Generate diagram
        diagram = await self._generate_pipeline(
            task=task, description=description, stages=stages, **kwargs
        )

        # Validate diagram
        validation = await self.validate(diagram, context=kwargs)

        if not validation["valid"]:
            logger.warning(f"Generated diagram failed validation: {validation['errors']}")

            # Try to fix errors if possible
            if strict_validation := kwargs.get("strict", self.strict_mode):
                # In strict mode, raise error
                raise ValueError(f"Pipeline validation failed: {validation['errors']}")

        # Learn from gold standards if enabled
        if hasattr(self, "learn_from_gold_standards"):
            diagram = await self.learn_from_gold_standards(task, diagram, **kwargs)

        return diagram

    async def _generate_pipeline(
        self, task: str, description: str, stages: List[str], **kwargs
    ) -> str:
        """
        Generate pipeline diagram.

        Args:
            task: Task description
            description: Pipeline description
            stages: List of pipeline stages
            **kwargs: Additional context

        Returns:
            Pipeline diagram code
        """
        # Get learned improvements if available
        if hasattr(self, "learned_improvements"):
            improvements = self.learned_improvements[:3]  # Top 3
            if improvements:
                # Use learned pattern if available
                for imp in improvements:
                    pattern = imp.get("teacher_output", "")
                    if pattern:
                        return pattern

        # Generate basic pipeline
        if self.output_format == "mermaid":
            mermaid = "graph LR\n"
            for i, stage in enumerate(stages):
                mermaid += f"    {chr(65+i)}[{stage}]\n"
            for i in range(len(stages) - 1):
                mermaid += f"    {chr(65+i)} --> {chr(65+i+1)}\n"
            return mermaid.strip()
        else:
            return f"Pipeline: {description}"

    async def _validate_pipeline(
        self, output: Any, expected: Optional[Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate pipeline diagram (domain validator for LearningCapability).

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

        # Check if it's Mermaid format
        if self.output_format == "mermaid":
            has_graph = "graph" in output_str.lower() or "flowchart" in output_str.lower()
            has_nodes = "[" in output_str
            has_arrow = "-->" in output_str
            syntax_valid = has_graph and has_nodes and has_arrow

            if not has_graph:
                errors.append("Missing graph declaration")
                score -= 0.3
            if not has_nodes:
                errors.append("Missing node definitions")
                score -= 0.3
            if not has_arrow:
                errors.append("Missing connections")
                score -= 0.3
        else:
            # For other formats, basic validation
            syntax_valid = len(output_str) > 10
            if not syntax_valid:
                errors.append("Diagram too short")
                score = 0.0

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
                "syntax_valid": len(errors) == 0,
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
        return await self._validate_pipeline(
            output=output,
            expected=expected,
            task=context.get("task", "") if context else "",
            context=context or {},
        )

    # =========================================================================
    # TRAINING AND VALIDATION DATA
    # =========================================================================

    def _get_default_training_cases(self) -> List[Dict[str, Any]]:
        """Get default training cases for pipelines."""
        if self.output_format == "mermaid":
            return [
                {
                    "input": "Generate CI/CD pipeline",
                    "output": """graph LR
    A[Build]
    B[Test]
    C[Deploy]
    A --> B
    B --> C""",
                    "context": {
                        "description": "Build, Test, Deploy",
                        "stages": ["Build", "Test", "Deploy"],
                    },
                },
                {
                    "input": "Generate complex pipeline",
                    "output": """graph TD
    A[Source]
    B[Build]
    C[Unit Tests]
    D[Integration Tests]
    E[Deploy Staging]
    F[Deploy Production]
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F""",
                    "context": {
                        "description": "Multi-stage with parallel steps",
                        "stages": [
                            "Source",
                            "Build",
                            "Unit Tests",
                            "Integration Tests",
                            "Deploy Staging",
                            "Deploy Production",
                        ],
                    },
                },
            ]
        else:
            return []

    def _get_default_validation_cases(self) -> List[Dict[str, Any]]:
        """Get default validation cases for pipelines."""
        if self.output_format == "mermaid":
            return [
                {
                    "input": "Generate deployment pipeline",
                    "output": """graph LR
    A[Code]
    B[Build]
    C[Test]
    D[Deploy]
    A --> B
    B --> C
    C --> D""",
                    "context": {
                        "description": "Deployment workflow",
                        "stages": ["Code", "Build", "Test", "Deploy"],
                    },
                }
            ]
        else:
            return []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_pipeline(
        self, stages: List[str], description: Optional[str] = None, **kwargs: Any
    ) -> str:
        """
        Generate a pipeline diagram.

        Args:
            stages: List of pipeline stage names
            description: Optional description
            **kwargs: Additional context

        Returns:
            Pipeline diagram code as string
        """
        result = await self.execute(
            task=f"Generate {self.output_format} pipeline diagram",
            stages=stages,
            description=description or f"Pipeline with {len(stages)} stages",
            **kwargs,
        )

        if result.success:
            return result.output
        else:
            raise ValueError(f"Pipeline generation failed: {result.error}")
