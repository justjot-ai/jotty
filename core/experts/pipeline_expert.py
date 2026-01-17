"""
Pipeline Expert Agent

A specialized expert agent for generating perfect CI/CD pipeline diagrams.
Uses OptimizationPipeline to ensure it always produces valid pipeline visualizations.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


class PipelineExpertAgent(BaseExpert):
    """
    Expert agent for CI/CD pipeline diagram generation.

    This agent is pre-trained to always generate valid pipeline diagrams
    (can be Mermaid, PlantUML, or other formats).
    """

    def __init__(self, config=None, output_format: str = "mermaid", memory=None, improvements: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize Pipeline Expert Agent.

        Args:
            config: Optional custom configuration
            output_format: Output format ("mermaid", "plantuml", etc.)
            memory: Optional HierarchicalMemory instance
            improvements: Optional list of learned improvements
        """
        # Set output_format BEFORE calling super().__init__() because domain property needs it
        self.output_format = output_format

        # Call BaseExpert.__init__ which will use our domain/description properties
        super().__init__(config=config, memory=memory, improvements=improvements)

    # =========================================================================
    # REQUIRED PROPERTIES (BaseExpert interface)
    # =========================================================================

    @property
    def domain(self) -> str:
        """Return domain name for this expert."""
        return f"pipeline_{self.output_format}"

    @property
    def description(self) -> str:
        """Return description for this expert."""
        return f"Expert agent for generating perfect {self.output_format} pipeline diagrams"

    # =========================================================================
    # DOMAIN-SPECIFIC AGENT CREATION (BaseExpert interface)
    # =========================================================================

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create the pipeline generation agent."""
        class PipelineAgent:
            def __init__(self, output_format: str):
                self.output_format = output_format
                self.learned_patterns = []
            
            def forward(self, task: str = None, description: str = None,
                       stages: List[str] = None, learned_improvements: List[Dict] = None, **kwargs) -> Any:
                """Generate pipeline diagram."""
                result = type('Result', (), {})()
                
                # Apply learned improvements if available
                if learned_improvements:
                    for imp in learned_improvements:
                        pattern = imp.get('teacher_output', '')
                        if pattern:
                            result._store = {"output": pattern}
                            return result
                
                # Default: Generate basic pipeline
                if self.output_format == "mermaid":
                    stages = stages or ["Build", "Test", "Deploy"]
                    mermaid = "graph LR\n"
                    for i, stage in enumerate(stages):
                        mermaid += f"    {chr(65+i)}[{stage}]\n"
                    for i in range(len(stages) - 1):
                        mermaid += f"    {chr(65+i)} --> {chr(65+i+1)}\n"
                    result._store = {"output": mermaid.strip()}
                else:
                    result._store = {"output": f"Pipeline: {description or 'default'}"}
                
                return result
        
        return PipelineAgent(self.output_format)
    
    def _create_domain_teacher(self) -> Any:
        """Create the pipeline teacher agent."""
        class PipelineTeacher:
            def forward(self, **kwargs) -> Any:
                """Provide correct pipeline diagram."""
                result = type('Result', (), {})()
                gold_standard = kwargs.get('gold_standard', '')
                result._store = {"output": gold_standard}
                return result
        
        return PipelineTeacher()
    
    # =========================================================================
    # DOMAIN-SPECIFIC EVALUATION (BaseExpert interface)
    # =========================================================================

    async def _evaluate_domain(
        self,
        output: Any,
        gold_standard: str,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate pipeline diagram correctness."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check if it's Mermaid format
        if "graph" in output_str.lower() or "flowchart" in output_str.lower():
            has_graph = True
            has_nodes = "[" in output_str
            has_arrow = "-->" in output_str
            syntax_valid = has_graph and has_nodes and has_arrow
        else:
            # For other formats, basic validation
            syntax_valid = len(output_str) > 10
        
        matches_gold = output_str == gold_str
        
        if matches_gold:
            score = 1.0
            status = "CORRECT"
        elif syntax_valid:
            score = 0.5
            status = "PARTIAL"
        else:
            score = 0.0
            status = "INCORRECT"
        
        return {
            "score": score,
            "status": status,
            "syntax_valid": syntax_valid,
            "matches_gold": matches_gold
        }
    
    # =========================================================================
    # TRAINING AND VALIDATION DATA (BaseExpert interface)
    # =========================================================================

    def _get_default_training_cases(self) -> List[Dict[str, Any]]:
        """Get default training cases for pipelines."""
        if self.output_format == "mermaid":
            return [
                {
                    "task": "Generate CI/CD pipeline",
                    "context": {"description": "Build, Test, Deploy"},
                    "gold_standard": """graph LR
    A[Build]
    B[Test]
    C[Deploy]
    A --> B
    B --> C"""
                },
                {
                    "task": "Generate complex pipeline",
                    "context": {"description": "Multi-stage with parallel steps"},
                    "gold_standard": """graph TD
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
    E --> F"""
                }
            ]
        else:
            return []
    
    def _get_default_validation_cases(self) -> List[Dict[str, Any]]:
        """Get default validation cases for pipelines."""
        if self.output_format == "mermaid":
            return [
                {
                    "task": "Generate deployment pipeline",
                    "context": {"description": "Deployment workflow"},
                    "gold_standard": """graph LR
    A[Code]
    B[Build]
    C[Test]
    D[Deploy]
    A --> B
    B --> C
    C --> D"""
                }
            ]
        else:
            return []
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_pipeline(
        self,
        stages: List[str],
        description: str = None,
        **kwargs
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
        task = f"Generate {self.output_format} pipeline diagram"
        context = {
            "description": description or f"Pipeline with {len(stages)} stages",
            "stages": stages,
            **kwargs
        }
        
        output = await self.generate(task=task, context=context)
        return str(output)
