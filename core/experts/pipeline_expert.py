"""
Pipeline Expert Agent

A specialized expert agent for generating perfect CI/CD pipeline diagrams.
Uses OptimizationPipeline to ensure it always produces valid pipeline visualizations.
"""

import logging
from typing import Any, Dict, List, Optional

from .expert_agent import ExpertAgent, ExpertAgentConfig

logger = logging.getLogger(__name__)


class PipelineExpertAgent(ExpertAgent):
    """
    Expert agent for CI/CD pipeline diagram generation.
    
    This agent is pre-trained to always generate valid pipeline diagrams
    (can be Mermaid, PlantUML, or other formats).
    """
    
    def __init__(self, config: Optional[ExpertAgentConfig] = None, output_format: str = "mermaid", memory=None):
        """
        Initialize Pipeline Expert Agent.
        
        Args:
            config: Optional custom configuration
            output_format: Output format ("mermaid", "plantuml", etc.)
            memory: Optional HierarchicalMemory instance
        """
        self.output_format = output_format
        
        if config is None:
            config = ExpertAgentConfig(
                name=f"pipeline_expert_{output_format}",
                domain=f"pipeline_{output_format}",
                description=f"Expert agent for generating perfect {output_format} pipeline diagrams",
                training_gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                evaluation_function=self._evaluate_pipeline,
                agent_module=lambda: self._create_pipeline_agent(),
                teacher_module=lambda: self._create_pipeline_teacher()
            )
        
        super().__init__(config, memory=memory)
    
    def _create_default_agent(self) -> Any:
        """Create default pipeline generation agent."""
        return self._create_pipeline_agent()
    
    def _create_pipeline_agent(self):
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
    
    def _create_pipeline_teacher(self):
        """Create the pipeline teacher agent."""
        class PipelineTeacher:
            def forward(self, **kwargs) -> Any:
                """Provide correct pipeline diagram."""
                result = type('Result', (), {})()
                gold_standard = kwargs.get('gold_standard', '')
                result._store = {"output": gold_standard}
                return result
        
        return PipelineTeacher()
    
    @staticmethod
    async def _evaluate_pipeline(output: Any, gold_standard: str, task: str, context: Dict) -> Dict[str, Any]:
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
