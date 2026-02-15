"""
Mermaid Expert Agent

A specialized expert agent for generating perfect Mermaid diagrams.
Uses OptimizationPipeline to ensure it always produces valid, correct Mermaid syntax.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)


class MermaidExpertAgent(BaseExpert):
    """
    Expert agent for Mermaid diagram generation.

    This agent is pre-trained to always generate valid Mermaid diagrams.
    It uses OptimizationPipeline internally to learn from mistakes.
    """

    # =========================================================================
    # REQUIRED PROPERTIES (BaseExpert interface)
    # =========================================================================

    @property
    def domain(self) -> str:
        """Return domain name for this expert."""
        return "mermaid"

    @property
    def description(self) -> str:
        """Return description for this expert."""
        return "Expert agent for generating perfect Mermaid diagrams"

    # =========================================================================
    # DOMAIN-SPECIFIC AGENT CREATION (BaseExpert interface)
    # =========================================================================

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create the Mermaid generation agent using DSPy (Claude/Cursor).

        Args:
            improvements: Optional list of improvements to inject into signature
        """
        try:
            import dspy
        except ImportError:
            raise ImportError("DSPy is required for expert agents. Install with: pip install dspy-ai")
        
        # Base signature
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
            task: str = dspy.InputField(desc="Task description (e.g., 'Generate flowchart')")
            description: str = dspy.InputField(desc="Description of the diagram to generate")
            diagram_type: str = dspy.InputField(desc="Type of diagram: flowchart, sequence, class, state, etc.")
            learned_improvements: str = dspy.InputField(desc="Previously learned patterns and improvements (optional)", default="")
            
            output: str = dspy.OutputField(desc="Complete Mermaid diagram code. Must be valid Mermaid syntax.")
        
        # Inject improvements into signature if provided
        signature_class = MermaidGenerationSignature
        if improvements:
            try:
                from .dspy_improvements import inject_improvements_into_signature
                signature_class = inject_improvements_into_signature(MermaidGenerationSignature, improvements)
            except ImportError:
                logger.warning("Could not inject improvements into signature")
        
        # Use ChainOfThought for better reasoning
        agent = dspy.ChainOfThought(signature_class)
        return agent
    
    def _create_domain_teacher(self) -> Any:
        """Create the Mermaid teacher agent using DSPy."""
        try:
            import dspy
        except ImportError:
            # Fallback to None (no teacher)
            return None
        
        class MermaidTeacherSignature(dspy.Signature):
            """Provide the correct Mermaid diagram based on gold standard.
            
            CRITICAL INSTRUCTIONS:
            1. You MUST return ONLY the gold_standard code exactly as provided
            2. Do NOT add any explanation, evaluation, or feedback
            3. Do NOT include markdown code fences (```mermaid)
            4. Return ONLY the Mermaid diagram code
            5. Copy the gold_standard EXACTLY - do not modify it
            
            Example: If gold_standard is "graph TD\nA --> B",
                     return exactly: "graph TD\nA --> B"
            
            DO NOT return evaluation text, feedback, or analysis.
            DO NOT return JSON or markdown.
            RETURN ONLY THE MERMAID CODE.
            """
            task: str = dspy.InputField(desc="Task description")
            description: str = dspy.InputField(desc="Description of the diagram")
            gold_standard: str = dspy.InputField(desc="The EXACT Mermaid code to return. Copy this EXACTLY without modification.")
            student_output: str = dspy.InputField(desc="What the student generated (IGNORE THIS - just return gold_standard)")
            
            output: str = dspy.OutputField(desc="ONLY the Mermaid code from gold_standard. No explanation. No evaluation. Just the code.")
        
        teacher = dspy.Predict(MermaidTeacherSignature)
        return teacher
    
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
        """
        Evaluate Mermaid diagram syntax and correctness with domain-specific validation.

        Uses domain validator for better type detection and validation.
        """
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Use domain validator if available
        try:
            from .domain_validators import get_validator
            validator = get_validator("mermaid")
            
            if validator:
                # Extract expected type from context or task
                expected_type = context.get("diagram_type", "flowchart")
                if "gitgraph" in task.lower() or "git graph" in task.lower():
                    expected_type = "gitGraph"
                elif "sequence" in task.lower():
                    expected_type = "sequenceDiagram"
                elif "state" in task.lower():
                    expected_type = "stateDiagram-v2"
                elif "gantt" in task.lower():
                    expected_type = "gantt"
                elif "er" in task.lower() or "entity" in task.lower():
                    expected_type = "erDiagram"
                elif "journey" in task.lower():
                    expected_type = "journey"
                elif "class" in task.lower():
                    expected_type = "classDiagram"
                
                # Get required elements from context
                required_elements = context.get("required_elements", [])
                
                is_valid, error_msg, metadata = validator.validate(
                    output=output_str,
                    expected_type=expected_type,
                    context={"task": task, "required_elements": required_elements}
                )
                
                # Calculate score based on validation
                score = 1.0 if is_valid else 0.5
                if metadata.get("type_match"):
                    score = min(1.0, score + 0.3)
                if metadata.get("element_coverage"):
                    score = min(1.0, score + metadata["element_coverage"] * 0.2)
                
                # Check if matches gold standard
                matches_gold = output_str == gold_str
                if matches_gold:
                    score = 1.0
                
                return {
                    "score": score,
                    "status": "CORRECT" if is_valid and matches_gold else ("PARTIAL" if is_valid else "INCORRECT"),
                    "syntax_valid": is_valid,
                    "matches_gold": matches_gold,
                    "error": error_msg if not is_valid else None,
                    "metadata": metadata,
                    "difference": error_msg if not is_valid else None
                }
        except Exception as e:
            logger.warning(f"Domain validator failed: {e}, using default evaluation")
        
        # Fallback to original evaluation
        # Syntax validation
        has_graph = "graph" in output_str.lower() or "flowchart" in output_str.lower()
        has_nodes = "[" in output_str or "{" in output_str
        has_arrow = "-->" in output_str
        
        syntax_valid = has_graph and has_nodes and has_arrow
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
        
        issues = []
        if not has_graph:
            issues.append("Missing 'graph' or 'flowchart' declaration")
        if not has_nodes:
            issues.append("Missing node definitions")
        if not has_arrow:
            issues.append("Missing arrow connections")
        if not matches_gold:
            issues.append("Output doesn't match gold standard")
        
        return {
            "score": score,
            "status": status,
            "syntax_valid": syntax_valid,
            "matches_gold": matches_gold,
            "issues": issues,
            "difference": "; ".join(issues) if issues else None
        }
    
    # =========================================================================
    # TRAINING AND VALIDATION DATA (BaseExpert interface)
    # =========================================================================

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """Get default training cases for Mermaid."""
        return [
            {
                "task": "Generate simple flowchart",
                "context": {"description": "Start to End flow"},
                "gold_standard": "graph TD\n    A[Start]\n    B[End]\n    A --> B"
            },
            {
                "task": "Generate decision flowchart",
                "context": {"description": "User login with validation"},
                "gold_standard": """graph TD
    A[User Login]
    B{Valid?}
    C[Show Dashboard]
    D[Show Error]
    A --> B
    B -->|Yes| C
    B -->|No| D"""
            },
            {
                "task": "Generate sequence diagram",
                "context": {"description": "Client-Server interaction"},
                "gold_standard": """sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: Request
    S-->>C: Response"""
            },
            {
                "task": "Generate class diagram",
                "context": {"description": "Basic class structure"},
                "gold_standard": """classDiagram
    class Animal {
        +name: string
        +speak()
    }
    class Dog {
        +breed: string
    }
    Animal <|-- Dog"""
            }
        ]
    
    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """Get default validation cases for Mermaid."""
        return [
            {
                "task": "Generate workflow diagram",
                "context": {"description": "Process workflow"},
                "gold_standard": """graph TD
    A[Start]
    B[Process]
    C[End]
    A --> B
    B --> C"""
            },
            {
                "task": "Generate state diagram",
                "context": {"description": "State transitions"},
                "gold_standard": """stateDiagram-v2
    [*] --> State1
    State1 --> State2
    State2 --> [*]"""
            }
        ]
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def generate_mermaid(self, description: str, diagram_type: str = 'flowchart', **kwargs: Any) -> str:
        """
        Generate a Mermaid diagram using the trained expert agent.

        The agent uses DSPy (Claude/Cursor) to generate diagrams from descriptions,
        and has been trained by OptimizationPipeline to always produce correct outputs.

        Args:
            description: Description of the diagram to generate
            diagram_type: Type of diagram (flowchart, sequence, class, etc.)
            **kwargs: Additional context

        Returns:
            Mermaid diagram code as string
        """
        task = f"Generate {diagram_type} diagram"
        context = {
            "description": description,
            "diagram_type": diagram_type,
            **kwargs
        }
        
        output = await self.generate(task=task, context=context)
        return str(output)
