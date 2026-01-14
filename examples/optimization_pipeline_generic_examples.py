"""
Generic Examples: OptimizationPipeline for Various Domains

This demonstrates how OptimizationPipeline works for ANY domain:
- Markdown generation
- Mermaid diagram generation
- PlantUML diagram generation
- Code generation
- Documentation generation
- Any other domain!

# ✅ GENERIC: No domain-specific assumptions
"""

import asyncio
import dspy
from pathlib import Path

from jotty.core.jotty import (
    OptimizationPipeline,
    OptimizationConfig,
    AgentConfig,
    create_optimization_pipeline
)


# =============================================================================
# Example 1: Markdown Documentation Generation
# =============================================================================

class MarkdownAnalyzer(dspy.Module):
    """Analyzes requirements and extracts structure."""
    def forward(self, requirements: str) -> str:
        return f"Structure: {requirements}"

class MarkdownGenerator(dspy.Module):
    """Generates markdown content."""
    def forward(self, structure: str) -> str:
        return f"# Generated Markdown\n\n{structure}\n\n## Content\n\nGenerated content here."

async def example_markdown_generation():
    """Optimize markdown generation."""
    agents = [
        AgentConfig(
            name="analyzer",
            agent=MarkdownAnalyzer(),
            outputs=["structure"]
        ),
        AgentConfig(
            name="generator",
            agent=MarkdownGenerator(),
            parameter_mappings={"structure": "structure"}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        output_path="./outputs/markdown_example"
    )
    
    def evaluate_markdown(output, gold_standard, task, context):
        """Evaluate markdown quality."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Simple comparison (in real use, you'd check structure, formatting, etc.)
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate_markdown
    
    result = await pipeline.optimize(
        task="Generate API documentation",
        context={"requirements": "Document REST endpoints"},
        gold_standard="# Generated Markdown\n\nStructure: Document REST endpoints\n\n## Content\n\nGenerated content here."
    )
    
    print(f"Markdown Generation: {'✓' if result['optimization_complete'] else '✗'}")
    return result


# =============================================================================
# Example 2: Mermaid Diagram Generation
# =============================================================================

class MermaidAnalyzer(dspy.Module):
    """Analyzes requirements for diagram structure."""
    def forward(self, description: str) -> str:
        return f"Entities: {description}"

class MermaidGenerator(dspy.Module):
    """Generates Mermaid diagram syntax."""
    def forward(self, entities: str) -> str:
        return f"graph TD\n    A[{entities}]\n    B[Process]\n    A --> B"

async def example_mermaid_generation():
    """Optimize Mermaid diagram generation."""
    agents = [
        AgentConfig(
            name="analyzer",
            agent=MermaidAnalyzer(),
            outputs=["entities"]
        ),
        AgentConfig(
            name="generator",
            agent=MermaidGenerator(),
            parameter_mappings={"entities": "entities"}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        output_path="./outputs/mermaid_example"
    )
    
    def evaluate_mermaid(output, gold_standard, task, context):
        """Evaluate Mermaid diagram syntax."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check if it's valid Mermaid syntax (basic check)
        is_valid = "graph" in output_str.lower() or "flowchart" in output_str.lower()
        matches_gold = output_str == gold_str
        
        score = 1.0 if (matches_gold and is_valid) else 0.5 if is_valid else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate_mermaid
    
    result = await pipeline.optimize(
        task="Generate workflow diagram",
        context={"description": "User login flow"},
        gold_standard="graph TD\n    A[User login flow]\n    B[Process]\n    A --> B"
    )
    
    print(f"Mermaid Generation: {'✓' if result['optimization_complete'] else '✗'}")
    return result


# =============================================================================
# Example 3: PlantUML Diagram Generation
# =============================================================================

class PlantUMLAnalyzer(dspy.Module):
    """Analyzes system architecture requirements."""
    def forward(self, architecture: str) -> str:
        return f"Components: {architecture}"

class PlantUMLGenerator(dspy.Module):
    """Generates PlantUML diagram syntax."""
    def forward(self, components: str) -> str:
        return f"@startuml\nclass {components} {{\n}}\n@enduml"

async def example_plantuml_generation():
    """Optimize PlantUML diagram generation."""
    agents = [
        AgentConfig(
            name="analyzer",
            agent=PlantUMLAnalyzer(),
            outputs=["components"]
        ),
        AgentConfig(
            name="generator",
            agent=PlantUMLGenerator(),
            parameter_mappings={"components": "components"}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        output_path="./outputs/plantuml_example"
    )
    
    def evaluate_plantuml(output, gold_standard, task, context):
        """Evaluate PlantUML diagram syntax."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check if it's valid PlantUML syntax
        is_valid = "@startuml" in output_str and "@enduml" in output_str
        matches_gold = output_str == gold_str
        
        score = 1.0 if (matches_gold and is_valid) else 0.5 if is_valid else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate_plantuml
    
    result = await pipeline.optimize(
        task="Generate class diagram",
        context={"architecture": "UserService"},
        gold_standard="@startuml\nclass UserService {\n}\n@enduml"
    )
    
    print(f"PlantUML Generation: {'✓' if result['optimization_complete'] else '✗'}")
    return result


# =============================================================================
# Example 4: Code Generation with Teacher Model
# =============================================================================

class CodeAnalyzer(dspy.Module):
    """Analyzes code requirements."""
    def forward(self, requirements: str) -> str:
        return f"Requirements: {requirements}"

class CodeGenerator(dspy.Module):
    """Generates code (student - makes mistakes)."""
    def forward(self, requirements: str) -> str:
        # Student makes a mistake
        return f"def function():\n    pass  # Missing implementation"

class CodeTeacher(dspy.Module):
    """Teacher that generates correct code."""
    def forward(self, requirements: str, student_output: str, gold_standard: str, evaluation_feedback: str) -> str:
        return gold_standard  # Teacher provides correct answer

async def example_code_generation_with_teacher():
    """Optimize code generation with teacher model."""
    agents = [
        AgentConfig(
            name="analyzer",
            agent=CodeAnalyzer(),
            outputs=["requirements"]
        ),
        AgentConfig(
            name="generator",
            agent=CodeGenerator(),
            parameter_mappings={"requirements": "requirements"}
        ),
        AgentConfig(
            name="teacher",
            agent=CodeTeacher(),
            metadata={"is_teacher": True}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=5,
        required_pass_count=2,
        enable_teacher_model=True,
        output_path="./outputs/code_example"
    )
    
    def evaluate_code(output, gold_standard, task, context):
        """Evaluate generated code."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check syntax and correctness
        has_implementation = "return" in output_str or "print" in output_str or "pass" not in output_str
        matches_gold = output_str == gold_str
        
        score = 1.0 if matches_gold else (0.5 if has_implementation else 0.0)
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate_code
    
    result = await pipeline.optimize(
        task="Generate Python function",
        context={"requirements": "Add two numbers"},
        gold_standard="def add(a, b):\n    return a + b"
    )
    
    teacher_used = any(it.get('has_teacher_output') for it in result['iterations'])
    print(f"Code Generation: {'✓' if result['optimization_complete'] else '✗'}")
    print(f"  Teacher used: {teacher_used}")
    return result


# =============================================================================
# Example 5: Multi-Format Content Generation
# =============================================================================

class ContentPlanner(dspy.Module):
    """Plans content structure."""
    def forward(self, topic: str) -> str:
        return f"Plan for: {topic}"

class FormatSelector(dspy.Module):
    """Selects appropriate format."""
    def forward(self, plan: str) -> str:
        return "markdown" if "doc" in plan.lower() else "diagram"

class ContentGenerator(dspy.Module):
    """Generates content in selected format."""
    def forward(self, plan: str, format: str) -> str:
        if format == "markdown":
            return f"# {plan}\n\nContent here."
        else:
            return f"graph TD\n    A[{plan}]"

async def example_multi_format_generation():
    """Optimize multi-format content generation."""
    agents = [
        AgentConfig(
            name="planner",
            agent=ContentPlanner(),
            outputs=["plan"]
        ),
        AgentConfig(
            name="format_selector",
            agent=FormatSelector(),
            parameter_mappings={"plan": "plan"},
            outputs=["format"]
        ),
        AgentConfig(
            name="generator",
            agent=ContentGenerator(),
            parameter_mappings={"plan": "plan", "format": "format"}
        )
    ]
    
    pipeline = create_optimization_pipeline(
        agents=agents,
        max_iterations=3,
        required_pass_count=1,
        output_path="./outputs/multi_format_example"
    )
    
    def evaluate_content(output, gold_standard, task, context):
        """Evaluate generated content."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    
    pipeline.config.evaluation_function = evaluate_content
    
    result = await pipeline.optimize(
        task="Generate documentation",
        context={"topic": "API Reference"},
        gold_standard="# Plan for: API Reference\n\nContent here."
    )
    
    print(f"Multi-Format Generation: {'✓' if result['optimization_complete'] else '✗'}")
    return result


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all generic examples."""
    print("=" * 80)
    print("Generic OptimizationPipeline Examples")
    print("=" * 80)
    
    print("\n1. Markdown Generation:")
    await example_markdown_generation()
    
    print("\n2. Mermaid Diagram Generation:")
    await example_mermaid_generation()
    
    print("\n3. PlantUML Diagram Generation:")
    await example_plantuml_generation()
    
    print("\n4. Code Generation with Teacher:")
    await example_code_generation_with_teacher()
    
    print("\n5. Multi-Format Content Generation:")
    await example_multi_format_generation()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Configure DSPy (use a real LM in production)
    try:
        dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))
    except:
        # Fallback if no LM configured
        print("Note: Configure DSPy LM for full functionality")
    
    # Run examples
    asyncio.run(main())
