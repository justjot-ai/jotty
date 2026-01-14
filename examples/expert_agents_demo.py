"""
Expert Agents Demo

Demonstrates how to use expert agents in Jotty.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import (
    get_mermaid_expert_async,
    get_pipeline_expert_async,
    MermaidExpertAgent,
    PipelineExpertAgent
)


async def demo_mermaid_expert():
    """Demonstrate Mermaid Expert Agent."""
    print("=" * 80)
    print("MERMAID EXPERT AGENT DEMO")
    print("=" * 80)
    print()
    
    # Get expert (auto-trains if needed)
    print("📚 Getting Mermaid Expert...")
    expert = await get_mermaid_expert_async(auto_train=True)
    
    print("✅ Expert ready!")
    print()
    
    # Generate various diagrams
    print("🎨 Generating Diagrams:")
    print()
    
    # Flowchart
    print("1. Flowchart:")
    flowchart = await expert.generate_mermaid(
        description="User registration process",
        diagram_type="flowchart"
    )
    print("```mermaid")
    print(flowchart)
    print("```")
    print()
    
    # Status
    status = expert.get_status()
    print(f"Expert Status:")
    print(f"  Trained: {status['trained']}")
    print(f"  Improvements Learned: {status['improvements_count']}")
    print()


async def demo_pipeline_expert():
    """Demonstrate Pipeline Expert Agent."""
    print("=" * 80)
    print("PIPELINE EXPERT AGENT DEMO")
    print("=" * 80)
    print()
    
    # Get expert
    print("📚 Getting Pipeline Expert...")
    expert = await get_pipeline_expert_async(output_format="mermaid", auto_train=True)
    
    print("✅ Expert ready!")
    print()
    
    # Generate pipeline
    print("🎨 Generating CI/CD Pipeline:")
    print()
    
    pipeline = await expert.generate_pipeline(
        stages=["Source", "Build", "Test", "Deploy", "Release"],
        description="Full CI/CD Pipeline"
    )
    
    print("```mermaid")
    print(pipeline)
    print("```")
    print()


async def demo_custom_training():
    """Demonstrate custom training."""
    print("=" * 80)
    print("CUSTOM TRAINING DEMO")
    print("=" * 80)
    print()
    
    # Create expert with custom training cases
    expert = MermaidExpertAgent()
    
    # Custom training cases
    custom_cases = [
        {
            "task": "Generate custom diagram",
            "context": {"description": "My custom flow"},
            "gold_standard": """graph TD
    A[Start]
    B[Custom Step]
    C[End]
    A --> B
    B --> C"""
        }
    ]
    
    print("📚 Training on custom cases...")
    results = await expert.train(gold_standards=custom_cases)
    
    print(f"Training Results:")
    print(f"  Passed: {results['passed_cases']}/{results['total_cases']}")
    print()
    
    # Generate using learned patterns
    print("🎨 Generating with learned patterns:")
    diagram = await expert.generate_mermaid(
        description="My custom flow",
        diagram_type="flowchart"
    )
    print("```mermaid")
    print(diagram)
    print("```")
    print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("EXPERT AGENTS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Mermaid Expert
    await demo_mermaid_expert()
    
    print("\n" + "=" * 80 + "\n")
    
    # Pipeline Expert
    await demo_pipeline_expert()
    
    print("\n" + "=" * 80 + "\n")
    
    # Custom Training
    await demo_custom_training()
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
