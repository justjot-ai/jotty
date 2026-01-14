"""
Test Expert Agents

Tests for MermaidExpertAgent and PipelineExpertAgent.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent, PipelineExpertAgent


async def test_mermaid_expert():
    """Test MermaidExpertAgent training and generation."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT AGENT")
    print("=" * 80)
    print()
    
    # Create expert agent
    expert = MermaidExpertAgent()
    
    print("📚 Training Mermaid Expert...")
    training_results = await expert.train()
    
    print(f"Training Results:")
    print(f"  Success: {training_results.get('overall_success')}")
    print(f"  Passed: {training_results.get('passed_cases')}/{training_results.get('total_cases')}")
    print()
    
    # Validate (skip if training didn't fully succeed)
    if training_results.get('overall_success') or training_results.get('passed_cases', 0) > 0:
        print("✅ Validating Mermaid Expert...")
        validation_results = await expert.validate(skip_if_not_trained=True)
        
        if validation_results.get('validated', True):
            print(f"Validation Results:")
            print(f"  Passed: {validation_results.get('passed_cases')}/{validation_results.get('total_cases')}")
            print(f"  Average Score: {validation_results.get('average_score', 0):.2f}")
        else:
            print(f"  Skipped: {validation_results.get('reason')}")
        print()
    
    # Generate
    print("🎨 Generating Mermaid Diagram...")
    diagram = await expert.generate_mermaid(
        description="User login flow with validation",
        diagram_type="flowchart"
    )
    
    print("Generated Diagram:")
    print("```mermaid")
    print(diagram)
    print("```")
    print()
    
    # Status
    status = expert.get_status()
    print("Expert Status:")
    print(f"  Trained: {status['trained']}")
    print(f"  Validated: {status['validation_passed']}")
    print(f"  Improvements: {status['improvements_count']}")
    print()
    
    return expert


async def test_pipeline_expert():
    """Test PipelineExpertAgent training and generation."""
    print("=" * 80)
    print("TESTING PIPELINE EXPERT AGENT")
    print("=" * 80)
    print()
    
    # Create expert agent
    expert = PipelineExpertAgent(output_format="mermaid")
    
    print("📚 Training Pipeline Expert...")
    training_results = await expert.train()
    
    print(f"Training Results:")
    print(f"  Success: {training_results.get('overall_success')}")
    print(f"  Passed: {training_results.get('passed_cases')}/{training_results.get('total_cases')}")
    print()
    
    # Generate
    print("🎨 Generating Pipeline Diagram...")
    pipeline = await expert.generate_pipeline(
        stages=["Build", "Test", "Deploy", "Release"],
        description="CI/CD Pipeline"
    )
    
    print("Generated Pipeline:")
    print("```mermaid")
    print(pipeline)
    print("```")
    print()
    
    return expert


async def main():
    """Run all expert agent tests."""
    print("\n" + "=" * 80)
    print("EXPERT AGENTS TEST SUITE")
    print("=" * 80)
    print()
    
    # Test Mermaid Expert
    mermaid_expert = await test_mermaid_expert()
    
    print("\n" + "=" * 80 + "\n")
    
    # Test Pipeline Expert
    pipeline_expert = await test_pipeline_expert()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)
    
    return mermaid_expert, pipeline_expert


if __name__ == "__main__":
    asyncio.run(main())
