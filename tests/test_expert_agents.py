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
    """Test MermaidExpertAgent (BaseExpert-based) training data and generation."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT AGENT")
    print("=" * 80)
    print()

    # Create expert agent
    expert = MermaidExpertAgent()

    # Verify training data is available (replaces old .train() call)
    print("Checking training data...")
    training_data = expert.get_training_data()
    print(f"Training Data:")
    print(f"  Cases available: {len(training_data)}")
    print()

    # Verify validation data is available
    validation_data = expert.get_validation_data()
    print(f"Validation Data:")
    print(f"  Cases available: {len(validation_data)}")
    print()

    # Status (uses get_stats() instead of old get_status())
    stats = expert.get_stats()
    print("Expert Stats:")
    print(f"  Domain: {stats['domain']}")
    print(f"  Expert Type: {stats['expert_type']}")
    print(f"  Training Cases: {stats['training_cases']}")
    print(f"  Validation Cases: {stats['validation_cases']}")
    print(f"  Improvements: {stats['improvements_count']}")
    print()

    return expert


async def test_pipeline_expert():
    """Test PipelineExpertAgent (BaseExpert-based) training data and stats."""
    print("=" * 80)
    print("TESTING PIPELINE EXPERT AGENT")
    print("=" * 80)
    print()

    # Create expert agent
    expert = PipelineExpertAgent(output_format="mermaid")

    # Verify training data
    training_data = expert.get_training_data()
    print(f"Training Data:")
    print(f"  Cases available: {len(training_data)}")
    print()

    # Stats
    stats = expert.get_stats()
    print("Expert Stats:")
    print(f"  Domain: {stats['domain']}")
    print(f"  Expert Type: {stats['expert_type']}")
    print(f"  Training Cases: {stats['training_cases']}")
    print(f"  Validation Cases: {stats['validation_cases']}")
    print(f"  Improvements: {stats['improvements_count']}")
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
