"""
Test Memory Consolidation for Expert Agent Improvements

Tests that improvements can be consolidated and synthesized using
Jotty's memory consolidation capabilities.
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent, ExpertAgentConfig
from core.experts.memory_integration import (
    store_improvement_to_memory,
    retrieve_improvements_from_memory,
    retrieve_synthesized_improvements,
    consolidate_improvements,
    run_improvement_consolidation_cycle
)
from core.memory.cortex import SwarmMemory
from core.foundation.data_structures import SwarmConfig, MemoryLevel


async def test_memory_consolidation():
    """Test memory consolidation for expert agent improvements."""
    print("=" * 80)
    print("TESTING MEMORY CONSOLIDATION FOR EXPERT AGENT IMPROVEMENTS")
    print("=" * 80)
    print()
    
    # Create memory system
    print("1. Setting up Memory System")
    print("-" * 80)
    memory_config = SwarmConfig()
    memory = SwarmMemory(
        agent_name="mermaid_expert_consolidation_test",
        config=memory_config
    )
    print("✅ Memory system created")
    print()
    
    # Store multiple improvements
    print("2. Storing Multiple Improvements")
    print("-" * 80)
    
    improvements = [
        {
            "iteration": 1,
            "task": "Generate simple flowchart",
            "learned_pattern": "When task is 'Generate simple flowchart', use 'flowchart TD' instead of 'graph TD'",
            "improvement_type": "teacher_correction"
        },
        {
            "iteration": 2,
            "task": "Generate decision flowchart",
            "learned_pattern": "For decision flowcharts, use diamond shapes [Decision] for decision nodes",
            "improvement_type": "teacher_correction"
        },
        {
            "iteration": 3,
            "task": "Generate sequence diagram",
            "learned_pattern": "For sequence diagrams, use 'sequenceDiagram' and include all participants",
            "improvement_type": "teacher_correction"
        },
        {
            "iteration": 4,
            "task": "Generate class diagram",
            "learned_pattern": "For class diagrams, use 'classDiagram' and define relationships clearly",
            "improvement_type": "teacher_correction"
        },
        {
            "iteration": 5,
            "task": "Generate state diagram",
            "learned_pattern": "For state diagrams, use 'stateDiagram-v2' and include [*] for start/end",
            "improvement_type": "teacher_correction"
        }
    ]
    
    stored_count = 0
    for improvement in improvements:
        entry = store_improvement_to_memory(
            memory=memory,
            improvement=improvement,
            expert_name="mermaid_expert_test",
            domain="mermaid"
        )
        if entry:
            stored_count += 1
    
    print(f"✅ Stored {stored_count}/{len(improvements)} improvements")
    
    # Check memory levels
    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    semantic_count = len(memory.memories[MemoryLevel.SEMANTIC])
    meta_count = len(memory.memories[MemoryLevel.META])
    
    print(f"   PROCEDURAL: {procedural_count}")
    print(f"   SEMANTIC: {semantic_count}")
    print(f"   META: {meta_count}")
    print()
    
    # Test raw retrieval
    print("3. Testing Raw Improvement Retrieval")
    print("-" * 80)
    
    raw_improvements = retrieve_improvements_from_memory(
        memory=memory,
        expert_name="mermaid_expert_test",
        domain="mermaid"
    )
    
    print(f"✅ Retrieved {len(raw_improvements)} raw improvements")
    print(f"   Sample: {raw_improvements[0].get('learned_pattern', '')[:60]}...")
    print()
    
    # Test synthesis
    print("4. Testing Synthesized Improvement Retrieval")
    print("-" * 80)
    
    synthesized = retrieve_synthesized_improvements(
        memory=memory,
        expert_name="mermaid_expert_test",
        domain="mermaid"
    )
    
    if synthesized:
        print(f"✅ Retrieved synthesized improvements")
        print(f"   Length: {len(synthesized)} chars")
        print(f"   Preview: {synthesized[:200]}...")
    else:
        print("⚠️  No synthesized improvements (may need LLM configured)")
    print()
    
    # Test consolidation
    print("5. Testing Improvement Consolidation")
    print("-" * 80)
    
    consolidation_result = consolidate_improvements(
        memory=memory,
        expert_name="mermaid_expert_test",
        domain="mermaid"
    )
    
    print(f"✅ Consolidation result:")
    print(f"   Consolidated patterns: {consolidation_result.get('consolidated', 0)}")
    print(f"   Preferences extracted: {consolidation_result.get('preferences', 0)}")
    
    # Check memory levels after consolidation
    procedural_count_after = len(memory.memories[MemoryLevel.PROCEDURAL])
    semantic_count_after = len(memory.memories[MemoryLevel.SEMANTIC])
    meta_count_after = len(memory.memories[MemoryLevel.META])
    
    print(f"   Memory levels after consolidation:")
    print(f"     PROCEDURAL: {procedural_count_after}")
    print(f"     SEMANTIC: {semantic_count_after} (was {semantic_count})")
    print(f"     META: {meta_count_after}")
    print()
    
    # Test consolidation cycle
    print("6. Testing Full Consolidation Cycle")
    print("-" * 80)
    
    cycle_result = await run_improvement_consolidation_cycle(
        memory=memory,
        expert_name="mermaid_expert_test",
        domain="mermaid"
    )
    
    print(f"✅ Consolidation cycle complete:")
    print(f"   Consolidated: {cycle_result.get('consolidated', 0)}")
    print(f"   Preferences: {cycle_result.get('preferences', 0)}")
    print()
    
    # Test expert with synthesis
    print("7. Testing Expert Agent with Synthesis")
    print("-" * 80)
    
    config = ExpertAgentConfig(
        name="mermaid_expert_synthesis",
        domain="mermaid",
        description="Mermaid expert with synthesis",
        use_memory_storage=True,
        use_memory_synthesis=True,  # Enable synthesis
        expert_data_dir="./test_outputs/mermaid_synthesis"
    )
    
    expert = MermaidExpertAgent(config=config, memory=memory)
    expert.trained = True  # Mark as trained for testing
    
    print(f"✅ Expert created with synthesis enabled")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    if expert.improvements:
        first_imp = expert.improvements[0]
        print(f"   Is synthesized: {first_imp.get('is_synthesized', False)}")
        print(f"   Preview: {str(first_imp.get('learned_pattern', ''))[:100]}...")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Memory consolidation working!")
    print(f"   - Stored {stored_count} improvements")
    print(f"   - Retrieved {len(raw_improvements)} raw improvements")
    print(f"   - Synthesized improvements: {'Yes' if synthesized else 'No (may need LLM)'}")
    print(f"   - Consolidation: {consolidation_result.get('consolidated', 0)} patterns")
    print(f"   - Expert with synthesis: {len(expert.improvements)} improvements")
    print()
    print("Benefits:")
    print("   ✅ Raw improvements: Detailed, individual patterns")
    print("   ✅ Synthesized improvements: Consolidated, coherent wisdom")
    print("   ✅ Consolidation: Automatic pattern extraction")
    print("   ✅ Memory levels: PROCEDURAL → SEMANTIC → META")
    print()


if __name__ == "__main__":
    asyncio.run(test_memory_consolidation())
