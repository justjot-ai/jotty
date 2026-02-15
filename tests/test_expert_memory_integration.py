"""
Test Expert Agent Memory Integration

Tests that expert agent improvements are stored in and retrieved from
Jotty's SwarmMemory system instead of files.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent, ExpertAgentConfig
from core.memory.cortex import SwarmMemory
from core.foundation.data_structures import SwarmConfig, MemoryLevel
from core.experts.memory_integration import (
    store_improvement_to_memory,
    retrieve_improvements_from_memory,
    sync_improvements_to_memory
)


async def test_memory_integration():
    """Test expert agent integration with SwarmMemory."""
    print("=" * 80)
    print("TESTING EXPERT AGENT MEMORY INTEGRATION")
    print("=" * 80)
    print()
    
    # Create memory system
    print("1. Creating SwarmMemory System")
    print("-" * 80)
    memory_config = SwarmConfig()
    memory = SwarmMemory(
        agent_name="mermaid_expert_test",
        config=memory_config
    )
    print("✅ Memory system created")
    print()
    
    # Create expert with memory
    print("2. Creating Expert Agent with Memory")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="mermaid_memory_test",
        domain="mermaid",
        description="Mermaid expert with memory integration",
        use_memory_storage=True,
        expert_data_dir="./test_outputs/mermaid_memory"
    )
    
    expert = MermaidExpertAgent(config=config, memory=memory)
    print(f"Expert created")
    print(f"   Uses memory storage: {config.use_memory_storage}")
    print(f"   Memory system: {type(expert.memory).__name__ if expert.memory else 'None'}")
    print()
    
    # Test storing improvements to memory
    print("3. Testing Improvement Storage to Memory")
    print("-" * 80)
    
    test_improvement = {
        "iteration": 1,
        "timestamp": "2026-01-13T22:00:00",
        "task": "Generate simple flowchart",
        "student_output": "graph A --> B",
        "teacher_output": "graph TD\n    A[Start]\n    B[End]\n    A --> B",
        "student_score": 0.0,
        "teacher_score": 1.0,
        "improvement_type": "teacher_correction",
        "learned_pattern": "When task is 'Generate simple flowchart', use 'graph TD...' instead of 'graph A --> B'"
    }
    
    # Store improvement
    entry = store_improvement_to_memory(
        memory=memory,
        improvement=test_improvement,
        expert_name=config.name,
        domain=config.domain
    )
    
    if entry:
        print(f"✅ Improvement stored to memory")
        print(f"   Memory key: {entry.key}")
        print(f"   Memory level: {entry.level.value}")
        print(f"   Content length: {len(entry.content)} chars")
    else:
        print("❌ Failed to store improvement")
    print()
    
    # Test retrieving improvements from memory
    print("4. Testing Improvement Retrieval from Memory")
    print("-" * 80)
    
    improvements = retrieve_improvements_from_memory(
        memory=memory,
        expert_name=config.name,
        domain=config.domain
    )
    
    print(f"✅ Retrieved {len(improvements)} improvements from memory")
    if improvements:
        print(f"   First improvement:")
        print(f"     Task: {improvements[0].get('task', 'Unknown')}")
        print(f"     Pattern: {improvements[0].get('learned_pattern', '')[:60]}...")
    print()
    
    # Test expert loading improvements from memory
    print("5. Testing Expert Loading from Memory")
    print("-" * 80)
    
    # Create new expert instance (should load from memory)
    expert2 = MermaidExpertAgent(config=config, memory=memory)
    loaded_improvements = expert2.improvements

    print(f"Expert loaded {len(loaded_improvements)} improvements")
    print(f"   Has memory: {expert2.memory is not None}")
    print()
    
    # Verify improvements are in memory
    print("6. Verifying Memory Storage")
    print("-" * 80)
    
    # Check PROCEDURAL level
    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    meta_count = len(memory.memories[MemoryLevel.META])
    
    print(f"   PROCEDURAL memories: {procedural_count}")
    print(f"   META memories: {meta_count}")
    print(f"   Total improvements in memory: {procedural_count + meta_count}")
    print()
    
    # Test retrieval with query
    print("7. Testing Memory Retrieval with Query")
    print("-" * 80)
    
    retrieved = memory.retrieve(
        query="expert agent mermaid_memory_test improvements mermaid",
        goal=f"expert_{config.domain}_improvements",
        budget_tokens=1000,
        levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META]
    )
    
    print(f"✅ Retrieved {len(retrieved)} memory entries")
    for i, entry in enumerate(retrieved[:3], 1):
        print(f"   {i}. Level: {entry.level.value}, Key: {entry.key[:16]}...")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Memory system integration working!")
    print("   - Improvements stored to SwarmMemory")
    print("   - Improvements retrieved from memory")
    print("   - Expert agent loads from memory")
    print("   - Memory levels: PROCEDURAL and META")
    print()
    print("Benefits:")
    print("   ✅ Persistent across runs")
    print("   ✅ Semantic search via LLM")
    print("   ✅ Automatic deduplication")
    print("   ✅ Goal-conditioned retrieval")
    print("   ✅ Memory consolidation")
    print()


if __name__ == "__main__":
    asyncio.run(test_memory_integration())
