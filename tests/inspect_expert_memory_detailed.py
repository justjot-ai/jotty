"""
Detailed Memory Inspection - Shows actual stored improvements

Loads improvements from files and syncs to memory, then shows contents.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts.memory_integration import (
    sync_improvements_to_memory,
    retrieve_improvements_from_memory,
    retrieve_synthesized_improvements
)
from core.memory.cortex import SwarmMemory
from core.foundation.data_structures import SwarmConfig, MemoryLevel


def show_file_improvements(expert_name: str, domain: str):
    """Show improvements from file."""
    data_dir = Path(f"./test_outputs/{domain}_expert")
    improvements_file = data_dir / "improvements.json"
    
    print(f"\n{'='*80}")
    print(f"FILE-BASED IMPROVEMENTS: {expert_name} ({domain})")
    print(f"{'='*80}\n")
    
    if not improvements_file.exists():
        print(f"❌ No improvements file found: {improvements_file}")
        return []
    
    try:
        with open(improvements_file, 'r') as f:
            improvements = json.load(f)
        
        print(f"✅ Found {len(improvements)} improvements in file")
        print(f"   File: {improvements_file}")
        print()
        
        for i, imp in enumerate(improvements, 1):
            print(f"Improvement {i}:")
            print(f"  Task: {imp.get('task', 'Unknown')}")
            print(f"  Iteration: {imp.get('iteration', 'N/A')}")
            print(f"  Timestamp: {imp.get('timestamp', 'N/A')}")
            print(f"  Type: {imp.get('improvement_type', 'unknown')}")
            print(f"  Student Score: {imp.get('student_score', 0.0)}")
            print(f"  Teacher Score: {imp.get('teacher_score', 1.0)}")
            print(f"  Learned Pattern:")
            pattern = imp.get('learned_pattern', '')
            if pattern:
                print(f"    {pattern[:200]}...")
            else:
                print(f"    (No pattern)")
            
            # Show student vs teacher output
            student = imp.get('student_output', '')
            teacher = imp.get('teacher_output', '')
            if student and teacher:
                print(f"  Student Output (first 100 chars):")
                print(f"    {str(student)[:100]}...")
                print(f"  Teacher Output (first 100 chars):")
                print(f"    {str(teacher)[:100]}...")
            
            print()
        
        return improvements
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []


def sync_and_show_memory(expert_name: str, domain: str, improvements: list):
    """Sync improvements to memory and show what's stored."""
    print(f"\n{'='*80}")
    print(f"MEMORY STORAGE: {expert_name} ({domain})")
    print(f"{'='*80}\n")
    
    # Create memory system
    memory_config = SwarmConfig()
    memory = SwarmMemory(
        agent_name=f"{expert_name}_memory",
        config=memory_config
    )
    
    if not improvements:
        print("⚠️  No improvements to sync")
        return memory
    
    # Sync improvements to memory
    print(f"Syncing {len(improvements)} improvements to memory...")
    synced_count = sync_improvements_to_memory(
        memory=memory,
        improvements=improvements,
        expert_name=expert_name,
        domain=domain
    )
    
    print(f"✅ Synced {synced_count}/{len(improvements)} improvements to memory")
    print()
    
    # Show memory levels
    procedural = memory.memories[MemoryLevel.PROCEDURAL]
    semantic = memory.memories[MemoryLevel.SEMANTIC]
    meta = memory.memories[MemoryLevel.META]
    
    print(f"Memory Levels After Sync:")
    print(f"  PROCEDURAL: {len(procedural)} entries")
    print(f"  SEMANTIC:   {len(semantic)} entries")
    print(f"  META:       {len(meta)} entries")
    print()
    
    # Show PROCEDURAL entries
    if procedural:
        print(f"PROCEDURAL Level Entries:")
        print("-" * 80)
        for i, (key, entry) in enumerate(list(procedural.items())[:5], 1):  # Show first 5
            print(f"\nEntry {i} (Key: {key[:16]}...):")
            print(f"  Created: {entry.created_at}")
            print(f"  Context: {json.dumps(entry.context, indent=4, default=str)}")
            print(f"  Content:")
            try:
                content_data = json.loads(entry.content)
                print(f"    {json.dumps(content_data, indent=4, default=str)}")
            except:
                print(f"    {entry.content[:300]}...")
        
        if len(procedural) > 5:
            print(f"\n  ... and {len(procedural) - 5} more entries")
    
    # Retrieve using integration
    print(f"\nRetrieved via Integration Function:")
    print("-" * 80)
    retrieved = retrieve_improvements_from_memory(
        memory=memory,
        expert_name=expert_name,
        domain=domain,
        max_results=10
    )
    
    print(f"  Retrieved {len(retrieved)} improvements")
    for i, imp in enumerate(retrieved[:3], 1):  # Show first 3
        print(f"\n  {i}. Task: {imp.get('task', 'Unknown')}")
        print(f"     Pattern: {imp.get('learned_pattern', '')[:150]}...")
    
    return memory


def main():
    """Inspect memory for both experts."""
    print("=" * 80)
    print("DETAILED EXPERT AGENT MEMORY INSPECTION")
    print("=" * 80)
    
    # PlantUML
    print("\n" + "="*80)
    print("PLANTUML EXPERT")
    print("="*80)
    
    plantuml_improvements = show_file_improvements("plantuml_expert_test", "plantuml")
    if plantuml_improvements:
        plantuml_memory = sync_and_show_memory(
            "plantuml_expert_test",
            "plantuml",
            plantuml_improvements
        )
    
    # Mermaid
    print("\n" + "="*80)
    print("MERMAID EXPERT")
    print("="*80)
    
    mermaid_improvements = show_file_improvements("mermaid_expert_test", "mermaid")
    if mermaid_improvements:
        mermaid_memory = sync_and_show_memory(
            "mermaid_expert_test",
            "mermaid",
            mermaid_improvements
        )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nPlantUML: {len(plantuml_improvements)} improvements in file")
    print(f"Mermaid:  {len(mermaid_improvements)} improvements in file")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
