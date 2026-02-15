"""
Inspect Expert Agent Memory Contents

Shows what's stored in memory for PlantUML and Mermaid experts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts.memory_integration import (
    retrieve_improvements_from_memory,
    retrieve_synthesized_improvements,
)
from core.foundation.data_structures import MemoryLevel, SwarmConfig
from core.memory.cortex import SwarmMemory


def inspect_memory_for_expert(memory: SwarmMemory, expert_name: str, domain: str):
    """Inspect memory contents for a specific expert."""
    print(f"\n{'='*80}")
    print(f"EXPERT: {expert_name} (Domain: {domain})")
    print(f"{'='*80}\n")

    # Check memory levels
    procedural = memory.memories[MemoryLevel.PROCEDURAL]
    semantic = memory.memories[MemoryLevel.SEMANTIC]
    meta = memory.memories[MemoryLevel.META]
    episodic = memory.memories[MemoryLevel.EPISODIC]
    causal = memory.memories[MemoryLevel.CAUSAL]

    print(f"Memory Levels:")
    print(f"  PROCEDURAL: {len(procedural)} entries")
    print(f"  SEMANTIC:   {len(semantic)} entries")
    print(f"  META:       {len(meta)} entries")
    print(f"  EPISODIC:   {len(episodic)} entries")
    print(f"  CAUSAL:     {len(causal)} entries")
    print()

    # Filter entries for this expert
    expert_procedural = [
        entry
        for entry in procedural.values()
        if entry.context.get("expert_name") == expert_name
        or entry.context.get("domain") == domain
        or f"expert_{domain}" in entry.context.get("expert_name", "")
    ]

    expert_semantic = [
        entry
        for entry in semantic.values()
        if entry.context.get("expert_name") == expert_name
        or entry.context.get("domain") == domain
        or f"expert_{domain}" in entry.context.get("expert_name", "")
    ]

    expert_meta = [
        entry
        for entry in meta.values()
        if entry.context.get("expert_name") == expert_name
        or entry.context.get("domain") == domain
        or f"expert_{domain}" in entry.context.get("expert_name", "")
    ]

    print(f"Expert-Specific Entries:")
    print(f"  PROCEDURAL: {len(expert_procedural)} entries")
    print(f"  SEMANTIC:   {len(expert_semantic)} entries")
    print(f"  META:       {len(expert_meta)} entries")
    print()

    # Show PROCEDURAL entries
    if expert_procedural:
        print(f"PROCEDURAL Level Entries ({len(expert_procedural)}):")
        print("-" * 80)
        for i, entry in enumerate(expert_procedural, 1):
            print(f"\nEntry {i}:")
            print(f"  Key: {entry.key}")
            print(f"  Created: {entry.created_at}")
            print(f"  Last Accessed: {entry.last_accessed}")
            print(f"  Access Count: {entry.access_count}")
            print(f"  Context: {json.dumps(entry.context, indent=4, default=str)}")
            print(f"  Content Preview:")
            try:
                content_data = json.loads(entry.content)
                print(f"    {json.dumps(content_data, indent=4, default=str)[:500]}...")
            except:
                print(f"    {entry.content[:500]}...")
        print()

    # Show SEMANTIC entries
    if expert_semantic:
        print(f"SEMANTIC Level Entries ({len(expert_semantic)}):")
        print("-" * 80)
        for i, entry in enumerate(expert_semantic, 1):
            print(f"\nEntry {i}:")
            print(f"  Key: {entry.key}")
            print(f"  Created: {entry.created_at}")
            print(f"  Context: {json.dumps(entry.context, indent=4, default=str)}")
            print(f"  Content Preview:")
            print(f"    {entry.content[:500]}...")
        print()

    # Show META entries
    if expert_meta:
        print(f"META Level Entries ({len(expert_meta)}):")
        print("-" * 80)
        for i, entry in enumerate(expert_meta, 1):
            print(f"\nEntry {i}:")
            print(f"  Key: {entry.key}")
            print(f"  Created: {entry.created_at}")
            print(f"  Context: {json.dumps(entry.context, indent=4, default=str)}")
            print(f"  Content Preview:")
            print(f"    {entry.content[:500]}...")
        print()

    # Retrieve using integration functions
    print(f"Retrieved Improvements (via integration):")
    print("-" * 80)
    improvements = retrieve_improvements_from_memory(
        memory=memory, expert_name=expert_name, domain=domain, max_results=20
    )

    print(f"  Found {len(improvements)} improvements")
    for i, imp in enumerate(improvements[:5], 1):  # Show first 5
        print(f"\n  Improvement {i}:")
        print(f"    Task: {imp.get('task', 'Unknown')}")
        print(f"    Pattern: {imp.get('learned_pattern', '')[:100]}...")
        print(f"    Type: {imp.get('improvement_type', 'unknown')}")
        print(f"    Source: {imp.get('source', 'unknown')}")

    if len(improvements) > 5:
        print(f"\n  ... and {len(improvements) - 5} more improvements")

    # Try synthesized
    print(f"\nSynthesized Improvements:")
    print("-" * 80)
    synthesized = retrieve_synthesized_improvements(
        memory=memory, expert_name=expert_name, domain=domain
    )

    if synthesized:
        print(f"  Length: {len(synthesized)} chars")
        print(f"  Preview: {synthesized[:300]}...")
    else:
        print(f"  No synthesized improvements available")


def main():
    """Inspect memory for both experts."""
    print("=" * 80)
    print("EXPERT AGENT MEMORY INSPECTION")
    print("=" * 80)

    # Create memory systems (or load existing)
    print("\n1. Creating Memory Systems")
    print("-" * 80)

    # For PlantUML
    plantuml_memory_config = SwarmConfig()
    plantuml_memory = SwarmMemory(agent_name="plantuml_expert_test", config=plantuml_memory_config)

    # For Mermaid
    mermaid_memory_config = SwarmConfig()
    mermaid_memory = SwarmMemory(agent_name="mermaid_expert_test", config=mermaid_memory_config)

    print("✅ Memory systems created")

    # Check if there are any existing memory files
    print("\n2. Checking for Existing Memory Data")
    print("-" * 80)

    # Try to load from expert data directories
    plantuml_data_dir = Path("./test_outputs/plantuml_expert")
    mermaid_data_dir = Path("./test_outputs/mermaid_expert")

    if plantuml_data_dir.exists():
        improvements_file = plantuml_data_dir / "improvements.json"
        if improvements_file.exists():
            print(f"✅ Found PlantUML improvements file: {improvements_file}")
            try:
                with open(improvements_file, "r") as f:
                    plantuml_improvements = json.load(f)
                    print(f"   Contains {len(plantuml_improvements)} improvements")
            except Exception as e:
                print(f"   Error reading: {e}")

    if mermaid_data_dir.exists():
        improvements_file = mermaid_data_dir / "improvements.json"
        if improvements_file.exists():
            print(f"✅ Found Mermaid improvements file: {improvements_file}")
            try:
                with open(improvements_file, "r") as f:
                    mermaid_improvements = json.load(f)
                    print(f"   Contains {len(mermaid_improvements)} improvements")
            except Exception as e:
                print(f"   Error reading: {e}")

    # Inspect PlantUML memory
    inspect_memory_for_expert(
        memory=plantuml_memory, expert_name="plantuml_expert_test", domain="plantuml"
    )

    # Inspect Mermaid memory
    inspect_memory_for_expert(
        memory=mermaid_memory, expert_name="mermaid_expert_test", domain="mermaid"
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    plantuml_procedural = len(
        [
            e
            for e in plantuml_memory.memories[MemoryLevel.PROCEDURAL].values()
            if e.context.get("domain") == "plantuml"
            or "plantuml" in str(e.context.get("expert_name", ""))
        ]
    )
    plantuml_semantic = len(
        [
            e
            for e in plantuml_memory.memories[MemoryLevel.SEMANTIC].values()
            if e.context.get("domain") == "plantuml"
            or "plantuml" in str(e.context.get("expert_name", ""))
        ]
    )

    mermaid_procedural = len(
        [
            e
            for e in mermaid_memory.memories[MemoryLevel.PROCEDURAL].values()
            if e.context.get("domain") == "mermaid"
            or "mermaid" in str(e.context.get("expert_name", ""))
        ]
    )
    mermaid_semantic = len(
        [
            e
            for e in mermaid_memory.memories[MemoryLevel.SEMANTIC].values()
            if e.context.get("domain") == "mermaid"
            or "mermaid" in str(e.context.get("expert_name", ""))
        ]
    )

    print(f"\nPlantUML Expert:")
    print(f"  PROCEDURAL: {plantuml_procedural} entries")
    print(f"  SEMANTIC:   {plantuml_semantic} entries")

    print(f"\nMermaid Expert:")
    print(f"  PROCEDURAL: {mermaid_procedural} entries")
    print(f"  SEMANTIC:   {mermaid_semantic} entries")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
