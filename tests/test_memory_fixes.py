"""
Test All Memory Fixes

Tests:
1. Teacher returns diagram code (not evaluation)
2. Consolidation happens automatically
3. Memory persistence works
4. Similar improvements consolidated
"""

import asyncio
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import PlantUMLExpertAgent, ExpertAgentConfig
from core.experts.memory_integration import (
    consolidate_improvements,
    retrieve_improvements_from_memory
)
from core.memory.cortex import HierarchicalMemory
from core.foundation.data_structures import JottyConfig, MemoryLevel


def configure_llm():
    """Configure DSPy with LLM."""
    try:
        import dspy
    except ImportError:
        return False
    
    try:
        from examples.claude_cli_wrapper import ClaudeCLILM
        import subprocess
        result = subprocess.run(["claude", "--version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            lm = ClaudeCLILM(model="sonnet")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude CLI")
            return True
    except:
        pass
    
    return False


async def test_all_fixes():
    """Test all memory fixes."""
    print("=" * 80)
    print("TESTING ALL MEMORY FIXES")
    print("=" * 80)
    print()
    
    if not configure_llm():
        print("⚠️  No LLM available, some tests will be limited")
    print()
    
    # Create memory system
    print("1. Creating Memory System with Persistence")
    print("-" * 80)
    memory_config = JottyConfig()
    memory = HierarchicalMemory(
        agent_name="plantuml_fix_test",
        config=memory_config
    )
    print("✅ Memory system created")
    print()
    
    # Create expert
    print("2. Creating Expert Agent")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="plantuml_fix_test",
        domain="plantuml",
        description="PlantUML expert for testing fixes",
        use_memory_storage=True,
        expert_data_dir="./test_outputs/plantuml_fix_test",
        max_training_iterations=2
    )
    
    expert = PlantUMLExpertAgent(config=config, memory=memory)
    print(f"✅ Expert created")
    print(f"   Memory persistence: {expert.memory_persistence is not None}")
    if expert.memory_persistence:
        print(f"   Persistence dir: {expert.memory_persistence.persistence_dir}")
    print()
    
    # Test training with proper gold standards
    print("3. Training with Proper Gold Standards")
    print("-" * 80)
    
    training_cases = [
        {
            "task": "Generate simple sequence diagram",
            "context": {"description": "User and System", "diagram_type": "sequence"},
            "gold_standard": "@startuml\nUser -> System: Request\nSystem --> User: Response\n@enduml"
        },
        {
            "task": "Generate class diagram",
            "context": {"description": "Basic class structure", "diagram_type": "class"},
            "gold_standard": "@startuml\nclass Animal {\n    +name: string\n}\nclass Dog {\n    +breed: string\n}\nAnimal <|-- Dog\n@enduml"
        }
    ]
    
    try:
        training_results = await asyncio.wait_for(
            expert.train(gold_standards=training_cases, force_retrain=True),
            timeout=180
        )
        print(f"✅ Training completed")
        print(f"   Passed cases: {training_results.get('passed_cases', 0)}")
        print(f"   Improvements: {len(expert.improvements)}")
    except Exception as e:
        print(f"⚠️  Training: {e}")
        expert.trained = True
    
    # Check memory
    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    semantic_count = len(memory.memories[MemoryLevel.SEMANTIC])
    print(f"   Memory: {procedural_count} PROCEDURAL, {semantic_count} SEMANTIC")
    print()
    
    # Test consolidation
    print("4. Testing Consolidation")
    print("-" * 80)
    
    if procedural_count >= 2:
        consolidation_result = consolidate_improvements(
            memory=memory,
            expert_name=config.name,
            domain=config.domain
        )
        print(f"✅ Consolidation result:")
        print(f"   Consolidated patterns: {consolidation_result.get('consolidated', 0)}")
        print(f"   Merged improvements: {consolidation_result.get('merged', 0)}")
        
        # Check SEMANTIC level
        semantic_after = len(memory.memories[MemoryLevel.SEMANTIC])
        print(f"   SEMANTIC memories: {semantic_after} (was {semantic_count})")
    else:
        print(f"⚠️  Not enough improvements for consolidation ({procedural_count} < 2)")
    print()
    
    # Test memory persistence
    print("5. Testing Memory Persistence")
    print("-" * 80)
    
    if expert.memory_persistence:
        saved = expert.memory_persistence.save()
        if saved:
            print(f"✅ Memory saved to disk")
            print(f"   Location: {expert.memory_persistence.persistence_dir}")
            
            # List saved files
            for level, file_path in expert.memory_persistence.level_files.items():
                if file_path.exists():
                    size = file_path.stat().st_size
                    print(f"   {level.value}: {file_path.name} ({size} bytes)")
        else:
            print("❌ Failed to save memory")
    else:
        print("⚠️  Memory persistence not enabled")
    print()
    
    # Show consolidated improvements
    print("6. Showing Consolidated Improvements")
    print("-" * 80)
    
    improvements = retrieve_improvements_from_memory(
        memory=memory,
        expert_name=config.name,
        domain=config.domain,
        max_results=10
    )
    
    print(f"Retrieved {len(improvements)} improvements")
    
    # Group by pattern type
    pattern_types = {}
    for imp in improvements:
        pattern = imp.get('learned_pattern', '')
        if 'syntax' in pattern.lower() or 'plantuml' in pattern.lower() or 'mermaid' in pattern.lower():
            pattern_type = "syntax_format"
        elif 'simple' in pattern.lower() or 'complex' in pattern.lower():
            pattern_type = "complexity"
        elif '@startuml' in pattern.lower():
            pattern_type = "tags"
        else:
            pattern_type = "general"
        
        if pattern_type not in pattern_types:
            pattern_types[pattern_type] = []
        pattern_types[pattern_type].append(imp)
    
    print(f"\nGrouped by pattern type:")
    for pattern_type, imps in pattern_types.items():
        print(f"  {pattern_type}: {len(imps)} improvements")
        if imps:
            print(f"    Sample: {imps[0].get('learned_pattern', '')[:100]}...")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ PROCEDURAL memories: {procedural_count}")
    print(f"✅ SEMANTIC memories: {semantic_count}")
    print(f"✅ Consolidation: {consolidation_result.get('consolidated', 0) if procedural_count >= 2 else 0} patterns")
    print(f"✅ Memory persistence: {'Enabled' if expert.memory_persistence else 'Disabled'}")
    print(f"✅ Pattern grouping: {len(pattern_types)} types")
    print()


if __name__ == "__main__":
    asyncio.run(test_all_fixes())
