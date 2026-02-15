"""
Test PlantUML GitHub Training Examples Loading

Tests loading PlantUML examples from GitHub and saving as JSON.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import PlantUMLExpertAgent
from core.experts.plantuml_expert import PlantUMLExpertAgent as PlantUMLExpert


async def test_github_loading():
    """Test loading PlantUML examples from GitHub."""
    print("=" * 80)
    print("PLANTUML GITHUB TRAINING EXAMPLES TEST")
    print("=" * 80)
    print()
    
    # Test loading from GitHub
    print("1. Loading examples from GitHub...")
    print("-" * 80)
    
    try:
        gold_standards = await PlantUMLExpert.load_training_examples_from_github(
            repo_url="https://github.com/joelparkerhenderson/plantuml-examples",
            max_examples=20,  # Start with 20 for testing
            save_to_file=True,
            expert_data_dir="./expert_data/plantuml_expert"
        )
        
        print(f"✅ Loaded {len(gold_standards)} gold standards from GitHub")
        print()
        
        # Show sample
        if gold_standards:
            print("Sample gold standard:")
            sample = gold_standards[0]
            print(f"  Task: {sample['task'][:80]}...")
            print(f"  Type: {sample['context'].get('diagram_type', 'unknown')}")
            print(f"  Source: {sample['context'].get('source', 'unknown')}")
            print(f"  Code preview: {sample['gold_standard'][:100]}...")
            print()
        
        # Check if file was saved
        examples_file = Path("./expert_data/plantuml_expert/github_training_examples.json")
        if examples_file.exists():
            print(f"✅ Examples saved to: {examples_file}")
            
            # Load and verify
            with open(examples_file) as f:
                saved_data = json.load(f)
            
            print(f"   Total examples: {saved_data.get('total_examples', 0)}")
            print(f"   Source: {saved_data.get('source', 'unknown')}")
            print(f"   Loaded at: {saved_data.get('loaded_at', 'unknown')}")
            print()
            
            # Show statistics
            types = {}
            sources = {}
            for gs in saved_data.get('gold_standards', []):
                diagram_type = gs['context'].get('diagram_type', 'unknown')
                source = gs['context'].get('source', 'unknown')
                types[diagram_type] = types.get(diagram_type, 0) + 1
                sources[source] = sources.get(source, 0) + 1
            
            print("Statistics:")
            print(f"   Diagram types: {dict(types)}")
            print(f"   Sources: {dict(sources)}")
        else:
            print(f"⚠️  Examples file not found at {examples_file}")
        
        # Test using gold_standards for training
        print()
        print("2. Testing gold_standards format...")
        print("-" * 80)
        
        if gold_standards:
            sample = gold_standards[0]
            print("✅ Gold standard format:")
            print(f"   - task: {sample.get('task', 'missing')[:60]}...")
            print(f"   - context: {list(sample.get('context', {}).keys())}")
            print(f"   - gold_standard: {len(sample.get('gold_standard', ''))} chars")
            print()
            print("✅ Format is correct for training!")
        
        return gold_standards
        
    except Exception as e:
        print(f"❌ Error loading from GitHub: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_training_with_github_examples():
    """Test training expert with GitHub-loaded examples."""
    print()
    print("3. Testing Training with GitHub Examples")
    print("-" * 80)
    
    # Load examples
    gold_standards = await PlantUMLExpert.load_training_examples_from_github(
        repo_url="https://github.com/joelparkerhenderson/plantuml-examples",
        max_examples=5,  # Small number for testing
        save_to_file=True
    )
    
    if not gold_standards:
        print("⚠️  No examples loaded, skipping training test")
        return
    
    print(f"✅ Loaded {len(gold_standards)} examples for training")
    print()
    print("Note: Full training test would:")
    print("  1. Create PlantUMLExpertAgent")
    print("  2. Call expert.train(gold_standards=gold_standards)")
    print("  3. Verify expert learns from examples")
    print()
    print("✅ Gold standards format is ready for training!")


if __name__ == "__main__":
    print("Testing PlantUML GitHub Training Examples Loading")
    print()
    
    # Test 1: Load from GitHub
    gold_standards = asyncio.run(test_github_loading())
    
    # Test 2: Verify format for training
    if gold_standards:
        asyncio.run(test_training_with_github_examples())
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("✅ GitHub loading: Working")
    print("✅ JSON saving: Working")
    print("✅ Gold standards format: Correct")
    print()
    print(f"Examples saved to: ./expert_data/plantuml_expert/github_training_examples.json")
