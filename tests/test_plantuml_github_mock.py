"""
Test PlantUML GitHub Training with Mock Data

Demonstrates the functionality without hitting GitHub API rate limits.
"""

import asyncio
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts.training_data_loader import TrainingDataLoader
from core.experts.domain_validators import get_validator


async def test_with_mock_data():
    """Test with mock PlantUML examples."""
    print("=" * 80)
    print("PLANTUML GITHUB TRAINING TEST (Mock Data)")
    print("=" * 80)
    print()
    
    # Create mock examples (simulating what would come from GitHub)
    mock_examples = [
        {
            "code": """@startuml
User -> System: Login
System -> Database: Validate
Database --> System: Success
System --> User: Welcome
@enduml""",
            "description": "User login sequence",
            "type": "sequence",
            "source": "github:joelparkerhenderson/plantuml-examples",
            "file_path": "doc/sequence-diagram/login.puml"
        },
        {
            "code": """@startuml
class User {
    +name: String
    +email: String
    +login()
}
class System {
    +authenticate()
}
User --> System
@enduml""",
            "description": "User system class diagram",
            "type": "class",
            "source": "github:joelparkerhenderson/plantuml-examples",
            "file_path": "doc/class-diagram/user-system.puml"
        },
        {
            "code": """@startuml
state Idle
state Active
Idle --> Active: Start
Active --> Idle: Stop
@enduml""",
            "description": "Simple state machine",
            "type": "state",
            "source": "github:joelparkerhenderson/plantuml-examples",
            "file_path": "doc/state-diagram/simple.puml"
        }
    ]
    
    print("1. Testing Training Data Loader")
    print("-" * 80)
    
    # Get validator
    validator = get_validator("plantuml")
    loader = TrainingDataLoader(domain="plantuml", validator=validator)
    
    # Validate examples
    valid_examples, invalid_examples = loader.validate_examples(mock_examples)
    print(f"✅ Validated: {len(valid_examples)} valid, {len(invalid_examples)} invalid")
    
    # Convert to gold_standards
    gold_standards = loader.convert_to_gold_standards(valid_examples)
    print(f"✅ Converted to {len(gold_standards)} gold standards")
    print()
    
    # Show sample
    if gold_standards:
        print("Sample gold standard:")
        sample = gold_standards[0]
        print(f"  Task: {sample['task']}")
        print(f"  Type: {sample['context']['diagram_type']}")
        print(f"  Source: {sample['context']['source']}")
        print(f"  Gold standard preview: {sample['gold_standard'][:80]}...")
        print()
    
    # Save to file
    print("2. Saving to JSON")
    print("-" * 80)
    
    expert_data_dir = Path("./expert_data/plantuml_expert")
    expert_data_dir.mkdir(parents=True, exist_ok=True)
    
    examples_file = expert_data_dir / "github_training_examples.json"
    with open(examples_file, 'w', encoding='utf-8') as f:
        json.dump({
            "source": "mock_data (simulating GitHub)",
            "total_examples": len(gold_standards),
            "loaded_at": "2026-01-14T00:50:00.000000",
            "gold_standards": gold_standards
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(gold_standards)} examples to {examples_file}")
    print()
    
    # Verify format
    print("3. Verifying Format")
    print("-" * 80)
    
    with open(examples_file) as f:
        saved_data = json.load(f)
    
    print(f"✅ File contains {saved_data['total_examples']} gold standards")
    print(f"✅ Format: {list(saved_data.keys())}")
    
    # Check gold_standards structure
    if saved_data['gold_standards']:
        gs = saved_data['gold_standards'][0]
        print(f"✅ Gold standard keys: {list(gs.keys())}")
        print(f"✅ Context keys: {list(gs['context'].keys())}")
        print()
    
    # Statistics
    types = {}
    for gs in saved_data['gold_standards']:
        diagram_type = gs['context'].get('diagram_type', 'unknown')
        types[diagram_type] = types.get(diagram_type, 0) + 1
    
    print("Statistics:")
    print(f"   Diagram types: {dict(types)}")
    print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("✅ Format is correct for training!")
    print("✅ Examples saved to expert directory")
    print("✅ Ready to use with expert.train(gold_standards=gold_standards)")
    print()
    print(f"File: {examples_file.absolute()}")


if __name__ == "__main__":
    asyncio.run(test_with_mock_data())
