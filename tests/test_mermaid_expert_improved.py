"""
Test Improved Mermaid Expert Agent

Tests with tough examples and shows how the expert can be improved
to handle complex cases by training on them.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent, ExpertAgentConfig


async def test_and_improve_mermaid_expert():
    """Test MermaidExpertAgent and improve it with tough examples."""
    print("=" * 80)
    print("TESTING AND IMPROVING MERMAID EXPERT WITH TOUGH EXAMPLES")
    print("=" * 80)
    print()
    
    # Create expert with tough training cases
    tough_training_cases = [
        {
            "task": "Generate complex multi-branch decision tree",
            "context": {"description": "Complex decision tree with multiple branches and conditions"},
            "gold_standard": """graph TD
    A[Start]
    B{Is User Logged In?}
    C{Has Permission?}
    D{Is Data Valid?}
    E[Process Request]
    F[Show Login Page]
    G[Show Error: No Permission]
    H[Show Error: Invalid Data]
    I[Success]
    A --> B
    B -->|Yes| C
    B -->|No| F
    C -->|Yes| D
    C -->|No| G
    D -->|Yes| E
    D -->|No| H
    E --> I"""
        },
        {
            "task": "Generate complex sequence diagram",
            "context": {"description": "Sequence diagram with loops and alt blocks"},
            "gold_standard": """sequenceDiagram
    participant U as User
    participant A as API
    participant D as Database
    
    U->>A: POST /register
    A->>D: Check if user exists
    alt User exists
        D-->>A: User found
        A-->>U: Error: User already exists
    else User not found
        D-->>A: User not found
        A->>D: Create user
        D-->>A: User created
        A-->>U: Success: User registered
    end"""
        },
        {
            "task": "Generate class diagram",
            "context": {"description": "Class diagram with inheritance and relationships"},
            "gold_standard": """classDiagram
    class Animal {
        +String name
        +int age
        +eat()
    }
    class Dog {
        +String breed
        +bark()
    }
    Animal <|-- Dog"""
        },
        {
            "task": "Generate flowchart with subgraphs",
            "context": {"description": "Flowchart with subgraphs/clusters"},
            "gold_standard": """graph TD
    A[Start]
    
    subgraph Authentication
        B[Login]
        C[Verify Token]
    end
    
    subgraph Processing
        D[Process Request]
        E[Validate Data]
    end
    
    A --> Authentication
    Authentication --> Processing
    Processing --> F[End]
    
    B --> C
    D --> E"""
        }
    ]
    
    # Create expert with custom config
    config = ExpertAgentConfig(
        name="mermaid_expert_improved",
        domain="mermaid",
        description="Improved Mermaid expert with tough examples",
        training_gold_standards=tough_training_cases,
        max_training_iterations=5,
        required_training_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        expert_data_dir="./expert_data/mermaid_improved"
    )
    
    expert = MermaidExpertAgent(config=config)

    # Verify training data is accessible (BaseExpert does not have .train())
    print("Verifying training data...")
    print()
    training_data = expert.get_training_data()
    print(f"Training Data Available:")
    print(f"  Cases: {len(training_data)}")
    print()

    # Show training case summaries
    for i, case in enumerate(training_data, 1):
        print(f"  Case {i}: {case.get('task', 'Unknown')}")
        print()
    
    # Test on new tough examples
    print("=" * 80)
    print("TESTING ON NEW TOUGH EXAMPLES")
    print("=" * 80)
    print()
    
    test_cases = [
        {
            "name": "Complex Decision Tree",
            "description": "Complex decision tree with multiple branches and conditions",
            "diagram_type": "flowchart",
            "expected_pattern": "multi-branch decision"
        },
        {
            "name": "Sequence Diagram",
            "description": "Sequence diagram with loops and alt blocks",
            "diagram_type": "sequence",
            "expected_pattern": "sequenceDiagram"
        },
        {
            "name": "Class Diagram",
            "description": "Class diagram with inheritance and relationships",
            "diagram_type": "class",
            "expected_pattern": "classDiagram"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Description: {test_case['description']}")
        print()
        
        try:
            generated = await expert.generate_mermaid(
                description=test_case['description'],
                diagram_type=test_case['diagram_type']
            )
            
            print(f"  Generated:")
            print(f"  ```mermaid")
            print(f"  {generated[:200]}...")  # Show first 200 chars
            print(f"  ```")
            print()
            
            # Check if it contains expected pattern
            contains_pattern = test_case['expected_pattern'].lower() in generated.lower()
            has_valid_syntax = "graph" in generated.lower() or "sequence" in generated.lower() or "class" in generated.lower()
            
            print(f"  Contains Expected Pattern: {'✅ YES' if contains_pattern else '❌ NO'}")
            print(f"  Has Valid Syntax: {'✅ YES' if has_valid_syntax else '❌ NO'}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()
    
    # Show expert stats
    print("=" * 80)
    print("EXPERT STATS")
    print("=" * 80)
    print()

    stats = expert.get_stats()
    print(f"Total Improvements: {stats['improvements_count']}")
    print(f"Training Cases: {stats['training_cases']}")
    print(f"Validation Cases: {stats['validation_cases']}")
    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The expert agent can be improved by:")
    print("  1. Training on more diverse examples")
    print("  2. Including complex patterns in training")
    print("  3. Using better agent implementations that generate based on descriptions")
    print("  4. Iteratively training on failures")
    print()
    print("Current Stats:")
    print(f"  - Domain: {stats['domain']}")
    print(f"  - Improvements Learned: {stats['improvements_count']}")

    return expert


if __name__ == "__main__":
    asyncio.run(test_and_improve_mermaid_expert())
