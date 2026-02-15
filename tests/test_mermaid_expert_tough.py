"""
Test Mermaid Expert Agent with Tough Examples

Tests the MermaidExpertAgent with challenging, complex examples to ensure
it can handle difficult cases correctly.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import MermaidExpertAgent


async def test_tough_mermaid_examples():
    """Test MermaidExpertAgent with challenging examples."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT WITH TOUGH EXAMPLES")
    print("=" * 80)
    print()
    
    # Create expert agent
    expert = MermaidExpertAgent()
    
    # Verify training data is available
    training_data = expert.get_training_data()
    print(f"Training data available: {len(training_data)} cases")
    print()
    
    # Tough test cases
    tough_cases = [
        {
            "name": "Complex Multi-Branch Decision Tree",
            "description": "Complex decision tree with multiple branches and conditions",
            "diagram_type": "flowchart",
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
            "name": "Complex Sequence Diagram with Loops",
            "description": "Sequence diagram with loops and alt blocks",
            "diagram_type": "sequence",
            "gold_standard": """sequenceDiagram
    participant U as User
    participant A as API
    participant D as Database
    participant E as Email Service
    
    U->>A: POST /register
    A->>D: Check if user exists
    alt User exists
        D-->>A: User found
        A-->>U: Error: User already exists
    else User not found
        D-->>A: User not found
        A->>D: Create user
        D-->>A: User created
        A->>E: Send welcome email
        E-->>A: Email sent
        A-->>U: Success: User registered
    end
    
    loop Every 5 minutes
        A->>D: Check pending emails
        D-->>A: Pending emails list
    end"""
        },
        {
            "name": "Complex Class Diagram with Relationships",
            "description": "Class diagram with inheritance, composition, and associations",
            "diagram_type": "class",
            "gold_standard": """classDiagram
    class Animal {
        +String name
        +int age
        +eat()
        +sleep()
    }
    
    class Mammal {
        +bool hasFur
        +giveBirth()
    }
    
    class Bird {
        +bool canFly
        +fly()
    }
    
    class Dog {
        +String breed
        +bark()
    }
    
    class Cat {
        +String color
        +meow()
    }
    
    class Owner {
        +String name
        +List~Pet~ pets
        +feedPets()
    }
    
    Animal <|-- Mammal
    Animal <|-- Bird
    Mammal <|-- Dog
    Mammal <|-- Cat
    Owner "1" --> "*" Animal : owns
    Dog ..> Owner : loves"""
        },
        {
            "name": "State Diagram with Nested States",
            "description": "State diagram with nested states and transitions",
            "diagram_type": "state",
            "gold_standard": """stateDiagram-v2
    [*] --> Idle
    
    Idle --> Processing : Start
    Processing --> Validating : Validate
    Processing --> Error : Invalid Input
    
    state Processing {
        [*] --> Reading
        Reading --> Parsing
        Parsing --> Transforming
        Transforming --> [*]
    }
    
    Validating --> Success : Valid
    Validating --> Error : Invalid
    
    Error --> Idle : Retry
    Success --> [*]"""
        },
        {
            "name": "Gantt Chart",
            "description": "Gantt chart for project timeline",
            "diagram_type": "gantt",
            "gold_standard": """gantt
    title Project Timeline
    dateFormat YYYY-MM-DD
    section Phase 1
    Design           :a1, 2024-01-01, 30d
    Review           :a2, after a1, 10d
    section Phase 2
    Development      :b1, 2024-02-01, 60d
    Testing          :b2, after b1, 20d
    section Phase 3
    Deployment       :c1, 2024-04-01, 15d
    Monitoring       :c2, after c1, 30d"""
        },
        {
            "name": "Git Graph",
            "description": "Git graph showing branch structure",
            "diagram_type": "gitgraph",
            "gold_standard": """gitgraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Feature A"
    commit id: "Feature B"
    checkout main
    commit id: "Hotfix"
    merge develop
    commit id: "Release"""
        },
        {
            "name": "Pie Chart",
            "description": "Pie chart with percentages",
            "diagram_type": "pie",
            "gold_standard": """pie title Sales by Region
    "North America" : 45
    "Europe" : 30
    "Asia" : 15
    "Other" : 10"""
        },
        {
            "name": "Complex Flowchart with Subgraphs",
            "description": "Flowchart with subgraphs/clusters",
            "diagram_type": "flowchart",
            "gold_standard": """graph TD
    A[Start]
    
    subgraph Authentication
        B[Login]
        C[Verify Token]
        D[Check Permissions]
    end
    
    subgraph Processing
        E[Process Request]
        F[Validate Data]
        G[Transform Data]
    end
    
    subgraph Response
        H[Format Response]
        I[Send Response]
    end
    
    A --> Authentication
    Authentication --> Processing
    Processing --> Response
    Response --> J[End]
    
    B --> C
    C --> D
    E --> F
    F --> G
    H --> I"""
        },
        {
            "name": "ER Diagram",
            "description": "Entity-relationship diagram",
            "diagram_type": "er",
            "gold_standard": """erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    PRODUCT ||--|{ LINE-ITEM : "ordered in"
    CUSTOMER {
        string name
        string email
        int id
    }
    ORDER {
        int orderNumber
        date orderDate
    }
    PRODUCT {
        string name
        float price
        int id
    }
    LINE-ITEM {
        int quantity
        float price
    }"""
        },
        {
            "name": "Journey Diagram",
            "description": "User journey diagram",
            "diagram_type": "journey",
            "gold_standard": """journey
    title User Journey
    section Discovery
      Visit Website: 5: User
      Browse Products: 4: User
      Read Reviews: 3: User
    section Purchase
      Add to Cart: 5: User
      Checkout: 4: User
      Payment: 3: User
    section Delivery
      Receive Email: 5: User
      Track Order: 4: User
      Receive Product: 5: User"""
        }
    ]
    
    print(f"Testing {len(tough_cases)} tough examples...")
    print()
    
    results = []
    
    for i, case in enumerate(tough_cases, 1):
        print(f"{'=' * 80}")
        print(f"TEST {i}/{len(tough_cases)}: {case['name']}")
        print(f"{'=' * 80}")
        print()
        print(f"Description: {case['description']}")
        print()
        print(f"Expected Output:")
        print("```mermaid")
        print(case['gold_standard'])
        print("```")
        print()
        
        # Generate diagram
        try:
            generated = await expert.generate_mermaid(
                description=case['description'],
                diagram_type=case['diagram_type']
            )
            
            print(f"Generated Output:")
            print("```mermaid")
            print(generated)
            print("```")
            print()
            
            # Evaluate
            eval_result = await expert._evaluate_domain(
                output=generated,
                gold_standard=case['gold_standard'],
                task=case['name'],
                context={"description": case['description']}
            )
            
            score = eval_result.get('score', 0.0)
            status = eval_result.get('status', 'UNKNOWN')
            matches = eval_result.get('matches_gold', False)
            
            print(f"Evaluation:")
            print(f"  Score: {score:.2f} / 1.0")
            print(f"  Status: {status}")
            print(f"  Matches Gold: {'✅ YES' if matches else '❌ NO'}")
            
            if eval_result.get('issues'):
                print(f"  Issues:")
                for issue in eval_result['issues']:
                    print(f"    - {issue}")
            
            results.append({
                "name": case['name'],
                "score": score,
                "status": status,
                "matches_gold": matches,
                "generated": generated,
                "expected": case['gold_standard']
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "name": case['name'],
                "score": 0.0,
                "status": "ERROR",
                "matches_gold": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total = len(results)
    passed = sum(1 for r in results if r['matches_gold'])
    avg_score = sum(r['score'] for r in results) / total if total > 0 else 0.0
    
    print(f"Total Tests: {total}")
    print(f"Perfect Matches: {passed} ({passed/total*100:.1f}%)")
    print(f"Average Score: {avg_score:.2f} / 1.0")
    print()
    
    print("Results by Test:")
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result['matches_gold'] else "⚠️" if result['score'] >= 0.5 else "❌"
        print(f"  {status_icon} {i}. {result['name']}: {result['score']:.2f} ({result['status']})")
    
    print()
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    asyncio.run(test_tough_mermaid_examples())
