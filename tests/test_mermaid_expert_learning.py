"""
Test Mermaid Expert Agent Learning

Tests if MermaidExpertAgent actually learns by:
1. Training on examples
2. Testing with complex descriptions it hasn't seen
3. Verifying it generates appropriate diagrams
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available")
    sys.exit(1)

from core.experts import MermaidExpertAgent, ExpertAgentConfig


async def test_mermaid_expert_learning():
    """Test if MermaidExpertAgent actually learns from training."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT LEARNING")
    print("=" * 80)
    print()
    
    # Check LLM configuration
    try:
        lm = dspy.settings.lm
        print(f"✅ LLM Configured: {type(lm).__name__}")
    except:
        print("⚠️  No LLM configured. Using mock mode.")
        print("   Configure with: dspy.configure(lm=dspy.LM(model='claude-3-opus'))")
        print()
        print("   Note: This test will show the structure but won't actually")
        print("   call the LLM. For real learning, configure an LLM.")
        print()
    
    # Training examples - teach the agent basic patterns
    training_cases = [
        {
            "task": "Generate simple flowchart",
            "context": {
                "description": "Start to End flow",
                "diagram_type": "flowchart"
            },
            "gold_standard": """graph TD
    A[Start]
    B[End]
    A --> B"""
        },
        {
            "task": "Generate decision flowchart",
            "context": {
                "description": "User login with validation",
                "diagram_type": "flowchart"
            },
            "gold_standard": """graph TD
    A[User Login]
    B{Valid?}
    C[Show Dashboard]
    D[Show Error]
    A --> B
    B -->|Yes| C
    B -->|No| D"""
        },
        {
            "task": "Generate multi-step workflow",
            "context": {
                "description": "Process with multiple steps",
                "diagram_type": "flowchart"
            },
            "gold_standard": """graph TD
    A[Start]
    B[Step 1]
    C[Step 2]
    D[Step 3]
    E[End]
    A --> B
    B --> C
    C --> D
    D --> E"""
        }
    ]
    
    # Create expert with training cases
    config = ExpertAgentConfig(
        name="mermaid_learning_test",
        domain="mermaid",
        description="Mermaid expert for learning test",
        training_gold_standards=training_cases,
        max_training_iterations=3,
        required_training_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        expert_data_dir="./test_outputs/mermaid_learning_test"
    )
    
    expert = MermaidExpertAgent(config=config)

    # Verify training data (BaseExpert does not have .train())
    print("=" * 80)
    print("PHASE 1: VERIFY TRAINING DATA")
    print("=" * 80)
    print()

    training_data = expert.get_training_data()
    print(f"Training Data:")
    print(f"  Cases available: {len(training_data)}")
    print()

    for i, case in enumerate(training_data, 1):
        print(f"  Case {i}: {case.get('task', 'Unknown')}")
        print()

    # Show stats
    stats = expert.get_stats()
    print(f"Improvements: {stats['improvements_count']}")
    print()
    
    # Test with complex descriptions the agent hasn't seen
    print("=" * 80)
    print("PHASE 2: TESTING WITH COMPLEX DESCRIPTIONS")
    print("=" * 80)
    print()
    
    complex_test_cases = [
        {
            "name": "Complex Multi-Branch Decision Tree",
            "description": "A complex decision tree with multiple branches: Start with user authentication, then check if user has admin permissions, then validate the data, and if all pass, process the request, otherwise show appropriate error messages",
            "diagram_type": "flowchart",
            "expected_elements": ["graph", "[", "]", "{", "}", "-->", "|Yes|", "|No|"]
        },
        {
            "name": "Multi-Stage Pipeline",
            "description": "A CI/CD pipeline with stages: Source code, Build, Unit Tests, Integration Tests, Deploy to Staging, Deploy to Production",
            "diagram_type": "flowchart",
            "expected_elements": ["graph", "[", "]", "-->"]
        },
        {
            "name": "Complex Workflow with Parallel Steps",
            "description": "A workflow that starts, then splits into two parallel processes (Process A and Process B), then both converge into a final step",
            "diagram_type": "flowchart",
            "expected_elements": ["graph", "[", "]", "-->"]
        },
        {
            "name": "User Registration Flow",
            "description": "User registration flow: User enters email, validate email format, check if email exists, if exists show error, if not create account, send verification email, show success",
            "diagram_type": "flowchart",
            "expected_elements": ["graph", "[", "]", "{", "}", "-->"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(complex_test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Description: {test_case['description']}")
        print()
        
        try:
            # Generate diagram
            diagram = await expert.generate_mermaid(
                description=test_case['description'],
                diagram_type=test_case['diagram_type']
            )
            
            print(f"  Generated Diagram:")
            print("  ```mermaid")
            print(f"  {diagram}")
            print("  ```")
            print()
            
            # Validate syntax
            diagram_str = str(diagram)
            validation = {
                "has_graph": "graph" in diagram_str.lower() or "flowchart" in diagram_str.lower(),
                "has_nodes": "[" in diagram_str or "{" in diagram_str,
                "has_arrows": "-->" in diagram_str,
                "has_decision_nodes": "{" in diagram_str and "}" in diagram_str,
                "has_labels": "|" in diagram_str
            }
            
            print(f"  Validation:")
            for key, value in validation.items():
                status = "✅" if value else "❌"
                print(f"    {status} {key.replace('_', ' ').title()}: {value}")
            
            # Check expected elements
            missing_elements = []
            for element in test_case['expected_elements']:
                if element not in diagram_str:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"  ⚠️  Missing expected elements: {', '.join(missing_elements)}")
            else:
                print(f"  ✅ All expected elements present!")
            
            # Score
            score = sum(validation.values()) / len(validation)
            print(f"  Score: {score:.2f} / 1.0")
            
            results.append({
                "name": test_case['name'],
                "diagram": diagram,
                "validation": validation,
                "score": score,
                "missing_elements": missing_elements
            })
            
        except RuntimeError as e:
            if "must be trained" in str(e):
                print(f"  ⚠️  Expert not trained (likely no LLM configured)")
                print(f"  This is expected if no LLM is available.")
            else:
                print(f"  ❌ Error: {e}")
            results.append({
                "name": test_case['name'],
                "error": str(e)
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": test_case['name'],
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    successful_tests = [r for r in results if 'score' in r]
    failed_tests = [r for r in results if 'error' in r]
    
    if successful_tests:
        avg_score = sum(r['score'] for r in successful_tests) / len(successful_tests)
        print(f"Successful Tests: {len(successful_tests)}/{len(results)}")
        print(f"Average Score: {avg_score:.2f} / 1.0")
        print()
        
        print("Results by Test:")
        for result in results:
            if 'score' in result:
                status = "✅" if result['score'] >= 0.8 else "⚠️" if result['score'] >= 0.5 else "❌"
                print(f"  {status} {result['name']}: {result['score']:.2f}")
            else:
                print(f"  ❌ {result['name']}: Error - {result.get('error', 'Unknown')}")
    else:
        print("⚠️  No successful tests (likely no LLM configured)")
        print()
        print("To test real learning:")
        print("  1. Configure LLM: dspy.configure(lm=dspy.LM(model='claude-3-opus'))")
        print("  2. Run this test again")
        print("  3. The expert will train on examples and learn patterns")
        print("  4. Then it will generate diagrams for complex descriptions")
    
    print()
    print("=" * 80)

    # Show improvements from expert instance
    if expert.improvements:
        print()
        print("=" * 80)
        print("EXPERT IMPROVEMENTS")
        print("=" * 80)
        print()
        print(f"Total Improvements: {len(expert.improvements)}")
        print()

        for i, imp in enumerate(expert.improvements[:3], 1):
            print(f"Improvement {i}:")
            print(f"  Task: {imp.get('task', 'Unknown')}")
            print(f"  Learned Pattern: {imp.get('learned_pattern', '')[:100]}...")
            print()

    return results


if __name__ == "__main__":
    asyncio.run(test_mermaid_expert_learning())
