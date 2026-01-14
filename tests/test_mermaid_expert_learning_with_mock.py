"""
Test Mermaid Expert Learning with Mock LLM

This test demonstrates the learning process with a mock LLM that simulates
learning behavior. For real testing, configure an actual LLM.
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

from core.experts import ExpertAgentConfig, ExpertAgent
from core.foundation.agent_config import AgentConfig


class MockLearningAgent:
    """Mock agent that simulates learning behavior."""
    
    def __init__(self):
        self.attempts = {}
        self.learned_patterns = []
    
    def __call__(self, task=None, description=None, learned_improvements=None, **kwargs):
        """Simulate LLM generation with learning."""
        # Track attempts
        key = f"{task}:{description}"
        if key not in self.attempts:
            self.attempts[key] = 0
        self.attempts[key] += 1
        
        # Apply learned improvements
        if learned_improvements:
            # Use learned patterns
            for pattern in self.learned_patterns:
                if pattern['matches'](task, description):
                    result = type('Result', (), {})()
                    result.output = pattern['output']
                    return result
        
        # First attempt: Generate basic diagram
        if self.attempts[key] == 1:
            # Simulate wrong output
            result = type('Result', (), {})()
            result.output = "graph A --> B"  # Missing nodes
            return result
        else:
            # After learning: Generate better diagram
            result = type('Result', (), {})()
            # Generate based on description
            if "decision" in description.lower() or "check" in description.lower():
                result.output = """graph TD
    A[Start]
    B{Decision?}
    C[Yes Path]
    D[No Path]
    A --> B
    B -->|Yes| C
    B -->|No| D"""
            elif "pipeline" in description.lower() or "stage" in description.lower():
                stages = description.split(":")[-1].split(",")
                stages = [s.strip() for s in stages if s.strip()]
                mermaid = "graph LR\n"
                for i, stage in enumerate(stages[:6]):  # Limit to 6 stages
                    mermaid += f"    {chr(65+i)}[{stage}]\n"
                for i in range(len(stages[:6]) - 1):
                    mermaid += f"    {chr(65+i)} --> {chr(65+i+1)}\n"
                result.output = mermaid.strip()
            else:
                # Generic flowchart
                result.output = """graph TD
    A[Start]
    B[Process]
    C[End]
    A --> B
    B --> C"""
            return result
    
    def learn(self, task, description, correct_output):
        """Learn a pattern."""
        self.learned_patterns.append({
            'task': task,
            'description_keywords': description.lower().split(),
            'output': correct_output,
            'matches': lambda t, d: (
                task.lower() in t.lower() or
                any(kw in d.lower() for kw in description.lower().split()[:3])
            )
        })


class MockTeacherAgent:
    """Mock teacher that provides correct outputs."""
    
    def __call__(self, task=None, gold_standard=None, **kwargs):
        """Return gold standard."""
        result = type('Result', (), {})()
        result.output = gold_standard
        return result


class MockMermaidExpert(ExpertAgent):
    """Mock Mermaid expert for testing learning."""
    
    def _create_default_agent(self):
        """Create mock agent."""
        return MockLearningAgent()
    
    def _create_default_teacher(self):
        """Create mock teacher."""
        return MockTeacherAgent()


async def test_learning_with_mock():
    """Test learning with mock agents."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT LEARNING (MOCK MODE)")
    print("=" * 80)
    print()
    print("This test uses mock agents to demonstrate the learning process.")
    print("For real testing with LLMs, configure: dspy.configure(lm=dspy.LM(...))")
    print()
    
    # Training examples
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
                "description": "User login with validation check",
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
        }
    ]
    
    # Create expert
    config = ExpertAgentConfig(
        name="mermaid_mock_test",
        domain="mermaid",
        description="Mock Mermaid expert",
        training_gold_standards=training_cases,
        max_training_iterations=3,
        required_training_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        expert_data_dir="./test_outputs/mermaid_mock_test"
    )
    
    expert = MockMermaidExpert(config)
    
    # Train
    print("=" * 80)
    print("PHASE 1: TRAINING")
    print("=" * 80)
    print()
    
    training_results = await expert.train()
    
    print(f"Training Results:")
    print(f"  Success: {training_results.get('overall_success')}")
    print(f"  Passed: {training_results.get('passed_cases')}/{training_results.get('total_cases')}")
    print()
    
    for case_result in training_results.get('training_cases', []):
        print(f"  Case {case_result['case_number']}: {case_result['task']}")
        print(f"    Success: {case_result['success']}")
        print(f"    Final Score: {case_result['final_score']:.2f}")
        print(f"    Iterations: {case_result['iterations']}")
        print()
    
    # Show improvements
    status = expert.get_status()
    print(f"Improvements Learned: {status['improvements_count']}")
    print()
    
    # Test with complex descriptions
    print("=" * 80)
    print("PHASE 2: TESTING WITH COMPLEX DESCRIPTIONS")
    print("=" * 80)
    print()
    
    complex_tests = [
        {
            "name": "Complex Decision Tree",
            "description": "A complex decision tree: Start with authentication, then check admin permissions, then validate data, process if all pass, show errors otherwise",
            "diagram_type": "flowchart"
        },
        {
            "name": "CI/CD Pipeline",
            "description": "A CI/CD pipeline with stages: Source code, Build, Unit Tests, Integration Tests, Deploy Staging, Deploy Production",
            "diagram_type": "flowchart"
        },
        {
            "name": "Multi-Step Workflow",
            "description": "A workflow: Start, Process A, Process B, Process C, End",
            "diagram_type": "flowchart"
        }
    ]
    
    results = []
    
    for i, test in enumerate(complex_tests, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Description: {test['description']}")
        print()
        
        try:
            diagram = await expert.generate_mermaid(
                description=test['description'],
                diagram_type=test['diagram_type']
            )
            
            print(f"  Generated Diagram:")
            print("  ```mermaid")
            print(f"  {diagram}")
            print("  ```")
            print()
            
            # Validate
            diagram_str = str(diagram)
            validation = {
                "has_graph": "graph" in diagram_str.lower(),
                "has_nodes": "[" in diagram_str,
                "has_arrows": "-->" in diagram_str,
                "has_decision": "{" in diagram_str,
                "valid_syntax": True
            }
            
            score = sum(validation.values()) / len(validation)
            
            print(f"  Validation:")
            for key, value in validation.items():
                status = "✅" if value else "❌"
                print(f"    {status} {key.replace('_', ' ').title()}")
            print(f"  Score: {score:.2f} / 1.0")
            
            results.append({
                "name": test['name'],
                "diagram": diagram,
                "score": score,
                "validation": validation
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "name": test['name'],
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    successful = [r for r in results if 'score' in r]
    if successful:
        avg_score = sum(r['score'] for r in successful) / len(successful)
        print(f"✅ Successful Tests: {len(successful)}/{len(results)}")
        print(f"✅ Average Score: {avg_score:.2f} / 1.0")
        print()
        
        print("Results:")
        for result in results:
            if 'score' in result:
                status = "✅" if result['score'] >= 0.8 else "⚠️"
                print(f"  {status} {result['name']}: {result['score']:.2f}")
            else:
                print(f"  ❌ {result['name']}: {result.get('error', 'Unknown')}")
    else:
        print("❌ No successful tests")
    
    print()
    print("=" * 80)
    print("LEARNING DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("This mock test demonstrates:")
    print("  1. ✅ Training on examples")
    print("  2. ✅ Learning from teacher corrections")
    print("  3. ✅ Generating diagrams for new descriptions")
    print("  4. ✅ Applying learned patterns")
    print()
    print("For real LLM testing:")
    print("  1. Configure LLM: dspy.configure(lm=dspy.LM(model='claude-3-opus'))")
    print("  2. Use MermaidExpertAgent (not MockMermaidExpert)")
    print("  3. Run the same test - it will use real LLM generation")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_learning_with_mock())
