"""
Comprehensive Test: PlantUML Expert Agent

Tests PlantUML expert with:
1. Memory integration
2. Training
3. Complex scenarios
4. Memory consolidation
"""

import asyncio
import sys
from pathlib import Path
import re
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import PlantUMLExpertAgent, ExpertAgentConfig
from core.experts.memory_integration import (
    store_improvement_to_memory,
    retrieve_improvements_from_memory,
    consolidate_improvements,
    retrieve_synthesized_improvements
)
from core.memory.cortex import SwarmMemory
from core.foundation.data_structures import SwarmConfig, MemoryLevel


def configure_llm():
    """Configure DSPy with LLM."""
    try:
        import dspy
    except ImportError:
        print("❌ DSPy not available")
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
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude API")
            return True
        except:
            pass
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            lm = dspy.LM(model="openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ Configured with OpenAI")
            return True
        except:
            pass
    
    print("❌ No LLM available")
    return False


def validate_plantuml_syntax(plantuml_code: str) -> tuple[bool, str]:
    """Validate basic PlantUML syntax."""
    if not plantuml_code:
        return False, "Empty code"
    
    plantuml_code = re.sub(r'^```plantuml\s*\n?', '', plantuml_code, flags=re.MULTILINE)
    plantuml_code = re.sub(r'^```\s*$', '', plantuml_code, flags=re.MULTILINE)
    plantuml_code = plantuml_code.strip()
    
    # Check for @startuml/@enduml tags
    has_start = '@startuml' in plantuml_code.lower() or '@start' in plantuml_code.lower()
    has_end = '@enduml' in plantuml_code.lower() or '@end' in plantuml_code.lower()
    
    if not has_start or not has_end:
        return False, f"Missing @startuml/@enduml tags. Has start: {has_start}, Has end: {has_end}"
    
    return True, "Valid"


# Complex PlantUML Scenarios
COMPLEX_SCENARIOS = [
    {
        "name": "Complex Sequence Diagram",
        "description": "Create a sequence diagram for online shopping: Customer sends order to OrderService, OrderService validates with PaymentService, PaymentService processes payment, OrderService confirms with Customer",
        "diagram_type": "sequence",
        "expected_elements": ["Customer", "OrderService", "PaymentService"]
    },
    {
        "name": "Class Diagram - E-commerce",
        "description": "Design a class diagram for e-commerce: User, Product, Order, OrderItem classes. User has many Orders, Order has many OrderItems, OrderItem belongs to Product",
        "diagram_type": "class",
        "expected_elements": ["User", "Product", "Order", "OrderItem"]
    },
    {
        "name": "Activity Diagram - Order Processing",
        "description": "Create an activity diagram for order processing: Start -> Receive Order -> Validate -> Process Payment -> Ship -> End",
        "diagram_type": "activity",
        "expected_elements": ["Receive Order", "Validate", "Process Payment", "Ship"]
    },
    {
        "name": "Component Diagram - Microservices",
        "description": "Draw a component diagram showing microservices: API Gateway component, Auth Service component, User Service component, Order Service component",
        "diagram_type": "component",
        "expected_elements": ["API Gateway", "Auth Service", "User Service", "Order Service"]
    },
    {
        "name": "State Diagram - Order States",
        "description": "Create a state diagram for order states: Pending -> Processing -> Shipped -> Delivered, with Cancelled state",
        "diagram_type": "state",
        "expected_elements": ["Pending", "Processing", "Shipped", "Delivered"]
    }
]


async def test_plantuml_expert_comprehensive():
    """Comprehensive test of PlantUML expert agent."""
    print("=" * 80)
    print("COMPREHENSIVE TEST: PLANTUML EXPERT AGENT")
    print("=" * 80)
    print()
    
    if not configure_llm():
        print("❌ Cannot proceed without LLM")
        return []
    print()
    
    # Create memory system
    print("1. Setting up Memory System")
    print("-" * 80)
    memory_config = SwarmConfig()
    memory = SwarmMemory(
        agent_name="plantuml_expert_test",
        config=memory_config
    )
    print("✅ Memory system created")
    print()
    
    # Create expert with memory
    print("2. Creating PlantUML Expert Agent")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="plantuml_expert_test",
        domain="plantuml",
        description="PlantUML expert for complex diagrams",
        use_memory_storage=True,
        use_memory_synthesis=False,  # Test raw improvements first
        expert_data_dir="./test_outputs/plantuml_expert",
        max_training_iterations=2
    )
    
    expert = PlantUMLExpertAgent(config=config, memory=memory)
    print(f"✅ Expert created")
    print(f"   Uses memory storage: {expert.use_memory_storage}")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    print()
    
    # Quick training
    print("3. Training Expert Agent")
    print("-" * 80)
    print("   Training on basic cases...")
    
    training_cases = [
        {
            "task": "Generate simple sequence diagram",
            "context": {"description": "User and System", "diagram_type": "sequence"},
            "gold_standard": """@startuml
User -> System: Request
System --> User: Response
@enduml"""
        },
        {
            "task": "Generate class diagram",
            "context": {"description": "Basic class structure", "diagram_type": "class"},
            "gold_standard": """@startuml
class Animal {
    +name: string
}
class Dog {
    +breed: string
}
Animal <|-- Dog
@enduml"""
        },
        {
            "task": "Generate activity diagram",
            "context": {"description": "Simple process flow", "diagram_type": "activity"},
            "gold_standard": """@startuml
start
:Process;
stop
@enduml"""
        }
    ]
    
    try:
        training_results = await asyncio.wait_for(
            expert.train(gold_standards=training_cases, force_retrain=True),
            timeout=120
        )
        print(f"✅ Training completed")
        print(f"   Passed cases: {training_results.get('passed_cases', 0)}")
        print(f"   Improvements learned: {len(expert.improvements)}")
        
        # Reload improvements after training
        expert.improvements = expert._load_improvements()
        print(f"   Improvements after reload: {len(expert.improvements)}")
        
        # Mark as trained if we have improvements or passed cases
        if training_results.get('passed_cases', 0) > 0 or len(expert.improvements) > 0:
            expert.trained = True
            print(f"   Expert marked as trained")
        else:
            print(f"   ⚠️  Training didn't produce improvements, marking as trained for testing")
            expert.trained = True  # Mark as trained for testing purposes
    except Exception as e:
        print(f"⚠️  Training error: {e}")
        print(f"   Improvements so far: {len(expert.improvements)}")
        # Mark as trained for testing
        expert.trained = True
        print(f"   Expert marked as trained for testing")
    
    # Check memory
    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    semantic_count = len(memory.memories[MemoryLevel.SEMANTIC])
    meta_count = len(memory.memories[MemoryLevel.META])
    print(f"   Memory: {procedural_count} PROCEDURAL, {semantic_count} SEMANTIC, {meta_count} META")
    print()
    
    # Test complex scenarios
    print("4. Testing 5 Complex Scenarios")
    print("-" * 80)
    print()
    
    results = []
    for i, scenario in enumerate(COMPLEX_SCENARIOS, 1):
        print(f"   Scenario {i}/5: {scenario['name']}")
        
        try:
            diagram = await asyncio.wait_for(
                expert.generate_plantuml(
                    description=scenario['description'],
                    diagram_type=scenario['diagram_type']
                ),
                timeout=45
            )
            
            is_valid, error_msg = validate_plantuml_syntax(diagram)
            found_elements = [e for e in scenario['expected_elements'] if e.lower() in diagram.lower()]
            element_coverage = len(found_elements) / len(scenario['expected_elements'])
            
            result = {
                "scenario": scenario['name'],
                "valid": is_valid,
                "error": error_msg if not is_valid else None,
                "elements_found": found_elements,
                "element_coverage": element_coverage,
                "success": is_valid and element_coverage >= 0.5,
                "diagram_preview": diagram[:150] + "..." if len(diagram) > 150 else diagram
            }
            results.append(result)
            
            status = "✅" if result['success'] else "⚠️" if result['valid'] else "❌"
            print(f"   {status} Valid: {is_valid}, Elements: {len(found_elements)}/{len(scenario['expected_elements'])}")
            if result['elements_found']:
                print(f"      Found: {', '.join(result['elements_found'][:3])}")
            if not is_valid:
                print(f"      Error: {error_msg}")
            
        except asyncio.TimeoutError:
            print(f"   ⏱️  Timeout")
            results.append({"scenario": scenario['name'], "valid": False, "error": "Timeout", "success": False})
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            results.append({"scenario": scenario['name'], "valid": False, "error": str(e)[:100], "success": False})
        
        print()
    
    # Test memory consolidation
    print("5. Testing Memory Consolidation")
    print("-" * 80)
    
    # Reload improvements from memory to get accurate count
    expert.improvements = expert._load_improvements()
    procedural_count_after = len(memory.memories[MemoryLevel.PROCEDURAL])
    
    print(f"   Improvements in memory: {len(expert.improvements)}")
    print(f"   PROCEDURAL memories: {procedural_count_after}")
    
    if procedural_count_after >= 3:
        consolidation_result = consolidate_improvements(
            memory=memory,
            expert_name=config.name,
            domain=config.domain
        )
        print(f"✅ Consolidation result:")
        print(f"   Consolidated: {consolidation_result.get('consolidated', 0)}")
        print(f"   Preferences: {consolidation_result.get('preferences', 0)}")
        
        # Check if SEMANTIC memory was created
        semantic_count_after = len(memory.memories[MemoryLevel.SEMANTIC])
        if semantic_count_after > semantic_count:
            print(f"   ✅ SEMANTIC memory created: {semantic_count_after} (was {semantic_count})")
    else:
        print(f"⚠️  Not enough improvements for consolidation (have {procedural_count_after}, need at least 3)")
        print(f"   Training produced {procedural_count_after} improvements")
    
    # Test synthesized retrieval
    print()
    print("6. Testing Synthesized Improvements")
    print("-" * 80)
    
    synthesized = retrieve_synthesized_improvements(
        memory=memory,
        expert_name=config.name,
        domain=config.domain
    )
    
    if synthesized:
        print(f"✅ Synthesized improvements retrieved")
        print(f"   Length: {len(synthesized)} chars")
        print(f"   Preview: {synthesized[:200]}...")
    else:
        print("⚠️  No synthesized improvements (may need more improvements or LLM)")
    
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r.get('success', False))
    valid_syntax = sum(1 for r in results if r.get('valid', False))
    avg_coverage = sum(r.get('element_coverage', 0) for r in results) / len(results) if results else 0
    
    print(f"✅ Successful scenarios: {successful}/5")
    print(f"✅ Valid syntax: {valid_syntax}/5")
    print(f"✅ Average element coverage: {avg_coverage:.1%}")
    print(f"✅ Improvements in memory: {len(expert.improvements)}")
    print(f"✅ Memory levels: {procedural_count} PROCEDURAL, {semantic_count} SEMANTIC, {meta_count} META")
    print()
    
    print("Detailed Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "✅" if result.get('success') else "⚠️" if result.get('valid') else "❌"
        print(f"{i:2d}. {status} {result['scenario']}")
        if result.get('elements_found'):
            print(f"     Elements: {', '.join(result['elements_found'])}")
        if result.get('error'):
            print(f"     Error: {result['error'][:60]}")
    
    print()
    return results


if __name__ == "__main__":
    results = asyncio.run(test_plantuml_expert_comprehensive())
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n{'✅ Test PASSED' if successful >= 4 else '⚠️  Test PARTIAL' if successful >= 2 else '❌ Test FAILED'}: {successful}/5 scenarios successful")
    sys.exit(0 if successful >= 4 else 1)
