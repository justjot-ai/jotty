"""
Simple Test: Mermaid Expert Agent with 10 Complex Scenarios
Tests memory integration - marks expert as trained for testing.
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import ExpertAgentConfig, MermaidExpertAgent
from core.foundation.data_structures import MemoryLevel, SwarmConfig
from core.memory.cortex import SwarmMemory


def configure_llm():
    """Configure DSPy with LLM."""
    try:
        import dspy
    except ImportError:
        return False

    try:
        import subprocess

        from examples.claude_cli_wrapper import ClaudeCLILM

        result = subprocess.run(["claude", "--version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            lm = ClaudeCLILM(model="sonnet")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude CLI")
            return True
    except Exception as e:
        logging.debug(f"Configuration failed: {e}")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude API")
            return True
        except Exception as e:
            logging.debug(f"Configuration failed: {e}")

    return False


def validate_mermaid_syntax(mermaid_code: str) -> tuple[bool, str]:
    """Validate basic Mermaid syntax."""
    if not mermaid_code:
        return False, "Empty code"

    mermaid_code = re.sub(r"^```mermaid\s*\n?", "", mermaid_code, flags=re.MULTILINE)
    mermaid_code = re.sub(r"^```\s*$", "", mermaid_code, flags=re.MULTILINE)
    mermaid_code = mermaid_code.strip()

    valid_types = [
        "graph",
        "flowchart",
        "sequenceDiagram",
        "classDiagram",
        "stateDiagram",
        "erDiagram",
        "gantt",
        "pie",
        "gitgraph",
        "journey",
        "C4Context",
        "mindmap",
    ]

    first_line = mermaid_code.split("\n")[0].strip()
    has_valid_type = any(first_line.startswith(t) for t in valid_types)

    if not has_valid_type:
        return False, f"Invalid diagram type: {first_line[:50]}"

    return True, "Valid"


# 10 Complex Scenarios
COMPLEX_SCENARIOS = [
    {
        "name": "Multi-level Organization Chart",
        "description": "Create a hierarchical organization chart with CEO at top, then CTO/CFO/CMO, then departments under each, with multiple levels",
        "expected_elements": ["CEO", "CTO", "CFO", "CMO"],
    },
    {
        "name": "Complex State Machine",
        "description": "Design a state diagram for an e-commerce order processing system with states: Pending, Payment, Processing, Shipped, Delivered, Cancelled",
        "expected_elements": ["Pending", "Payment", "Processing", "Shipped"],
    },
    {
        "name": "Database ER Diagram",
        "description": "Create an entity-relationship diagram with Users, Orders, Products, OrderItems tables. Users have one-to-many relationship with Orders",
        "expected_elements": ["Users", "Orders", "Products"],
    },
    {
        "name": "Microservices Architecture",
        "description": "Draw a flowchart showing microservices architecture: API Gateway connects to Auth Service, User Service, Order Service, Payment Service",
        "expected_elements": ["API Gateway", "Auth Service", "User Service"],
    },
    {
        "name": "CI/CD Pipeline Flow",
        "description": "Create a flowchart for CI/CD pipeline: Code Commit triggers Build, then Test, then Deploy to Staging, then Production",
        "expected_elements": ["Commit", "Build", "Test", "Deploy"],
    },
    {
        "name": "Sequence Diagram - User Authentication",
        "description": "Create a sequence diagram showing user login flow: User sends credentials to Frontend, Frontend sends to Auth Service, Auth Service validates with Database",
        "expected_elements": ["User", "Frontend", "Auth Service"],
    },
    {
        "name": "Class Diagram - Library System",
        "description": "Design a class diagram for a library management system with classes: Book, Author, Member, Loan, Library",
        "expected_elements": ["Book", "Author", "Member"],
    },
    {
        "name": "Complex Flowchart - Decision Tree",
        "description": "Create a flowchart for a customer support routing system: Start -> Is Premium Customer? -> Yes: Route to VIP Support, No: Route to Standard Queue",
        "expected_elements": ["Premium Customer", "VIP Support", "Standard Queue"],
    },
    {
        "name": "Git Workflow Diagram",
        "description": "Draw a gitgraph showing Git Flow workflow: main branch, develop branch, feature branches, release branches",
        "expected_elements": ["main", "develop", "feature"],
    },
    {
        "name": "Journey Map - Customer Onboarding",
        "description": "Create a journey diagram showing customer onboarding: Sign Up -> Email Verification -> Profile Setup -> First Purchase",
        "expected_elements": ["Sign Up", "Email Verification", "Profile Setup"],
    },
]


async def test_complex_scenarios():
    """Test expert agent with 10 complex Mermaid scenarios."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT WITH 10 COMPLEX SCENARIOS")
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
    memory = SwarmMemory(agent_name="mermaid_expert_test", config=memory_config)
    print("✅ Memory system created")
    print()

    # Create expert with memory
    print("2. Creating Expert Agent")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="mermaid_expert_test",
        domain="mermaid",
        description="Mermaid expert for complex diagrams",
        use_memory_storage=True,
        expert_data_dir="./test_outputs/mermaid_test_memory",
    )

    expert = MermaidExpertAgent(config=config, memory=memory)

    # Add some mock improvements to memory for testing
    from core.experts.memory_integration import store_improvement_to_memory

    test_improvement = {
        "learned_pattern": "When generating flowcharts, use 'flowchart TD' instead of 'graph TD'",
        "task": "Generate flowchart",
        "improvement_type": "teacher_correction",
    }
    store_improvement_to_memory(memory, test_improvement, config.name, config.domain)

    # Reload improvements from memory
    expert.improvements = expert._load_improvements()

    # Mark as trained for testing (since training is slow)
    expert.trained = True

    print(f"✅ Expert created")
    print(f"   Uses memory storage: {expert.use_memory_storage}")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    print(f"   Marked as trained: {expert.trained}")
    print()

    # Check memory
    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    meta_count = len(memory.memories[MemoryLevel.META])
    print(f"   Memory: {procedural_count} PROCEDURAL, {meta_count} META")
    print()

    # Test scenarios
    print("3. Testing 10 Complex Scenarios")
    print("-" * 80)
    print()

    results = []
    for i, scenario in enumerate(COMPLEX_SCENARIOS, 1):
        print(f"   Scenario {i}/10: {scenario['name']}")

        try:
            diagram = await asyncio.wait_for(
                expert.generate_mermaid(description=scenario["description"]), timeout=45
            )

            is_valid, error_msg = validate_mermaid_syntax(diagram)
            found_elements = [
                e for e in scenario["expected_elements"] if e.lower() in diagram.lower()
            ]
            element_coverage = len(found_elements) / len(scenario["expected_elements"])

            result = {
                "scenario": scenario["name"],
                "valid": is_valid,
                "error": error_msg if not is_valid else None,
                "elements_found": found_elements,
                "element_coverage": element_coverage,
                "success": is_valid and element_coverage >= 0.5,
                "diagram_preview": diagram[:100] + "..." if len(diagram) > 100 else diagram,
            }
            results.append(result)

            status = "✅" if result["success"] else "⚠️" if result["valid"] else "❌"
            print(
                f"   {status} Valid: {is_valid}, Elements: {len(found_elements)}/{len(scenario['expected_elements'])}"
            )
            if result["elements_found"]:
                print(f"      Found: {', '.join(result['elements_found'][:3])}")

        except asyncio.TimeoutError:
            print(f"   ⏱️  Timeout")
            results.append(
                {"scenario": scenario["name"], "valid": False, "error": "Timeout", "success": False}
            )
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            results.append(
                {
                    "scenario": scenario["name"],
                    "valid": False,
                    "error": str(e)[:100],
                    "success": False,
                }
            )

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r.get("success", False))
    valid_syntax = sum(1 for r in results if r.get("valid", False))
    avg_coverage = (
        sum(r.get("element_coverage", 0) for r in results) / len(results) if results else 0
    )

    print(f"✅ Successful scenarios: {successful}/10")
    print(f"✅ Valid syntax: {valid_syntax}/10")
    print(f"✅ Average element coverage: {avg_coverage:.1%}")
    print(f"✅ Improvements in memory: {len(expert.improvements)}")
    print()

    print("Detailed Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "✅" if result.get("success") else "⚠️" if result.get("valid") else "❌"
        print(f"{i:2d}. {status} {result['scenario']}")
        if result.get("elements_found"):
            print(f"     Elements: {', '.join(result['elements_found'])}")
        if result.get("error"):
            print(f"     Error: {result['error'][:60]}")

    print()
    return results


if __name__ == "__main__":
    results = asyncio.run(test_complex_scenarios())
    successful = sum(1 for r in results if r.get("success", False))
    print(
        f"\n{'✅ Test PASSED' if successful >= 7 else '⚠️  Test PARTIAL' if successful >= 4 else '❌ Test FAILED'}: {successful}/10 scenarios successful"
    )
    sys.exit(0 if successful >= 7 else 1)
