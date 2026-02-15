"""
Test Mermaid Expert Agent with 10 Complex Scenarios
Verifies that learned improvements from memory are used correctly.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import ExpertAgentConfig, MermaidExpertAgent
from core.foundation.data_structures import SwarmConfig
from core.memory.cortex import SwarmMemory


def configure_llm():
    """Configure DSPy with LLM (Claude CLI preferred, fallback to API)."""
    try:
        import dspy

        DSPY_AVAILABLE = True
    except ImportError:
        print("❌ DSPy not available. Install with: pip install dspy-ai")
        return False

    # Try Claude CLI first (no API key needed)
    try:
        import subprocess

        from examples.claude_cli_wrapper import ClaudeCLILM

        # Check if Claude CLI is available
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Claude CLI available: {result.stdout.strip()}")
            lm = ClaudeCLILM(model="sonnet")
            dspy.configure(lm=lm)
            print("✅ Configured DSPy with Claude CLI wrapper")
            return True
        else:
            print("⚠️  Claude CLI not working")
    except FileNotFoundError:
        print("⚠️  Claude CLI not found")
        print("   Install from: https://github.com/anthropics/claude-code")
    except Exception as e:
        print(f"⚠️  Failed to configure Claude CLI: {e}")

    # Fallback to API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        print("🔧 Configuring DSPy with Claude API...")
        try:
            lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude API")
            return True
        except Exception as e:
            print(f"⚠️  Failed to configure Claude API: {e}")

    if openai_key:
        print("🔧 Configuring DSPy with OpenAI...")
        try:
            lm = dspy.LM(model="openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ Configured with OpenAI")
            return True
        except Exception as e:
            print(f"⚠️  Failed to configure OpenAI: {e}")

    # Check if already configured
    try:
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            print(f"✅ DSPy already configured with: {type(dspy.settings.lm).__name__}")
            return True
    except (AttributeError, TypeError):
        pass

    print("❌ Could not configure LLM")
    print("   Options:")
    print("   1. Install Claude CLI: https://github.com/anthropics/claude-code")
    print("   2. Set ANTHROPIC_API_KEY environment variable")
    print("   3. Set OPENAI_API_KEY environment variable")
    return False


def validate_mermaid_syntax(mermaid_code: str) -> tuple[bool, str]:
    """Validate basic Mermaid syntax."""
    if not mermaid_code:
        return False, "Empty code"

    # Remove markdown code fences if present
    mermaid_code = re.sub(r"^```mermaid\s*\n?", "", mermaid_code, flags=re.MULTILINE)
    mermaid_code = re.sub(r"^```\s*$", "", mermaid_code, flags=re.MULTILINE)
    mermaid_code = mermaid_code.strip()

    # Check for valid diagram types
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
        return False, f"Invalid diagram type. First line: {first_line[:50]}"

    # Check for balanced brackets/parentheses
    open_brackets = mermaid_code.count("[") + mermaid_code.count("(") + mermaid_code.count("{")
    close_brackets = mermaid_code.count("]") + mermaid_code.count(")") + mermaid_code.count("}")

    if open_brackets != close_brackets:
        return False, f"Unbalanced brackets: {open_brackets} open, {close_brackets} close"

    # Check for arrows (basic connectivity)
    if "graph" in first_line or "flowchart" in first_line:
        if "-->" not in mermaid_code and "---" not in mermaid_code and "==>" not in mermaid_code:
            return False, "No connections found in graph/flowchart"

    return True, "Valid"


# 10 Complex Mermaid Scenarios
COMPLEX_SCENARIOS = [
    {
        "name": "Multi-level Organization Chart",
        "description": "Create a hierarchical organization chart with CEO at top, then CTO/CFO/CMO, then departments under each, with multiple levels",
        "expected_elements": ["CEO", "CTO", "CFO", "CMO", "departments"],
    },
    {
        "name": "Complex State Machine",
        "description": "Design a state diagram for an e-commerce order processing system with states: Pending, Payment, Processing, Shipped, Delivered, Cancelled, and transitions between them including error states",
        "expected_elements": [
            "Pending",
            "Payment",
            "Processing",
            "Shipped",
            "Delivered",
            "Cancelled",
        ],
    },
    {
        "name": "Database ER Diagram",
        "description": "Create an entity-relationship diagram with Users, Orders, Products, OrderItems tables. Users have one-to-many relationship with Orders. Orders have many-to-many with Products through OrderItems. Include primary keys and foreign keys",
        "expected_elements": ["Users", "Orders", "Products", "OrderItems"],
    },
    {
        "name": "Microservices Architecture",
        "description": "Draw a flowchart showing microservices architecture: API Gateway connects to Auth Service, User Service, Order Service, Payment Service, and Notification Service. Each service connects to its own database",
        "expected_elements": [
            "API Gateway",
            "Auth Service",
            "User Service",
            "Order Service",
            "Payment Service",
        ],
    },
    {
        "name": "CI/CD Pipeline Flow",
        "description": "Create a flowchart for CI/CD pipeline: Code Commit triggers Build, then Test, then Security Scan, then Deploy to Staging, then Integration Tests, then Deploy to Production, with rollback paths",
        "expected_elements": ["Commit", "Build", "Test", "Deploy", "Staging", "Production"],
    },
    {
        "name": "Sequence Diagram - User Authentication",
        "description": "Create a sequence diagram showing user login flow: User sends credentials to Frontend, Frontend sends to Auth Service, Auth Service validates with Database, returns JWT token, Frontend stores token",
        "expected_elements": ["User", "Frontend", "Auth Service", "Database"],
    },
    {
        "name": "Class Diagram - Library System",
        "description": "Design a class diagram for a library management system with classes: Book, Author, Member, Loan, Library. Book has many Authors, Member has many Loans, Loan connects Book and Member",
        "expected_elements": ["Book", "Author", "Member", "Loan", "Library"],
    },
    {
        "name": "Complex Flowchart - Decision Tree",
        "description": "Create a flowchart for a customer support routing system: Start -> Is Premium Customer? -> Yes: Route to VIP Support, No: Is Issue Urgent? -> Yes: Route to Priority Queue, No: Route to Standard Queue. Each queue has multiple steps",
        "expected_elements": [
            "Premium Customer",
            "VIP Support",
            "Priority Queue",
            "Standard Queue",
        ],
    },
    {
        "name": "Git Workflow Diagram",
        "description": "Draw a gitgraph showing Git Flow workflow: main branch, develop branch, feature branches, release branches, hotfix branches, with merges between them",
        "expected_elements": ["main", "develop", "feature", "release", "hotfix"],
    },
    {
        "name": "Journey Map - Customer Onboarding",
        "description": "Create a journey diagram showing customer onboarding: Sign Up -> Email Verification -> Profile Setup -> First Purchase -> Payment Setup -> Order Confirmation -> Welcome Email",
        "expected_elements": ["Sign Up", "Email Verification", "Profile Setup", "First Purchase"],
    },
]


async def test_complex_scenarios():
    """Test expert agent with 10 complex Mermaid scenarios."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT WITH 10 COMPLEX SCENARIOS")
    print("=" * 80)
    print()

    # Configure LLM
    print("0. Configuring LLM")
    print("-" * 80)
    if not configure_llm():
        print("❌ Cannot proceed without LLM")
        return []
    print()

    # Create memory system
    print("1. Setting up Memory System")
    print("-" * 80)
    memory_config = SwarmConfig()
    memory = SwarmMemory(agent_name="mermaid_expert_complex_test", config=memory_config)
    print("✅ Memory system created")
    print()

    # Create expert with memory
    print("2. Creating Expert Agent with Memory")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="mermaid_expert_complex",
        domain="mermaid",
        description="Mermaid expert for complex diagrams",
        use_memory_storage=True,
        expert_data_dir="./test_outputs/mermaid_complex_memory",
        max_training_iterations=3,
    )

    expert = MermaidExpertAgent(config=config, memory=memory)
    print(f"✅ Expert created")
    print(f"   Uses memory storage: {expert.use_memory_storage}")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    print()

    # Train expert first (to learn patterns and store to memory)
    print("3. Training Expert Agent")
    print("-" * 80)
    print("   Training on basic cases to learn patterns...")
    print("   (This may take a few minutes with real LLM)")

    training_cases = [
        {
            "task": "Generate a simple flowchart with Start and End nodes",
            "expected_output": "graph TD\n    Start[Start]\n    End[End]\n    Start --> End",
        },
        {
            "task": "Create a basic sequence diagram with User and System",
            "expected_output": "sequenceDiagram\n    User->>System: Request\n    System-->>User: Response",
        },
    ]

    try:
        training_results = await asyncio.wait_for(
            expert.train(gold_standards=training_cases, force_retrain=True),
            timeout=300,  # 5 minutes timeout
        )
        print(f"✅ Training completed")
        print(f"   Passed cases: {training_results.get('passed_cases', 0)}")
        print(f"   Total iterations: {training_results.get('total_iterations', 0)}")
        print(f"   Improvements learned: {len(expert.improvements)}")
    except asyncio.TimeoutError:
        print("⚠️  Training timed out (may still have learned some patterns)")
        print(f"   Improvements learned so far: {len(expert.improvements)}")
    except Exception as e:
        print(f"⚠️  Training error: {e}")
        print(f"   Improvements learned so far: {len(expert.improvements)}")

    print()

    # Check memory storage
    print("4. Verifying Memory Storage")
    print("-" * 80)
    from core.foundation.data_structures import MemoryLevel

    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    meta_count = len(memory.memories[MemoryLevel.META])
    print(f"   PROCEDURAL memories: {procedural_count}")
    print(f"   META memories: {meta_count}")
    print(f"   Total improvements in memory: {procedural_count + meta_count}")
    print()

    # Test 10 complex scenarios
    print("5. Testing 10 Complex Scenarios")
    print("-" * 80)
    print()

    results = []
    for i, scenario in enumerate(COMPLEX_SCENARIOS, 1):
        print(f"   Scenario {i}/10: {scenario['name']}")
        print(f"   Description: {scenario['description'][:60]}...")

        try:
            # Generate diagram (with timeout)
            diagram = await asyncio.wait_for(
                expert.generate_mermaid(
                    description=scenario["description"], context={"scenario": scenario["name"]}
                ),
                timeout=60,  # 1 minute per generation
            )

            # Validate syntax
            is_valid, error_msg = validate_mermaid_syntax(diagram)

            # Check for expected elements
            found_elements = []
            for element in scenario["expected_elements"]:
                if element.lower() in diagram.lower():
                    found_elements.append(element)

            element_coverage = len(found_elements) / len(scenario["expected_elements"])

            # Store result
            result = {
                "scenario": scenario["name"],
                "valid": is_valid,
                "error": error_msg if not is_valid else None,
                "diagram": diagram[:200] + "..." if len(diagram) > 200 else diagram,
                "elements_found": found_elements,
                "element_coverage": element_coverage,
                "success": is_valid and element_coverage >= 0.5,  # At least 50% elements found
            }
            results.append(result)

            # Print result
            if result["success"]:
                print(
                    f"   ✅ SUCCESS - Valid syntax, {len(found_elements)}/{len(scenario['expected_elements'])} elements found"
                )
            else:
                print(
                    f"   ⚠️  PARTIAL - Valid: {is_valid}, Elements: {len(found_elements)}/{len(scenario['expected_elements'])}"
                )
                if not is_valid:
                    print(f"      Error: {error_msg}")

        except Exception as e:
            print(f"   ❌ ERROR: {str(e)[:100]}")
            results.append(
                {"scenario": scenario["name"], "valid": False, "error": str(e), "success": False}
            )

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    successful = sum(1 for r in results if r.get("success", False))
    valid_syntax = sum(1 for r in results if r.get("valid", False))
    avg_coverage = (
        sum(r.get("element_coverage", 0) for r in results) / len(results) if results else 0
    )

    print(f"✅ Successful scenarios: {successful}/10")
    print(f"✅ Valid syntax: {valid_syntax}/10")
    print(f"✅ Average element coverage: {avg_coverage:.1%}")
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
    print("Memory System Status:")
    print("-" * 80)
    print(f"   Improvements in memory: {len(expert.improvements)}")
    print(f"   PROCEDURAL memories: {procedural_count}")
    print(f"   META memories: {meta_count}")
    print()

    # Verify improvements are being used
    print("Verification:")
    print("-" * 80)
    if len(expert.improvements) > 0:
        print("✅ Improvements loaded from memory")
        print(
            f"   Sample improvement: {expert.improvements[0].get('learned_pattern', 'N/A')[:80]}..."
        )
    else:
        print("⚠️  No improvements loaded (may need more training)")

    if procedural_count + meta_count > 0:
        print("✅ Improvements stored in memory system")
    else:
        print("⚠️  No improvements in memory (may need training)")

    print()
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = asyncio.run(test_complex_scenarios())

    # Exit with error code if not all successful
    successful = sum(1 for r in results if r.get("success", False))
    if successful < 7:  # At least 7/10 should succeed
        print(f"\n⚠️  Only {successful}/10 scenarios fully successful")
        sys.exit(1)
    else:
        print(f"\n✅ {successful}/10 scenarios successful - Test passed!")
        sys.exit(0)
