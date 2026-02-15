"""
Quick Test: Mermaid Expert Agent with 5 Complex Scenarios
Tests memory integration with fewer scenarios for faster execution.
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experts import ExpertAgentConfig, MermaidExpertAgent
from core.foundation.data_structures import SwarmConfig
from core.memory.cortex import SwarmMemory


def configure_llm():
    """Configure DSPy with LLM."""
    try:
        import dspy
    except ImportError:
        print("❌ DSPy not available")
        return False

    # Try Claude CLI first
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
        logging.debug(f"Claude CLI configuration failed: {e}")

    # Fallback to API
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude API")
            return True
        except Exception as e:
            logging.debug(f"Anthropic API configuration failed: {e}")

    if os.getenv("OPENAI_API_KEY"):
        try:
            lm = dspy.LM(model="openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ Configured with OpenAI")
            return True
        except Exception as e:
            logging.debug(f"OpenAI configuration failed: {e}")

    print("❌ No LLM available")
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


# 5 Complex Scenarios (subset for faster testing)
COMPLEX_SCENARIOS = [
    {
        "name": "Multi-level Organization Chart",
        "description": "Create a hierarchical organization chart with CEO at top, then CTO/CFO/CMO, then departments under each",
        "expected_elements": ["CEO", "CTO", "CFO"],
    },
    {
        "name": "Complex State Machine",
        "description": "Design a state diagram for an e-commerce order: Pending, Payment, Processing, Shipped, Delivered, Cancelled",
        "expected_elements": ["Pending", "Payment", "Shipped"],
    },
    {
        "name": "Database ER Diagram",
        "description": "Create an ER diagram with Users, Orders, Products tables. Users have one-to-many with Orders",
        "expected_elements": ["Users", "Orders", "Products"],
    },
    {
        "name": "Microservices Architecture",
        "description": "Draw a flowchart: API Gateway connects to Auth Service, User Service, Order Service",
        "expected_elements": ["API Gateway", "Auth Service", "User Service"],
    },
    {
        "name": "Sequence Diagram",
        "description": "Create a sequence diagram: User sends credentials to Frontend, Frontend sends to Auth Service",
        "expected_elements": ["User", "Frontend", "Auth Service"],
    },
]


async def test_complex_scenarios():
    """Test expert agent with 5 complex Mermaid scenarios."""
    print("=" * 80)
    print("QUICK TEST: MERMAID EXPERT WITH 5 COMPLEX SCENARIOS")
    print("=" * 80)
    print()

    # Configure LLM
    if not configure_llm():
        print("❌ Cannot proceed without LLM")
        return []
    print()

    # Create memory system
    print("1. Setting up Memory System")
    print("-" * 80)
    memory_config = SwarmConfig()
    memory = SwarmMemory(agent_name="mermaid_expert_quick_test", config=memory_config)
    print("✅ Memory system created")
    print()

    # Create expert with memory
    print("2. Creating Expert Agent")
    print("-" * 80)
    config = ExpertAgentConfig(
        name="mermaid_expert_quick",
        domain="mermaid",
        description="Mermaid expert for complex diagrams",
        use_memory_storage=True,
        expert_data_dir="./test_outputs/mermaid_quick_memory",
        max_training_iterations=2,  # Reduced for speed
    )

    expert = MermaidExpertAgent(config=config, memory=memory)
    print(f"✅ Expert created")
    print(f"   Uses memory storage: {expert.use_memory_storage}")
    print(f"   Improvements loaded: {len(expert.improvements)}")
    print()

    # Quick training
    print("3. Quick Training (1 case)")
    print("-" * 80)
    training_cases = [
        {
            "task": "Generate a simple flowchart with Start and End",
            "expected_output": "graph TD\n    Start[Start]\n    End[End]\n    Start --> End",
        }
    ]

    try:
        training_results = await asyncio.wait_for(
            expert.train(gold_standards=training_cases, force_retrain=True),
            timeout=120,  # 2 minutes
        )
        print(f"✅ Training completed")
        print(f"   Improvements learned: {len(expert.improvements)}")
    except Exception as e:
        print(f"⚠️  Training: {e}")
        print(f"   Improvements so far: {len(expert.improvements)}")

    # Check memory
    from core.foundation.data_structures import MemoryLevel

    procedural_count = len(memory.memories[MemoryLevel.PROCEDURAL])
    meta_count = len(memory.memories[MemoryLevel.META])
    print(f"   Memory: {procedural_count} PROCEDURAL, {meta_count} META")
    print()

    # Test scenarios
    print("4. Testing 5 Complex Scenarios")
    print("-" * 80)
    print()

    results = []
    for i, scenario in enumerate(COMPLEX_SCENARIOS, 1):
        print(f"   Scenario {i}/5: {scenario['name']}")

        try:
            diagram = await asyncio.wait_for(
                expert.generate_mermaid(description=scenario["description"]), timeout=30
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
            }
            results.append(result)

            status = "✅" if result["success"] else "⚠️" if result["valid"] else "❌"
            print(
                f"   {status} Valid: {is_valid}, Elements: {len(found_elements)}/{len(scenario['expected_elements'])}"
            )

        except asyncio.TimeoutError:
            print(f"   ⏱️  Timeout")
            results.append(
                {"scenario": scenario["name"], "valid": False, "error": "Timeout", "success": False}
            )
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:60]}")
            results.append(
                {"scenario": scenario["name"], "valid": False, "error": str(e), "success": False}
            )

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r.get("success", False))
    valid_syntax = sum(1 for r in results if r.get("valid", False))

    print(f"✅ Successful: {successful}/5")
    print(f"✅ Valid syntax: {valid_syntax}/5")
    print(f"✅ Improvements in memory: {len(expert.improvements)}")
    print()

    return results


if __name__ == "__main__":
    results = asyncio.run(test_complex_scenarios())
    successful = sum(1 for r in results if r.get("success", False))
    sys.exit(0 if successful >= 3 else 1)
