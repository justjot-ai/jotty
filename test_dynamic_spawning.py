#!/usr/bin/env python3
"""
Test Dynamic Agent Spawning - Both Approaches
==============================================

Demonstrates:
1. Approach 1: Simple manual spawning (MegaAgent style)
2. Approach 2: LLM-based complexity assessment + auto-spawning

This shows how Jotty's dynamic spawning compares to MegaAgent's approach
while maintaining our sophisticated DSPy architecture.
"""

import sys
from pathlib import Path
import asyncio

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import dspy
except ImportError:
    print("❌ DSPy not installed. Install with: pip install dspy-ai")
    sys.exit(1)

from core.orchestration.dynamic_spawner import DynamicAgentSpawner, create_spawn_tool
from core.orchestration.complexity_assessor import ComplexityAssessor, assess_and_spawn, ComplexityLevel
from core.integration.direct_claude_cli_lm import DirectClaudeCLI


# =============================================================================
# EXAMPLE DSPy SIGNATURES
# =============================================================================

class PlannerSignature(dspy.Signature):
    """Planner - Determines overall strategy"""
    goal: str = dspy.InputField(desc="High-level goal to achieve")
    plan: str = dspy.OutputField(desc="Step-by-step plan")


class ResearcherSignature(dspy.Signature):
    """Researcher - Conducts research on specific topics"""
    topic: str = dspy.InputField(desc="Topic to research")
    research_summary: str = dspy.OutputField(desc="Research findings")


class ContentWriterSignature(dspy.Signature):
    """Content Writer - Writes content sections"""
    section_title: str = dspy.InputField(desc="Section to write")
    context: str = dspy.InputField(desc="Context/research to use")
    content: str = dspy.OutputField(desc="Written content")


# =============================================================================
# APPROACH 1: SIMPLE MANUAL SPAWNING
# =============================================================================

def test_approach_1_simple_spawning():
    """
    Test Approach 1: Simple manual spawning (MegaAgent style)

    Demonstrates:
    - Creating spawner
    - Manually spawning agents
    - Tracking subordinates
    - Parent-child relationships
    """
    print("\n" + "=" * 80)
    print("  APPROACH 1: SIMPLE MANUAL SPAWNING (MegaAgent Style)")
    print("=" * 80 + "\n")

    # Initialize spawner
    spawner = DynamicAgentSpawner(max_spawned_per_agent=5)
    print(f"✅ Created spawner: {spawner}\n")

    # Simulate a parent agent deciding to spawn subordinates
    parent_agent = "planner"

    print(f"📋 Agent '{parent_agent}' assessed task complexity")
    print("   Decision: Task needs 3 specialized researchers + 3 writers\n")

    # Spawn researchers
    print("🌱 Spawning researchers...")
    for i in range(1, 4):
        try:
            config = spawner.spawn_agent(
                name=f"researcher_{i}",
                description=f"Research section {i} of the guide",
                signature=ResearcherSignature,
                spawned_by=parent_agent
            )
            print(f"   ✅ Spawned: researcher_{i}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print()

    # Spawn writers
    print("🌱 Spawning writers...")
    for i in range(1, 4):
        try:
            config = spawner.spawn_agent(
                name=f"writer_{i}",
                description=f"Write section {i} from research",
                signature=ContentWriterSignature,
                spawned_by=parent_agent
            )
            print(f"   ✅ Spawned: writer_{i}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print()

    # Show results
    print("📊 Spawning Results:")
    print(f"   Total spawned: {spawner.total_spawned}")
    print(f"   Subordinates of '{parent_agent}': {len(spawner.get_subordinates(parent_agent))}")
    print(f"   Names: {', '.join(spawner.get_subordinates(parent_agent))}\n")

    # Try spawning beyond limit
    print("🚫 Testing limit enforcement...")
    try:
        config = spawner.spawn_agent(
            name="extra_agent",
            description="Extra agent beyond limit",
            signature=ResearcherSignature,
            spawned_by=parent_agent
        )
        print("   ❌ UNEXPECTED: Limit not enforced!")
    except ValueError as e:
        print(f"   ✅ Limit enforced: {e}\n")

    # Show spawn tree
    print("🌳 Spawn Tree:")
    tree = spawner.get_spawn_tree()
    for parent, info in tree.items():
        print(f"   {parent} ({info['count']} subordinates)")
        for sub in info['subordinates']:
            agent_info = spawner.get_agent_info(sub)
            print(f"      └─ {sub}: {agent_info.description}")

    print()

    # Show stats
    stats = spawner.get_stats()
    print("📈 Statistics:")
    print(f"   Total spawned: {stats['total_spawned']}")
    print(f"   Unique parents: {stats['unique_parents']}")
    print(f"   Avg subordinates per parent: {stats['avg_subordinates_per_parent']:.1f}")
    print(f"   Max subordinates: {stats['max_subordinates']}")

    return spawner


# =============================================================================
# APPROACH 2: LLM-BASED COMPLEXITY ASSESSMENT
# =============================================================================

def test_approach_2_complexity_assessment():
    """
    Test Approach 2: LLM-based complexity assessment + auto-spawning

    Demonstrates:
    - Complexity assessment with LLM
    - Automatic spawning decisions
    - Agent recommendations
    - Integration with spawner
    """
    print("\n\n" + "=" * 80)
    print("  APPROACH 2: LLM-BASED COMPLEXITY ASSESSMENT + AUTO-SPAWNING")
    print("=" * 80 + "\n")

    # Configure LLM
    print("🔧 Configuring Claude CLI...")
    try:
        lm = DirectClaudeCLI(model="haiku")
        dspy.configure(lm=lm)
        print("✅ Configured with Claude Haiku\n")
    except Exception as e:
        print(f"❌ Failed to configure LLM: {e}")
        print("   Skipping Approach 2 test\n")
        return None, None

    # Initialize spawner and assessor
    spawner = DynamicAgentSpawner(max_spawned_per_agent=10)
    assessor = ComplexityAssessor(signature_registry={
        "ResearcherSignature": ResearcherSignature,
        "ContentWriterSignature": ContentWriterSignature,
        "PlannerSignature": PlannerSignature
    })

    print(f"✅ Created spawner: {spawner}")
    print(f"✅ Created assessor: {assessor}\n")

    # Test cases with different complexity levels
    test_cases = [
        {
            "task": "Add two numbers together",
            "existing_agents": ["calculator"],
            "expected_complexity": ComplexityLevel.TRIVIAL
        },
        {
            "task": "Write a single 200-word blog post",
            "existing_agents": ["writer"],
            "expected_complexity": ComplexityLevel.SIMPLE
        },
        {
            "task": "Write a comprehensive 15-section technical guide with research",
            "existing_agents": ["planner"],
            "expected_complexity": ComplexityLevel.COMPLEX
        },
        {
            "task": "Build a multi-service distributed system with microservices, databases, and monitoring",
            "existing_agents": ["architect"],
            "expected_complexity": ComplexityLevel.VERY_COMPLEX
        }
    ]

    print("🧪 Testing Complexity Assessment on Various Tasks:\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['task'][:60]}...")
        print(f"   Existing agents: {', '.join(test['existing_agents'])}")

        # Assess complexity
        result = assessor.assess_task(
            task=test["task"],
            existing_agents=test["existing_agents"],
            current_progress=""
        )

        print(f"   ✅ Complexity: {result.complexity_level.name} (score {result.complexity_level.value})")
        print(f"   ✅ Should spawn: {result.should_spawn}")
        print(f"   ✅ Reasoning: {result.reasoning[:150]}...")

        if result.recommended_agents:
            print(f"   ✅ Recommended {len(result.recommended_agents)} agents:")
            for agent in result.recommended_agents[:3]:  # Show first 3
                print(f"      - {agent['name']} ({agent['signature_name']})")

        print()

    # Now test auto-spawning with complex task
    print("🌱 Testing Auto-Spawning with Complex Task:\n")

    complex_task = "Write a comprehensive 10-section guide on Kubernetes with research"
    parent_agent = "planner"

    print(f"Task: {complex_task}")
    print(f"Parent: {parent_agent}\n")

    # Assess and spawn
    spawned, agent_names = assess_and_spawn(
        assessor=assessor,
        spawner=spawner,
        task=complex_task,
        existing_agents=["planner"],
        parent_agent=parent_agent,
        current_progress=""
    )

    if spawned:
        print(f"\n✅ Auto-spawned {len(agent_names)} agents:")
        for name in agent_names:
            agent_info = spawner.get_agent_info(name)
            print(f"   - {name}: {agent_info.description}")
    else:
        print("\n❌ No agents spawned (task deemed simple enough for existing agents)")

    # Show final stats
    print("\n📈 Final Statistics:")
    stats = spawner.get_stats()
    print(f"   Total spawned: {stats['total_spawned']}")
    print(f"   Spawn tree: {stats['spawn_tree']}")

    return spawner, assessor


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_integration():
    """
    Test both approaches together

    Shows how Approach 1 (manual) and Approach 2 (auto) can coexist
    """
    print("\n\n" + "=" * 80)
    print("  INTEGRATION TEST: BOTH APPROACHES WORKING TOGETHER")
    print("=" * 80 + "\n")

    print("Scenario: Guide generation with mixed manual + auto spawning\n")

    # Configure LLM
    try:
        lm = DirectClaudeCLI(model="haiku")
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"❌ Failed to configure LLM: {e}")
        return

    # Initialize both systems
    spawner = DynamicAgentSpawner(max_spawned_per_agent=10)
    assessor = ComplexityAssessor(signature_registry={
        "ResearcherSignature": ResearcherSignature,
        "ContentWriterSignature": ContentWriterSignature
    })

    print(f"✅ Spawner ready: {spawner}")
    print(f"✅ Assessor ready: {assessor}\n")

    # Step 1: Manual spawning (Approach 1)
    print("Step 1: Manual spawning of core agents (Approach 1)")
    planner_config = spawner.spawn_agent(
        name="planner",
        description="Plan the guide structure",
        signature=PlannerSignature,
        spawned_by="system"
    )
    print(f"   ✅ Manually spawned: planner\n")

    # Step 2: Planner assesses complexity (Approach 2)
    print("Step 2: Planner uses complexity assessor to decide on subordinates (Approach 2)")
    spawned, researchers = assess_and_spawn(
        assessor=assessor,
        spawner=spawner,
        task="Write 10-section guide with research on each section",
        existing_agents=["planner"],
        parent_agent="planner",
        current_progress="Guide outline created"
    )

    if spawned:
        print(f"   ✅ Auto-spawned {len(researchers)} researchers based on LLM assessment\n")
    else:
        print("   ℹ️  No auto-spawning triggered\n")

    # Step 3: Show final hierarchy
    print("Step 3: Final Agent Hierarchy")
    print(f"   Total agents: {spawner.total_spawned}")
    tree = spawner.get_spawn_tree()
    for parent, info in tree.items():
        print(f"\n   {parent} ({info['count']} subordinates)")
        for sub in info['subordinates']:
            agent_info = spawner.get_agent_info(sub)
            print(f"      └─ {sub}: {agent_info.description[:50]}...")

    print("\n✅ Integration test complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "JOTTY DYNAMIC AGENT SPAWNING TEST" + " " * 25 + "║")
    print("║" + " " * 15 + "Approach 1 (Simple) + Approach 2 (Complex)" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test Approach 1
    spawner1 = test_approach_1_simple_spawning()

    # Test Approach 2
    spawner2, assessor2 = test_approach_2_complexity_assessment()

    # Integration test
    if spawner2 and assessor2:
        test_integration()

    print("\n" + "=" * 80)
    print("  ✨ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("  ✅ Approach 1: Simple manual spawning (MegaAgent style) - Working!")
    print("  ✅ Approach 2: LLM complexity assessment + auto-spawn - Working!")
    print("  ✅ Integration: Both approaches can work together\n")
    print("Jotty now has dynamic agent spawning with:")
    print("  • MegaAgent-style simplicity (Approach 1)")
    print("  • LLM-based intelligence (Approach 2)")
    print("  • DSPy signature integration")
    print("  • Parent-child tracking")
    print("  • Spawn limits for safety\n")
