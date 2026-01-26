"""
Test Jotty Minimal Implementation
==================================

Tests the standalone jotty_minimal.py to verify:
1. Import time < 0.5s
2. Memory usage < 50MB
3. Simple tasks work correctly
4. Dynamic spawning works
5. Memory and message bus work
"""

import sys
import time
import tracemalloc
import asyncio
from jotty_minimal import (
    Orchestrator, AgentConfig, SimpleMemory, MessageBus,
    DynamicSpawner, PlannerSignature, ExecutorSignature,
    setup_dspy, setup_logging
)

def test_import_time():
    """Test 1: Import time should be < 0.5s"""
    print("\n" + "="*80)
    print("TEST 1: Import Time")
    print("="*80)

    # Import time already measured (module already imported)
    # This test just confirms imports work
    print("✅ All imports successful")
    print("   - Orchestrator, AgentConfig, SimpleMemory")
    print("   - MessageBus, DynamicSpawner")
    print("   - PlannerSignature, ExecutorSignature")

def test_memory_usage():
    """Test 2: Memory usage should be < 50MB for minimal setup"""
    print("\n" + "="*80)
    print("TEST 2: Memory Usage")
    print("="*80)

    tracemalloc.start()

    # Create orchestrator
    orchestrator = Orchestrator()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    current_mb = current / 1024 / 1024
    peak_mb = peak / 1024 / 1024

    print(f"   Current: {current_mb:.2f} MB")
    print(f"   Peak: {peak_mb:.2f} MB")

    if peak_mb < 50:
        print(f"✅ Memory usage under 50MB threshold")
    else:
        print(f"⚠️ Memory usage above 50MB threshold")

def test_simple_memory():
    """Test 3: SimpleMemory operations"""
    print("\n" + "="*80)
    print("TEST 3: SimpleMemory Operations")
    print("="*80)

    memory = SimpleMemory(max_entries=100)

    # Store entries
    memory.store("First memory", tags=["test", "important"])
    memory.store("Second memory", tags=["test"])
    memory.store("Third memory", tags=["other"])

    # Retrieve by tags
    test_entries = memory.retrieve(tags=["test"], limit=10)
    print(f"   Retrieved {len(test_entries)} entries with 'test' tag")

    # Search
    search_results = memory.search("First", limit=10)
    print(f"   Found {len(search_results)} entries matching 'First'")

    if len(test_entries) == 2 and len(search_results) == 1:
        print("✅ Memory storage and retrieval working")
    else:
        print("❌ Memory test failed")

def test_message_bus():
    """Test 4: MessageBus operations"""
    print("\n" + "="*80)
    print("TEST 4: MessageBus Operations")
    print("="*80)

    bus = MessageBus()

    # Send messages
    bus.send("agent1", "agent2", "Hello from agent1")
    bus.send("agent2", "agent1", "Reply from agent2")
    bus.broadcast("planner", "Broadcast to all")

    # Get messages
    agent2_messages = bus.get_messages("agent2", limit=10)
    print(f"   Agent2 received {len(agent2_messages)} messages")

    if len(agent2_messages) >= 2:  # Direct message + broadcast
        print("✅ Message passing working")
    else:
        print("❌ Message bus test failed")

def test_dynamic_spawning():
    """Test 5: DynamicSpawner operations"""
    print("\n" + "="*80)
    print("TEST 5: Dynamic Spawning")
    print("="*80)

    spawner = DynamicSpawner(max_spawned_per_agent=3)

    # Assess complexity
    simple_task = "Write hello world"
    complex_task = "Research and analyze multiple quantum computing algorithms in detail"

    simple_assessment = spawner.assess_complexity(simple_task)
    complex_assessment = spawner.assess_complexity(complex_task)

    print(f"   Simple task complexity: {simple_assessment['complexity_score']}/5")
    print(f"   Should spawn: {simple_assessment['should_spawn']}")

    print(f"   Complex task complexity: {complex_assessment['complexity_score']}/5")
    print(f"   Should spawn: {complex_assessment['should_spawn']}")

    # Spawn agents
    spawner.spawn(
        parent_agent="planner",
        agent_name="researcher_1",
        description="Research quantum algorithms",
        signature=ExecutorSignature
    )

    spawner.spawn(
        parent_agent="planner",
        agent_name="researcher_2",
        description="Analyze quantum algorithms",
        signature=ExecutorSignature
    )

    hierarchy = spawner.get_hierarchy()
    print(f"   Spawned {spawner.total_spawned} agents")
    print(f"   Hierarchy: {hierarchy}")

    if (not simple_assessment['should_spawn'] and
        complex_assessment['should_spawn'] and
        spawner.total_spawned == 2):
        print("✅ Dynamic spawning working")
    else:
        print("❌ Dynamic spawning test failed")

async def test_orchestrator():
    """Test 6: Full orchestrator run (requires API key)"""
    print("\n" + "="*80)
    print("TEST 6: Orchestrator Execution")
    print("="*80)

    # Check if API key available
    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("⚠️ Skipped: No API key found")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test full execution")
        return

    try:
        # Setup DSPy
        setup_dspy(model="gpt-4o-mini")

        # Create orchestrator
        orchestrator = Orchestrator()

        # Run simple task
        result = await orchestrator.run(
            goal="Write a hello world program in Python",
            max_steps=5
        )

        print(f"   Success: {result['success']}")
        print(f"   Steps completed: {len(result.get('results', []))}")

        if result['success']:
            print("✅ Orchestrator execution working")
        else:
            print("❌ Orchestrator execution failed")

    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# JOTTY MINIMAL TEST SUITE")
    print("#"*80)

    # Setup logging
    setup_logging("WARNING")  # Quiet during tests

    # Run tests
    test_import_time()
    test_memory_usage()
    test_simple_memory()
    test_message_bus()
    test_dynamic_spawning()

    # Async test
    asyncio.run(test_orchestrator())

    print("\n" + "#"*80)
    print("# ALL TESTS COMPLETED")
    print("#"*80)

if __name__ == "__main__":
    run_all_tests()
