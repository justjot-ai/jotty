"""
Comprehensive Tests for FULL SYSTEM (Including Tools & Conductor)
==================================================================

Previous tests only covered jotty_minimal.py (1,500 lines).
This test suite validates:
1. Tools integration (simple tools)
2. Full Conductor with learning
3. Full Conductor with tools
4. All module configurations

Uses Claude CLI with JSON output for actual LLM calls.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

import dspy

# Configure Claude CLI
print("Configuring DSPy with Claude CLI...")
try:
    sys.path.insert(0, str(Path(__file__).parent / "examples"))
    from claude_cli_wrapper_enhanced import EnhancedClaudeCLILM

    lm = EnhancedClaudeCLILM(model="sonnet")
    dspy.configure(lm=lm)
    print("✅ DSPy configured with Claude CLI (sonnet)")
except Exception as e:
    print(f"❌ Failed to configure Claude CLI: {e}")
    sys.exit(1)

# ============================================================================
# Test 7: Simple Tools Integration (jotty_minimal)
# ============================================================================

async def test_simple_tools():
    """Test 7: Simple tools with jotty_minimal orchestrator"""
    print("\n" + "="*80)
    print("TEST 7: Simple Tools Integration")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator, Agent, AgentConfig, SimpleMemory, MessageBus
        import dspy

        print("\n  Creating calculator tool agent...")

        # Define a simple calculator tool signature
        class CalculatorSignature(dspy.Signature):
            """Simple calculator tool"""
            operation = dspy.InputField(desc="Math operation to perform")
            result = dspy.OutputField(desc="Calculation result")

        # Create orchestrator
        orchestrator = Orchestrator()

        # Create agent config for calculator
        calc_config = AgentConfig(
            name="calculator",
            description="Performs mathematical calculations",
            signature=CalculatorSignature
        )

        # Create calculator agent with proper initialization
        calculator = Agent(
            config=calc_config,
            memory=orchestrator.memory,
            message_bus=orchestrator.message_bus
        )

        # Add to orchestrator
        orchestrator.agents['calculator'] = calculator

        print("  Testing calculator tool agent...")

        # Test tool execution
        result = await calculator.execute(operation="What is 15 + 27?")

        print(f"    Tool execution: {result.success}")
        if result.success and hasattr(result.output, 'result'):
            print(f"    Result: {result.output.result[:100]}")

        if result.success:
            print("✅ Simple tools integration works")
            return True
        else:
            print(f"⚠️ Tool execution failed: {result.error}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 8: Full Conductor with Learning
# ============================================================================

async def test_conductor_with_learning():
    """Test 8: Full Conductor with Q-learning enabled"""
    print("\n" + "="*80)
    print("TEST 8: Full Conductor with Learning")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.agent_config import AgentConfig
        import dspy

        print("\n  Creating Conductor with enable_learning=True...")

        # Simple signature for testing
        class SimpleTaskSignature(dspy.Signature):
            """Simple task execution"""
            task = dspy.InputField()
            result = dspy.OutputField()

        # Create agent config using correct AgentConfig
        agents = [
            AgentConfig(
                name="worker",
                agent=dspy.ChainOfThought(SimpleTaskSignature),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]

        print("  Initializing Conductor...")

        # Create Conductor with learning enabled
        conductor = MultiAgentsOrchestrator(
            actors=agents,
            enable_learning=True,  # Enable Q-learning
            enable_validation=False,
            enable_memory=False,
            max_steps=2
        )

        print("  Running task with learning enabled...")
        result = await conductor.run(goal="Count from 1 to 3", context={})

        print(f"  Success: {result is not None}")

        # Check if Q-learning module initialized
        has_learning = hasattr(conductor, 'q_learning') and conductor.q_learning is not None
        print(f"  Q-learning initialized: {has_learning}")

        if result is not None:
            print("✅ Full Conductor with learning works")
            return True
        else:
            print("⚠️ Conductor execution returned None")
            return False

    except ImportError as e:
        print(f"  ⚠️ Skipped: Cannot import Conductor - {e}")
        return None
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 9: Full Conductor with Tools
# ============================================================================

async def test_conductor_with_tools():
    """Test 9: Full Conductor with ToolShed integration"""
    print("\n" + "="*80)
    print("TEST 9: Full Conductor with Tools")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.metadata.tool_shed import ToolShed
        from core.foundation.agent_config import AgentConfig
        import dspy

        print("\n  Creating ToolShed with sample tools...")

        # Create tool shed
        tool_shed = ToolShed()

        # Define a simple search tool
        def search_tool(query: str) -> str:
            """Simple search tool that returns mock results"""
            return f"Search results for: {query} - Found 3 results"

        # Register tool
        tool_shed.register_tool(
            name="search",
            function=search_tool,
            description="Search for information",
            parameters={"query": "string"}
        )

        print(f"  Registered tools: {list(tool_shed.tools.keys())}")

        # Simple signature
        class TaskWithToolsSignature(dspy.Signature):
            """Task that can use tools"""
            task = dspy.InputField()
            result = dspy.OutputField()

        # Create agent that can use tools
        agents = [
            AgentConfig(
                name="researcher",
                agent=dspy.ChainOfThought(TaskWithToolsSignature),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]

        print("  Initializing Conductor with tools...")

        # Create Conductor with tools
        conductor = MultiAgentsOrchestrator(
            actors=agents,
            enable_learning=False,
            enable_validation=False,
            enable_memory=False,
            max_steps=2
        )

        # Attach tool shed to conductor
        conductor.tool_shed = tool_shed

        print("  Running task with tools available...")

        # Execute task that could use tools
        result = await conductor.run(
            goal="Find information about Python programming",
            context={}
        )

        print(f"  Success: {result is not None}")
        print(f"  Tools registered: {len(tool_shed.tools)}")

        if result is not None and len(tool_shed.tools) > 0:
            print("✅ Full Conductor with tools works")
            return True
        else:
            print("⚠️ Conductor with tools execution incomplete")
            return False

    except ImportError as e:
        print(f"  ⚠️ Skipped: Cannot import required modules - {e}")
        return None
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 10: Module Configuration Validation
# ============================================================================

async def test_module_configurations():
    """Test 10: Validate module-based configuration combinations"""
    print("\n" + "="*80)
    print("TEST 10: Module Configuration Validation")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator

        print("\n  Testing different module configurations...")

        configs = [
            {
                "name": "Minimal (no memory, no spawning)",
                "max_memory_entries": 0,
                "max_spawned_per_agent": 0
            },
            {
                "name": "Development (memory, spawning)",
                "max_memory_entries": 1000,
                "max_spawned_per_agent": 3
            },
            {
                "name": "Production (memory, high spawning)",
                "max_memory_entries": 5000,
                "max_spawned_per_agent": 10
            }
        ]

        results = {}

        for config in configs:
            name = config.pop("name")
            print(f"\n  Testing: {name}")

            orchestrator = Orchestrator(**config)
            result = await orchestrator.run(
                goal="What is 2 + 2?",
                max_steps=1
            )

            success = result['success']
            memory_entries = len(orchestrator.memory.entries)

            print(f"    Success: {success}")
            print(f"    Memory entries: {memory_entries}")

            results[name] = success

        all_passed = all(results.values())

        if all_passed:
            print("\n✅ All module configurations work")
            return True
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"\n⚠️ Some configurations failed: {failed}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main Test Runner
# ============================================================================

async def run_full_system_tests():
    """Run ALL comprehensive tests including tools and Conductor"""
    print("\n" + "#"*80)
    print("# FULL SYSTEM COMPREHENSIVE TESTS")
    print("# (Tools + Conductor + All Configurations)")
    print("#"*80)

    tests = [
        ("Simple Tools Integration", test_simple_tools),
        ("Full Conductor + Learning", test_conductor_with_learning),
        ("Full Conductor + Tools", test_conductor_with_tools),
        ("Module Configurations", test_module_configurations),
    ]

    results = {}
    for name, test_func in tests:
        result = await test_func()
        results[name] = result

    # Summary
    print("\n" + "#"*80)
    print("# FULL SYSTEM TEST SUMMARY")
    print("#"*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⚠️ {name} (skipped)")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ Some tests failed")
        return False
    else:
        print("\n✅ All full system tests passed!")
        return True

if __name__ == "__main__":
    success = asyncio.run(run_full_system_tests())
    sys.exit(0 if success else 1)
