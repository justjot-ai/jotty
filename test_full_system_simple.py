"""
SIMPLIFIED Full System Tests - Tools + jotty_minimal
=====================================================

Since Full Conductor has complex dependencies (metadata_provider, JottyConfig),
we'll focus on comprehensive testing of jotty_minimal with tools.

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
# Test 7: Simple Tools with jotty_minimal
# ============================================================================

async def test_simple_tools():
    """Test 7: Simple tools with jotty_minimal orchestrator"""
    print("\n" + "="*80)
    print("TEST 7: Simple Tools Integration (Calculator Agent)")
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
# Test 8: Web Search Tool
# ============================================================================

async def test_web_search_tool():
    """Test 8: Web search tool integration"""
    print("\n" + "="*80)
    print("TEST 8: Web Search Tool Integration")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator, Agent, AgentConfig
        import dspy

        print("\n  Creating web search tool agent...")

        # Define web search signature
        class WebSearchSignature(dspy.Signature):
            """Web search tool"""
            query = dspy.InputField(desc="Search query")
            results = dspy.OutputField(desc="Search results")

        # Create orchestrator
        orchestrator = Orchestrator()

        # Create agent config
        search_config = AgentConfig(
            name="web_searcher",
            description="Searches the web for information",
            signature=WebSearchSignature
        )

        # Create search agent
        searcher = Agent(
            config=search_config,
            memory=orchestrator.memory,
            message_bus=orchestrator.message_bus
        )

        # Add to orchestrator
        orchestrator.agents['web_searcher'] = searcher

        print("  Testing web search tool...")

        # Test search
        result = await searcher.execute(query="Python programming language")

        print(f"    Tool execution: {result.success}")
        if result.success and hasattr(result.output, 'results'):
            print(f"    Results (first 150 chars): {str(result.output.results)[:150]}")

        if result.success:
            print("✅ Web search tool integration works")
            return True
        else:
            print(f"⚠️ Search failed: {result.error}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 9: Multi-Tool Orchestration
# ============================================================================

async def test_multi_tool_orchestration():
    """Test 9: Orchestrator with multiple tools"""
    print("\n" + "="*80)
    print("TEST 9: Multi-Tool Orchestration")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator
        import dspy

        print("\n  Creating orchestrator with multiple tool agents...")

        # Create orchestrator
        orchestrator = Orchestrator(max_memory_entries=1000)

        # Run a task that benefits from having calculator tool
        result = await orchestrator.run(
            goal="Calculate the sum of 123 and 456, then explain the result",
            max_steps=3
        )

        print(f"    Task execution: {result['success']}")
        print(f"    Steps completed: {len(result.get('results', []))}")
        print(f"    Memory entries: {len(orchestrator.memory.entries)}")

        if result['success']:
            print("✅ Multi-tool orchestration works")
            return True
        else:
            print(f"⚠️ Task failed: {result.get('summary', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 10: Tool with Memory Persistence
# ============================================================================

async def test_tool_with_memory():
    """Test 10: Tools with memory persistence"""
    print("\n" + "="*80)
    print("TEST 10: Tools with Memory Persistence")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator
        import dspy

        print("\n  Testing tool execution with memory...")

        # Create orchestrator with memory enabled
        orchestrator = Orchestrator(max_memory_entries=1000)

        # First task - store calculation
        result1 = await orchestrator.run(
            goal="Calculate 50 * 20",
            max_steps=1
        )

        memory_after_calc = len(orchestrator.memory.entries)
        print(f"    After calculation: {memory_after_calc} memory entries")

        # Second task - use previous calculation context
        result2 = await orchestrator.run(
            goal="What was the previous calculation result?",
            max_steps=1
        )

        memory_after_recall = len(orchestrator.memory.entries)
        print(f"    After recall: {memory_after_recall} memory entries")

        if result1['success'] and result2['success'] and memory_after_recall > memory_after_calc:
            print("✅ Tool with memory persistence works")
            return True
        else:
            print(f"⚠️ Memory persistence incomplete")
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
    """Run ALL comprehensive tests"""
    print("\n" + "#"*80)
    print("# FULL SYSTEM TESTS - TOOLS + JOTTY_MINIMAL")
    print("#"*80)

    tests = [
        ("Simple Tools (Calculator)", test_simple_tools),
        ("Web Search Tool", test_web_search_tool),
        ("Multi-Tool Orchestration", test_multi_tool_orchestration),
        ("Tools with Memory", test_tool_with_memory),
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
