#!/usr/bin/env python3
"""
Basic DSPy Agent Test (No External Dependencies)
Tests the most basic DSPy agent functionality
"""
def test_dspy_imports():
    """Test that all required imports work"""
    print("=" * 80)
    print("TEST 1: Basic Imports")
    print("=" * 80)

    try:
        import dspy
        print("✅ dspy imported successfully")
    except ImportError as e:
        print(f"❌ dspy import failed: {e}")
        print("   Install: pip install dspy-ai")
        return False

    try:
        from Jotty.core.infrastructure.integration.mcp_tool_executor import MCPToolExecutor
        print("✅ MCPToolExecutor imported successfully")
    except ImportError as e:
        print(f"❌ MCPToolExecutor import failed: {e}")
        return False

    try:
        from Jotty.core.modes.agent.dspy_mcp_agent import DSPyMCPAgent
        print("✅ DSPyMCPAgent imported successfully")
    except ImportError as e:
        print(f"❌ DSPyMCPAgent import failed: {e}")
        return False

    print()
    return True


def test_dspy_signature():
    """Test basic DSPy signature"""
    print("=" * 80)
    print("TEST 2: DSPy Signature")
    print("=" * 80)

    try:
        import dspy

        class SimpleSignature(dspy.Signature):
            """Basic question answering"""
            question = dspy.InputField(desc="A question")
            answer = dspy.OutputField(desc="The answer")

        print("✅ DSPy signature created successfully")
        print(f"   Input fields: {list(SimpleSignature.input_fields.keys())}")
        print(f"   Output fields: {list(SimpleSignature.output_fields.keys())}")
        print()
        return True

    except Exception as e:
        print(f"❌ DSPy signature failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcp_tool_executor_creation():
    """Test MCP tool executor creation (without actually calling tools)"""
    print("=" * 80)
    print("TEST 3: MCP Tool Executor Creation")
    print("=" * 80)

    try:
        from Jotty.core.infrastructure.integration.mcp_tool_executor import MCPToolExecutor

        executor = MCPToolExecutor(base_url="http://localhost:3000")
        print("✅ MCPToolExecutor created successfully")
        print(f"   Base URL: {executor.base_url}")
        print()
        return True

    except Exception as e:
        print(f"❌ MCPToolExecutor creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcp_tool_discovery():
    """Test MCP tool discovery (without network calls)"""
    print("=" * 80)
    print("TEST 4: MCP Tool Discovery")
    print("=" * 80)

    try:
        import asyncio
        from Jotty.core.infrastructure.integration.mcp_tool_executor import MCPToolExecutor

        async def discover():
            executor = MCPToolExecutor(base_url="http://localhost:3000")
            tools = await executor.discover_tools()
            return executor, tools

        executor, tools = asyncio.run(discover())

        print("✅ Tool discovery completed")
        print(f"   Tools discovered: {len(tools)}")
        for tool in tools[:3]:  # Show first 3
            print(f"   - {tool.name}: {tool.description[:60]}...")
        print()

        # Test formatting for DSPy
        formatted = executor.format_tools_for_dspy()
        print(f"✅ Tools formatted for DSPy ({len(formatted)} characters)")
        print()
        return True

    except Exception as e:
        print(f"❌ Tool discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dspy_agent_creation():
    """Test DSPy agent creation (without execution)"""
    print("=" * 80)
    print("TEST 5: DSPy Agent Creation")
    print("=" * 80)

    try:
        import asyncio
        from Jotty.core.modes.agent.dspy_mcp_agent import DSPyMCPAgent

        async def create_agent():
            agent = DSPyMCPAgent(
                name="Test Agent",
                description="A simple test agent",
                base_url="http://localhost:3000"
            )
            await agent.initialize()
            return agent

        agent = asyncio.run(create_agent())

        print("✅ DSPy agent created and initialized")
        print(f"   Name: {agent.name}")
        print(f"   Description: {agent.description}")
        print(f"   Tools available: {len(agent.mcp_executor.available_tools)}")
        print()
        return True

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claude_cli_availability():
    """Test if Claude CLI is available"""
    print("=" * 80)
    print("TEST 6: Claude CLI Availability")
    print("=" * 80)

    try:
        from Jotty.examples.claude_cli_wrapper import ClaudeCLILM
        import dspy

        lm = ClaudeCLILM(model="sonnet")
        dspy.configure(lm=lm)

        print("✅ Claude CLI configured successfully")
        print(f"   Model: sonnet")
        print()
        return True

    except Exception as e:
        print(f"⚠️  Claude CLI not available: {e}")
        print("   This is OK - Claude CLI needed for actual execution")
        print("   Install: npm install -g @anthropic-ai/claude-code")
        print()
        return False


def run_all_tests():
    """Run all basic tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "JOTTY DSPy + MCP BASIC TESTS" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Imports", test_dspy_imports()))
    results.append(("DSPy Signature", test_dspy_signature()))
    results.append(("MCP Tool Executor Creation", test_mcp_tool_executor_creation()))
    results.append(("MCP Tool Discovery", test_mcp_tool_discovery()))
    results.append(("DSPy Agent Creation", test_dspy_agent_creation()))
    results.append(("Claude CLI Availability", test_claude_cli_availability()))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("🎉 All tests passed!")
        print()
        print("Next steps:")
        print("  1. Run with Claude CLI: python tests/test_dspy_mcp_integration.py")
        print("  2. Test memory MCP server: python mcp_server/memory_server.py --mode http")
        print("  3. Integrate with JustJot agents")
    elif passed >= total - 1:
        print("✅ Core components working!")
        print("   (Claude CLI is optional for development)")
    else:
        print("❌ Some core components failed - check errors above")

    print()


if __name__ == "__main__":
    run_all_tests()
