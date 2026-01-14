"""
Test DSPy Agent with MCP Tools
Tests Phase 1: DSPy + MCP Tools Integration
"""
import asyncio
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agents.dspy_mcp_agent import DSPyMCPAgent
from examples.claude_cli_wrapper import ClaudeCLILM
import dspy


async def test_dspy_mcp_basic():
    """Test basic DSPy agent with MCP tools"""
    print("=" * 80)
    print("TEST: DSPy + MCP Tools Integration")
    print("=" * 80)
    print()

    # Step 1: Configure DSPy with Claude CLI
    print("Step 1: Configuring DSPy with Claude CLI")
    print("-" * 80)
    try:
        lm = ClaudeCLILM(model="sonnet")
        dspy.configure(lm=lm)
        print("✅ Claude CLI configured")
    except Exception as e:
        print(f"❌ Failed to configure Claude CLI: {e}")
        print("   Make sure Claude CLI is installed: npm install -g @anthropic-ai/claude-code")
        return
    print()

    # Step 2: Create DSPy agent
    print("Step 2: Creating DSPy MCP Agent")
    print("-" * 80)
    agent = DSPyMCPAgent(
        name="Research Assistant",
        description="Helps research and analyze ideas in JustJot",
        base_url="http://localhost:3000"  # Assuming JustJot running locally
    )
    await agent.initialize()
    print()

    # Step 3: Test simple query (tool discovery)
    print("Step 3: Testing Tool Discovery")
    print("-" * 80)
    print(f"Available tools: {agent.mcp_executor.get_tool_names()}")
    print(f"Total: {len(agent.mcp_executor.available_tools)}")
    print()

    # Step 4: Test agent execution with tool call
    print("Step 4: Testing Agent Execution (Search Ideas)")
    print("-" * 80)
    try:
        result = await agent.execute(
            query="Search for ideas about 'machine learning' and summarize what you find",
            conversation_history=""
        )

        print("✅ Agent execution completed")
        print()
        print("Reasoning:")
        print(result.get("reasoning", ""))
        print()
        print(f"Tool Calls: {len(result.get('tool_calls', []))}")
        for i, call in enumerate(result.get("tool_calls", []), 1):
            print(f"  {i}. {call.get('name')}: {call.get('arguments')}")
        print()
        print(f"Tool Results: {len(result.get('tool_results', []))}")
        for i, res in enumerate(result.get("tool_results", []), 1):
            success = "✅" if res.get("success") else "❌"
            print(f"  {success} {i}. {res.get('tool')}")
        print()
        print("Response:")
        print(result.get("response", ""))
        print()

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ DSPy agent created")
    print(f"✅ MCP tools discovered: {len(agent.mcp_executor.available_tools)}")
    print(f"✅ Agent executed with {result.get('iterations', 0)} iteration(s)")
    print(f"✅ Tool calls made: {len(result.get('tool_calls', []))}")
    print()


async def test_dspy_mcp_with_memory():
    """Test DSPy agent with memory retrieval"""
    print("=" * 80)
    print("TEST: DSPy + Memory Integration")
    print("=" * 80)
    print()

    # Configure DSPy
    lm = ClaudeCLILM(model="sonnet")
    dspy.configure(lm=lm)

    # Create agent
    agent = DSPyMCPAgent(
        name="Memory-Enhanced Assistant",
        description="Uses Jotty memory for context-aware responses"
    )
    await agent.initialize()

    # Test query that should use memory
    result = await agent.execute(
        query="What ideas have I created recently about AI and ML?",
        conversation_history="User previously asked about machine learning tutorials"
    )

    print("Memory-Enhanced Response:")
    print(result.get("response", ""))
    print()


if __name__ == "__main__":
    print("Testing DSPy + MCP Tools Integration (Phase 1)")
    print()

    # Run basic test
    asyncio.run(test_dspy_mcp_basic())

    # Uncomment to test memory integration (Phase 2)
    # asyncio.run(test_dspy_mcp_with_memory())
