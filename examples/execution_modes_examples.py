"""
Jotty Execution Modes Examples
================================

Demonstrates WorkflowMode and ChatMode usage.
"""

import asyncio
from Jotty import Conductor, AgentConfig
from Jotty.core.orchestration import (
    WorkflowMode,
    ChatMode,
    ChatMessage,
    create_workflow,
    create_chat
)
import dspy


# ============================================
# Setup: Create Conductor with Agents
# ============================================

async def setup_conductor():
    """Create conductor with sample agents."""

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4', api_key='your-key')
    dspy.configure(lm=lm)

    # Create agents
    research_agent = AgentConfig(
        name="Research",
        agent=dspy.ChainOfThought("question -> answer"),
        description="Research specialist"
    )

    writer_agent = AgentConfig(
        name="Writer",
        agent=dspy.ChainOfThought("topic -> article"),
        description="Content writer"
    )

    editor_agent = AgentConfig(
        name="Editor",
        agent=dspy.ChainOfThought("content -> improved_content"),
        description="Content editor"
    )

    # Create conductor
    conductor = Conductor(
        actors=[research_agent, writer_agent, editor_agent],
        config={
            "memory": {"enabled": True},
            "learning": {"enabled": True}
        }
    )

    return conductor


# ============================================
# Example 1: WorkflowMode - Dynamic Routing
# ============================================

async def example_workflow_dynamic():
    """
    Dynamic workflow: Agents adaptively collaborate.
    Use case: Complex task requiring intelligent routing.
    """
    print("\n" + "="*50)
    print("Example 1: WorkflowMode - Dynamic Routing")
    print("="*50)

    conductor = await setup_conductor()

    # Create workflow with dynamic mode
    workflow = create_workflow(conductor, mode="dynamic")

    # Execute workflow
    result = await workflow.execute(
        goal="Write article about AI safety",
        context={
            "target_audience": "technical",
            "word_count": 1000
        }
    )

    print(f"\n✅ Workflow Complete!")
    print(f"Success: {result['success']}")
    print(f"Output: {result['final_output'][:200]}...")
    print(f"Agents Used: {list(result['actor_outputs'].keys())}")


# ============================================
# Example 2: WorkflowMode - Static Pipeline
# ============================================

async def example_workflow_static():
    """
    Static workflow: Predefined agent order.
    Use case: ETL pipeline, report generation.
    """
    print("\n" + "="*50)
    print("Example 2: WorkflowMode - Static Pipeline")
    print("="*50)

    conductor = await setup_conductor()

    # Create workflow with static order
    workflow = create_workflow(
        conductor,
        mode="static",
        agent_order=["Research", "Writer", "Editor"]  # Explicit order
    )

    # Execute workflow
    result = await workflow.execute(
        goal="Create quarterly report",
        context={"quarter": "Q4", "year": 2026}
    )

    print(f"\n✅ Workflow Complete!")
    print(f"Agents executed in order: Research → Writer → Editor")
    print(f"Output: {result['final_output'][:200]}...")


# ============================================
# Example 3: WorkflowMode - Streaming
# ============================================

async def example_workflow_streaming():
    """
    Streaming workflow: Real-time progress updates.
    Use case: Long-running tasks with UI progress.
    """
    print("\n" + "="*50)
    print("Example 3: WorkflowMode - Streaming")
    print("="*50)

    conductor = await setup_conductor()
    workflow = create_workflow(conductor, mode="dynamic")

    # Stream workflow execution
    print("\n📊 Streaming workflow events:\n")

    async for event in workflow.execute_stream(
        goal="Analyze market trends",
        context={"market": "tech", "timeframe": "2026-Q1"}
    ):
        event_type = event.get("type")

        if event_type == "agent_start":
            print(f"  🚀 Starting: {event['agent']}")

        elif event_type == "agent_complete":
            agent = event['agent']
            result = event['result']
            print(f"  ✅ Completed: {agent}")
            print(f"     Output preview: {str(result)[:80]}...")


# ============================================
# Example 4: ChatMode - Single Agent
# ============================================

async def example_chat_single_agent():
    """
    Single-agent chat: One expert conversation.
    Use case: Specialized chatbot (e.g., research assistant).
    """
    print("\n" + "="*50)
    print("Example 4: ChatMode - Single Agent")
    print("="*50)

    conductor = await setup_conductor()

    # Create single-agent chat
    chat = create_chat(conductor, agent_id="Research")

    # Conversation history
    history = [
        ChatMessage(role="user", content="What is reinforcement learning?"),
        ChatMessage(role="assistant", content="Reinforcement learning is...")
    ]

    # Stream response
    print("\n💬 User: Tell me about transformers in AI")
    print("🤖 Assistant: ", end="", flush=True)

    async for event in chat.stream(
        message="Tell me about transformers in AI",
        history=history
    ):
        if event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)

        elif event["type"] == "done":
            print("\n\n✅ Chat complete!")


# ============================================
# Example 5: ChatMode - Multi-Agent
# ============================================

async def example_chat_multi_agent():
    """
    Multi-agent chat: Collaborative response.
    Use case: Complex questions requiring multiple experts.
    """
    print("\n" + "="*50)
    print("Example 5: ChatMode - Multi-Agent")
    print("="*50)

    conductor = await setup_conductor()

    # Create multi-agent chat with dynamic routing
    chat = create_chat(conductor, mode="dynamic")

    # Stream response with multiple agents
    print("\n💬 User: Write a technical article about quantum computing")
    print("\n🤖 Multi-Agent Response:\n")

    async for event in chat.stream(
        message="Write a technical article about quantum computing"
    ):
        if event["type"] == "agent_selected":
            print(f"\n  🎯 Agent: {event['agent']}")

        elif event["type"] == "reasoning":
            print(f"  💭 Reasoning: {event['content'][:100]}...")

        elif event["type"] == "tool_call":
            print(f"  🔧 Tool Call: {event['tool']}")

        elif event["type"] == "tool_result":
            print(f"  📦 Tool Result: {str(event['result'])[:80]}...")

        elif event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)


# ============================================
# Example 6: ChatMode - With Tool Execution
# ============================================

async def example_chat_with_tools():
    """
    Chat with tool execution (via MCP).
    Use case: Chat that can perform actions (e.g., create ideas, search).
    """
    print("\n" + "="*50)
    print("Example 6: ChatMode - With Tool Execution")
    print("="*50)

    conductor = await setup_conductor()
    chat = create_chat(conductor, agent_id="Research")

    print("\n💬 User: Create a new idea about climate change")
    print("\n🤖 Assistant:\n")

    async for event in chat.stream(
        message="Create a new idea about climate change"
    ):
        if event["type"] == "reasoning":
            print(f"  💭 {event['content']}")

        elif event["type"] == "tool_call":
            tool = event['tool']
            args = event['args']
            print(f"\n  🔧 Calling tool: {tool}")
            print(f"     Args: {args}")

        elif event["type"] == "tool_result":
            result = event['result']
            print(f"  ✅ Tool result: {str(result)[:100]}...")

        elif event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)

        elif event["type"] == "done":
            print("\n\n✅ Chat complete!")


# ============================================
# Example 7: WorkflowMode vs ChatMode
# ============================================

async def example_comparison():
    """
    Direct comparison: Same task in both modes.
    """
    print("\n" + "="*50)
    print("Example 7: WorkflowMode vs ChatMode Comparison")
    print("="*50)

    conductor = await setup_conductor()

    task = "Generate report on AI trends"

    # ---- WorkflowMode ----
    print("\n📊 WorkflowMode (batch execution):")
    workflow = create_workflow(conductor, mode="dynamic")
    workflow_result = await workflow.execute(goal=task)
    print(f"  Output: {workflow_result['final_output'][:100]}...")

    # ---- ChatMode ----
    print("\n💬 ChatMode (conversational):")
    chat = create_chat(conductor, mode="dynamic")

    print("  Response: ", end="", flush=True)
    async for event in chat.stream(message=task):
        if event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)

    print("\n\n📝 Key Differences:")
    print("  - WorkflowMode: Returns complete result at end")
    print("  - ChatMode: Streams progressive response")


# ============================================
# Run All Examples
# ============================================

async def run_all_examples():
    """Run all examples."""

    await example_workflow_dynamic()
    await example_workflow_static()
    await example_workflow_streaming()
    await example_chat_single_agent()
    await example_chat_multi_agent()
    await example_chat_with_tools()
    await example_comparison()


if __name__ == "__main__":
    print("="*50)
    print("Jotty Execution Modes Examples")
    print("="*50)

    asyncio.run(run_all_examples())

    print("\n" + "="*50)
    print("All examples completed!")
    print("="*50)
