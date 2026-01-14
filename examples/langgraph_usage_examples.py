"""
LangGraph Usage Examples
========================

Demonstrates consistent API for both dynamic and static modes.
"""

import asyncio
import dspy
from Jotty import Conductor, AgentConfig, JottyConfig

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4", api_key="your-key"))


# Example 1: Dynamic Mode (Jotty's Dependency Graph)
# ===================================================

async def example_dynamic_mode():
    """Use Jotty's DynamicDependencyGraph with LangGraph orchestration."""
    
    # Define agents
    class ResearchAgent(dspy.Module):
        def forward(self, query: str) -> str:
            return f"Research results for: {query}"
    
    class AnalyzeAgent(dspy.Module):
        def forward(self, research_data: str) -> str:
            return f"Analysis of: {research_data}"
    
    agents = [
        AgentConfig(
            name="ResearchAgent",
            agent=ResearchAgent(),
            dependencies=["AnalyzeAgent"],  # AnalyzeAgent depends on ResearchAgent
        ),
        AgentConfig(
            name="AnalyzeAgent",
            agent=AnalyzeAgent(),
        ),
    ]
    
    # Create conductor with dynamic LangGraph mode
    conductor = Conductor(
        actors=agents,
        metadata_provider=None,
        config=JottyConfig(),
        use_langgraph=True,
        langgraph_mode="dynamic"  # Use Jotty's DynamicDependencyGraph
    )
    
    # Run - LangGraph orchestrates based on dependencies
    result = await conductor.run(
        goal="Research and analyze AI trends",
        query="What are the latest AI trends?"
    )
    
    print(f"Result: {result.final_output}")


# Example 2: Static Mode (Explicit Agent Order)
# =============================================

async def example_static_mode():
    """Use static LangGraph with explicit agent order."""
    
    # Define agents (same as before)
    class ResearchAgent(dspy.Module):
        def forward(self, query: str) -> str:
            return f"Research results for: {query}"
    
    class AnalyzeAgent(dspy.Module):
        def forward(self, research_data: str) -> str:
            return f"Analysis of: {research_data}"
    
    class ReportAgent(dspy.Module):
        def forward(self, analysis: str) -> str:
            return f"Report: {analysis}"
    
    agents = [
        AgentConfig(name="ResearchAgent", agent=ResearchAgent()),
        AgentConfig(name="AnalyzeAgent", agent=AnalyzeAgent()),
        AgentConfig(name="ReportAgent", agent=ReportAgent()),
    ]
    
    # Create conductor with static LangGraph mode
    conductor = Conductor(
        actors=agents,
        metadata_provider=None,
        config=JottyConfig(),
        use_langgraph=True,
        langgraph_mode="static",
        agent_order=["ResearchAgent", "AnalyzeAgent", "ReportAgent"]  # Explicit order
    )
    
    # Run - LangGraph executes in exact order specified
    result = await conductor.run(
        goal="Research, analyze, and report on AI trends",
        query="What are the latest AI trends?"
    )
    
    print(f"Result: {result.final_output}")


# Example 3: Runtime Mode Override
# ================================

async def example_runtime_override():
    """Override mode at runtime."""
    
    agents = [
        AgentConfig(name="Agent1", agent=...),
        AgentConfig(name="Agent2", agent=...),
        AgentConfig(name="Agent3", agent=...),
    ]
    
    conductor = Conductor(
        actors=agents,
        metadata_provider=None,
        config=JottyConfig(),
        use_langgraph=True,
        langgraph_mode="dynamic"  # Default to dynamic
    )
    
    # Run with dynamic mode (default)
    result1 = await conductor.run(goal="Task 1")
    
    # Override to static mode at runtime
    result2 = await conductor.run(
        goal="Task 2",
        mode="static",
        agent_order=["Agent1", "Agent2", "Agent3"]  # Explicit order
    )
    
    # Back to dynamic mode
    result3 = await conductor.run(goal="Task 3")


# Example 4: Streaming Support
# ============================

async def example_streaming():
    """Use streaming with LangGraph."""
    
    agents = [
        AgentConfig(name="ResearchAgent", agent=...),
        AgentConfig(name="AnalyzeAgent", agent=...),
    ]
    
    conductor = Conductor(
        actors=agents,
        metadata_provider=None,
        config=JottyConfig(),
        use_langgraph=True,
        langgraph_mode="static",
        agent_order=["ResearchAgent", "AnalyzeAgent"]
    )
    
    # Stream execution events
    async for event in conductor.langgraph_orchestrator.run_stream(
        goal="Research and analyze",
        context={"query": "AI trends"}
    ):
        print(f"Event: {event}")


# Example 5: Consistent API Comparison
# ====================================

async def example_consistent_api():
    """
    Demonstrates consistent API for both modes.
    
    Same interface, just different parameters:
    - Dynamic: mode="dynamic" (uses dependencies)
    - Static: mode="static", agent_order=[...] (explicit order)
    """
    
    agents = [
        AgentConfig(name="Agent1", agent=...),
        AgentConfig(name="Agent2", agent=...),
        AgentConfig(name="Agent3", agent=...),
    ]
    
    # Dynamic mode - simple
    conductor_dynamic = Conductor(
        actors=agents,
        metadata_provider=None,
        config=JottyConfig(),
        use_langgraph=True,
        langgraph_mode="dynamic"  # Just specify mode
    )
    
    result_dynamic = await conductor_dynamic.run(goal="Task")
    
    # Static mode - specify order
    conductor_static = Conductor(
        actors=agents,
        metadata_provider=None,
        config=JottyConfig(),
        use_langgraph=True,
        langgraph_mode="static",
        agent_order=["Agent1", "Agent2", "Agent3"]  # Specify order
    )
    
    result_static = await conductor_static.run(goal="Task")
    
    # Both use same API!
    print(f"Dynamic result: {result_dynamic.final_output}")
    print(f"Static result: {result_static.final_output}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_dynamic_mode())
    asyncio.run(example_static_mode())
    asyncio.run(example_consistent_api())
