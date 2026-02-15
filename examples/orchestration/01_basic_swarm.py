"""
Example 1: Basic Swarm Creation

Demonstrates:
- Creating a swarm from natural language
- Zero-config agent creation
- Task execution with coordination
- Understanding swarm results
"""
import asyncio
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

async def main():
    import dspy
    from Jotty.core.infrastructure.foundation.direct_anthropic_lm import DirectAnthropicLM
    from Jotty.core.intelligence.orchestration import Orchestrator

    # Setup LLM
    lm = DirectAnthropicLM(model="haiku")
    dspy.configure(lm=lm)

    print("=== Creating Your First Swarm ===\n")

    # Zero-config swarm: Just describe what you need
    swarm = Orchestrator(
        agents="Researcher + Analyst + Writer"
    )

    print("✅ Created swarm with 3 agents:")
    print("  1. Researcher: Gathers information")
    print("  2. Analyst: Analyzes data and finds insights")
    print("  3. Writer: Creates well-structured output")
    print()

    # Execute a task
    print("=== Executing Task ===\n")
    print("Task: Research and analyze top AI trends")
    print("Status: Running...")
    print()

    result = await swarm.run(
        goal="Research the top 3 AI trends in 2026 and create a brief summary"
    )

    print("=== Result ===\n")
    print(f"Success: {result.success}")
    print(f"Duration: {result.latency_ms / 1000:.2f}s")
    print(f"LLM calls: {result.llm_calls}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print()

    if result.success:
        print("Output:")
        print("-" * 60)
        print(result.output)
        print("-" * 60)
    else:
        print(f"Error: {result.error}")

    print("\n=== What Happened Behind the Scenes ===\n")
    print("1. Task Router analyzed the goal")
    print("2. Selected best agent(s) for the task")
    print("3. Agents coordinated using swarm intelligence:")
    print("   - Researcher gathered information")
    print("   - Analyst processed and analyzed")
    print("   - Writer created final summary")
    print("4. Result returned with metadata (cost, timing, etc.)")

    print("\n=== Learning for Future Tasks ===\n")
    print("The swarm learned from this execution:")
    print("- Which agents performed well")
    print("- Estimated time and cost for similar tasks")
    print("- Best coordination pattern for research tasks")
    print()
    print("Next time you run a similar task:")
    print("→ Routing will be FASTER (RL-based selection)")
    print("→ Agent selection will be SMARTER (learned preferences)")
    print("→ Coordination will be BETTER (learned patterns)")

    print("\n✅ Example complete!")

    print("\n💡 Key Concepts:")
    print("- Swarms > Single agents for complex tasks")
    print("- Zero-config: Describe agents, don't configure them")
    print("- Learning: Each run improves future performance")
    print("- Intelligence: Stored in ~/jotty/intelligence/")


if __name__ == "__main__":
    asyncio.run(main())
