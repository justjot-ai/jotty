"""
Example 1: TD-Lambda Training

Demonstrates:
- Creating a TD-Lambda learner
- Recording state transitions with rewards
- Tracking learning progress
- Understanding eligibility traces
"""

import asyncio

from Jotty.core.intelligence.learning import get_td_lambda


async def main():
    # Get TD-Lambda learner (gamma=0.99, lambda=0.95)
    td = get_td_lambda()

    print("=== TD-Lambda Reinforcement Learning ===\n")
    print("Goal: Learn which actions lead to successful task completion\n")

    # Simulate a task execution episode
    print("Episode 1: Research task\n")

    # Step 1: Start with web search
    td.update(
        state={"task": "research", "step": 1},
        action={"tool": "web-search", "query": "AI trends 2026"},
        reward=0.5,  # Partial reward (found some info)
        next_state={"task": "research", "step": 2},
    )
    print("✅ Step 1: Web search (reward=0.5)")

    # Step 2: Scrape website
    td.update(
        state={"task": "research", "step": 2},
        action={"tool": "web-scraper", "url": "https://example.com"},
        reward=0.3,  # Partial reward
        next_state={"task": "research", "step": 3},
    )
    print("✅ Step 2: Web scraper (reward=0.3)")

    # Step 3: Analyze and summarize
    td.update(
        state={"task": "research", "step": 3},
        action={"tool": "summarizer"},
        reward=1.0,  # Full reward (task completed successfully!)
        next_state={"task": "research", "step": 4, "done": True},
    )
    print("✅ Step 3: Summarize (reward=1.0 - SUCCESS!)\n")

    print("=== What TD-Lambda Learned ===\n")
    print("Eligibility traces (how much credit each step gets):")
    print("- Step 1 (web-search): High trace (started the success)")
    print("- Step 2 (web-scraper): Medium trace (contributed)")
    print("- Step 3 (summarizer): Highest trace (completed task)")
    print()
    print("Value estimates updated for:")
    print("- State-action pair: (research, web-search) → higher value")
    print("- State-action pair: (research, web-scraper) → higher value")
    print("- State-action pair: (research, summarizer) → highest value")
    print()

    print("=== Next Episode Benefits ===\n")
    print("When faced with a similar research task:")
    print("1. TD-Lambda will favor web-search as first action (learned it works)")
    print("2. Will prefer this sequence: search → scrape → summarize")
    print("3. Faster decision-making (exploits learned value estimates)")

    print("\n✅ Example complete!")
    print("\nKey Concept: TD-Lambda learns from DELAYED rewards")
    print("Even though final reward came at step 3, earlier steps get credit too!")


if __name__ == "__main__":
    asyncio.run(main())
