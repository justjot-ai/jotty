"""
Jotty V3 Quick Start Examples
==============================

Shows all 4 tiers and convenience methods.
"""

import asyncio
from Jotty import Jotty, ExecutionTier


async def main():
    print("=" * 60)
    print("JOTTY V3 - QUICK START")
    print("=" * 60)

    jotty = Jotty(log_level="INFO")

    # =========================================================================
    # Example 1: Auto-Detection (Recommended)
    # =========================================================================
    print("\n1. AUTO-DETECTION")
    print("-" * 60)

    # Simple query → Tier 1 (DIRECT)
    result = await jotty.run("What is 2+2?")
    print(f"Result: {result.output}")
    print(f"Tier: {result.tier.name} | LLM calls: {result.llm_calls}")

    # Multi-step → Tier 2 (AGENTIC)
    result = await jotty.run("Search for Python tutorials and summarize")
    print(f"\nResult: {result.output[:100]}...")
    print(f"Tier: {result.tier.name} | LLM calls: {result.llm_calls}")

    # =========================================================================
    # Example 2: Explicit Tiers
    # =========================================================================
    print("\n2. EXPLICIT TIERS")
    print("-" * 60)

    # Force Tier 1 (DIRECT)
    result = await jotty.run(
        "Calculate 15 * 23",
        tier=ExecutionTier.DIRECT
    )
    print(f"Tier 1 (DIRECT): {result.output}")

    # Force Tier 2 (AGENTIC)
    result = await jotty.run(
        "Plan a Python learning path",
        tier=ExecutionTier.AGENTIC
    )
    print(f"\nTier 2 (AGENTIC): {len(result.steps)} steps planned")

    # =========================================================================
    # Example 3: Convenience Methods
    # =========================================================================
    print("\n3. CONVENIENCE METHODS")
    print("-" * 60)

    # Chat (Tier 1)
    response = await jotty.chat("What is the capital of France?")
    print(f"Chat: {response}")

    # Plan (Tier 2)
    result = await jotty.plan("Create a simple calculator")
    print(f"\nPlan: {len(result.steps)} steps")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step.description}")

    # Learn (Tier 3) - with memory
    print("\nLearn with memory:")
    result = await jotty.learn("Analyze sample data")
    print(f"  Success: {result.success}")
    print(f"  Used memory: {result.used_memory}")
    if result.validation:
        print(f"  Validated: {result.validation.success}")

    # =========================================================================
    # Example 4: Tier Explanation
    # =========================================================================
    print("\n4. TIER EXPLANATION")
    print("-" * 60)

    goals = [
        "What is 2+2?",
        "Research AI trends and create report",
        "Optimize this code with learning",
        "Experiment with different approaches",
    ]

    for goal in goals:
        explanation = jotty.explain_tier(goal)
        print(f"\nGoal: {goal}")
        print(explanation)

    # =========================================================================
    # Example 5: V2 Compatibility (No Breakage!)
    # =========================================================================
    print("\n5. V2 COMPATIBILITY")
    print("-" * 60)

    # V2 still works via Tier 4 (RESEARCH)
    result = await jotty.research(
        "Complex task requiring full V2 features",
        enable_td_lambda=True,
        enable_hierarchical_memory=True,
    )
    print(f"V2 Tier 4 (RESEARCH): {result.tier.name}")
    print(f"V2 episode included: {result.v2_episode is not None}")

    # =========================================================================
    # Example 6: Progress Callback
    # =========================================================================
    print("\n6. PROGRESS CALLBACK")
    print("-" * 60)

    def progress_callback(stage: str, detail: str):
        print(f"  [{stage.upper()}] {detail}")

    result = await jotty.run(
        "Research Python best practices",
        tier=ExecutionTier.AGENTIC,
        status_callback=progress_callback
    )

    print(f"\nCompleted with {result.llm_calls} LLM calls")

    # =========================================================================
    # Stats
    # =========================================================================
    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)

    stats = jotty.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
