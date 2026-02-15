#!/usr/bin/env python3
"""
Multi-Swarm Coordination - Basic Example
=========================================

Shows how to execute multiple swarms in parallel with different merge strategies.

Zero wrapper code needed - just use SwarmAdapter!
"""

import asyncio
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

async def main():
    from Jotty.core.intelligence.orchestration import (
        SwarmAdapter,
        get_multi_swarm_coordinator,
        MergeStrategy
    )

    print("\n" + "="*70)
    print("EXAMPLE: Multi-Swarm Coordination")
    print("="*70 + "\n")

    # Create coordinator
    coordinator = get_multi_swarm_coordinator()

    # =========================================================================
    # EXAMPLE 1: Voting Strategy (Consensus)
    # =========================================================================

    print("Example 1: Voting Strategy (2/3 Consensus)\n")

    # Create 3 swarms with ONE line
    swarms = SwarmAdapter.quick_swarms([
        ("Optimist", "You're optimistic about AI. Answer in 1 sentence."),
        ("Optimist2", "You're optimistic about AI. Answer in 1 sentence."),
        ("Pessimist", "You're pessimistic about AI. Answer in 1 sentence."),
    ])

    task = "Will AI agents be widely adopted in 2026?"

    result = await coordinator.execute_parallel(
        swarms=swarms,
        task=task,
        merge_strategy=MergeStrategy.VOTING  # Majority wins
    )

    print(f"Task: {task}")
    print(f"Result: {result.output}")
    print(f"Confidence: {result.confidence:.0%} (2/3 voted for this)")
    print()

    # =========================================================================
    # EXAMPLE 2: Concatenation Strategy (Show All Perspectives)
    # =========================================================================

    print("="*70)
    print("Example 2: Concatenation Strategy (All Perspectives)\n")

    swarms = SwarmAdapter.quick_swarms([
        ("Technical", "You're a technical expert. 1 sentence."),
        ("Business", "You're a business expert. 1 sentence."),
        ("Ethics", "You're an ethics expert. 1 sentence."),
    ])

    task = "What's the most important consideration for AI agents?"

    result = await coordinator.execute_parallel(
        swarms=swarms,
        task=task,
        merge_strategy=MergeStrategy.CONCATENATE  # Show all
    )

    print(f"Task: {task}\n")
    print("Result (all perspectives):")
    print(result.output)
    print()

    # =========================================================================
    # EXAMPLE 3: Best-of-N Strategy (Highest Confidence)
    # =========================================================================

    print("="*70)
    print("Example 3: Best-of-N Strategy (Highest Confidence)\n")

    swarms = SwarmAdapter.quick_swarms([
        ("Expert1", "You're an AI expert. Be confident. 1 sentence."),
        ("Expert2", "You're an AI expert. Be confident. 1 sentence."),
        ("Expert3", "You're an AI expert. Be confident. 1 sentence."),
    ])

    task = "What will AI agents cost in 2026?"

    result = await coordinator.execute_parallel(
        swarms=swarms,
        task=task,
        merge_strategy=MergeStrategy.BEST_OF_N  # Highest confidence wins
    )

    print(f"Task: {task}")
    print(f"Result: {result.output}")
    print(f"Confidence: {result.confidence:.0%}")
    print()

    # =========================================================================
    # STATS
    # =========================================================================

    print("="*70)
    print("Coordinator Statistics\n")

    stats = coordinator.get_stats()
    print(f"Total executions: {stats['total_executions']}")
    print(f"Merge strategies used: {stats['merge_strategy_usage']}")
    print()

    print("="*70)
    print("✅ Examples complete!")
    print("="*70 + "\n")

    print("Key Takeaways:")
    print("  1. SwarmAdapter.quick_swarms() - Create swarms in 1 line")
    print("  2. coordinator.execute_parallel() - Execute in parallel")
    print("  3. MergeStrategy - Choose how to combine results")
    print("     - VOTING: Majority consensus")
    print("     - CONCATENATE: All perspectives")
    print("     - BEST_OF_N: Highest confidence")
    print("     - ENSEMBLE: Weighted average (numeric)")
    print("     - FIRST_SUCCESS: First successful result")
    print()


if __name__ == '__main__':
    asyncio.run(main())
