#!/usr/bin/env python3
"""
Cost-Aware Learning Example
============================

Shows how to use cost-aware learning to train agents to prefer cheaper strategies.
"""

import asyncio

async def main():
    from Jotty.core.learning import get_cost_aware_td_lambda
    from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator, MergeStrategy

    print("\n" + "="*70)
    print("EXAMPLE: Cost-Aware Learning")
    print("="*70 + "\n")

    # Create cost-aware learner
    learner = get_cost_aware_td_lambda(cost_sensitivity=0.5)

    print("Cost-Aware Learner initialized")
    print(f"  Cost sensitivity: {learner.cost_sensitivity}")
    print(f"  Formula: adjusted_reward = reward - (cost / {learner.cost_sensitivity})")
    print()

    # Simulate different strategies
    strategies = [
        ("cheap_fast", 2, 0.05),   # 2 swarms, $0.05 total
        ("expensive_slow", 5, 0.25),  # 5 swarms, $0.25 total
        ("balanced", 3, 0.10),     # 3 swarms, $0.10 total
    ]

    print("Simulating 3 strategies:\n")

    for strategy_name, num_swarms, cost in strategies:
        # Simulate task success
        reward = 1.0  # All succeed

        # Update learner
        learner.update(
            state={"strategy": strategy_name},
            action={"num_swarms": num_swarms},
            reward=reward,
            next_state={"strategy": "complete"},
            cost_usd=cost
        )

        # Calculate adjusted reward
        cost_penalty = cost / learner.cost_sensitivity
        adjusted_reward = reward - cost_penalty

        print(f"{strategy_name}:")
        print(f"  Swarms: {num_swarms}, Cost: ${cost:.2f}")
        print(f"  Task reward: {reward:.2f}")
        print(f"  Cost penalty: -{cost_penalty:.2f}")
        print(f"  Adjusted reward: {adjusted_reward:.2f}")
        print(f"  → {'✅ Good' if adjusted_reward > 0 else '❌ Too expensive'}")
        print()

    # Show learning stats
    print("="*70)
    print("Learning Statistics\n")

    stats = learner.get_stats()
    print(f"Total updates: {stats['updates']}")
    print(f"Cost saved: ${stats['total_cost_saved_usd']:.4f}")
    print(f"Avg cost saved/update: ${stats['avg_cost_saved_per_update']:.4f}")
    print()

    print("="*70)
    print("✅ Example complete!")
    print("="*70 + "\n")

    print("Key Insights:")
    print("  1. System learns to prefer cheaper strategies")
    print("  2. expensive_slow gets negative reward (1.0 - 0.50 = 0.50)")
    print("  3. cheap_fast gets high reward (1.0 - 0.10 = 0.90)")
    print("  4. Agent will learn to avoid expensive strategies")
    print()


if __name__ == '__main__':
    asyncio.run(main())
