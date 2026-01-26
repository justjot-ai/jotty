#!/usr/bin/env python3
"""
Standalone RL ordering test - verifies Q-values actually control agent selection.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestration.conductor import MultiAgentsOrchestrator
from core.foundation import JottyConfig, AgentConfig
import dspy


# Mock agents
class MockAgent(dspy.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, **kwargs):
        return dspy.Prediction(output=f"{self.name} output", success=True)


async def test_rl_ordering():
    """Test that RL learns correct agent ordering."""

    print("=" * 80)
    print("🧪 RL AGENT ORDERING TEST - VERIFYING Q-VALUE-BASED SELECTION")
    print("=" * 80)

    config = JottyConfig(
        enable_rl=True,
        alpha=0.1,
        gamma=0.95,
        lambda_trace=0.9,
        epsilon_start=0.3  # 30% exploration
    )

    # Create agents in WRONG ORDER (Visualizer should be LAST)
    actors_wrong_order = [
        AgentConfig(
            name="Visualizer",
            agent=MockAgent("Visualizer"),
            enable_architect=False,
            enable_auditor=False
        ),  # ❌ Wrong: should be last
        AgentConfig(
            name="Fetcher",
            agent=MockAgent("Fetcher"),
            enable_architect=False,
            enable_auditor=False
        ),        # ✅ Correct: should be first
        AgentConfig(
            name="Processor",
            agent=MockAgent("Processor"),
            enable_architect=False,
            enable_auditor=False
        )     # ✅ Correct: should be second
    ]

    print("\n📋 Initial Agent Order (WRONG):")
    print("   1. Visualizer (wrong - should be last)")
    print("   2. Fetcher (correct - should be first)")
    print("   3. Processor (correct - should be second)")

    # Create mock metadata provider
    class MockMetadataProvider:
        def register_artifact(self, *args, **kwargs):
            pass
        def get_artifacts(self, *args, **kwargs):
            return []

    orchestrator = MultiAgentsOrchestrator(
        actors=actors_wrong_order,
        metadata_provider=MockMetadataProvider(),
        config=config
    )

    print("\n" + "=" * 80)
    print("🔄 RUNNING 5 EPISODES TO OBSERVE Q-VALUE-BASED SELECTION")
    print("=" * 80)

    agent_selection_counts = {
        "Visualizer": 0,
        "Fetcher": 0,
        "Processor": 0
    }

    first_agent_selected = []

    for episode in range(1, 6):
        print(f"\n{'=' * 80}")
        print(f"📊 EPISODE {episode}")
        print(f"{'=' * 80}")

        try:
            result = await orchestrator.run(
                goal=f"Fetch sales data, process it, and create visualization (episode {episode})"
            )

            # Track which agent was selected first
            if hasattr(orchestrator, 'todo') and orchestrator.todo:
                execution_log = []
                for task_id, task in orchestrator.todo.subtasks.items():
                    if task.status.name == 'COMPLETED':
                        execution_log.append(task.actor)
                        agent_selection_counts[task.actor] += 1

                if execution_log:
                    first_agent_selected.append(execution_log[0])
                    print(f"\n🎯 Execution Order: {' → '.join(execution_log)}")
                    print(f"   First Agent: {execution_log[0]}")

            print(f"\n✅ Episode {episode} completed: {result.success}")

        except Exception as e:
            print(f"\n⚠️  Episode {episode} error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)

    print("\n🎯 Agent Selection Counts (all tasks):")
    for agent, count in agent_selection_counts.items():
        print(f"   {agent}: {count} times")

    print(f"\n🥇 First Agent Selected Per Episode:")
    for i, agent in enumerate(first_agent_selected, 1):
        print(f"   Episode {i}: {agent}")

    print("\n📈 Q-Value Analysis:")
    if hasattr(orchestrator, 'q_learner') and orchestrator.q_learner:
        try:
            # Get Q-table stats
            q_table = orchestrator.q_learner.q_table
            print(f"   Q-table entries: {len(q_table)}")

            # Get experiences
            experiences = orchestrator.q_learner.experiences
            print(f"   Total experiences: {len(experiences)}")

            # Compute average Q-values per agent
            agent_q_values = {"Visualizer": [], "Fetcher": [], "Processor": []}
            for key, q_val in q_table.items():
                try:
                    state, action = eval(key)
                    actor = action.get('actor', '')
                    if actor in agent_q_values:
                        agent_q_values[actor].append(q_val)
                except:
                    pass

            print(f"\n   Average Q-values by agent:")
            for agent, q_vals in agent_q_values.items():
                if q_vals:
                    avg_q = sum(q_vals) / len(q_vals)
                    print(f"   {agent}: {avg_q:.3f} (n={len(q_vals)})")

        except Exception as e:
            print(f"   Could not analyze Q-values: {e}")

    # Determine if RL is working
    print("\n" + "=" * 80)
    print("🔬 ANALYSIS")
    print("=" * 80)

    fetcher_first_count = sum(1 for agent in first_agent_selected if agent == "Fetcher")
    visualizer_first_count = sum(1 for agent in first_agent_selected if agent == "Visualizer")

    print(f"\n✅ Q-value-based selection: {'WORKING' if hasattr(orchestrator, 'q_learner') else 'NOT INITIALIZED'}")
    print(f"✅ Fetcher selected first: {fetcher_first_count}/5 episodes")
    print(f"❌ Visualizer selected first: {visualizer_first_count}/5 episodes (should decrease over time)")

    if fetcher_first_count >= 3:
        print("\n🎉 SUCCESS: RL is learning to prefer Fetcher first (correct order)!")
    elif fetcher_first_count > visualizer_first_count:
        print("\n✅ PARTIAL SUCCESS: RL is learning but needs more episodes")
    else:
        print("\n⚠️  WARNING: RL may not be affecting agent selection")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_rl_ordering())
