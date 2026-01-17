#!/usr/bin/env python3
"""
RL Learning Test with REAL Claude CLI (Working Version)
========================================================

Uses the working UnifiedLMProvider integration from original Jotty.
"""

import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.foundation.unified_lm_provider import UnifiedLMProvider
from core.orchestration import SingleAgentOrchestrator, MultiAgentsOrchestrator
from core.foundation import JottyConfig, AgentConfig
import dspy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Agent signatures
class FetchDataSignature(dspy.Signature):
    """Fetch sales data from database."""
    query: str = dspy.InputField(desc="What data to fetch")
    data: str = dspy.OutputField(desc="Fetched sales data in JSON format")


class ProcessDataSignature(dspy.Signature):
    """Process raw sales data into summary."""
    data: str = dspy.InputField(desc="Raw sales data")
    summary: str = dspy.OutputField(desc="Processed summary statistics")


class VisualizeDataSignature(dspy.Signature):
    """Create visualization description from summary."""
    summary: str = dspy.InputField(desc="Summary statistics")
    visualization: str = dspy.OutputField(desc="Description of chart/visualization")


async def test_rl_with_real_claude_cli():
    """Test RL with real Claude CLI using UnifiedLMProvider."""

    print("\n" + "="*80)
    print("🧪 RL LEARNING TEST - REAL CLAUDE CLI (Working Version)")
    print("="*80)

    # Configure DSPy with Claude CLI using UnifiedLMProvider
    print("\n🔧 Configuring DSPy with Claude CLI...")
    try:
        lm = UnifiedLMProvider.create_lm('claude-cli', model='sonnet')
        dspy.configure(lm=lm)
        print("✅ DSPy configured with Claude CLI (via UnifiedLMProvider)")
    except Exception as e:
        print(f"❌ Failed to configure Claude CLI: {e}")
        print("\nRequirements:")
        print("  1. Claude CLI installed: npm install -g @anthropic-ai/claude-code")
        print("  2. API key set: export ANTHROPIC_API_KEY=your_key")
        return

    # Create config with RL enabled
    config = JottyConfig(
        enable_rl=True,
        alpha=0.2,  # Higher learning rate
        gamma=0.95,
        lambda_trace=0.9,
        epsilon_start=0.3  # 30% exploration
    )

    print("\n📋 Configuration:")
    print(f"   RL Enabled: {config.enable_rl}")
    print(f"   Learning Rate (alpha): {config.alpha}")
    print(f"   Exploration (epsilon): {config.epsilon_start}")
    print(f"   Claude CLI: Using real LLM calls")

    # Create REAL agents with Claude CLI
    print("\n🔧 Creating agents with real Claude CLI LLM...")

    fetcher = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(FetchDataSignature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config
    )

    processor = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(ProcessDataSignature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config
    )

    visualizer = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(VisualizeDataSignature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config
    )

    # Define actors in WRONG ORDER (intentionally)
    actors_wrong_order = [
        AgentConfig(name="Visualizer", agent=visualizer, enable_architect=False, enable_auditor=False),
        AgentConfig(name="Fetcher", agent=fetcher, enable_architect=False, enable_auditor=False),
        AgentConfig(name="Processor", agent=processor, enable_architect=False, enable_auditor=False)
    ]

    print("\n📋 Agent Order (INTENTIONALLY WRONG):")
    print("   Initial order: Visualizer → Fetcher → Processor")
    print("   Expected correct: Fetcher → Processor → Visualizer")

    # Mock metadata provider
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

    print("\n🚀 Running 10 episodes with REAL Claude CLI LLM...")
    print("   This will make actual LLM calls and learn from results")
    print("="*80)

    episode_data = []

    for episode in range(1, 11):
        print(f"\n{'─'*80}")
        print(f"📊 EPISODE {episode}/10")
        print(f"{'─'*80}")

        try:
            result = await orchestrator.run(
                goal=f"Fetch sales data, process it, and create visualization (episode {episode})"
            )

            # Extract first agent selected
            first_agent = None
            if hasattr(orchestrator, 'todo') and orchestrator.todo:
                for task_id, task in orchestrator.todo.subtasks.items():
                    if task.status.name == 'COMPLETED' or task.status.name == 'FAILED':
                        first_agent = task.actor
                        break

            # Get Q-values
            q_values = {}
            if hasattr(orchestrator, 'q_learner') and orchestrator.q_learner:
                try:
                    if hasattr(orchestrator.q_learner, 'q_table'):
                        for key, value in orchestrator.q_learner.q_table.items():
                            try:
                                state, action = eval(key)
                                actor = action.get('actor', 'unknown')
                                if actor not in q_values or value > q_values[actor]:
                                    q_values[actor] = value
                            except:
                                pass
                except Exception as e:
                    logger.debug(f"Could not extract Q-values: {e}")

            episode_data.append({
                'episode': episode,
                'first_agent': first_agent,
                'success': result.success,
                'q_values': q_values
            })

            print(f"\n✓ Episode {episode} completed")
            print(f"  First agent: {first_agent}")
            print(f"  Success: {result.success}")
            if q_values:
                print(f"  Q-values: {q_values}")

        except Exception as e:
            print(f"⚠️  Episode {episode} error: {e}")
            import traceback
            traceback.print_exc()

    # Analyze results
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS")
    print(f"{'='*80}")

    print("\n🎯 First Agent Selected Per Episode:")
    for data in episode_data:
        print(f"   Episode {data['episode']}: {data['first_agent']}")

    print("\n📈 Q-Value Progression:")
    agents = ['Visualizer', 'Fetcher', 'Processor']
    for agent in agents:
        values = []
        for data in episode_data:
            if agent in data['q_values']:
                values.append(f"{data['q_values'][agent]:.3f}")
            else:
                values.append("N/A")
        print(f"   {agent}: {' → '.join(values)}")

    # Count agent selections
    first_agent_counts = {}
    for data in episode_data:
        agent = data['first_agent']
        if agent:
            first_agent_counts[agent] = first_agent_counts.get(agent, 0) + 1

    print("\n📊 Agent Selection Frequency:")
    for agent, count in first_agent_counts.items():
        pct = (count / len(episode_data)) * 100
        print(f"   {agent}: {count}/10 episodes ({pct:.0f}%)")

    # Check if learning occurred
    fetcher_count = first_agent_counts.get('Fetcher', 0)
    visualizer_count = first_agent_counts.get('Visualizer', 0)

    print(f"\n{'='*80}")
    print("🎓 LEARNING ANALYSIS")
    print(f"{'='*80}")

    if fetcher_count >= 6:
        print("\n✅ SUCCESS: RL learned to prefer Fetcher first!")
        print(f"   Fetcher selected {fetcher_count}/10 times (correct)")
    elif fetcher_count > visualizer_count:
        print("\n⚡ PARTIAL SUCCESS: RL is learning but needs more episodes")
        print(f"   Fetcher selected {fetcher_count}/10 times")
        print(f"   Visualizer selected {visualizer_count}/10 times")
    else:
        print("\n⚠️  Learning not evident yet - may need more episodes (50-100)")
        print(f"   Fetcher: {fetcher_count}/10")
        print(f"   Visualizer: {visualizer_count}/10")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    asyncio.run(test_rl_with_real_claude_cli())
