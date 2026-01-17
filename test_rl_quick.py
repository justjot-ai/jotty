#!/usr/bin/env python3
"""Quick RL test with logging to see what's happening."""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.foundation.unified_lm_provider import UnifiedLMProvider
from core.orchestration import SingleAgentOrchestrator, MultiAgentsOrchestrator
from core.foundation import JottyConfig, AgentConfig
import dspy

# Track first agent globally
FIRST_AGENT_PER_EPISODE = []

# Signatures
class FetcherSignature(dspy.Signature):
    """Fetch sales data."""
    sales_data: str = dspy.OutputField(desc="Raw sales data")
    success: bool = dspy.OutputField(desc="Success")

class ProcessorSignature(dspy.Signature):
    """Process sales data."""
    sales_data: str = dspy.InputField(desc="Raw data")
    summary: str = dspy.OutputField(desc="Summary")
    success: bool = dspy.OutputField(desc="Success")

# Agents
class FetcherAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = FetcherSignature

    def forward(self, **kwargs) -> dspy.Prediction:
        print(f"🔵 FETCHER")
        return dspy.Prediction(
            sales_data='{"sales": 1000}',
            success=True,
            _reasoning="Fetched data"
        )

class ProcessorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = ProcessorSignature

    def forward(self, sales_data: str = '', **kwargs) -> dspy.Prediction:
        has_data = bool(sales_data)
        status = "✅ HAS DATA" if has_data else "❌ NO DATA"
        print(f"🟢 PROCESSOR ({status})")

        if not sales_data:
            return dspy.Prediction(
                summary='',
                success=False,
                _reasoning="No sales_data"
            )

        return dspy.Prediction(
            summary="Summary of sales",
            success=True,
            _reasoning="Processed"
        )

async def main():
    print("🧪 RL Learning Test - 30 Episodes\n")

    lm = UnifiedLMProvider.create_lm('claude-cli', model='sonnet')
    dspy.configure(lm=lm)

    config = JottyConfig(
        enable_rl=True,
        alpha=0.3,
        epsilon_start=0.3,
        allow_partial_execution=True,
        max_validation_retries=0,  # 🔥 CRITICAL: Disable retries for natural dependency learning!
        q_value_mode="simple"  # "simple" (average reward) or "llm" (semantic prediction)
    )

    # Create configs
    processor_config = AgentConfig(name="Processor", agent=None, enable_architect=False, enable_auditor=False)
    fetcher_config = AgentConfig(name="Fetcher", agent=None, enable_architect=False, enable_auditor=False)

    # Wrap agents
    processor = SingleAgentOrchestrator(
        agent=ProcessorAgent(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=processor_config
    )

    fetcher = SingleAgentOrchestrator(
        agent=FetcherAgent(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=fetcher_config
    )

    processor_config.agent = processor
    fetcher_config.agent = fetcher

    # WRONG order (Processor before Fetcher)
    actors = [processor_config, fetcher_config]

    print("📋 Order: Processor → Fetcher (WRONG!)")
    print("   Expected: Processor fails (no data), Fetcher succeeds\n")

    class MockProvider:
        def register_artifact(self, *args, **kwargs): pass
        def get_artifacts(self, *args, **kwargs): return []

    orchestrator = MultiAgentsOrchestrator(
        actors=actors,
        metadata_provider=MockProvider(),
        config=config
    )

    success_count = 0
    fetcher_first_count = 0
    episode_results = []

    for i in range(1, 11):  # 10 episodes for quick test
        print(f"\n====== EPISODE {i} ======")
        result = await orchestrator.run(
            goal=f"Process sales (episode {i})",
            max_iterations=2  # 🔥 CRITICAL: 2 agents = 2 iterations max (one per agent), NO RETRIES!
        )

        # Track which agent ran first
        first_agent = None
        if hasattr(orchestrator, 'trajectory') and orchestrator.trajectory:
            first_step = orchestrator.trajectory[0]
            first_agent = first_step.get('actor', 'Unknown')

        if result.success:
            success_count += 1

        if first_agent == 'Fetcher':
            fetcher_first_count += 1

        episode_results.append({
            'episode': i,
            'first_agent': first_agent,
            'success': result.success
        })

        # Print progress every 5 episodes
        if i % 5 == 0:
            recent_success = sum(1 for e in episode_results[-5:] if e['success'])
            recent_fetcher_first = sum(1 for e in episode_results[-5:] if e['first_agent'] == 'Fetcher')
            print(f"Episode {i:2d}: Success rate (last 5) = {recent_success}/5, Fetcher-first = {recent_fetcher_first}/5")

    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS (30 episodes)")
    print('='*60)
    print(f"Overall success rate: {success_count}/30 = {success_count/30*100:.1f}%")
    print(f"Fetcher selected first: {fetcher_first_count}/30 = {fetcher_first_count/30*100:.1f}%")
    print()

    # Analyze learning trend
    early_success = sum(1 for e in episode_results[:10] if e['success'])
    mid_success = sum(1 for e in episode_results[10:20] if e['success'])
    late_success = sum(1 for e in episode_results[20:30] if e['success'])

    early_fetcher = sum(1 for e in episode_results[:10] if e['first_agent'] == 'Fetcher')
    mid_fetcher = sum(1 for e in episode_results[10:20] if e['first_agent'] == 'Fetcher')
    late_fetcher = sum(1 for e in episode_results[20:30] if e['first_agent'] == 'Fetcher')

    print(f"📈 LEARNING TREND:")
    print(f"   Episodes  1-10: {early_success}/10 success ({early_success*10}%), {early_fetcher}/10 Fetcher-first ({early_fetcher*10}%)")
    print(f"   Episodes 11-20: {mid_success}/10 success ({mid_success*10}%), {mid_fetcher}/10 Fetcher-first ({mid_fetcher*10}%)")
    print(f"   Episodes 21-30: {late_success}/10 success ({late_success*10}%), {late_fetcher}/10 Fetcher-first ({late_fetcher*10}%)")
    print()

    if late_fetcher >= 7:
        print("✅ SUCCESS: RL learned optimal order! (Fetcher-first ≥70% in late episodes)")
    elif late_fetcher > early_fetcher:
        print("⚡ LEARNING IN PROGRESS: Fetcher-first selection increasing")
    else:
        print("⚠️  Need more episodes or lower epsilon for convergence")
    print('='*60)

if __name__ == "__main__":
    asyncio.run(main())
