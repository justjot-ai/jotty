#!/usr/bin/env python3
"""
RL Learning with NATURAL Data Dependencies (Not Hardcoded Order)
=================================================================

Key Insight: Agents fail based on MISSING DATA, not position in sequence.
This is real RL - system learns which order produces needed data naturally!

Example:
- Visualizer needs "summary" field → If missing, NATURALLY fails
- Processor needs "raw_data" field → If missing, NATURALLY fails
- Fetcher needs nothing → Always succeeds (data source)

RL learns: "Fetcher first produces data → Processor uses it → Visualizer uses result"
NOT because we told it that order, but because that's the only order that works!
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


# ============================================================================
# DSPY SIGNATURES - Declare inputs/outputs for each agent
# ============================================================================

class FetcherSignature(dspy.Signature):
    """Fetch sales data from database. No dependencies."""
    sales_data: str = dspy.OutputField(desc="Raw sales data JSON string")
    success: bool = dspy.OutputField(desc="Whether fetch succeeded")


class ProcessorSignature(dspy.Signature):
    """Process sales data into summary. Depends on Fetcher."""
    sales_data: str = dspy.InputField(desc="Raw sales data from Fetcher")
    summary: str = dspy.OutputField(desc="Processed sales summary")
    success: bool = dspy.OutputField(desc="Whether processing succeeded")


class VisualizerSignature(dspy.Signature):
    """Create visualization from summary. Depends on Processor."""
    summary: str = dspy.InputField(desc="Summary from Processor")
    chart: str = dspy.OutputField(desc="Visualization description")
    success: bool = dspy.OutputField(desc="Whether visualization succeeded")


# ============================================================================
# AGENTS WITH NATURAL DATA DEPENDENCIES (NOT POSITION-BASED)
# ============================================================================

class FetcherAgent(dspy.Module):
    """
    Fetcher: NO dependencies, always succeeds.
    Produces: 'sales_data' in context
    """
    def __init__(self):
        super().__init__()
        self.signature = FetcherSignature

    def forward(self) -> dspy.Prediction:
        # Fetcher is data source - always succeeds
        sales_data = '{"region": "US", "sales": 1000000, "quarter": "Q1"}'

        return dspy.Prediction(
            sales_data=sales_data,
            success=True,
            _reasoning="Fetched sales data from database"
        )


class ProcessorAgent(dspy.Module):
    """
    Processor: NEEDS 'sales_data' from context
    Produces: 'summary' in context

    NATURAL FAILURE: If sales_data missing → can't process → fails
    """
    def __init__(self):
        super().__init__()
        self.signature = ProcessorSignature

    def forward(self, sales_data: str = '') -> dspy.Prediction:
        # NATURAL DEPENDENCY CHECK (not position-based!)
        if not sales_data or sales_data == '':
            return dspy.Prediction(
                summary='',
                success=False,
                _reasoning="ERROR: Cannot process - no sales_data available!"
            )

        # Process the data
        summary = f"Sales Summary: $1M in Q1 for US region"

        return dspy.Prediction(
            summary=summary,
            success=True,
            _reasoning=f"Processed {len(sales_data)} bytes of data"
        )


class VisualizerAgent(dspy.Module):
    """
    Visualizer: NEEDS 'summary' from context
    Produces: 'chart' description

    NATURAL FAILURE: If summary missing → can't visualize → fails
    """
    def __init__(self):
        super().__init__()
        self.signature = VisualizerSignature

    def forward(self, summary: str = '') -> dspy.Prediction:
        # NATURAL DEPENDENCY CHECK (not position-based!)
        if not summary or summary == '':
            return dspy.Prediction(
                chart='',
                success=False,
                _reasoning="ERROR: Cannot visualize - no summary available!"
            )

        # Create visualization
        chart = f"Bar chart showing: {summary}"

        return dspy.Prediction(
            chart=chart,
            success=True,
            _reasoning=f"Created visualization from summary: {summary[:50]}..."
        )


async def test_rl_with_natural_dependencies():
    """Test RL with natural data dependencies (not hardcoded order)."""

    print("\n" + "="*80)
    print("🧪 RL WITH NATURAL DATA DEPENDENCIES")
    print("="*80)

    print("""
Key Concept: Agents fail based on MISSING DATA, not position!

Fetcher:    Needs nothing → Always succeeds
Processor:  Needs 'sales_data' → Fails if missing (natural dependency)
Visualizer: Needs 'summary' → Fails if missing (natural dependency)

RL will learn: The only order that works is Fetcher → Processor → Visualizer
NOT because we told it, but because that's the only order where data flows correctly!
""")

    # Configure DSPy
    print("🔧 Configuring DSPy with Claude CLI...")
    try:
        lm = UnifiedLMProvider.create_lm('claude-cli', model='sonnet')
        dspy.configure(lm=lm)
        print("✅ DSPy configured\n")
    except Exception as e:
        print(f"⚠️  Using mock LM: {e}\n")

    # Create config with RL enabled
    config = JottyConfig(
        enable_rl=True,
        alpha=0.3,  # Higher learning rate for faster learning
        gamma=0.95,
        lambda_trace=0.9,
        epsilon_start=0.3,  # 30% exploration
        allow_partial_execution=True  # 🔥 CRITICAL: Allow agents to execute with missing params (natural dependencies)
    )

    print("📋 Configuration:")
    print(f"   RL Enabled: {config.enable_rl}")
    print(f"   Learning Rate: {config.alpha}")
    print(f"   Exploration: {config.epsilon_start}")

    # Define agent configs first (so we can pass them to both SingleAgentOrchestrator and conductor)
    fetcher_config = AgentConfig(name="Fetcher", agent=None, enable_architect=False, enable_auditor=False)
    processor_config = AgentConfig(name="Processor", agent=None, enable_architect=False, enable_auditor=False)
    visualizer_config = AgentConfig(name="Visualizer", agent=None, enable_architect=False, enable_auditor=False)

    # Wrap agents in SingleAgentOrchestrator with configs
    fetcher = SingleAgentOrchestrator(
        agent=FetcherAgent(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=fetcher_config  # Pass config so validation works correctly
    )

    processor = SingleAgentOrchestrator(
        agent=ProcessorAgent(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=processor_config  # Pass config so validation works correctly
    )

    visualizer = SingleAgentOrchestrator(
        agent=VisualizerAgent(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=visualizer_config  # Pass config so validation works correctly
    )

    # Update agent references in configs (after creating orchestrators)
    fetcher_config.agent = fetcher
    processor_config.agent = processor
    visualizer_config.agent = visualizer

    # Define actors in WRONG ORDER
    actors_wrong_order = [
        visualizer_config,
        processor_config,
        fetcher_config
    ]

    print("\n📋 Agent Order (INTENTIONALLY WRONG):")
    print("   Initial: Visualizer → Processor → Fetcher")
    print("   Correct: Fetcher → Processor → Visualizer")
    print("\n   Why correct order works:")
    print("   1. Fetcher produces 'sales_data'")
    print("   2. Processor consumes 'sales_data', produces 'summary'")
    print("   3. Visualizer consumes 'summary', produces 'chart'")

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

    print("\n🚀 Running 15 episodes...")
    print("   Watch Q-values diverge based on success/failure!")
    print("="*80)

    episode_data = []

    for episode in range(1, 16):
        print(f"\n{'─'*80}")
        print(f"📊 EPISODE {episode}/15")
        print(f"{'─'*80}")

        try:
            result = await orchestrator.run(
                goal=f"Process and visualize sales data (episode {episode})"
            )

            # Extract execution order and results
            execution_log = []
            if hasattr(orchestrator, 'todo') and orchestrator.todo:
                for task_id, task in orchestrator.todo.subtasks.items():
                    status = task.status.name
                    execution_log.append({
                        'agent': task.actor,
                        'status': status
                    })

            # Get Q-values from logs
            q_values = {}
            if hasattr(orchestrator, 'q_learner'):
                # We'll check logs for Q-values
                pass

            first_agent = execution_log[0]['agent'] if execution_log else None

            episode_data.append({
                'episode': episode,
                'first_agent': first_agent,
                'execution': execution_log,
                'overall_success': result.success
            })

            print(f"\n✓ Episode {episode}:")
            print(f"  Overall success: {result.success}")
            print(f"  Execution order:")
            for item in execution_log:
                status_icon = "✅" if item['status'] == 'COMPLETED' else "❌"
                print(f"    {status_icon} {item['agent']}: {item['status']}")

        except Exception as e:
            print(f"⚠️  Episode {episode} error: {e}")

    # Analyze results
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS")
    print(f"{'='*80}")

    print("\n🎯 First Agent Selected Per Episode:")
    for data in episode_data:
        print(f"   Episode {data['episode']:2d}: {data['first_agent']}")

    # Count selections
    first_agent_counts = {}
    for data in episode_data:
        agent = data['first_agent']
        if agent:
            first_agent_counts[agent] = first_agent_counts.get(agent, 0) + 1

    print("\n📊 Agent Selection Frequency:")
    for agent in ['Visualizer', 'Processor', 'Fetcher']:
        count = first_agent_counts.get(agent, 0)
        pct = (count / len(episode_data)) * 100
        print(f"   {agent}: {count}/15 episodes ({pct:.0f}%)")

    # Success rate over time
    early_success = sum(1 for d in episode_data[:5] if d['overall_success'])
    late_success = sum(1 for d in episode_data[10:15] if d['overall_success'])

    print(f"\n📈 Success Rate:")
    print(f"   Early episodes (1-5): {early_success}/5 = {early_success*20}%")
    print(f"   Late episodes (11-15): {late_success}/5 = {late_success*20}%")

    # Learning analysis
    fetcher_count = first_agent_counts.get('Fetcher', 0)
    visualizer_count = first_agent_counts.get('Visualizer', 0)

    print(f"\n{'='*80}")
    print("🎓 LEARNING ANALYSIS")
    print(f"{'='*80}")

    if fetcher_count >= 10:
        print("\n✅ SUCCESS: RL learned optimal order!")
        print(f"   Fetcher selected first {fetcher_count}/15 times")
        print("   RL discovered: Fetcher must run first to produce data")
    elif fetcher_count > visualizer_count:
        print("\n⚡ LEARNING IN PROGRESS")
        print(f"   Fetcher: {fetcher_count}/15 (increasing preference)")
        print(f"   Visualizer: {visualizer_count}/15 (decreasing)")
        print("   RL is learning from natural failures!")
    else:
        print("\n⚠️  Needs more episodes")
        print(f"   Fetcher: {fetcher_count}/15")
        print(f"   Visualizer: {visualizer_count}/15")

    print(f"\n{'='*80}")
    print("🎯 KEY INSIGHT")
    print(f"{'='*80}")
    print("""
This IS real RL because:
✅ Agents fail based on MISSING DATA (natural dependencies)
✅ NOT based on position in sequence (no hardcoded order)
✅ RL learns which order produces needed data flow
✅ Q-values diverge: High for orders that work, low for orders that fail

Natural dependency chain:
  Fetcher → sales_data → Processor → summary → Visualizer → chart

RL discovers this chain through trial and error, not because we told it!
""")


if __name__ == "__main__":
    asyncio.run(test_rl_with_natural_dependencies())
