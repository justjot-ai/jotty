#!/usr/bin/env python3
"""
RL Natural Dependencies - DEBUG VERSION
========================================

Enhanced logging to understand why agents aren't failing when they should.
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

# VERBOSE logging to see everything
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s [%(name)s] %(message)s')
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
# AGENTS WITH SIGNATURES
# ============================================================================

class FetcherAgentDebug(dspy.Module):
    """Fetcher with verbose logging."""

    def __init__(self):
        super().__init__()
        self.signature = FetcherSignature

    def forward(self) -> dspy.Prediction:
        logger.info("=" * 80)
        logger.info("🔍 FETCHER AGENT CALLED")
        logger.info("=" * 80)

        sales_data = '{"region": "US", "sales": 1000000, "quarter": "Q1"}'

        result = dspy.Prediction(
            sales_data=sales_data,
            success=True,
            _reasoning="Fetched sales data from database"
        )

        logger.info(f"✅ FETCHER returning: sales_data={sales_data[:50]}..., success=True")
        return result


class ProcessorAgentDebug(dspy.Module):
    """Processor with verbose logging and natural dependency check."""

    def __init__(self):
        super().__init__()
        self.signature = ProcessorSignature

    def forward(self, sales_data: str = '') -> dspy.Prediction:
        logger.info("=" * 80)
        logger.info("🔍 PROCESSOR AGENT CALLED")
        logger.info(f"📊 sales_data value: '{sales_data}' (type: {type(sales_data)})")
        logger.info(f"📊 sales_data bool check: bool(sales_data) = {bool(sales_data)}")
        logger.info(f"📊 sales_data == '' check: sales_data == '' = {sales_data == ''}")
        logger.info("=" * 80)

        # NATURAL DEPENDENCY CHECK
        if not sales_data or sales_data == '':
            logger.error("❌ PROCESSOR FAILING: No sales_data available!")
            result = dspy.Prediction(
                summary='',
                success=False,
                _reasoning="ERROR: Cannot process - no sales_data available!"
            )
            logger.info(f"❌ PROCESSOR returning: summary='', success=False")
            return result

        # Process the data
        summary = f"Sales Summary: $1M in Q1 for US region"
        result = dspy.Prediction(
            summary=summary,
            success=True,
            _reasoning=f"Processed {len(sales_data)} bytes of data"
        )

        logger.info(f"✅ PROCESSOR returning: summary='{summary}', success=True")
        return result


class VisualizerAgentDebug(dspy.Module):
    """Visualizer with verbose logging and natural dependency check."""

    def __init__(self):
        super().__init__()
        self.signature = VisualizerSignature

    def forward(self, summary: str = '') -> dspy.Prediction:
        logger.info("=" * 80)
        logger.info("🔍 VISUALIZER AGENT CALLED")
        logger.info(f"📊 summary value: '{summary}' (type: {type(summary)})")
        logger.info(f"📊 summary bool check: bool(summary) = {bool(summary)}")
        logger.info(f"📊 summary == '' check: summary == '' = {summary == ''}")
        logger.info("=" * 80)

        # NATURAL DEPENDENCY CHECK
        if not summary or summary == '':
            logger.error("❌ VISUALIZER FAILING: No summary available!")
            result = dspy.Prediction(
                chart='',
                success=False,
                _reasoning="ERROR: Cannot visualize - no summary available!"
            )
            logger.info(f"❌ VISUALIZER returning: chart='', success=False")
            return result

        # Create visualization
        chart = f"Bar chart showing: {summary}"
        result = dspy.Prediction(
            chart=chart,
            success=True,
            _reasoning=f"Created visualization from summary: {summary[:50]}..."
        )

        logger.info(f"✅ VISUALIZER returning: chart='{chart[:50]}...', success=True")
        return result


async def test_debug():
    """Run 3 episodes with verbose logging to understand data flow."""

    print("\n" + "="*80)
    print("🐛 RL NATURAL DEPENDENCIES - DEBUG MODE")
    print("="*80)
    print("\nRunning 3 episodes with VERBOSE logging to see:")
    print("  1. What kwargs each agent receives")
    print("  2. Whether dependency checks trigger")
    print("  3. Why agents succeed when they should fail\n")

    # Configure DSPy (optional LM)
    try:
        lm = UnifiedLMProvider.create_lm('claude-cli', model='sonnet')
        dspy.configure(lm=lm)
        print("✅ Using Claude CLI\n")
    except Exception as e:
        print(f"⚠️  Using mock LM: {e}\n")

    # RL config
    config = JottyConfig(
        enable_rl=True,
        alpha=0.3,
        gamma=0.95,
        lambda_trace=0.9,
        epsilon_start=0.3,
        allow_partial_execution=True  # 🔥 CRITICAL: Allow agents to execute with missing params (natural dependencies)
    )

    # Define agent configs first (so we can pass them to both SingleAgentOrchestrator and conductor)
    fetcher_config = AgentConfig(name="Fetcher", agent=None, enable_architect=False, enable_auditor=False)
    processor_config = AgentConfig(name="Processor", agent=None, enable_architect=False, enable_auditor=False)
    visualizer_config = AgentConfig(name="Visualizer", agent=None, enable_architect=False, enable_auditor=False)

    # Wrap agents with their respective configs
    fetcher = SingleAgentOrchestrator(
        agent=FetcherAgentDebug(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=fetcher_config  # Pass config so validation works correctly
    )

    processor = SingleAgentOrchestrator(
        agent=ProcessorAgentDebug(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=processor_config  # Pass config so validation works correctly
    )

    visualizer = SingleAgentOrchestrator(
        agent=VisualizerAgentDebug(),
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

    # WRONG order (should fail!)
    actors_wrong_order = [
        visualizer_config,
        processor_config,
        fetcher_config
    ]

    print("📋 Agent Order: Visualizer → Processor → Fetcher (WRONG!)")
    print("   Expected behavior:")
    print("     - Visualizer runs first → no 'summary' → FAILS ❌")
    print("     - Processor runs → no 'sales_data' → FAILS ❌")
    print("     - Fetcher runs → succeeds ✅")
    print("   Over episodes: RL should learn to run Fetcher first\n")

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

    print("🚀 Running 3 episodes with VERBOSE logging...")
    print("="*80 + "\n")

    episode_data = []

    for episode in range(1, 4):
        print(f"\n{'#'*80}")
        print(f"# EPISODE {episode}/3")
        print(f"{'#'*80}\n")

        try:
            result = await orchestrator.run(
                goal=f"Process and visualize sales data (episode {episode})"
            )

            # Extract execution order
            execution_log = []
            if hasattr(orchestrator, 'todo') and orchestrator.todo:
                for task_id, task in orchestrator.todo.subtasks.items():
                    execution_log.append({
                        'agent': task.actor,
                        'status': task.status.name
                    })

            first_agent = execution_log[0]['agent'] if execution_log else None

            episode_data.append({
                'episode': episode,
                'first_agent': first_agent,
                'execution': execution_log,
                'overall_success': result.success
            })

            print(f"\n{'='*80}")
            print(f"📊 EPISODE {episode} RESULTS:")
            print(f"{'='*80}")
            print(f"Overall success: {result.success}")
            print(f"First agent: {first_agent}")
            print("Execution order:")
            for item in execution_log:
                status_icon = "✅" if item['status'] == 'COMPLETED' else "❌"
                print(f"  {status_icon} {item['agent']}: {item['status']}")

        except Exception as e:
            print(f"⚠️  Episode {episode} error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")

    print("\n🎯 First Agent Selected:")
    for data in episode_data:
        print(f"   Episode {data['episode']}: {data['first_agent']}")

    print("\n📈 Overall Success:")
    for data in episode_data:
        print(f"   Episode {data['episode']}: {data['overall_success']}")

    print("\n🔍 ANALYSIS:")
    all_succeeded = all(d['overall_success'] for d in episode_data)

    if all_succeeded:
        print("   ⚠️  ALL episodes succeeded - agents NOT failing as expected!")
        print("   📝 Check logs above to see:")
        print("      - What kwargs agents received")
        print("      - Whether dependency checks ran")
        print("      - Where data is coming from")
    else:
        print("   ✅ Some episodes failed - natural dependencies working!")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(test_debug())
