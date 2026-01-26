#!/usr/bin/env python3
"""
Simple RL ordering test - checks conductor logs to verify Q-value-based selection.
"""

import asyncio
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestration.conductor import MultiAgentsOrchestrator
from core.foundation import JottyConfig, AgentConfig
import dspy

# Configure logging to capture all INFO logs
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/rl_ordering_test.log', mode='w')
    ]
)


# Mock agents
class MockAgent(dspy.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, **kwargs):
        return dspy.Prediction(output=f"{self.name} output", success=True)


async def test_single_episode():
    """Run single episode with detailed logging."""

    print("=" * 80)
    print("🧪 RL AGENT ORDERING TEST - SINGLE EPISODE WITH DETAILED LOGS")
    print("=" * 80)

    config = JottyConfig(
        enable_rl=True,
        alpha=0.1,
        gamma=0.95,
        lambda_trace=0.9,
        epsilon_start=0.3,  # 30% exploration
        log_level="INFO"  # Enable INFO logging
    )

    # Create agents in WRONG ORDER
    actors_wrong_order = [
        AgentConfig(
            name="Visualizer",
            agent=MockAgent("Visualizer"),
            enable_architect=False,
            enable_auditor=False
        ),
        AgentConfig(
            name="Fetcher",
            agent=MockAgent("Fetcher"),
            enable_architect=False,
            enable_auditor=False
        ),
        AgentConfig(
            name="Processor",
            agent=MockAgent("Processor"),
            enable_architect=False,
            enable_auditor=False
        )
    ]

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

    print("\n📋 Initial Agent Order (WRONG):")
    print("   1. Visualizer (wrong - should be last)")
    print("   2. Fetcher (correct - should be first)")
    print("   3. Processor (correct - should be second)")
    print("\n🔍 Running single episode...")
    print("📝 Check /tmp/rl_ordering_test.log for detailed logs")
    print("=" * 80)

    try:
        result = await orchestrator.run(
            goal="Fetch sales data, process it, and create visualization"
        )
        print(f"\n✅ Episode completed: {result.success}")
    except Exception as e:
        print(f"\n⚠️  Episode error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("📋 Check /tmp/rl_ordering_test.log for logs showing:")
    print("   - 🔍 [get_next_task] logs showing if Q-predictor was passed")
    print("   - 🎯 [get_next_task] logs showing Q-value-based selection")
    print("   - 📊 [get_next_task] logs showing Q-values for each agent")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_single_episode())
