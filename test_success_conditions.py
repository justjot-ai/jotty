#!/usr/bin/env python3
"""
Simple test to check success condition logic in SingleAgentOrchestrator.
"""

import sys
import os
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.foundation.unified_lm_provider import UnifiedLMProvider
from core.orchestration import SingleAgentOrchestrator
from core.foundation import JottyConfig, AgentConfig
import dspy

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)


class SimpleSignature(dspy.Signature):
    """Simple agent that returns success=True."""
    result: str = dspy.OutputField(desc="Result")
    success: bool = dspy.OutputField(desc="Success flag")


class SimpleAgent(dspy.Module):
    """Agent that always succeeds."""

    def __init__(self):
        super().__init__()
        self.signature = SimpleSignature

    def forward(self, **kwargs) -> dspy.Prediction:
        logger.info(f"🔍 SimpleAgent executing with kwargs: {list(kwargs.keys())}")
        return dspy.Prediction(
            result="Task completed",
            success=True,
            _reasoning="Success"
        )


async def test_success_conditions():
    """Test that success conditions are properly checked."""

    print("\n" + "="*80)
    print("🧪 SUCCESS CONDITION TEST")
    print("="*80)

    # Configure DSPy
    try:
        lm = UnifiedLMProvider.create_lm('claude-cli', model='sonnet')
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"⚠️  Using mock LM: {e}")

    # Config with architect/auditor disabled
    config = JottyConfig(
        enable_rl=False,
        enable_validation=False  # Disable architect/auditor for this test
    )

    # Agent config with validation disabled
    agent_config = AgentConfig(
        name="SimpleAgent",
        agent=SimpleAgent(),
        enable_architect=False,
        enable_auditor=False
    )

    # Wrap agent
    agent_wrapped = SingleAgentOrchestrator(
        agent=SimpleAgent(),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        agent_config=agent_config  # Pass agent_config so validation is properly disabled
    )

    print("\n🚀 Running simple agent...")
    result = await agent_wrapped.arun(goal="Test success conditions")

    print(f"\n📊 Result:")
    print(f"   success: {result.success}")
    print(f"   output: {result.output}")

    if result.success:
        print("\n✅ Agent succeeded as expected!")
    else:
        print("\n❌ Agent failed unexpectedly!")
        print("   Check logs above for success condition breakdown")


if __name__ == "__main__":
    asyncio.run(test_success_conditions())
