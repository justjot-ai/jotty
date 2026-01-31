#!/usr/bin/env python3
"""
ONE-LINER DEMO: Just give a goal, swarm does everything.
Uses Claude CLI via DSPy - no API keys needed!
"""
import asyncio
import sys
import logging
import dspy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

async def main():
    print("\n" + "="*70)
    print("  ONE-LINER SWARM DEMO: Just describe what you want")
    print("  Using Claude CLI (no API key needed)")
    print("="*70 + "\n")

    # Configure DSPy with Claude CLI
    print("🔧 Configuring Claude CLI as LLM backend...")
    from core.integration.direct_claude_cli_lm import DirectClaudeCLI

    lm = DirectClaudeCLI(model="sonnet")
    dspy.configure(lm=lm)
    print("✅ Claude CLI configured!\n")

    # Import the swarm
    from core.orchestration.v2.swarm_manager import SwarmManager

    # ONE LINE - just describe what you want in natural language
    goal = "Analyze stocks AAPL and NVDA - give buy/sell recommendation with reasons"

    print(f"🎯 GOAL: {goal}")
    print("-"*70 + "\n")

    # Create swarm with natural language (zero-config mode)
    print("🚀 Creating swarm from natural language...")
    swarm = SwarmManager(agents=goal, enable_zero_config=True)

    # Run it - everything happens automatically
    print("⚡ Running swarm... (this calls Claude CLI under the hood)\n")
    result = await swarm.run(goal=goal)

    print("\n" + "="*70)
    print("  RESULT")
    print("="*70)
    print(f"\n✅ Success: {result.success}")
    print(f"\n📊 Output:\n{result.output}")

    # Show what the swarm learned
    print("\n" + "="*70)
    print("  SWARM LEARNING (World-Class Features)")
    print("="*70)

    specs = swarm.get_agent_specializations()
    print(f"\n🧠 Agent Specializations: {specs}")

    # Show stigmergy signals
    si = swarm.swarm_intelligence
    print(f"🐜 Stigmergy Signals: {len(si.stigmergy.signals)}")

    # Show adaptive weights
    print(f"⚖️  Credit Weights: {swarm.credit_weights}")

    # Show trust scores
    print(f"🛡️  Agent Trust Scores:")
    for name, profile in si.agent_profiles.items():
        print(f"   - {name}: {profile.trust_score:.2f}")

    return 0

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
