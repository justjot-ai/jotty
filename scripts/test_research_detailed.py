#!/usr/bin/env python3
"""Test ResearchSwarm and show DETAILED output."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


async def test():
    from Jotty.core.intelligence.orchestration.swarms.research_swarm import ResearchSwarm
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.types import ResearchConfig

    print("Testing ResearchSwarm with FULL output...\n")

    config = ResearchConfig(
        send_telegram=False,
        include_charts=False,
        include_sentiment=False,
        include_peers=False,
        use_llm_analysis=True,  # Enable LLM analysis
        exchange="NSE",
    )

    swarm = ResearchSwarm(config)
    result = await swarm.research("RELIANCE", exchange="NSE")

    print("\n" + "=" * 80)
    print("DETAILED RESULT ANALYSIS")
    print("=" * 80)

    print(f"\n1. Basic Info:")
    print(f"   Success: {result.success}")
    print(f"   Company: {result.company_name}")
    print(f"   Ticker: {result.ticker}")
    print(f"   Price: {result.current_price}")
    print(f"   Target: {result.target_price}")

    print(f"\n2. Analysis:")
    print(f"   Rating: {result.rating}")
    print(f"   Confidence: {result.rating_confidence}")
    print(f"   Investment Thesis: {result.investment_thesis}")
    print(f"   Key Risks: {result.key_risks}")

    print(f"\n3. Data Sources:")
    print(f"   Sources: {result.data_sources}")
    print(f"   Screener Success: {result.screener_data.get('success', False)}")

    print(f"\n4. Outputs:")
    print(f"   MD Report: {result.md_path}")
    print(f"   PDF Report: {result.pdf_path}")

    print(f"\n5. Errors:")
    print(f"   Error: {result.error or 'None'}")

    print(f"\n6. Agent Contributions:")
    for agent, contrib in result.agent_contributions.items():
        print(f"   {agent}: {contrib}")

    print(f"\n7. Execution Time: {result.execution_time:.2f}s")

    # Check if this was REAL analysis or just basic fetch
    has_real_analysis = (
        len(result.investment_thesis) > 1
        and result.investment_thesis != ["Based on fundamental analysis"]
        and result.rating_confidence > 0.3
    )

    print(f"\n{'='*80}")
    if has_real_analysis:
        print("✅ REAL LLM ANALYSIS PERFORMED")
    else:
        print("⚠️  BASIC DATA FETCH ONLY - No deep LLM analysis")
    print("=" * 80)


asyncio.run(test())
