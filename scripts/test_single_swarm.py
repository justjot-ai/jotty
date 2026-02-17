#!/usr/bin/env python3
"""Test a single swarm with detailed logging."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.anthropic")
load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key: {api_key[:20] if api_key else 'NOT FOUND'}...")


async def test():
    print("\n1. Importing ResearchSwarm...")
    from Jotty.core.intelligence.orchestration.swarms.research_swarm import ResearchSwarm
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.types import ResearchConfig

    print("2. Creating config...")
    config = ResearchConfig(
        send_telegram=False, include_charts=False, include_sentiment=False, include_peers=False
    )

    print("3. Creating swarm instance...")
    swarm = ResearchSwarm(config)

    print("4. Calling research() with minimal task...")
    result = await swarm.research("AAPL")

    print(f"5. Got result: {str(result)[:200]}...")
    return result


if __name__ == "__main__":
    print("=" * 80)
    print("SINGLE SWARM TEST - ResearchSwarm")
    print("=" * 80)
    result = asyncio.run(test())
    print("\n✅ TEST PASSED")
    print(f"Result: {result}")
