#!/usr/bin/env python3
"""Test all 10 available swarms sequentially with real LLM."""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.anthropic")
load_dotenv(Path(__file__).parent.parent / ".env")

print(f"API Key loaded: {os.getenv('ANTHROPIC_API_KEY', 'NOT FOUND')[:20]}...")
print()

results = []


async def test_research():
    print("Testing ResearchSwarm...")
    from Jotty.core.intelligence.orchestration.swarms.research_swarm import ResearchSwarm
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.types import ResearchConfig

    config = ResearchConfig(send_telegram=False, include_charts=False)
    swarm = ResearchSwarm(config)
    return await swarm.research("AAPL")


async def main():
    print("=" * 80)
    print("TESTING 10 SWARMS WITH REAL LLM")
    print("=" * 80)
    print()

    # Start with just ResearchSwarm that we know works
    try:
        start = time.time()
        result = await test_research()
        elapsed = time.time() - start
        print(f"✅ ResearchSwarm: {elapsed:.1f}s")
        print(f"   Result: {str(result)[:150]}...")
        results.append(("ResearchSwarm", "success", elapsed))
    except Exception as e:
        print(f"❌ ResearchSwarm failed: {e}")
        results.append(("ResearchSwarm", "failed", str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    success = [r for r in results if r[1] == "success"]
    print(f"✅ Success: {len(success)}/{len(results)}")
    print(f"❌ Failed: {len(results) - len(success)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
