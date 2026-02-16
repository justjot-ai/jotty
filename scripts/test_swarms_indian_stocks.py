#!/usr/bin/env python3
"""Test swarms with INDIAN stocks - NSE exchange."""

import asyncio
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

print("=" * 80)
print("SWARM TEST - INDIAN STOCKS (NSE)")
print("=" * 80)

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key: {api_key[:20] if api_key else 'NOT FOUND'}...")


async def test_research():
    print("\n>>> Testing ResearchSwarm with Indian stock...")
    from Jotty.core.execution.swarms.research_swarm import ResearchSwarm
    from Jotty.core.execution.swarms.research_swarm.types import ResearchConfig

    print(">>> Creating config for NSE...")
    config = ResearchConfig(
        send_telegram=False,
        include_charts=False,
        include_sentiment=False,
        include_peers=False,
        exchange="NSE",  # Indian exchange
    )

    print(">>> Creating swarm...")
    swarm = ResearchSwarm(config)

    print(">>> Researching RELIANCE (Reliance Industries) on NSE...")
    result = await swarm.research("RELIANCE", exchange="NSE")

    print(f"\n>>> Result:")
    print(f"    Success: {result.success}")
    print(f"    Company: {result.company_name}")
    print(f"    Price: {result.current_price}")
    print(f"    Rating: {result.rating}")
    print(f"    Error: {result.error or 'None'}")

    return result


async def test_olympiad():
    print("\n>>> Testing OlympiadLearningSwarm...")
    from Jotty.core.execution.swarms.olympiad_learning_swarm import OlympiadLearningSwarm
    from Jotty.core.execution.swarms.olympiad_learning_swarm.types import (
        OlympiadLearningConfig,
        Subject,
        LessonDepth,
        DifficultyTier,
    )

    print(">>> Creating config...")
    config = OlympiadLearningConfig(
        subject=Subject.MATHEMATICS, send_telegram=False, generate_pdf=False, generate_html=False
    )

    print(">>> Creating swarm...")
    swarm = OlympiadLearningSwarm(config)

    print(">>> Teaching: What is 2+2?...")
    result = await swarm.teach(
        topic="What is 2+2?",
        subject=Subject.MATHEMATICS,
        student_name="Test Student",
        depth=LessonDepth.QUICK,
        target_tier=DifficultyTier.FOUNDATION,
        send_telegram=False,
    )

    print(f"\n>>> Result: {type(result)}")
    return result


async def main():
    tests = [
        ("ResearchSwarm (NSE/RELIANCE)", test_research),
        ("OlympiadLearningSwarm", test_olympiad),
    ]

    results = []

    for name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"TESTING: {name}")
        print("=" * 80)

        try:
            result = await test_func()

            # Check success
            success = getattr(result, "success", True)
            error = getattr(result, "error", "")

            if success and not error:
                print(f"\n✅ {name} - FULL SUCCESS")
                results.append((name, "success"))
            elif success:
                print(f"\n⚠️  {name} - PARTIAL (executed but with warnings)")
                results.append((name, "partial"))
            else:
                print(f"\n❌ {name} - FAILED: {error}")
                results.append((name, "failed"))
                print("\n>>> STOPPING ON FIRST FAILURE <<<")
                break

        except Exception as e:
            print(f"\n❌ {name} - EXCEPTION")
            print(f"Error: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            results.append((name, "exception"))
            print("\n>>> STOPPING ON FIRST EXCEPTION <<<")
            break

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    success = [r for r in results if r[1] == "success"]
    partial = [r for r in results if r[1] == "partial"]
    failed = [r for r in results if r[1] in ("failed", "exception")]

    print(f"✅ Success: {len(success)}/{len(results)}")
    print(f"⚠️  Partial: {len(partial)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
