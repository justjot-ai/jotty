#!/usr/bin/env python3
"""Test swarms with immediate error reporting."""

import asyncio
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

print("=" * 80)
print("SWARM DEBUG TEST - STOP ON FIRST ERROR")
print("=" * 80)

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key: {api_key[:20] if api_key else 'NOT FOUND'}...")


async def test_research():
    print("\n>>> Importing ResearchSwarm...")
    from Jotty.core.intelligence.orchestration.swarms.research_swarm import ResearchSwarm
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.types import ResearchConfig

    print(">>> Creating config...")
    config = ResearchConfig(
        send_telegram=False,
        include_charts=False,
        include_sentiment=False,
        include_peers=False,
        exchange="NASDAQ",
    )

    print(">>> Creating swarm instance...")
    swarm = ResearchSwarm(config)

    print(">>> Calling research() for AAPL...")
    result = await swarm.research("AAPL", exchange="NASDAQ")

    print(f">>> Got result: success={result.success}")
    print(f">>> Error (if any): {result.error}")
    return result


async def test_olympiad():
    print("\n>>> Importing OlympiadLearningSwarm...")
    from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm import (
        OlympiadLearningSwarm,
    )
    from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.types import (
        OlympiadLearningConfig,
        Subject,
        LessonDepth,
        DifficultyTier,
    )

    print(">>> Creating config...")
    config = OlympiadLearningConfig(
        subject=Subject.MATHEMATICS, send_telegram=False, generate_pdf=False, generate_html=False
    )

    print(">>> Creating swarm instance...")
    swarm = OlympiadLearningSwarm(config)

    print(">>> Calling teach() for basic addition...")
    result = await swarm.teach(
        topic="What is 2+2?",
        subject=Subject.MATHEMATICS,
        student_name="Test",
        depth=LessonDepth.QUICK,
        target_tier=DifficultyTier.FOUNDATION,
        send_telegram=False,
    )

    print(f">>> Got result type: {type(result)}")
    return result


async def main():
    tests = [
        ("ResearchSwarm", test_research),
        ("OlympiadLearningSwarm", test_olympiad),
    ]

    for name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"TESTING: {name}")
        print("=" * 80)

        try:
            result = await test_func()
            print(f"\n✅ {name} COMPLETED")

        except Exception as e:
            print(f"\n❌ {name} FAILED")
            print(f"Error: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("\n>>> STOPPING ON FIRST ERROR <<<")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
