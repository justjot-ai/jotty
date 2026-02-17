#!/usr/bin/env python3
"""
Test All 10 Available Swarms Systematically
============================================

Actually tests every swarm with minimal real tasks.
No claims until validated.

Note: DeploymentSwarm, DebugSwarm, and MarketingSwarm do not exist in codebase.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ No API key")
    sys.exit(1)

print(f"API Key: {api_key[:20]}...")

results = []


async def test_swarm(name: str, test_func):
    """Test a single swarm and record result."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print("=" * 80)

    try:
        start = time.time()
        result = await test_func()
        elapsed = time.time() - start

        print(f"✅ SUCCESS - {elapsed:.1f}s")
        results.append({"name": name, "status": "success", "time": elapsed, "result": result})
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.append({"name": name, "status": "failed", "error": str(e)})
        return False


async def test_olympiad():
    from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm import (
        OlympiadLearningSwarm,
    )
    from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.types import (
        OlympiadLearningConfig,
        Subject,
        LessonDepth,
        DifficultyTier,
    )

    config = OlympiadLearningConfig(
        subject=Subject.MATHEMATICS, send_telegram=False, generate_pdf=False, generate_html=False
    )
    swarm = OlympiadLearningSwarm(config)
    return await swarm.teach(
        topic="Addition",
        subject=Subject.MATHEMATICS,
        student_name="Test",
        depth=LessonDepth.QUICK,
        target_tier=DifficultyTier.FOUNDATION,
        send_telegram=False,
    )


async def test_research():
    from Jotty.core.intelligence.orchestration.swarms.research_swarm import ResearchSwarm
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.types import ResearchConfig

    config = ResearchConfig(
        send_telegram=False, include_charts=False, include_sentiment=False, include_peers=False
    )
    swarm = ResearchSwarm(config)
    return await swarm.research("AAPL", exchange="NASDAQ")


async def test_arxiv():
    from Jotty.core.intelligence.orchestration.swarms.arxiv_learning_swarm import learn_paper

    return await learn_paper(paper_id="1706.03762", send_telegram=False)


async def test_coding():
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm import CodingSwarm
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

    config = CodingConfig()
    swarm = CodingSwarm(config)
    # Find the right method by checking what's available
    return await swarm.execute(prompt="Write hello world in Python")


async def test_perspective():
    from Jotty.core.intelligence.orchestration.swarms.perspective_learning_swarm import (
        PerspectiveLearningSwarm,
    )
    from Jotty.core.intelligence.orchestration.swarms.perspective_learning_swarm.types import (
        PerspectiveLearningConfig,
    )

    config = PerspectiveLearningConfig(send_telegram=False)
    swarm = PerspectiveLearningSwarm(config)
    return await swarm.execute(topic="Python basics", perspectives=["beginner"])


async def test_pilot():
    from Jotty.core.intelligence.orchestration.swarms.pilot_swarm import PilotSwarm
    from Jotty.core.intelligence.orchestration.swarms.pilot_swarm.types import PilotConfig

    config = PilotConfig()
    swarm = PilotSwarm(config)
    return await swarm.execute(goal="Test task")


# Template swarms
async def test_testing():
    from Jotty.core.intelligence.orchestration.swarms.templates.testing import TestingSwarm

    swarm = TestingSwarm()
    return await swarm.execute(code="def add(a,b): return a+b")


async def test_review():
    from Jotty.core.intelligence.orchestration.swarms.templates.review import ReviewSwarm

    swarm = ReviewSwarm()
    return await swarm.execute(code="def add(a,b): return a+b")


async def test_data_analysis():
    from Jotty.core.intelligence.orchestration.swarms.templates.data_analysis_swarm import (
        DataAnalysisSwarm,
    )

    swarm = DataAnalysisSwarm()
    return await swarm.execute(data=[1, 2, 3])


async def test_devops():
    from Jotty.core.intelligence.orchestration.swarms.templates.devops_swarm import DevOpsSwarm

    swarm = DevOpsSwarm()
    return await swarm.execute(task="health check")


async def main():
    """Test all 10 available swarms."""
    print("\n" + "=" * 80)
    print("SYSTEMATIC TEST OF ALL 10 SWARMS")
    print("=" * 80)

    swarms = [
        ("OlympiadLearningSwarm", test_olympiad),
        ("ResearchSwarm", test_research),
        ("ArxivLearningSwarm", test_arxiv),
        ("CodingSwarm", test_coding),
        ("PerspectiveLearningSwarm", test_perspective),
        ("PilotSwarm", test_pilot),
        ("TestingSwarm", test_testing),
        ("ReviewSwarm", test_review),
        ("DataAnalysisSwarm", test_data_analysis),
        ("DevOpsSwarm", test_devops),
    ]

    for name, func in swarms:
        await test_swarm(name, func)

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n✅ Success: {len(success)}/10")
    for r in success:
        print(f"   • {r['name']}: {r.get('time', 0):.1f}s")

    print(f"\n❌ Failed: {len(failed)}/10")
    for r in failed:
        print(f"   • {r['name']}: {r.get('error', 'Unknown')[:80]}")

    print(f"\n{'='*80}")
    print(f"SUCCESS RATE: {len(success)}/10 ({len(success)/10*100:.0f}%)")
    print("=" * 80)

    print("\nNote: DeploymentSwarm, DebugSwarm, MarketingSwarm do not exist in codebase")

    return len(success) == 10


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
