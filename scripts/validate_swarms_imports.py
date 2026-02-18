#!/usr/bin/env python3
"""
Validate Swarm Imports
======================

Quick validation that all swarms can be imported and instantiated.
Does not run actual LLM calls - just checks structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

results = []


def test_import(name: str, import_func):
    """Test if a swarm can be imported and instantiated."""
    try:
        import_func()
        print(f"✅ {name}")
        results.append({"name": name, "status": "success"})
        return True
    except Exception as e:
        print(f"❌ {name}: {e}")
        results.append({"name": name, "status": "failed", "error": str(e)})
        return False


def test_olympiad():
    from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm import (
        OlympiadLearningSwarm,
    )
    from Jotty.core.intelligence.orchestration.swarms.olympiad_learning_swarm.types import (
        OlympiadLearningConfig,
        Subject,
    )

    config = OlympiadLearningConfig(subject=Subject.MATHEMATICS, send_telegram=False)
    swarm = OlympiadLearningSwarm(config)


def test_research():
    from Jotty.core.intelligence.orchestration.swarms.research_swarm import ResearchSwarm
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.types import ResearchConfig

    config = ResearchConfig(send_telegram=False)
    swarm = ResearchSwarm(config)


def test_arxiv():
    from Jotty.core.intelligence.orchestration.swarms.arxiv_learning_swarm import learn_paper


def test_coding():
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm import CodingSwarm
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

    config = CodingConfig()
    swarm = CodingSwarm(config)


def test_perspective():
    from Jotty.core.intelligence.orchestration.swarms.perspective_learning_swarm import (
        PerspectiveLearningSwarm,
    )
    from Jotty.core.intelligence.orchestration.swarms.perspective_learning_swarm.types import (
        PerspectiveLearningConfig,
    )

    config = PerspectiveLearningConfig(send_telegram=False)
    swarm = PerspectiveLearningSwarm(config)


def test_pilot():
    from Jotty.core.intelligence.orchestration.swarms.pilot_swarm import PilotSwarm
    from Jotty.core.intelligence.orchestration.swarms.pilot_swarm.types import PilotConfig

    config = PilotConfig()
    swarm = PilotSwarm(config)


def test_testing():
    from Jotty.core.intelligence.orchestration.swarms.templates.testing_swarm import TestingSwarm

    swarm = TestingSwarm()


def test_review():
    from Jotty.core.intelligence.orchestration.swarms.templates.review_swarm import ReviewSwarm

    swarm = ReviewSwarm()


def test_data_analysis():
    from Jotty.core.intelligence.orchestration.swarms.templates.data_analysis_swarm import (
        DataAnalysisSwarm,
    )

    swarm = DataAnalysisSwarm()


def test_devops():
    from Jotty.core.intelligence.orchestration.swarms.templates.devops_swarm import DevOpsSwarm

    swarm = DevOpsSwarm()


def main():
    """Validate all 10 available swarms."""
    print("=" * 80)
    print("VALIDATING SWARM IMPORTS (10 swarms)")
    print("=" * 80)
    print()

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
        test_import(name, func)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n✅ Success: {len(success)}/10")
    print(f"❌ Failed: {len(failed)}/10")

    if failed:
        print("\nFailed swarms:")
        for r in failed:
            print(f"   • {r['name']}: {r.get('error', 'Unknown')[:100]}")

    print(f"\n{'=' * 80}")
    print(f"VALIDATION RATE: {len(success)}/10 ({len(success)/10*100:.0f}%)")
    print("=" * 80)

    print("\nNote: DeploymentSwarm, DebugSwarm, and MarketingSwarm do not exist in codebase")

    return len(success) == 10


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
