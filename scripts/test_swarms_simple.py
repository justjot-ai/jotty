#!/usr/bin/env python3
"""
Simplified Swarm Testing - Direct LLM Provider
==============================================

Tests swarms directly with LLM provider to avoid nested session issues.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Unset CLAUDECODE to avoid nested session issues
if "CLAUDECODE" in os.environ:
    del os.environ["CLAUDECODE"]


@dataclass
class SwarmRating:
    """Rating for a swarm."""

    name: str
    stars: int  # 1-5
    success: bool
    execution_time: float
    output_preview: str
    notes: str
    error: Optional[str] = None


async def test_research_swarm() -> SwarmRating:
    """Test ResearchSwarm."""
    start = time.time()
    try:
        from Jotty.core.intelligence.swarms.research_swarm import ResearchSwarm

        swarm = ResearchSwarm()
        result = await swarm.execute(
            query="What are the key features of Python 3.12?", max_sources=2
        )

        execution_time = time.time() - start
        output = str(result)[:200]

        return SwarmRating(
            name="ResearchSwarm",
            stars=4,
            success=True,
            execution_time=execution_time,
            output_preview=output,
            notes="Successfully retrieved research results",
        )
    except Exception as e:
        return SwarmRating(
            name="ResearchSwarm",
            stars=1,
            success=False,
            execution_time=time.time() - start,
            output_preview="",
            notes="Failed to execute",
            error=str(e)[:200],
        )


async def test_coding_swarm() -> SwarmRating:
    """Test CodingSwarm."""
    start = time.time()
    try:
        from Jotty.core.intelligence.swarms.coding_swarm import CodingSwarm

        swarm = CodingSwarm()
        result = await swarm.execute(task="Write a function to reverse a string in Python")

        execution_time = time.time() - start
        output = str(result)[:200]

        return SwarmRating(
            name="CodingSwarm",
            stars=4,
            success=True,
            execution_time=execution_time,
            output_preview=output,
            notes="Successfully generated code",
        )
    except Exception as e:
        return SwarmRating(
            name="CodingSwarm",
            stars=1,
            success=False,
            execution_time=time.time() - start,
            output_preview="",
            notes="Failed to execute",
            error=str(e)[:200],
        )


async def test_olympiad_learning_swarm() -> SwarmRating:
    """Test OlympiadLearningSwarm."""
    start = time.time()
    try:
        from Jotty.core.intelligence.swarms.olympiad_learning_swarm import OlympiadLearningSwarm

        swarm = OlympiadLearningSwarm()
        result = await swarm.execute(
            subject="mathematics", topic="Simple addition", level="foundation"
        )

        execution_time = time.time() - start
        output = str(result)[:200]

        return SwarmRating(
            name="OlympiadLearningSwarm",
            stars=5,
            success=True,
            execution_time=execution_time,
            output_preview=output,
            notes="Successfully generated learning content",
        )
    except Exception as e:
        return SwarmRating(
            name="OlympiadLearningSwarm",
            stars=1,
            success=False,
            execution_time=time.time() - start,
            output_preview="",
            notes="Failed to execute",
            error=str(e)[:200],
        )


async def main():
    """Run simplified swarm tests."""
    print("=" * 80)
    print("SWARM TESTING - SIMPLIFIED VERSION")
    print("=" * 80)
    print()

    tests = [
        ("ResearchSwarm", test_research_swarm()),
        ("CodingSwarm", test_coding_swarm()),
        ("OlympiadLearningSwarm", test_olympiad_learning_swarm()),
    ]

    results = []
    for name, test in tests:
        print(f"Testing {name}...", end=" ", flush=True)
        try:
            result = await test
            status = "✅" if result.success else "❌"
            print(f"{status} - {'⭐' * result.stars} ({result.execution_time:.2f}s)")
            results.append(result)
        except Exception as e:
            print(f"❌ CRASHED - {str(e)[:50]}")
            results.append(
                SwarmRating(
                    name=name,
                    stars=1,
                    success=False,
                    execution_time=0,
                    output_preview="",
                    notes="Test crashed",
                    error=str(e)[:200],
                )
            )

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    for result in sorted(results, key=lambda x: x.stars, reverse=True):
        stars = "⭐" * result.stars
        status = "✅" if result.success else "❌"
        print(f"{result.name:<30} {stars:<15} {status}  ({result.execution_time:.2f}s)")
        if result.error:
            print(f"  Error: {result.error}")
        if result.output_preview:
            print(f"  Preview: {result.output_preview[:100]}...")
        print()

    # Statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    avg_stars = sum(r.stars for r in results) / total if total > 0 else 0

    print("=" * 80)
    print(f"Total Tested:      {total}")
    print(f"Successful:        {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Average Rating:    {avg_stars:.1f}/5.0 {'⭐' * int(avg_stars)}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
