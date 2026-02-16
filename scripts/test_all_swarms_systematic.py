#!/usr/bin/env python3
"""
Systematic Swarm Testing with Real LLM
======================================

Tests each swarm with appropriate minimal tasks and real API calls.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found")
    sys.exit(1)

print(f"✅ API key loaded: {api_key[:20]}...")


class SwarmTester:
    """Systematic swarm tester."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()

    async def test_olympiad_swarm(self) -> Dict[str, Any]:
        """Test OlympiadLearningSwarm."""
        print("\n" + "=" * 80)
        print("🎓 Testing OlympiadLearningSwarm")
        print("=" * 80)

        try:
            from Jotty.core.intelligence.swarms.olympiad_learning_swarm import OlympiadLearningSwarm
            from Jotty.core.intelligence.swarms.olympiad_learning_swarm.types import (
                OlympiadLearningConfig,
                Subject,
                LessonDepth,
                DifficultyTier,
            )

            config = OlympiadLearningConfig(
                subject=Subject.MATHEMATICS,
                send_telegram=False,
                generate_pdf=False,  # Skip PDF for speed
                generate_html=False,  # Skip HTML for speed
            )
            swarm = OlympiadLearningSwarm(config=config)

            start = time.time()
            result = await swarm.teach(
                topic="Simple Addition",
                subject=Subject.MATHEMATICS,
                student_name="TestStudent",
                depth=LessonDepth.QUICK,
                target_tier=DifficultyTier.FOUNDATION,
                send_telegram=False,
            )
            elapsed = time.time() - start

            print(f"✅ SUCCESS - {elapsed:.1f}s")
            return {
                "swarm": "OlympiadLearningSwarm",
                "status": "success",
                "time": elapsed,
                "result": result,
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback

            traceback.print_exc()
            return {"swarm": "OlympiadLearningSwarm", "status": "failed", "error": str(e)}

    async def test_arxiv_swarm(self) -> Dict[str, Any]:
        """Test ArxivLearningSwarm."""
        print("\n" + "=" * 80)
        print("📚 Testing ArxivLearningSwarm")
        print("=" * 80)

        try:
            from Jotty.core.intelligence.swarms.arxiv_learning_swarm import learn_paper

            start = time.time()
            result = await learn_paper(
                paper_id="1706.03762",  # Attention Is All You Need (use paper_id not arxiv_id)
                send_telegram=False,
            )
            elapsed = time.time() - start

            print(f"✅ SUCCESS - {elapsed:.1f}s")
            return {
                "swarm": "ArxivLearningSwarm",
                "status": "success",
                "time": elapsed,
                "result": result,
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback

            traceback.print_exc()
            return {"swarm": "ArxivLearningSwarm", "status": "failed", "error": str(e)}

    async def test_research_swarm(self) -> Dict[str, Any]:
        """Test ResearchSwarm."""
        print("\n" + "=" * 80)
        print("🔬 Testing ResearchSwarm")
        print("=" * 80)

        try:
            from Jotty.core.intelligence.swarms.research_swarm import ResearchSwarm
            from Jotty.core.intelligence.swarms.research_swarm.types import ResearchConfig

            config = ResearchConfig(
                send_telegram=False,
                include_charts=False,  # Skip charts for speed
                include_sentiment=False,  # Skip sentiment for speed
                include_peers=False,  # Skip peers for speed
            )
            swarm = ResearchSwarm(config=config)

            start = time.time()
            result = await swarm.research(query="AAPL", exchange="NASDAQ", send_telegram=False)
            elapsed = time.time() - start

            print(f"✅ SUCCESS - {elapsed:.1f}s")
            return {
                "swarm": "ResearchSwarm",
                "status": "success",
                "time": elapsed,
                "result": result,
            }

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback

            traceback.print_exc()
            return {"swarm": "ResearchSwarm", "status": "failed", "error": str(e)}

    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all swarm tests."""
        print("\n" + "=" * 80)
        print("🚀 SYSTEMATIC SWARM TESTING")
        print("=" * 80)
        print(f"API Provider: Anthropic Claude")
        print(f"API Key: {api_key[:20]}...")

        # Test swarms one by one
        self.results.append(await self.test_olympiad_swarm())
        self.results.append(await self.test_arxiv_swarm())
        self.results.append(await self.test_research_swarm())

        return self.results

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)

        success = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "failed"]

        total_time = time.time() - self.start_time

        print(f"\n✅ Successful: {len(success)}/{len(self.results)}")
        for r in success:
            print(f"   • {r['swarm']}: {r.get('time', 0):.1f}s")

        if failed:
            print(f"\n❌ Failed: {len(failed)}/{len(self.results)}")
            for r in failed:
                error = r.get("error", "Unknown")
                print(f"   • {r['swarm']}: {error[:100]}")

        print(f"\n⏱️  Total Time: {total_time:.1f}s")
        print(f"🎯 Success Rate: {len(success)/len(self.results)*100:.1f}%")


async def main():
    """Main entry point."""
    tester = SwarmTester()

    try:
        await tester.run_all_tests()
        tester.print_summary()

        # Exit code based on results
        failed = [r for r in tester.results if r["status"] == "failed"]
        if failed:
            print(f"\n💥 {len(failed)} swarm(s) failed")
            sys.exit(1)
        else:
            print("\n🎉 All swarms passed!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
