#!/usr/bin/env python3
"""
Test Swarm Templates with Real LLM via Unified LM Provider
===========================================================

Tests key swarms with actual LLM calls and rates them.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Verify API key is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found in environment")
    sys.exit(1)
print(f"✅ API key loaded: {api_key[:20]}...")


async def test_research_swarm():
    """Test ResearchSwarm with a simple query."""
    print("\n" + "=" * 80)
    print("🔬 Testing ResearchSwarm")
    print("=" * 80)

    try:
        from Jotty.core.execution.swarms.research_swarm import ResearchSwarm
        from Jotty.core.execution.swarms.research_swarm.types import ResearchConfig

        # Create swarm with correct config
        config = ResearchConfig(send_telegram=False)
        swarm = ResearchSwarm(config=config)

        # Indian stock on NSE (yfinance needs .NS suffix)
        result = await swarm.research(
            query="Reliance Industries", ticker="RELIANCE.NS", exchange="NSE", send_telegram=False
        )

        print(f"\n✅ ResearchSwarm Success!")
        print(f"   Ticker: {getattr(result, 'ticker', 'N/A')}")
        print(f"   Rating: {getattr(result, 'rating', 'N/A')}")
        print(f"   PDF: {getattr(result, 'pdf_path', 'N/A')}")
        print(f"   Success: {getattr(result, 'success', 'N/A')}")

        return {"status": "success", "swarm": "ResearchSwarm", "result": result}

    except Exception as e:
        print(f"\n❌ ResearchSwarm Failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "swarm": "ResearchSwarm", "error": str(e)}


async def test_arxiv_swarm():
    """Test ArxivLearningSwarm with a simple paper research task."""
    print("\n" + "=" * 80)
    print("📚 Testing ArxivLearningSwarm")
    print("=" * 80)

    try:
        from Jotty.core.execution.swarms.arxiv_learning_swarm import learn_paper

        # Simple arxiv research task  # "Attention is All You Need" paper
        result = await learn_paper(paper_id="1706.03762", send_telegram=False)

        print(f"\n✅ ArxivLearningSwarm Success!")
        print(f"   Paper: Attention is All You Need (arXiv:1706.03762)")
        if hasattr(result, "pdf_path") and result.pdf_path:
            print(f"   PDF: {result.pdf_path}")
        if hasattr(result, "content") and result.content:
            print(f"   Content: {str(result.content)[:100]}...")

        return {"status": "success", "swarm": "ArxivLearningSwarm", "result": result}

    except Exception as e:
        print(f"\n❌ ArxivLearningSwarm Failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "swarm": "ArxivLearningSwarm", "error": str(e)}


async def test_olympiad_swarm():
    """Test OlympiadLearningSwarm with a simple educational task."""
    print("\n" + "=" * 80)
    print("🎓 Testing OlympiadLearningSwarm")
    print("=" * 80)

    try:
        from Jotty.core.execution.swarms.olympiad_learning_swarm import OlympiadLearningSwarm
        from Jotty.core.execution.swarms.olympiad_learning_swarm.types import (
            OlympiadLearningConfig,
            Subject,
            LessonDepth,
            DifficultyTier,
        )

        # Create swarm with correct config
        config = OlympiadLearningConfig(subject=Subject.MATHEMATICS, send_telegram=False)
        swarm = OlympiadLearningSwarm(config=config)

        # Simple teaching task
        result = await swarm.teach(
            topic="Basic Addition",
            subject=Subject.MATHEMATICS,
            student_name="TestStudent",
            depth=LessonDepth.QUICK,
            target_tier=DifficultyTier.FOUNDATION,
            send_telegram=False,
        )

        print(f"\n✅ OlympiadLearningSwarm Success!")
        print(f"   Subject: Mathematics")
        print(f"   Topic: Basic Addition")
        if hasattr(result, "pdf_path"):
            print(f"   PDF: {result.pdf_path}")
        if hasattr(result, "html_path"):
            print(f"   HTML: {result.html_path}")

        return {"status": "success", "swarm": "OlympiadLearningSwarm", "result": result}

    except Exception as e:
        print(f"\n❌ OlympiadLearningSwarm Failed: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "failed", "swarm": "OlympiadLearningSwarm", "error": str(e)}


async def main():
    """Run all swarm tests."""
    print("\n" + "=" * 80)
    print("🚀 SWARM TEMPLATE TESTING WITH REAL LLM")
    print("=" * 80)
    print(f"API Key: {api_key[:20]}...")
    print(f"Provider: Anthropic Claude")

    results = []

    # Test swarms sequentially
    results.append(await test_research_swarm())
    results.append(await test_arxiv_swarm())
    results.append(await test_olympiad_swarm())

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)

    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"{status_emoji} {result['swarm']}: {result['status']}")
        if result["status"] == "failed":
            print(f"   Error: {result.get('error', 'Unknown')[:100]}")

    print(
        f"\n🎯 Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)"
    )

    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
