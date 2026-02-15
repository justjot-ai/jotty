"""
Test ResearchTemplate with real LLM - Quick version.

This tests the actual LLM-powered research workflow (simplified for speed).
"""

import asyncio
import os
from core.intelligence.swarms.templates.research import ResearchTemplate
from core.intelligence.swarms.research_swarm.types import ResearchConfig

# Requires ANTHROPIC_API_KEY env var
assert os.getenv("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY to run real LLM tests"


async def test_research_template_quick():
    """Quick test with minimal features enabled."""
    print("=" * 80)
    print("RESEARCH TEMPLATE - REAL LLM TEST (Quick)")
    print("=" * 80)

    # Minimal config for fast test
    config = ResearchConfig(
        exchange="US",
        use_llm_analysis=True,
        include_sentiment=True,
        include_peers=False,  # Skip to speed up
        include_charts=False,  # Skip to speed up
        include_technical=False,  # Skip to speed up
        use_screener=False,  # Skip (Indian stocks only)
        include_social_sentiment=False,  # Skip to speed up
        send_telegram=False,  # Don't send
        output_dir="/tmp/jotty_test",
        max_web_results=5,  # Reduce for speed
        parallel_fetch=True,
    )

    template = ResearchTemplate(config)

    print(f"\n🔍 Testing with ticker: AAPL (Apple Inc.)")
    print(f"⚡ Quick mode: Minimal features for fast execution\n")

    try:
        result = await template.execute(
            query="AAPL",
            ticker="AAPL",
            exchange="US",
        )

        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"✅ Success: {result.success}")
        print(f"✅ Ticker: {result.ticker}")
        print(f"✅ Company: {result.company_name}")
        print(f"✅ Current Price: ${result.current_price:.2f}")
        print(f"✅ Target Price: ${result.target_price:.2f}")
        print(f"✅ Rating: {result.rating} (confidence: {result.rating_confidence:.1%})")
        print(f"✅ Sentiment: {result.sentiment_label} (score: {result.sentiment_score:+.2f})")
        print(f"✅ News Count: {result.news_count}")
        print(f"✅ Execution Time: {result.execution_time:.2f}s")

        if result.investment_thesis:
            print(f"\n📊 Investment Thesis ({len(result.investment_thesis)} points):")
            for i, point in enumerate(result.investment_thesis[:3], 1):
                print(f"  {i}. {point}")

        if result.key_risks:
            print(f"\n⚠️  Key Risks ({len(result.key_risks)} points):")
            for i, risk in enumerate(result.key_risks[:3], 1):
                print(f"  {i}. {risk}")

        print("\n" + "=" * 80)
        print("✅ RESEARCH TEMPLATE TEST PASSED!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_research_template_quick())
    exit(0 if result else 1)
