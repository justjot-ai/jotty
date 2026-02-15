#!/usr/bin/env python
"""
Standalone demo of ResearchTemplate with real LLM.

Usage: python demo_research_template.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    print("❌ No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    sys.exit(1)

print("✓ API key loaded")

from Jotty.core.intelligence.swarms.research_swarm.types import ResearchConfig

# Import ResearchTemplate
from Jotty.core.intelligence.swarms.templates.research import ResearchTemplate

print("✓ Imports successful")


async def main():
    """Run ResearchTemplate demonstration."""
    print("\n" + "=" * 80)
    print("RESEARCH TEMPLATE DEMONSTRATION")
    print("=" * 80)

    # Create configuration
    config = ResearchConfig(
        exchange="US",
        use_llm_analysis=True,
        include_sentiment=True,
        include_peers=False,  # Skip to speed up
        include_charts=False,  # Skip to speed up
        include_technical=False,  # Skip to speed up
        use_screener=False,  # Skip (Indian stocks only)
        include_social_sentiment=False,  # Skip to speed up
        send_telegram=False,
        output_dir="/tmp/jotty_research_demo",
        max_web_results=10,
        parallel_fetch=True,
    )

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Create template
    print("\n📋 Creating ResearchTemplate...")
    template = ResearchTemplate(config)
    print(f"✓ Template created with {len(template.STAGES)} stages")

    # Execute research on Apple
    ticker = "AAPL"
    print(f"\n🔍 Executing research for {ticker}...")
    print(f"   Config: LLM={config.use_llm_analysis}, Sentiment={config.include_sentiment}")

    try:
        result = await template.execute(
            query=ticker,
            ticker=ticker,
            exchange="US",
            send_telegram=False,
        )

        # Display results
        print("\n" + "=" * 80)
        print("RESEARCH RESULTS")
        print("=" * 80)
        print(f"✓ Success: {result.success}")
        print(f"📊 Ticker: {result.ticker}")
        print(f"🏢 Company: {result.company_name}")
        print(f"⭐ Rating: {result.rating} (confidence: {result.rating_confidence:.2%})")
        print(f"💰 Price: ${result.current_price:.2f} → Target: ${result.target_price:.2f}")
        print(f"😊 Sentiment: {result.sentiment_label} (score: {result.sentiment_score:+.2f})")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        print(f"📰 News Count: {result.news_count}")

        if result.investment_thesis:
            print("\n📊 Investment Thesis:")
            for i, point in enumerate(result.investment_thesis[:3], 1):
                print(f"  {i}. {point}")

        if result.key_risks:
            print("\n⚠️  Key Risks:")
            for i, risk in enumerate(result.key_risks[:3], 1):
                print(f"  {i}. {risk}")

        print("\n" + "=" * 80)
        print("✅ DEMONSTRATION COMPLETE!")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Research execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
