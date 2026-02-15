"""
Test Research Template - Real LLM Integration Test

This test demonstrates the ResearchTemplate with actual LLM execution.
It validates the 4-phase workflow with CUSTOM pattern and STAGES.

Run with: pytest tests/test_research_template.py -v -s
"""

import asyncio
import logging
import os
from pathlib import Path

import pytest

from Jotty.core.intelligence.swarms.research_swarm.types import ResearchConfig
from Jotty.core.intelligence.swarms.templates.research import ResearchTemplate

# Setup logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.fixture
def research_config():
    """Create test configuration."""
    return ResearchConfig(
        exchange="US",
        use_llm_analysis=True,
        include_sentiment=True,
        include_peers=False,  # Skip to speed up test
        include_charts=False,  # Skip to speed up test
        include_technical=False,  # Skip to speed up test
        use_screener=False,  # Skip (works for Indian stocks only)
        include_social_sentiment=False,  # Skip to speed up test
        send_telegram=False,  # Don't send during test
        output_dir="/tmp/jotty_research_test",
        max_web_results=10,  # Reduce for faster test
        parallel_fetch=True,  # Test parallel execution
    )


@pytest.fixture
def output_dir():
    """Create and return test output directory."""
    out_dir = Path("/tmp/jotty_research_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_research_template_instantiation(research_config):
    """Test that ResearchTemplate can be instantiated."""
    template = ResearchTemplate(research_config)

    assert template is not None
    assert template.config == research_config
    assert template.TASK_TYPE == "research"
    assert len(template.DEFAULT_TOOLS) == 10
    assert template.AGENT_TEAM is not None
    assert len(template.STAGES) == 4

    logger.info("✅ ResearchTemplate instantiation successful")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_research_template_stages_validation(research_config):
    """Test that STAGES are properly configured."""
    template = ResearchTemplate(research_config)

    # Validate stage structure
    assert len(template.STAGES) == 4

    # Phase 1: Data collection (parallel)
    stage1 = template.STAGES[0]
    assert stage1.name == "data_collection"
    assert stage1.parallel is True
    assert len(stage1.agents) == 4
    assert stage1.needs == []

    # Phase 2: Analysis (parallel, depends on data collection)
    stage2 = template.STAGES[1]
    assert stage2.name == "analysis"
    assert stage2.parallel is True
    assert len(stage2.agents) == 4
    assert "data_collection" in stage2.needs

    # Phase 3: Charts (sequential, depends on data collection)
    stage3 = template.STAGES[2]
    assert stage3.name == "charts"
    assert stage3.parallel is False
    assert len(stage3.agents) == 1
    assert "data_collection" in stage3.needs

    # Phase 4: Report (sequential, depends on analysis and charts)
    stage4 = template.STAGES[3]
    assert stage4.name == "report"
    assert stage4.parallel is False
    assert len(stage4.agents) == 1
    assert "analysis" in stage4.needs
    assert "charts" in stage4.needs

    logger.info("✅ STAGES validation successful")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="Requires LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
)
async def test_research_template_execution_real_llm(research_config, output_dir):
    """
    Test ResearchTemplate with real LLM execution.

    This is the MAIN TEST that demonstrates the complete workflow.
    Uses a well-known stock (AAPL) to ensure data availability.
    """
    logger.info("=" * 80)
    logger.info("Starting ResearchTemplate Real LLM Test")
    logger.info("=" * 80)

    # Create template
    template = ResearchTemplate(research_config)

    # Execute research on Apple (well-known stock with lots of data)
    ticker = "AAPL"
    logger.info(f"\n🔍 Executing research for {ticker}...")

    try:
        result = await template.execute(
            query=ticker,
            ticker=ticker,
            exchange="US",
            send_telegram=False,
        )

        # Validate result structure
        assert result is not None
        assert (
            result.success is True
        ), f"Research failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        assert result.ticker == ticker
        assert result.company_name is not None

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("RESEARCH RESULTS")
        logger.info("=" * 80)
        logger.info(f"Ticker: {result.ticker}")
        logger.info(f"Company: {result.company_name}")
        logger.info(f"Rating: {result.rating} (confidence: {result.rating_confidence:.2%})")
        logger.info(f"Price: ${result.current_price:.2f} → Target: ${result.target_price:.2f}")
        logger.info(f"Sentiment: {result.sentiment_label} (score: {result.sentiment_score:+.2f})")
        logger.info(f"Execution Time: {result.execution_time:.2f}s")
        logger.info(f"News Count: {result.news_count}")

        if result.investment_thesis:
            logger.info("\n📊 Investment Thesis:")
            for i, point in enumerate(result.investment_thesis[:3], 1):
                logger.info(f"  {i}. {point}")

        if result.key_risks:
            logger.info("\n⚠️  Key Risks:")
            for i, risk in enumerate(result.key_risks[:3], 1):
                logger.info(f"  {i}. {risk}")

        if result.agent_contributions:
            logger.info("\n🤖 Agent Contributions:")
            for agent, contribution in result.agent_contributions.items():
                if contribution > 0:
                    logger.info(f"  {agent}: {contribution:.1%}")

        logger.info("=" * 80)

        # Validate required fields
        assert result.rating in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
        assert 0 <= result.rating_confidence <= 1
        assert result.sentiment_label in ["POSITIVE", "NEUTRAL", "NEGATIVE", "BULLISH", "BEARISH"]
        assert -1 <= result.sentiment_score <= 1
        assert result.execution_time > 0

        # Validate that at least some data was fetched
        assert result.current_price > 0, "Current price should be fetched"
        assert result.news_count > 0, "Should have fetched some news"

        logger.info("\n✅ All validations passed!")

        return result

    except Exception as e:
        logger.error(f"\n❌ Research execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@pytest.mark.asyncio
@pytest.mark.integration
async def test_research_template_error_handling(research_config):
    """Test error handling with invalid ticker."""
    template = ResearchTemplate(research_config)

    # Use an obviously invalid ticker
    result = await template.execute(
        query="INVALID_TICKER_XYZ123",
        ticker="INVALID_TICKER_XYZ123",
        exchange="US",
        send_telegram=False,
    )

    # Should still return a result (even if with limited data)
    assert result is not None
    assert result.ticker == "INVALID_TICKER_XYZ123"

    logger.info("✅ Error handling test passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_research_template_ticker_extraction():
    """Test ticker extraction from query."""
    template = ResearchTemplate()

    # Test various query formats
    assert template._extract_ticker("AAPL") == "AAPL"
    assert template._extract_ticker("Research AAPL stock") == "AAPL"
    assert template._extract_ticker("Latest news for TSLA") == "TSLA"
    assert template._extract_ticker("Analyze MSFT company") == "MSFT"

    logger.info("✅ Ticker extraction test passed")


def test_research_template_backward_compatibility():
    """Test backward compatibility alias."""
    from Jotty.core.intelligence.swarms.templates.research import ResearchSwarm

    assert ResearchSwarm is ResearchTemplate

    logger.info("✅ Backward compatibility test passed")


if __name__ == "__main__":
    """Run tests directly with real LLM."""
    print("\n" + "=" * 80)
    print("RESEARCH TEMPLATE - REAL LLM TEST")
    print("=" * 80)

    # Setup
    config = ResearchConfig(
        exchange="US",
        use_llm_analysis=True,
        include_sentiment=True,
        include_peers=False,
        include_charts=False,
        include_technical=False,
        use_screener=False,
        include_social_sentiment=False,
        send_telegram=False,
        output_dir="/tmp/jotty_research_test",
        max_web_results=10,
        parallel_fetch=True,
    )

    output_dir = Path("/tmp/jotty_research_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run main test
    print("\n📋 Running: test_research_template_execution_real_llm")
    result = asyncio.run(test_research_template_execution_real_llm(config, str(output_dir)))

    print("\n✅ All tests completed successfully!")
    print("=" * 80)
