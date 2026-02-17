"""
Research Template - Multi-phase stock research with parallel data collection and analysis.

Demonstrates CUSTOM pattern with STAGES for complex, multi-phase workflows.

Architecture:
    Phase 1: Parallel Data Collection (DataFetcher + WebSearch + Screener + Technical)
    Phase 2: Parallel Analysis (Sentiment + SocialSentiment + Peers + LLM)
    Phase 3: Chart Generation (sequential)
    Phase 4: Report Generation (sequential)

Example:
    from Jotty.core.intelligence.orchestration.swarms.templates.research import ResearchTemplate

    swarm = ResearchTemplate()
    result = await swarm.execute("AAPL")
"""

from typing import Any, Dict, Optional

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

from .._base.stage_config import StageConfig
from ..base import SwarmTemplate, TeamCoordinator
from ..research_swarm.agents import (
    DataFetcherAgent,
    EnhancedChartGeneratorAgent,
    LLMAnalysisAgent,
    PeerComparisonAgent,
    ReportGeneratorAgent,
    ScreenerAgent,
    SentimentAgent,
    SocialSentimentAgent,
    TechnicalAnalysisAgent,
    WebSearchAgent,
)
from ..research_swarm.types import ResearchConfig, ResearchResult


class ResearchTemplate(SwarmTemplate):
    """
    Stock research template with 4-phase workflow.

    Uses CUSTOM coordination pattern with STAGES to orchestrate:
    1. Parallel data fetching
    2. Parallel analysis
    3. Chart generation
    4. Report generation

    All 10 agents work together with proper dependency resolution.
    """

    # Multi-phase workflow definition (define before AGENT_TEAM)
    STAGES = [
        # Phase 1: Parallel data collection
        StageConfig(
            name="data_collection",
            agents=["_data_fetcher", "_web_searcher", "_screener_agent", "_technical_analyzer"],
            parallel=True,
            output_key="raw_data",
        ),
        # Phase 2: Parallel analysis (depends on data collection)
        StageConfig(
            name="analysis",
            agents=[
                "_sentiment_analyzer",
                "_social_sentiment_agent",
                "_peer_comparator",
                "_llm_analyzer",
            ],
            needs=["data_collection"],
            parallel=True,
            output_key="analysis_results",
        ),
        # Phase 3: Chart generation (depends on data collection)
        StageConfig(
            name="charts",
            agents=["_chart_generator"],
            needs=["data_collection"],
            parallel=False,
            output_key="chart_paths",
        ),
        # Phase 4: Report generation (depends on analysis and charts)
        StageConfig(
            name="report",
            agents=["_report_generator"],
            needs=["analysis", "charts"],
            parallel=False,
            output_key="final_report",
        ),
    ]

    # Declarative agent team with CUSTOM pattern + STAGES
    AGENT_TEAM = TeamCoordinator.define(
        (DataFetcherAgent, "DataFetcher", "_data_fetcher"),
        (WebSearchAgent, "WebSearch", "_web_searcher"),
        (SentimentAgent, "Sentiment", "_sentiment_analyzer"),
        (LLMAnalysisAgent, "LLMAnalyzer", "_llm_analyzer"),
        (PeerComparisonAgent, "PeerComparator", "_peer_comparator"),
        (EnhancedChartGeneratorAgent, "ChartGenerator", "_chart_generator"),
        (ReportGeneratorAgent, "ReportGenerator", "_report_generator"),
        (TechnicalAnalysisAgent, "TechnicalAnalyzer", "_technical_analyzer"),
        (ScreenerAgent, "Screener", "_screener_agent"),
        (SocialSentimentAgent, "SocialSentiment", "_social_sentiment_agent"),
        pattern=CoordinationPattern.CUSTOM,
        stages=STAGES,  # Pass STAGES to TeamCoordinator
    )

    TASK_TYPE = "research"
    DEFAULT_TOOLS = [
        "data_fetch",
        "web_search",
        "sentiment_analysis",
        "llm_analysis",
        "peer_comparison",
        "chart_generation",
        "report_generation",
        "technical_analysis",
        "screener",
        "social_sentiment",
    ]

    def __init__(self, config: Optional[ResearchConfig] = None) -> None:
        """Initialize research template."""
        super().__init__(config or ResearchConfig())
        self._initialized = False

    def _init_shared_resources(self) -> None:
        """Initialize shared swarm resources and DSPy modules."""
        if self._initialized:
            return

        # Auto-configure DSPy LM if needed
        import logging

        import dspy

        logger = logging.getLogger(__name__)

        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            try:
                from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                    configure_dspy_lm,
                )

                lm = configure_dspy_lm()
                logger.info("✓ Auto-configured DSPy via UnifiedLMProvider")
            except Exception as e:
                logger.warning(f"Could not configure DSPy LM: {e}")

        # Initialize shared resources
        try:
            from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
            from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import (
                SwarmResources,
            )

            jotty_config = SwarmConfig()
            resources = SwarmResources.get_instance(jotty_config)

            self._memory = resources.memory
            self._context = resources.context
            self._bus = resources.bus

            logger.info("✓ Shared swarm resources initialized")
        except Exception as e:
            logger.warning(f"SwarmResources not available: {e}")

        # Initialize DSPy modules
        if self.config.use_llm_analysis:  # type: ignore[attr-defined]
            from ..research_swarm.signatures import (
                PeerSelectionSignature,
                SentimentAnalysisSignature,
                SocialSentimentSignature,
                StockAnalysisSignature,
                TechnicalSignalsSignature,
            )

            self._stock_analyzer = dspy.ChainOfThought(StockAnalysisSignature)
            self._sentiment_module = dspy.ChainOfThought(SentimentAnalysisSignature)
            self._peer_selector = dspy.Predict(PeerSelectionSignature)
            self._social_sentiment_module = dspy.ChainOfThought(SocialSentimentSignature)
            self._technical_signals_module = dspy.Predict(TechnicalSignalsSignature)

        self._initialized = True

    async def _execute_domain(self, query: str, **kwargs: Any) -> ResearchResult:
        """
        Execute domain-specific research logic (called by SwarmTemplate.execute()).

        Args:
            query: Company name or ticker
            **kwargs: Additional arguments (ticker, exchange, send_telegram, etc.)

        Returns:
            ResearchResult with all data and analysis
        """
        return await self.execute(query, **kwargs)

    async def execute(
        self,
        query: str,
        ticker: Optional[str] = None,
        exchange: Optional[str] = None,
        send_telegram: Optional[bool] = None,
        **kwargs: Any,
    ) -> ResearchResult:
        """
        Execute comprehensive stock research.

        Args:
            query: Company name or ticker
            ticker: Explicit ticker symbol
            exchange: Exchange (NSE, BSE, NYSE)
            send_telegram: Override Telegram setting

        Returns:
            ResearchResult with all data and analysis
        """
        # Initialize resources
        self._init_shared_resources()

        # Parse inputs
        ticker = ticker or self._extract_ticker(query)
        exchange = exchange or getattr(self.config, "exchange", "NSE")
        send_tg = (
            send_telegram
            if send_telegram is not None
            else getattr(self.config, "send_telegram", True)
        )

        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Research Template starting for {ticker} ({exchange})")
        logger.info(
            f"   Config: LLM={self.config.use_llm_analysis}, "  # type: ignore[attr-defined]
            f"Peers={self.config.include_peers}, "
            f"Sentiment={self.config.include_sentiment}"
        )

        # Prepare context for stage execution
        context = {
            "query": query,
            "ticker": ticker,
            "exchange": exchange,
            "send_telegram": send_tg,
            "config": self.config,
        }

        # Execute using CUSTOM pattern with STAGES
        # SwarmTemplate.execute_team will handle the multi-phase workflow
        result = await self.execute_team(
            task=f"Research {ticker}",
            context=context,
            tools_used=self.DEFAULT_TOOLS,
        )

        # Convert team result to ResearchResult
        return self._build_research_result(ticker, result, context)

    def _extract_ticker(self, query: str) -> str:
        """Extract ticker from query."""
        import re

        query_upper = query.upper().strip()
        words = query_upper.split()

        skip_words = {
            "RESEARCH",
            "STOCK",
            "COMPANY",
            "ANALYZE",
            "ANALYSIS",
            "LATEST",
            "NEWS",
            "PDF",
            "TELEGRAM",
            "SEND",
            "TO",
            "THE",
            "AND",
            "WITH",
            "FOR",
            "ABOUT",
            "GET",
            "SHOW",
            "FIND",
        }

        for word in words:
            clean = re.sub(r"[,.\-:]", "", word)
            if clean and clean not in skip_words and len(clean) <= 15:
                return clean

        return words[0] if words else "UNKNOWN"

    def _build_research_result(
        self,
        ticker: str,
        team_result: Any,  # TeamResult object
        context: Dict[str, Any],
    ) -> ResearchResult:
        """
        Build ResearchResult from team execution results.

        Args:
            ticker: Stock ticker
            team_result: TeamResult from execute_team
            context: Execution context

        Returns:
            ResearchResult with all data
        """
        # Extract agent outputs from TeamResult
        outputs = team_result.outputs if hasattr(team_result, "outputs") else {}

        # DEBUG
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"DEBUG all output keys: {list(outputs.keys())}")
        logger.info(f"DEBUG _data_fetcher value: {outputs.get('_data_fetcher')}")

        # Merge data from agents
        financial_data = outputs.get("_data_fetcher", {}) or {}
        web_data = outputs.get("_web_searcher", {}) or {}
        screener_data = outputs.get("_screener_agent", {}) or {}
        technical_data = outputs.get("_technical_analyzer", {}) or {}

        # DEBUG
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"DEBUG financial_data keys: {list(financial_data.keys()) if financial_data else 'None'}"
        )
        logger.info(f"DEBUG current_price: {financial_data.get('current_price', 'N/A')}")

        # Merge analysis results
        sentiment_result = outputs.get("_sentiment_analyzer", {}) or {}
        social_sentiment_result = outputs.get("_social_sentiment_agent", {}) or {}
        peer_result = outputs.get("_peer_comparator", {}) or {}
        llm_result = outputs.get("_llm_analyzer", {}) or {}

        # Chart and report outputs
        chart_data = outputs.get("_chart_generator", {}) or {}
        chart_paths = chart_data.get("chart_paths", []) if isinstance(chart_data, dict) else []
        final_report = outputs.get("_report_generator", {}) or {}

        # Build final result
        return ResearchResult(
            success=True,
            ticker=ticker,
            company_name=financial_data.get("company_name", ticker),
            current_price=financial_data.get("current_price", 0),
            target_price=financial_data.get("target_mean_price", 0),
            rating=llm_result.get("rating", "HOLD"),
            rating_confidence=llm_result.get("confidence", 0.5),
            investment_thesis=llm_result.get("thesis", []),
            key_risks=llm_result.get("risks", []),
            sentiment_score=sentiment_result.get("sentiment_score", 0),
            sentiment_label=sentiment_result.get("sentiment_label", "NEUTRAL"),
            peers=peer_result.get("peers", []),
            peer_comparison=peer_result.get("comparison", {}),
            pdf_path=final_report.get("pdf_path", ""),
            md_path=final_report.get("md_path", ""),
            chart_paths=chart_paths,
            telegram_sent=final_report.get("telegram_sent", False),
            data_sources=financial_data.get("sources", []),
            news_count=web_data.get("news_count", 0),
            execution_time=(
                getattr(team_result, "execution_time", 0)
                if hasattr(team_result, "execution_time")
                else 0
            ),
            agent_contributions={},  # Will be calculated from TeamResult metadata
            technical_signals=technical_data.get("signals", {}),
            support_levels=technical_data.get("support_levels", []),
            resistance_levels=technical_data.get("resistance_levels", []),
            trend=technical_data.get("trend", "NEUTRAL"),
            screener_data=screener_data,
            social_sentiment_score=social_sentiment_result.get("overall_sentiment", 0),
            sentiment_drivers=social_sentiment_result.get("sentiment_drivers", {}),
        )


# Convenience function
async def research(query: str, **kwargs: Any) -> ResearchResult:
    """
    One-liner research function.

    Example:
        from Jotty.core.intelligence.orchestration.swarms.templates.research import research
        result = await research("AAPL", send_telegram=True)
    """
    template = ResearchTemplate()
    return await template.execute(query, **kwargs)


# Backward compatibility
ResearchSwarm = ResearchTemplate
__all__ = ["ResearchTemplate", "ResearchSwarm", "research"]
