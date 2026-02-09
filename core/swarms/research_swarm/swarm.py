"""Research Swarm - Main swarm orchestrator."""

import asyncio
import logging
import os
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import dspy

from ..base import DomainSwarm, AgentTeam

from .types import RatingType, ResearchConfig, ResearchResult
from .agents import (
    BaseResearchAgent, DataFetcherAgent, WebSearchAgent,
    SentimentAgent, LLMAnalysisAgent, PeerComparisonAgent,
    ChartGeneratorAgent, TechnicalAnalysisAgent,
    EnhancedChartGeneratorAgent, ScreenerAgent,
    SocialSentimentAgent, ReportGeneratorAgent,
)

logger = logging.getLogger(__name__)

class ResearchSwarm(DomainSwarm):
    """
    World-Class Research Swarm with parallel agents and LLM analysis.

    Agent Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     ResearchSwarm                            │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │DataFetcher  │  │ WebSearch   │  │  Screener   │         │
    │  │   Agent     │  │   Agent     │  │   Agent     │         │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
    │         │                │                │                 │
    │         └────────────────┼────────────────┘                 │
    │                          ▼                                  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │ Sentiment   │  │   LLM       │  │    Peer     │         │
    │  │   Agent     │  │  Analyzer   │  │  Comparator │         │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
    │         │                │                │                 │
    │         └────────────────┼────────────────┘                 │
    │                          ▼                                  │
    │  ┌─────────────┐  ┌─────────────┐                          │
    │  │   Chart     │  │   Report    │──────► PDF + Telegram    │
    │  │  Generator  │  │  Generator  │                          │
    │  └─────────────┘  └─────────────┘                          │
    └─────────────────────────────────────────────────────────────┘
    """

    # Declarative agent team - defined after agent classes at end of file
    AGENT_TEAM = None  # Set after agent class definitions

    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize research swarm."""
        super().__init__(config or ResearchConfig())

        # DSPy modules
        self._stock_analyzer = None
        self._sentiment_module = None
        self._peer_selector = None
        # New DSPy modules
        self._social_sentiment_module = None
        self._technical_signals_module = None

    def _init_shared_resources(self):
        """Initialize shared swarm resources."""
        if self._initialized:
            return

        # Auto-configure DSPy LM if needed
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                lm = DirectClaudeCLI()  # model resolved from config_defaults
                dspy.configure(lm=lm)
                logger.info("🔧 Auto-configured DSPy with DirectClaudeCLI")
            except Exception as e:
                logger.warning(f"Could not configure DSPy LM: {e}")

        # Initialize shared resources
        try:
            from ..agents.dag_agents import SwarmResources
            from ..foundation.data_structures import JottyConfig

            jotty_config = JottyConfig()
            resources = SwarmResources.get_instance(jotty_config)

            self._memory = resources.memory
            self._context = resources.context
            self._bus = resources.bus

            logger.info("✅ Shared swarm resources initialized")
        except Exception as e:
            logger.warning(f"SwarmResources not available: {e}")

        # Initialize DSPy modules
        if self.config.use_llm_analysis:
            self._stock_analyzer = dspy.ChainOfThought(StockAnalysisSignature)
            self._sentiment_module = dspy.ChainOfThought(SentimentAnalysisSignature)
            self._peer_selector = dspy.Predict(PeerSelectionSignature)
            self._social_sentiment_module = dspy.ChainOfThought(SocialSentimentSignature)
            self._technical_signals_module = dspy.Predict(TechnicalSignalsSignature)

        self._initialized = True

    async def _execute_domain(
        self,
        query: str,
        **kwargs
    ) -> ResearchResult:
        """Execute research (called by DomainSwarm.execute())."""
        return await self.research(query, **kwargs)

    async def research(
        self,
        query: str,
        ticker: Optional[str] = None,
        exchange: Optional[str] = None,
        send_telegram: Optional[bool] = None
    ) -> ResearchResult:
        """
        Execute comprehensive research on a company/stock.

        Args:
            query: Company name or ticker
            ticker: Explicit ticker symbol
            exchange: Exchange (NSE, BSE, NYSE)
            send_telegram: Override Telegram setting

        Returns:
            ResearchResult with all data and analysis
        """
        start_time = datetime.now()

        # Note: Pre-execution learning and agent init handled by DomainSwarm.execute()

        # Parse inputs
        ticker = ticker or self._extract_ticker(query)
        exchange = exchange or self.config.exchange
        send_tg = send_telegram if send_telegram is not None else self.config.send_telegram

        logger.info(f"🚀 Research Swarm starting for {ticker} ({exchange})")
        logger.info(f"   Config: LLM={self.config.use_llm_analysis}, Peers={self.config.include_peers}, Sentiment={self.config.include_sentiment}")

        try:
            # =================================================================
            # PHASE 1: PARALLEL DATA COLLECTION
            # =================================================================
            logger.info("📥 Phase 1: Parallel data collection...")

            if self.config.parallel_fetch:
                # Run data fetching in parallel - now includes Screener and Technical Analysis
                fetch_tasks = [
                    self._data_fetcher.fetch(ticker, exchange),
                    self._web_searcher.search(ticker, self.config.max_web_results)
                ]

                # Add Screener.in fetching if enabled
                if self.config.use_screener:
                    fetch_tasks.append(self._screener_agent.fetch(ticker))

                # Add Technical Analysis if enabled
                if self.config.include_technical:
                    fetch_tasks.append(
                        self._technical_analyzer.analyze(ticker, self.config.technical_timeframes)
                    )

                fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                financial_data = fetch_results[0] if not isinstance(fetch_results[0], Exception) else {}
                web_data = fetch_results[1] if not isinstance(fetch_results[1], Exception) else {}

                # Extract Screener data if fetched
                screener_data = {}
                if self.config.use_screener and len(fetch_results) > 2:
                    screener_data = fetch_results[2] if not isinstance(fetch_results[2], Exception) else {}

                # Extract Technical data if fetched
                technical_data = {}
                tech_idx = 3 if self.config.use_screener else 2
                if self.config.include_technical and len(fetch_results) > tech_idx:
                    technical_data = fetch_results[tech_idx] if not isinstance(fetch_results[tech_idx], Exception) else {}
            else:
                financial_data = await self._data_fetcher.fetch(ticker, exchange)
                web_data = await self._web_searcher.search(ticker, self.config.max_web_results)
                screener_data = await self._screener_agent.fetch(ticker) if self.config.use_screener else {}
                technical_data = await self._technical_analyzer.analyze(
                    ticker, self.config.technical_timeframes
                ) if self.config.include_technical else {}

            # Merge data
            merged_data = {**financial_data, **web_data}
            merged_data['ticker'] = ticker
            merged_data['exchange'] = exchange
            merged_data['screener_data'] = screener_data
            merged_data['technical_data'] = technical_data

            # Store in shared context
            if self._context:
                self._context.set(f"research:{ticker}:raw_data", merged_data)

            # =================================================================
            # PHASE 2: PARALLEL ANALYSIS
            # =================================================================
            logger.info("🧠 Phase 2: Parallel analysis...")

            analysis_tasks = []

            # Helper for creating async results (Python 3.11+ compatible)
            async def async_result(value):
                return value

            # Sentiment analysis (if enabled)
            if self.config.include_sentiment and web_data.get('news_text'):
                analysis_tasks.append(
                    self._sentiment_analyzer.analyze(
                        merged_data.get('company_name', ticker),
                        web_data.get('news_text', '')
                    )
                )
            else:
                analysis_tasks.append(async_result({'sentiment_score': 0, 'sentiment_label': 'NEUTRAL'}))

            # Social Sentiment analysis (if enabled) - NEW
            if self.config.include_social_sentiment and web_data.get('news_text'):
                analysis_tasks.append(
                    self._social_sentiment_agent.analyze(
                        merged_data.get('company_name', ticker),
                        web_data.get('news_text', ''),
                        ''  # Forum text - can be extended to scrape forums
                    )
                )
            else:
                analysis_tasks.append(async_result({'overall_sentiment': 0, 'sentiment_label': 'NEUTRAL', 'sentiment_drivers': {}}))

            # Peer comparison (if enabled)
            if self.config.include_peers:
                analysis_tasks.append(
                    self._peer_comparator.compare(
                        ticker,
                        merged_data.get('sector', ''),
                        merged_data.get('industry', ''),
                        exchange
                    )
                )
            else:
                analysis_tasks.append(async_result({'peers': [], 'comparison': {}}))

            # LLM analysis
            if self.config.use_llm_analysis:
                analysis_tasks.append(
                    self._llm_analyzer.analyze(merged_data, web_data.get('news_text', ''))
                )
            else:
                analysis_tasks.append(async_result(self._rule_based_analysis(merged_data)))

            # Run analysis in parallel
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            sentiment_result = analysis_results[0] if not isinstance(analysis_results[0], Exception) else {}
            social_sentiment_result = analysis_results[1] if not isinstance(analysis_results[1], Exception) else {}
            peer_result = analysis_results[2] if not isinstance(analysis_results[2], Exception) else {}
            llm_result = analysis_results[3] if not isinstance(analysis_results[3], Exception) else {}

            # =================================================================
            # PHASE 3: CHART GENERATION (if enabled) - ENHANCED
            # =================================================================
            chart_paths = []
            if self.config.include_charts:
                logger.info("📊 Phase 3: Generating enhanced multi-panel charts...")
                # Use technical data for multi-timeframe charts if available
                chart_result = await self._chart_generator.generate(
                    ticker,
                    merged_data,
                    self.config.output_dir,
                    timeframes=self.config.technical_timeframes,
                    technical_data=technical_data,
                    include_heiken_ashi=self.config.include_heiken_ashi
                )
                chart_paths = chart_result.get('chart_paths', [])

            # =================================================================
            # PHASE 4: REPORT GENERATION + OUTPUT
            # =================================================================
            logger.info("📝 Phase 4: Generating report...")

            report_result = await self._report_generator.generate(
                ticker=ticker,
                data=merged_data,
                analysis=llm_result,
                sentiment=sentiment_result,
                peers=peer_result,
                chart_paths=chart_paths,
                output_dir=self.config.output_dir,
                send_telegram=send_tg
            )

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            # Parse technical signals for support/resistance
            support_levels = []
            resistance_levels = []
            trend = "NEUTRAL"
            if technical_data:
                support_levels = technical_data.get('support_levels', [])
                resistance_levels = technical_data.get('resistance_levels', [])
                trend = technical_data.get('trend', 'NEUTRAL')

            result = ResearchResult(
                success=True,
                ticker=ticker,
                company_name=merged_data.get('company_name', ticker),
                current_price=merged_data.get('current_price', 0),
                target_price=merged_data.get('target_mean_price', 0),
                rating=llm_result.get('rating', 'HOLD'),
                rating_confidence=llm_result.get('confidence', 0.5),
                investment_thesis=llm_result.get('thesis', []),
                key_risks=llm_result.get('risks', []),
                sentiment_score=sentiment_result.get('sentiment_score', 0),
                sentiment_label=sentiment_result.get('sentiment_label', 'NEUTRAL'),
                peers=peer_result.get('peers', []),
                peer_comparison=peer_result.get('comparison', {}),
                pdf_path=report_result.get('pdf_path', ''),
                md_path=report_result.get('md_path', ''),
                chart_paths=chart_paths,
                telegram_sent=report_result.get('telegram_sent', False),
                data_sources=merged_data.get('sources', []),
                news_count=web_data.get('news_count', 0),
                execution_time=exec_time,
                agent_contributions={
                    'DataFetcher': 0.15,
                    'WebSearch': 0.10,
                    'Sentiment': 0.05 if self.config.include_sentiment else 0,
                    'SocialSentiment': 0.05 if self.config.include_social_sentiment else 0,
                    'LLMAnalysis': 0.20 if self.config.use_llm_analysis else 0,
                    'PeerComparison': 0.08 if self.config.include_peers else 0,
                    'TechnicalAnalysis': 0.15 if self.config.include_technical else 0,
                    'Screener': 0.07 if self.config.use_screener else 0,
                    'ChartGenerator': 0.05 if self.config.include_charts else 0,
                    'ReportGenerator': 0.10
                },
                # New enhanced fields
                technical_signals=technical_data.get('signals', {}),
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend=trend,
                screener_data=screener_data,
                social_sentiment_score=social_sentiment_result.get('overall_sentiment', 0),
                sentiment_drivers=social_sentiment_result.get('sentiment_drivers', {})
            )

            # Store in memory for learning
            if self.config.learn_from_research and self._memory:
                await self._store_learning(result)

            logger.info(f"✅ Research complete: {ticker} = {result.rating} (₹{result.current_price:,.2f})")
            logger.info(f"   Sentiment: {result.sentiment_label} ({result.sentiment_score:+.2f})")
            logger.info(f"   Time: {exec_time:.1f}s | Telegram: {result.telegram_sent}")

            return result

        except Exception as e:
            logger.error(f"❌ Research swarm error: {e}")
            import traceback
            traceback.print_exc()
            return ResearchResult(
                success=False,
                ticker=ticker,
                company_name=ticker,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _extract_ticker(self, query: str) -> str:
        """Extract ticker from query."""
        query_upper = query.upper().strip()
        words = query_upper.split()

        skip_words = {'RESEARCH', 'STOCK', 'COMPANY', 'ANALYZE', 'ANALYSIS',
                      'LATEST', 'NEWS', 'PDF', 'TELEGRAM', 'SEND', 'TO', 'THE',
                      'AND', 'WITH', 'FOR', 'ABOUT', 'GET', 'SHOW', 'FIND'}

        for word in words:
            clean = re.sub(r'[,.\-:]', '', word)
            if clean and clean not in skip_words and len(clean) <= 15:
                return clean

        return words[0] if words else "UNKNOWN"

    def _rule_based_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based analysis when LLM not available."""
        score = 0

        # Upside potential
        current = data.get('current_price', 0)
        target = data.get('target_mean_price', 0)
        if target and current:
            upside = (target - current) / current * 100
            if upside > 20: score += 30
            elif upside > 10: score += 20
            elif upside > 0: score += 10

        # Growth
        growth = data.get('revenue_growth', 0)
        if growth > 20: score += 25
        elif growth > 10: score += 15
        elif growth > 0: score += 5

        # Profitability
        margin = data.get('profit_margin', 0)
        if margin > 15: score += 25
        elif margin > 10: score += 15
        elif margin > 0: score += 5

        # Rating
        if score >= 60: rating = "BUY"
        elif score >= 40: rating = "HOLD"
        else: rating = "SELL"

        return {
            'rating': rating,
            'confidence': min(score / 100, 1.0),
            'thesis': ["Based on quantitative analysis"],
            'risks': ["Market volatility", "Sector headwinds"],
            'reasoning': f"Score-based rating: {score}/100"
        }

    async def _store_learning(self, result: ResearchResult):
        """Store research results in memory for learning."""
        try:
            from ..foundation.data_structures import MemoryLevel

            content = f"""Research: {result.ticker}
Rating: {result.rating} (confidence: {result.rating_confidence:.2f})
Price: ₹{result.current_price:,.2f} → Target: ₹{result.target_price:,.2f}
Sentiment: {result.sentiment_label} ({result.sentiment_score:+.2f})
Execution: {result.execution_time:.1f}s
Success: {result.success}"""

            self._memory.store(
                content=content,
                level=MemoryLevel.EPISODIC,
                context={
                    'ticker': result.ticker,
                    'rating': result.rating,
                    'sentiment': result.sentiment_label,
                    'success': result.success
                },
                goal=f"Research {result.ticker}"
            )
        except Exception as e:
            logger.debug(f"Memory store failed: {e}")


# =============================================================================
# AGENTS
# =============================================================================



async def research(query: str, **kwargs) -> ResearchResult:
    """
    One-liner research function.

    Usage:
        from core.swarms import research
        result = await research("Paytm", send_telegram=True)
    """
    swarm = ResearchSwarm()
    return await swarm.research(query, **kwargs)


def research_sync(query: str, **kwargs) -> ResearchResult:
    """
    Synchronous research function.

    Usage:
        from core.swarms import research_sync
        result = research_sync("Paytm")
    """
    return asyncio.run(research(query, **kwargs))


# =============================================================================
# AGENT_TEAM ASSIGNMENT (after all agent classes defined)
# =============================================================================

# Set AGENT_TEAM now that all agent classes are defined
ResearchSwarm.AGENT_TEAM = AgentTeam.define(
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
)
