"""
World-Class Research Swarm
==========================

Production-grade multi-agent swarm for comprehensive stock research with:
- LLM-powered analysis (DSPy Chain of Thought)
- Multi-source data fusion (Yahoo, Web Search, Screener.in)
- Peer comparison with sector analysis
- Technical analysis with multi-timeframe indicators (pandas_ta)
- Professional multi-panel charts (mplfinance)
- News and social sentiment scoring
- Shared memory for learning
- Parallel agent execution

Usage:
    from core.swarms import ResearchSwarm

    swarm = ResearchSwarm()
    result = await swarm.research("Paytm", send_telegram=True)
"""

import asyncio
import logging
import os
import dspy
import glob
import gzip
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class RatingType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class ResearchConfig:
    """Configuration for research swarm."""
    output_dir: str = field(default_factory=lambda: os.path.expanduser('~/jotty/reports'))
    send_telegram: bool = True
    include_charts: bool = True
    include_peers: bool = True
    include_sentiment: bool = True
    target_pages: int = 12
    max_web_results: int = 25
    max_peers: int = 5
    exchange: str = "NSE"
    use_llm_analysis: bool = True
    parallel_fetch: bool = True
    learn_from_research: bool = True
    # New enhanced options
    include_technical: bool = True
    technical_timeframes: List[str] = field(default_factory=lambda: ["60minute", "Day"])
    use_screener: bool = True
    include_social_sentiment: bool = True
    include_heiken_ashi: bool = False
    nse_data_path: str = "/var/www/sites/personal/stock_market/common/Data/NSE/"


@dataclass
class ResearchResult:
    """Result from research swarm."""
    success: bool
    ticker: str
    company_name: str
    current_price: float = 0.0
    target_price: float = 0.0
    rating: str = "HOLD"
    rating_confidence: float = 0.0
    investment_thesis: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "NEUTRAL"
    peers: List[str] = field(default_factory=list)
    peer_comparison: Dict[str, Any] = field(default_factory=dict)
    pdf_path: str = ""
    md_path: str = ""
    chart_paths: List[str] = field(default_factory=list)
    telegram_sent: bool = False
    data_sources: List[str] = field(default_factory=list)
    news_count: int = 0
    error: str = ""
    execution_time: float = 0.0
    agent_contributions: Dict[str, float] = field(default_factory=dict)
    # New enhanced fields
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend: str = "NEUTRAL"  # BULLISH/NEUTRAL/BEARISH
    screener_data: Dict[str, Any] = field(default_factory=dict)
    social_sentiment_score: float = 0.0
    sentiment_drivers: Dict[str, List[str]] = field(default_factory=dict)


# =============================================================================
# DSPy SIGNATURES FOR LLM-POWERED ANALYSIS
# =============================================================================

class StockAnalysisSignature(dspy.Signature):
    """Analyze stock data and provide investment recommendation.

    You are a senior equity research analyst at a top investment bank.
    Analyze the provided financial data and news to generate a professional rating.
    """
    ticker: str = dspy.InputField(desc="Stock ticker symbol")
    company_name: str = dspy.InputField(desc="Company name")
    financial_data: str = dspy.InputField(desc="JSON financial metrics")
    news_summary: str = dspy.InputField(desc="Recent news and developments")

    rating: str = dspy.OutputField(desc="Rating: STRONG BUY, BUY, HOLD, SELL, or STRONG SELL")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
    thesis: str = dspy.OutputField(desc="3-5 bullet points for investment thesis, separated by |")
    risks: str = dspy.OutputField(desc="3-5 key risks, separated by |")
    reasoning: str = dspy.OutputField(desc="2-3 sentence reasoning for the rating")


class SentimentAnalysisSignature(dspy.Signature):
    """Analyze sentiment from news headlines and content.

    You are a sentiment analysis expert. Score the overall sentiment
    from the provided news about a company.
    """
    company: str = dspy.InputField(desc="Company name")
    news_text: str = dspy.InputField(desc="News headlines and snippets")

    sentiment_score: float = dspy.OutputField(desc="Score from -1.0 (very negative) to 1.0 (very positive)")
    sentiment_label: str = dspy.OutputField(desc="VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, or VERY_POSITIVE")
    key_themes: str = dspy.OutputField(desc="Top 3 themes in news, separated by |")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for sentiment score")


class PeerSelectionSignature(dspy.Signature):
    """Select appropriate peer companies for comparison.

    Given a company's sector and industry, identify the most relevant
    publicly traded peer companies for comparison.
    """
    company: str = dspy.InputField(desc="Company name")
    sector: str = dspy.InputField(desc="Company sector")
    industry: str = dspy.InputField(desc="Company industry")
    exchange: str = dspy.InputField(desc="Stock exchange (NSE, NYSE, etc.)")

    peers: str = dspy.OutputField(desc="5 peer ticker symbols separated by comma (e.g., INFY,TCS,WIPRO,HCLTECH,TECHM)")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for peer selection")


class SocialSentimentSignature(dspy.Signature):
    """Analyze social sentiment from multiple sources.

    You are a sentiment analysis expert analyzing news, forums, and analyst discussions
    about a company to provide comprehensive sentiment analysis.
    """
    company: str = dspy.InputField(desc="Company name")
    news_text: str = dspy.InputField(desc="News headlines and snippets")
    forum_text: str = dspy.InputField(desc="Forum discussions and social media posts")

    overall_sentiment: float = dspy.OutputField(desc="Score from -1.0 (very bearish) to 1.0 (very bullish)")
    sentiment_label: str = dspy.OutputField(desc="BEARISH, NEUTRAL, or BULLISH")
    key_themes: str = dspy.OutputField(desc="Top 3-5 themes discussed, separated by |")
    sentiment_drivers: str = dspy.OutputField(desc="Key positive factors | Key negative factors (separated by ||)")


class TechnicalSignalsSignature(dspy.Signature):
    """Analyze technical indicators to generate trading signals.

    You are a technical analysis expert. Analyze the indicator values and generate
    a comprehensive technical outlook.
    """
    ticker: str = dspy.InputField(desc="Stock ticker symbol")
    indicator_summary: str = dspy.InputField(desc="JSON summary of technical indicators")
    price_data: str = dspy.InputField(desc="Recent price levels and changes")

    trend: str = dspy.OutputField(desc="BULLISH, NEUTRAL, or BEARISH")
    signal_strength: float = dspy.OutputField(desc="Signal strength 0.0-1.0")
    support_levels: str = dspy.OutputField(desc="Key support levels separated by comma")
    resistance_levels: str = dspy.OutputField(desc="Key resistance levels separated by comma")
    key_observations: str = dspy.OutputField(desc="Top 3 technical observations, separated by |")


# =============================================================================
# RESEARCH SWARM
# =============================================================================

class ResearchSwarm:
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

    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize research swarm."""
        self.config = config or ResearchConfig()
        self._initialized = False

        # Shared resources
        self._memory = None
        self._context = None
        self._bus = None

        # Agents
        self._data_fetcher = None
        self._web_searcher = None
        self._sentiment_analyzer = None
        self._llm_analyzer = None
        self._peer_comparator = None
        self._chart_generator = None
        self._report_generator = None
        # New enhanced agents
        self._technical_analyzer = None
        self._screener_agent = None
        self._social_sentiment_agent = None

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
                lm = DirectClaudeCLI(model="sonnet")
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

    def _init_agents(self):
        """Initialize all agents."""
        self._init_shared_resources()

        self._data_fetcher = DataFetcherAgent(self._memory, self._context, self._bus)
        self._web_searcher = WebSearchAgent(self._memory, self._context, self._bus)
        self._sentiment_analyzer = SentimentAgent(self._memory, self._context, self._bus, self._sentiment_module)
        self._llm_analyzer = LLMAnalysisAgent(self._memory, self._context, self._bus, self._stock_analyzer)
        self._peer_comparator = PeerComparisonAgent(self._memory, self._context, self._bus, self._peer_selector)
        self._chart_generator = EnhancedChartGeneratorAgent(self._memory, self._context, self._bus)
        self._report_generator = ReportGeneratorAgent(self._memory, self._context, self._bus)
        # New enhanced agents
        self._technical_analyzer = TechnicalAnalysisAgent(
            self._memory, self._context, self._bus,
            self._technical_signals_module,
            self.config.nse_data_path
        )
        self._screener_agent = ScreenerAgent(self._memory, self._context, self._bus)
        self._social_sentiment_agent = SocialSentimentAgent(
            self._memory, self._context, self._bus,
            self._social_sentiment_module
        )

        logger.info("✅ All research agents initialized (including enhanced agents)")

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

        # Initialize agents if needed
        if not self._data_fetcher:
            self._init_agents()

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

class BaseAgent:
    """Base class for all agents with shared resources."""

    def __init__(self, memory=None, context=None, bus=None):
        self.memory = memory
        self.context = context
        self.bus = bus

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass


class DataFetcherAgent(BaseAgent):
    """Fetches financial data from Yahoo Finance and other APIs."""

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._fetcher = None

    async def fetch(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Fetch financial data."""
        try:
            from ..skills.research.data_fetcher import ResearchDataFetcher

            if not self._fetcher:
                self._fetcher = ResearchDataFetcher()

            data = await self._fetcher.fetch_company_data(ticker, exchange)
            data['sources'] = data.get('sources', [])

            self._broadcast("data_fetched", {'ticker': ticker, 'has_data': bool(data.get('current_price'))})

            return data
        except Exception as e:
            logger.error(f"DataFetcherAgent error: {e}")
            return {'error': str(e), 'sources': []}


class WebSearchAgent(BaseAgent):
    """Searches web for news and updates."""

    def __init__(self, memory=None, context=None, bus=None):
        super().__init__(memory, context, bus)
        self._search_tool = None

    async def search(self, ticker: str, max_results: int = 25) -> Dict[str, Any]:
        """Search web for company news."""
        try:
            # Initialize search tool
            if not self._search_tool:
                try:
                    from Jotty.core.registry.skills_registry import get_skills_registry
                except ImportError:
                    from ..registry.skills_registry import get_skills_registry

                registry = get_skills_registry()
                registry.init()
                web_skill = registry.get_skill('web-search')
                if web_skill:
                    self._search_tool = web_skill.tools.get('search_web_tool')

            if not self._search_tool:
                return {'news_text': '', 'news_count': 0}

            # Search queries
            queries = [
                f"{ticker} stock latest news 2024 2025",
                f"{ticker} quarterly results earnings",
                f"{ticker} analyst rating target price"
            ]

            all_results = []
            import inspect

            for query in queries:
                try:
                    if inspect.iscoroutinefunction(self._search_tool):
                        result = await self._search_tool({'query': query, 'max_results': max_results // 3})
                    else:
                        result = self._search_tool({'query': query, 'max_results': max_results // 3})

                    if result.get('success') and result.get('results'):
                        all_results.extend(result['results'])
                except Exception:
                    pass

            # Deduplicate
            seen = set()
            unique = []
            for r in all_results:
                url = r.get('url', '')
                if url and url not in seen:
                    seen.add(url)
                    unique.append(r)

            # Build news text
            news_text = "\n".join([
                f"• {r.get('title', '')}: {r.get('snippet', '')[:200]}"
                for r in unique[:20]
            ])

            self._broadcast("web_search_complete", {'ticker': ticker, 'count': len(unique)})

            return {
                'news_text': news_text,
                'news_items': unique[:20],
                'news_count': len(unique)
            }
        except Exception as e:
            logger.error(f"WebSearchAgent error: {e}")
            return {'news_text': '', 'news_count': 0}


class SentimentAgent(BaseAgent):
    """Analyzes sentiment from news."""

    def __init__(self, memory=None, context=None, bus=None, llm_module=None):
        super().__init__(memory, context, bus)
        self._llm = llm_module

    async def analyze(self, company: str, news_text: str) -> Dict[str, Any]:
        """Analyze sentiment from news."""
        if not news_text:
            return {'sentiment_score': 0, 'sentiment_label': 'NEUTRAL', 'key_themes': []}

        try:
            if self._llm:
                # LLM-based sentiment
                result = self._llm(company=company, news_text=news_text[:3000])

                score = float(result.sentiment_score) if result.sentiment_score else 0
                score = max(-1, min(1, score))  # Clamp

                return {
                    'sentiment_score': score,
                    'sentiment_label': str(result.sentiment_label).upper(),
                    'key_themes': str(result.key_themes).split('|') if result.key_themes else [],
                    'reasoning': str(result.reasoning) if result.reasoning else ''
                }
            else:
                # Rule-based fallback
                return self._rule_based_sentiment(news_text)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return self._rule_based_sentiment(news_text)

    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based sentiment."""
        text_lower = text.lower()

        positive = ['profit', 'growth', 'beat', 'surge', 'strong', 'buy', 'upgrade', 'record', 'success']
        negative = ['loss', 'decline', 'miss', 'fall', 'weak', 'sell', 'downgrade', 'concern', 'risk']

        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return {'sentiment_score': 0, 'sentiment_label': 'NEUTRAL', 'key_themes': []}

        score = (pos_count - neg_count) / total

        if score > 0.3: label = 'POSITIVE'
        elif score < -0.3: label = 'NEGATIVE'
        else: label = 'NEUTRAL'

        return {'sentiment_score': score, 'sentiment_label': label, 'key_themes': []}


class LLMAnalysisAgent(BaseAgent):
    """LLM-powered stock analysis."""

    def __init__(self, memory=None, context=None, bus=None, llm_module=None):
        super().__init__(memory, context, bus)
        self._llm = llm_module

    async def analyze(self, data: Dict[str, Any], news_text: str) -> Dict[str, Any]:
        """Analyze stock using LLM."""
        ticker = data.get('ticker', 'UNKNOWN')
        company = data.get('company_name', ticker)

        if not self._llm:
            return self._fallback_analysis(data)

        try:
            # Prepare financial data summary
            financial_summary = json.dumps({
                'current_price': data.get('current_price', 0),
                'target_price': data.get('target_mean_price', 0),
                'pe_ratio': data.get('pe_ratio', 0),
                'pb_ratio': data.get('pb_ratio', 0),
                'revenue_growth': data.get('revenue_growth', 0),
                'profit_margin': data.get('profit_margin', 0),
                'market_cap': data.get('market_cap', 0),
                'sector': data.get('sector', ''),
                'industry': data.get('industry', ''),
                'analyst_count': data.get('num_analysts', 0)
            }, indent=2)

            result = self._llm(
                ticker=ticker,
                company_name=company,
                financial_data=financial_summary,
                news_summary=news_text[:2000]
            )

            # Parse response
            rating = str(result.rating).upper().strip() if result.rating else 'HOLD'
            if rating not in ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']:
                rating = 'HOLD'

            confidence = float(result.confidence) if result.confidence else 0.5
            confidence = max(0, min(1, confidence))

            thesis = [t.strip() for t in str(result.thesis).split('|')] if result.thesis else []
            risks = [r.strip() for r in str(result.risks).split('|')] if result.risks else []

            self._broadcast("llm_analysis_complete", {'ticker': ticker, 'rating': rating})

            return {
                'rating': rating,
                'confidence': confidence,
                'thesis': thesis[:5],
                'risks': risks[:5],
                'reasoning': str(result.reasoning) if result.reasoning else ''
            }
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(data)

    def _fallback_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback."""
        score = 0

        current = data.get('current_price', 0)
        target = data.get('target_mean_price', 0)
        if target and current and target > current:
            score += 30

        growth = data.get('revenue_growth', 0)
        if growth > 15: score += 25
        elif growth > 5: score += 10

        margin = data.get('profit_margin', 0)
        if margin > 15: score += 25
        elif margin > 5: score += 10

        if score >= 60: rating = "BUY"
        elif score >= 30: rating = "HOLD"
        else: rating = "SELL"

        return {
            'rating': rating,
            'confidence': min(score / 100, 1.0),
            'thesis': ["Based on fundamental analysis"],
            'risks': ["Market and sector risks apply"],
            'reasoning': f"Quantitative score: {score}/100"
        }


class PeerComparisonAgent(BaseAgent):
    """Compares stock with sector peers."""

    def __init__(self, memory=None, context=None, bus=None, llm_module=None):
        super().__init__(memory, context, bus)
        self._llm = llm_module
        self._fetcher = None

    async def compare(self, ticker: str, sector: str, industry: str, exchange: str) -> Dict[str, Any]:
        """Compare with peer companies."""
        try:
            # Get peer suggestions
            peers = await self._get_peers(ticker, sector, industry, exchange)

            if not peers:
                return {'peers': [], 'comparison': {}}

            # Clean peer tickers - remove any suffix like .NS, .BO
            cleaned_peers = []
            for p in peers[:5]:
                clean = p.strip()
                # Remove common exchange suffixes
                for suffix in ['.NS', '.BO', '.NSE', '.BSE']:
                    if clean.upper().endswith(suffix):
                        clean = clean[:-len(suffix)]
                        break
                if clean:
                    cleaned_peers.append(clean)

            # Fetch peer data
            if not self._fetcher:
                from ..skills.research.data_fetcher import ResearchDataFetcher
                self._fetcher = ResearchDataFetcher()

            peer_data = {}
            for peer in cleaned_peers:
                try:
                    data = await self._fetcher.fetch_company_data(peer, exchange)
                    if data.get('current_price'):
                        peer_data[peer] = {
                            'price': data.get('current_price', 0),
                            'pe': data.get('pe_ratio', 0),
                            'pb': data.get('pb_ratio', 0),
                            'market_cap': data.get('market_cap', 0)
                        }
                except Exception:
                    pass

            self._broadcast("peer_comparison_complete", {'ticker': ticker, 'peers': list(peer_data.keys())})

            return {
                'peers': list(peer_data.keys()),
                'comparison': peer_data
            }
        except Exception as e:
            logger.warning(f"Peer comparison failed: {e}")
            return {'peers': [], 'comparison': {}}

    async def _get_peers(self, ticker: str, sector: str, industry: str, exchange: str) -> List[str]:
        """Get peer suggestions."""
        # Known peer mappings for common sectors
        SECTOR_PEERS = {
            'Technology': {
                'NSE': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
                'US': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
            },
            'Financial Services': {
                'NSE': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN'],
                'US': ['JPM', 'BAC', 'GS', 'MS', 'C']
            },
            'Consumer Cyclical': {
                'NSE': ['TITAN', 'TRENT', 'PAGEIND', 'ABFRL', 'VEDL'],
                'US': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX']
            }
        }

        # Try LLM first
        if self._llm and sector:
            try:
                result = self._llm(
                    company=ticker,
                    sector=sector or 'Unknown',
                    industry=industry or 'Unknown',
                    exchange=exchange
                )
                peers = [p.strip() for p in str(result.peers).split(',')]
                peers = [p for p in peers if p and p != ticker]
                if peers:
                    return peers[:5]
            except Exception:
                pass

        # Fallback to known peers
        exchange_type = 'US' if exchange.upper() in ('US', 'NYSE', 'NASDAQ') else 'NSE'
        sector_peers = SECTOR_PEERS.get(sector, {}).get(exchange_type, [])
        return [p for p in sector_peers if p != ticker][:5]


class ChartGeneratorAgent(BaseAgent):
    """Generates technical analysis charts."""

    async def generate(self, ticker: str, data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Generate price and technical charts."""
        chart_paths = []

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            prices = data.get('price_history', [])
            dates = data.get('dates', [])

            if not prices or len(prices) < 10:
                return {'chart_paths': []}

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 1. Price Chart with Moving Averages
            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(prices))
            ax.plot(x, prices, label='Price', linewidth=2, color='#2196F3')

            # Moving averages
            if len(prices) >= 20:
                ma20 = self._moving_average(prices, 20)
                ax.plot(x[19:], ma20, label='MA20', linewidth=1, color='#FF9800', linestyle='--')

            if len(prices) >= 50:
                ma50 = self._moving_average(prices, 50)
                ax.plot(x[49:], ma50, label='MA50', linewidth=1, color='#4CAF50', linestyle='--')

            ax.set_title(f'{ticker} - Price Chart', fontsize=14, fontweight='bold')
            ax.set_xlabel('Trading Days')
            ax.set_ylabel('Price (₹)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            price_chart = output_path / f'{ticker}_price_{timestamp}.png'
            plt.savefig(price_chart, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(price_chart))

            # 2. Volume Chart (if available)
            volumes = data.get('volume_history', [])
            if volumes and len(volumes) == len(prices):
                fig, ax = plt.subplots(figsize=(12, 4))

                colors = ['#4CAF50' if i > 0 and prices[i] >= prices[i-1] else '#F44336'
                          for i in range(len(prices))]
                ax.bar(x, volumes, color=colors, alpha=0.7)

                ax.set_title(f'{ticker} - Volume', fontsize=14, fontweight='bold')
                ax.set_xlabel('Trading Days')
                ax.set_ylabel('Volume')
                ax.grid(True, alpha=0.3)

                volume_chart = output_path / f'{ticker}_volume_{timestamp}.png'
                plt.savefig(volume_chart, dpi=150, bbox_inches='tight')
                plt.close()
                chart_paths.append(str(volume_chart))

            logger.info(f"📊 Generated {len(chart_paths)} charts for {ticker}")

            self._broadcast("charts_generated", {'ticker': ticker, 'count': len(chart_paths)})

            return {'chart_paths': chart_paths}

        except ImportError:
            logger.warning("matplotlib not available for chart generation")
            return {'chart_paths': []}
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return {'chart_paths': []}

    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate simple moving average."""
        return [sum(data[i:i+window])/window for i in range(len(data)-window+1)]


class TechnicalAnalysisAgent(BaseAgent):
    """
    Technical Analysis Agent with multi-timeframe support.

    Loads NSE data and calculates comprehensive technical indicators using pandas_ta.
    Supports timeframes: 15minute, 30minute, 60minute, Day, Week
    """

    TIMEFRAME_DIRS = {
        '15minute': '15minuteData',
        '30minute': '30minuteData',
        '60minute': '60minuteData',
        'day': 'DayData',
        'week': 'WeekData',
    }

    # Indicator periods by category
    INDICATOR_CONFIG = {
        'trend': {
            'sma': [20, 50, 200],
            'ema': [9, 21],
            'wma': [20],
            'adx': 14,
            'supertrend': {'length': 10, 'multiplier': 3},
        },
        'momentum': {
            'rsi': 14,
            'stoch': {'k': 14, 'd': 3, 'smooth_k': 3},
            'cci': 20,
            'willr': 14,
            'mfi': 14,
            'roc': 12,
        },
        'volatility': {
            'bbands': {'length': 20, 'std': 2},
            'atr': 14,
            'kc': {'length': 20, 'scalar': 1.5},
        },
        'volume': {
            'obv': True,
            'vwap': True,
            'pvt': True,
            'ad': True,
        },
        'overlap': {
            'ichimoku': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
        }
    }

    def __init__(self, memory=None, context=None, bus=None, llm_module=None, data_path: str = None):
        super().__init__(memory, context, bus)
        self._llm = llm_module
        self.data_path = data_path or "/var/www/sites/personal/stock_market/common/Data/NSE/"

    async def analyze(self, ticker: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """
        Analyze ticker across multiple timeframes.

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE')
            timeframes: List of timeframes to analyze (default: ['60minute', 'Day'])

        Returns:
            Dict with technical signals, support/resistance, trend analysis
        """
        timeframes = timeframes or ['60minute', 'Day']
        result = {
            'ticker': ticker,
            'timeframes': {},
            'signals': {},
            'support_levels': [],
            'resistance_levels': [],
            'trend': 'NEUTRAL',
            'summary': {}
        }

        try:
            import pandas as pd
            # Try pandas_ta first, fallback to ta library
            try:
                import pandas_ta as ta
                self._use_pandas_ta = True
            except ImportError:
                import ta
                self._use_pandas_ta = False
                logger.info("Using 'ta' library instead of pandas_ta")
        except ImportError:
            logger.warning("No technical analysis library available")
            return result

        all_signals = []

        for timeframe in timeframes:
            try:
                df = self._load_data(ticker, timeframe.lower())
                if df is None or df.empty or len(df) < 60:
                    logger.warning(f"Insufficient data for {ticker} @ {timeframe}")
                    continue

                # Add all technical indicators
                df = self._add_all_indicators(df)

                # Generate signals
                signals = self._generate_signals(df, timeframe)
                result['timeframes'][timeframe] = {
                    'last_price': float(df['close'].iloc[-1]) if 'close' in df.columns else 0,
                    'indicators': self._get_latest_indicators(df),
                    'signals': signals
                }

                # Collect signals for overall trend
                if signals.get('trend_signal'):
                    all_signals.append(signals['trend_signal'])

                # Extract support/resistance from pivot points
                if 'pivot' in df.columns:
                    s1 = df['s1'].iloc[-1] if 's1' in df.columns else None
                    s2 = df['s2'].iloc[-1] if 's2' in df.columns else None
                    r1 = df['r1'].iloc[-1] if 'r1' in df.columns else None
                    r2 = df['r2'].iloc[-1] if 'r2' in df.columns else None

                    if s1 and not pd.isna(s1): result['support_levels'].append(float(s1))
                    if s2 and not pd.isna(s2): result['support_levels'].append(float(s2))
                    if r1 and not pd.isna(r1): result['resistance_levels'].append(float(r1))
                    if r2 and not pd.isna(r2): result['resistance_levels'].append(float(r2))

            except Exception as e:
                logger.warning(f"Technical analysis error for {ticker} @ {timeframe}: {e}")
                continue

        # Determine overall trend
        if all_signals:
            bullish = sum(1 for s in all_signals if s > 0)
            bearish = sum(1 for s in all_signals if s < 0)
            if bullish > bearish:
                result['trend'] = 'BULLISH'
            elif bearish > bullish:
                result['trend'] = 'BEARISH'

        # Deduplicate and sort support/resistance
        result['support_levels'] = sorted(list(set(result['support_levels'])), reverse=True)[:5]
        result['resistance_levels'] = sorted(list(set(result['resistance_levels'])))[:5]

        # Generate summary using LLM if available
        if self._llm and result['timeframes']:
            try:
                indicator_summary = json.dumps({
                    tf: data.get('indicators', {})
                    for tf, data in result['timeframes'].items()
                }, indent=2)[:2000]

                price_data = f"Support: {result['support_levels']}, Resistance: {result['resistance_levels']}"

                llm_result = self._llm(
                    ticker=ticker,
                    indicator_summary=indicator_summary,
                    price_data=price_data
                )

                result['summary'] = {
                    'trend': str(llm_result.trend),
                    'signal_strength': float(llm_result.signal_strength) if llm_result.signal_strength else 0.5,
                    'key_observations': [obs.strip() for obs in str(llm_result.key_observations).split('|')]
                }
            except Exception as e:
                logger.debug(f"LLM technical summary failed: {e}")

        self._broadcast("technical_analysis_complete", {
            'ticker': ticker,
            'trend': result['trend'],
            'timeframes': list(result['timeframes'].keys())
        })

        return result

    def _load_data(self, ticker: str, timeframe: str) -> 'pd.DataFrame':
        """Load OHLCV data from NSE data files."""
        import pandas as pd

        timeframe_dir = self.TIMEFRAME_DIRS.get(timeframe.lower())
        if not timeframe_dir:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return None

        data_dir = Path(self.data_path) / timeframe_dir
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return None

        # Find matching files for the ticker
        pattern = f"*-{ticker}-*.csv.gz"
        files = list(data_dir.glob(pattern))

        # Also try without hyphen (for indices like NIFTY 50)
        if not files:
            pattern = f"*{ticker}*.csv.gz"
            files = list(data_dir.glob(pattern))

        if not files:
            logger.warning(f"No data files found for {ticker} in {data_dir}")
            return None

        # Sort by year (filename starts with year)
        files = sorted(files, key=lambda f: f.name)

        # Load recent files (last 2-3 years for daily, last year for intraday)
        recent_files = files[-3:] if timeframe == 'day' else files[-1:]

        dfs = []
        for file in recent_files:
            try:
                df = pd.read_csv(file, compression='gzip', on_bad_lines='skip')
                dfs.append(df)
            except Exception as e:
                logger.debug(f"Failed to read {file}: {e}")

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)

        # Standardize column names (lowercase)
        df.columns = df.columns.str.lower()

        # Remove duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure required columns
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.warning(f"Missing required columns in data for {ticker}")
            return None

        # Parse dates and set index
        date_col = df['date'].copy()
        # Handle timezone-aware dates
        if date_col.dtype == 'object':
            date_col = pd.to_datetime(date_col, errors='coerce', utc=True)
            if date_col.dt.tz is not None:
                date_col = date_col.dt.tz_localize(None)
        else:
            date_col = pd.to_datetime(date_col, errors='coerce')

        df['date'] = date_col
        df = df.dropna(subset=['date'])
        df = df.set_index('date')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # Return recent data
        return df.tail(500)

    def _add_all_indicators(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add all technical indicators using pandas_ta or ta library."""
        try:
            if getattr(self, '_use_pandas_ta', True):
                return self._add_indicators_pandas_ta(df)
            else:
                return self._add_indicators_ta(df)
        except Exception as e:
            logger.warning(f"Error adding indicators: {e}")
            return df

    def _add_indicators_pandas_ta(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add indicators using pandas_ta library."""
        import pandas_ta as ta

        try:
            # Trend Indicators
            for period in self.INDICATOR_CONFIG['trend']['sma']:
                df[f'sma_{period}'] = ta.sma(df['close'], length=period)

            for period in self.INDICATOR_CONFIG['trend']['ema']:
                df[f'ema_{period}'] = ta.ema(df['close'], length=period)

            # ADX
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_df is not None:
                df = df.join(adx_df)

            # MACD
            macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_df is not None:
                df = df.join(macd_df)

            # SuperTrend
            st_config = self.INDICATOR_CONFIG['trend']['supertrend']
            st_df = ta.supertrend(df['high'], df['low'], df['close'],
                                  length=st_config['length'], multiplier=st_config['multiplier'])
            if st_df is not None:
                df = df.join(st_df)

            # Momentum Indicators
            df['rsi'] = ta.rsi(df['close'], length=14)

            stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            if stoch_df is not None:
                df = df.join(stoch_df)

            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            df['roc'] = ta.roc(df['close'], length=12)

            # Volatility Indicators
            bb_df = ta.bbands(df['close'], length=20, std=2)
            if bb_df is not None:
                df = df.join(bb_df)

            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            kc_df = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
            if kc_df is not None:
                df = df.join(kc_df)

            # Volume Indicators
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            df['pvt'] = ta.pvt(df['close'], df['volume'])
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])

            # Ichimoku Cloud
            ichi_df = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
            if ichi_df is not None and len(ichi_df) == 2:
                df = df.join(ichi_df[0])

            # Pivot Points for support/resistance
            pivot_df = ta.pivot(df['high'], df['low'], df['close'])
            if pivot_df is not None:
                pivot_df.columns = [c.lower().replace('_', '') for c in pivot_df.columns]
                for col in pivot_df.columns:
                    df[col] = pivot_df[col]

        except Exception as e:
            logger.warning(f"Error adding pandas_ta indicators: {e}")

        return df

    def _add_indicators_ta(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add indicators using ta library (fallback)."""
        import ta
        import numpy as np

        try:
            # Trend Indicators
            for period in self.INDICATOR_CONFIG['trend']['sma']:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)

            for period in self.INDICATOR_CONFIG['trend']['ema']:
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)

            # ADX
            adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['ADX_14'] = adx_indicator.adx()
            df['DMP_14'] = adx_indicator.adx_pos()
            df['DMN_14'] = adx_indicator.adx_neg()

            # MACD
            macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD_12_26_9'] = macd_indicator.macd()
            df['MACDs_12_26_9'] = macd_indicator.macd_signal()
            df['MACDh_12_26_9'] = macd_indicator.macd_diff()

            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['STOCHk_14_3_3'] = stoch.stoch()
            df['STOCHd_14_3_3'] = stoch.stoch_signal()

            # CCI
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

            # Williams %R
            df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

            # MFI
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)

            # ROC
            df['roc'] = ta.momentum.roc(df['close'], window=12)

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BBL_20_2.0'] = bb.bollinger_lband()
            df['BBM_20_2.0'] = bb.bollinger_mavg()
            df['BBU_20_2.0'] = bb.bollinger_hband()

            # ATR
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

            # Keltner Channels
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
            df['KCL_20'] = kc.keltner_channel_lband()
            df['KCM_20'] = kc.keltner_channel_mband()
            df['KCU_20'] = kc.keltner_channel_hband()

            # Volume Indicators
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
            df['ITS_9'] = ichimoku.ichimoku_conversion_line()
            df['IKS_26'] = ichimoku.ichimoku_base_line()
            df['ISA_9'] = ichimoku.ichimoku_a()
            df['ISB_26'] = ichimoku.ichimoku_b()

            # Simple pivot points calculation
            pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            df['pivot'] = pp
            df['s1'] = 2 * pp - df['high'].shift(1)
            df['s2'] = pp - (df['high'].shift(1) - df['low'].shift(1))
            df['r1'] = 2 * pp - df['low'].shift(1)
            df['r2'] = pp + (df['high'].shift(1) - df['low'].shift(1))

            # SuperTrend calculation (manual since ta library doesn't have it)
            atr = df['atr']
            hl2 = (df['high'] + df['low']) / 2
            multiplier = 3
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            supertrend = np.zeros(len(df))
            supertrend[0] = upper_band.iloc[0]

            for i in range(1, len(df)):
                if df['close'].iloc[i] > supertrend[i-1]:
                    supertrend[i] = max(lower_band.iloc[i], supertrend[i-1]) if supertrend[i-1] == lower_band.iloc[i-1] or supertrend[i-1] < lower_band.iloc[i] else lower_band.iloc[i]
                else:
                    supertrend[i] = min(upper_band.iloc[i], supertrend[i-1]) if supertrend[i-1] == upper_band.iloc[i-1] or supertrend[i-1] > upper_band.iloc[i] else upper_band.iloc[i]

            df['SUPERT_10_3.0'] = supertrend

        except Exception as e:
            logger.warning(f"Error adding ta library indicators: {e}")

        return df

    def _generate_signals(self, df: 'pd.DataFrame', timeframe: str) -> Dict[str, Any]:
        """Generate trading signals from indicators."""
        signals = {
            'trend_signal': 0,  # -1 (bearish) to 1 (bullish)
            'momentum_signal': 0,
            'volatility_signal': 0,
            'buy_signals': [],
            'sell_signals': [],
            'observations': []
        }

        if df.empty:
            return signals

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        try:
            # Trend signals
            trend_score = 0

            # SMA crossover (golden/death cross)
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                if latest['sma_50'] > latest['sma_200']:
                    trend_score += 1
                    if prev['sma_50'] <= prev['sma_200']:
                        signals['buy_signals'].append('Golden Cross (SMA50 > SMA200)')
                else:
                    trend_score -= 1
                    if prev['sma_50'] >= prev['sma_200']:
                        signals['sell_signals'].append('Death Cross (SMA50 < SMA200)')

            # Price vs SMA200
            if 'sma_200' in df.columns:
                if latest['close'] > latest['sma_200']:
                    trend_score += 0.5
                else:
                    trend_score -= 0.5

            # MACD
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    trend_score += 0.5
                    if prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']:
                        signals['buy_signals'].append('MACD Bullish Crossover')
                else:
                    trend_score -= 0.5
                    if prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']:
                        signals['sell_signals'].append('MACD Bearish Crossover')

            # ADX
            if 'ADX_14' in df.columns:
                adx = latest['ADX_14']
                if adx > 25:
                    signals['observations'].append(f'Strong trend (ADX={adx:.1f})')
                elif adx < 20:
                    signals['observations'].append(f'Weak trend (ADX={adx:.1f})')

            # SuperTrend
            supert_col = [c for c in df.columns if c.startswith('SUPERT_')]
            if supert_col:
                if latest['close'] > latest[supert_col[0]]:
                    trend_score += 0.5
                else:
                    trend_score -= 0.5

            signals['trend_signal'] = max(-1, min(1, trend_score / 3))

            # Momentum signals
            momentum_score = 0

            # RSI
            if 'rsi' in df.columns:
                rsi = latest['rsi']
                if rsi > 70:
                    momentum_score -= 1
                    signals['sell_signals'].append(f'RSI Overbought ({rsi:.1f})')
                elif rsi < 30:
                    momentum_score += 1
                    signals['buy_signals'].append(f'RSI Oversold ({rsi:.1f})')
                elif rsi > 50:
                    momentum_score += 0.3
                else:
                    momentum_score -= 0.3

            # MFI
            if 'mfi' in df.columns:
                mfi = latest['mfi']
                if mfi > 80:
                    momentum_score -= 0.5
                elif mfi < 20:
                    momentum_score += 0.5

            # Stochastic
            if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
                stoch_k = latest['STOCHk_14_3_3']
                stoch_d = latest['STOCHd_14_3_3']
                if stoch_k < 20 and stoch_k > stoch_d:
                    momentum_score += 0.5
                    signals['buy_signals'].append('Stochastic Bullish (oversold)')
                elif stoch_k > 80 and stoch_k < stoch_d:
                    momentum_score -= 0.5
                    signals['sell_signals'].append('Stochastic Bearish (overbought)')

            signals['momentum_signal'] = max(-1, min(1, momentum_score / 2))

            # Volatility signals
            if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                bb_lower = latest['BBL_20_2.0']
                bb_upper = latest['BBU_20_2.0']
                bb_mid = latest.get('BBM_20_2.0', (bb_lower + bb_upper) / 2)

                if latest['close'] <= bb_lower:
                    signals['volatility_signal'] = 1  # Potential bounce
                    signals['observations'].append('Price at lower Bollinger Band')
                elif latest['close'] >= bb_upper:
                    signals['volatility_signal'] = -1  # Potential pullback
                    signals['observations'].append('Price at upper Bollinger Band')

        except Exception as e:
            logger.debug(f"Signal generation error: {e}")

        return signals

    def _get_latest_indicators(self, df: 'pd.DataFrame') -> Dict[str, Any]:
        """Extract latest indicator values."""
        import pandas as pd

        if df.empty:
            return {}

        latest = df.iloc[-1]
        indicators = {}

        indicator_cols = [
            'rsi', 'mfi', 'cci', 'willr', 'roc', 'atr', 'obv', 'vwap',
            'sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21'
        ]

        for col in indicator_cols:
            if col in df.columns and not pd.isna(latest[col]):
                indicators[col] = round(float(latest[col]), 2)

        # MACD
        if 'MACD_12_26_9' in df.columns:
            indicators['macd'] = round(float(latest['MACD_12_26_9']), 2) if not pd.isna(latest['MACD_12_26_9']) else None
            indicators['macd_signal'] = round(float(latest.get('MACDs_12_26_9', 0)), 2) if not pd.isna(latest.get('MACDs_12_26_9')) else None

        # ADX
        if 'ADX_14' in df.columns:
            indicators['adx'] = round(float(latest['ADX_14']), 2) if not pd.isna(latest['ADX_14']) else None

        # Bollinger Bands
        if 'BBL_20_2.0' in df.columns:
            indicators['bb_lower'] = round(float(latest['BBL_20_2.0']), 2) if not pd.isna(latest['BBL_20_2.0']) else None
            indicators['bb_upper'] = round(float(latest['BBU_20_2.0']), 2) if not pd.isna(latest['BBU_20_2.0']) else None

        # Stochastic
        if 'STOCHk_14_3_3' in df.columns:
            indicators['stoch_k'] = round(float(latest['STOCHk_14_3_3']), 2) if not pd.isna(latest['STOCHk_14_3_3']) else None
            indicators['stoch_d'] = round(float(latest.get('STOCHd_14_3_3', 0)), 2) if not pd.isna(latest.get('STOCHd_14_3_3')) else None

        return indicators


class EnhancedChartGeneratorAgent(BaseAgent):
    """
    Enhanced Chart Generator following HourlyReport.py patterns.

    Creates professional multi-panel charts with:
    - Panel 0: Price + Moving Averages + SuperTrend
    - Panel 1: Volume / PVT
    - Panel 2: RSI (with 45/55 fill zones)
    - Panel 3: MACD (with histogram)
    """

    FIGURE_SIZE = (8.27, 11.69)  # A4 paper
    DPI = 300
    STYLE = "yahoo"

    async def generate(
        self,
        ticker: str,
        data: Dict[str, Any],
        output_dir: str,
        timeframes: List[str] = None,
        technical_data: Dict[str, Any] = None,
        include_heiken_ashi: bool = False
    ) -> Dict[str, Any]:
        """Generate professional multi-panel charts."""
        chart_paths = []
        timeframes = timeframes or ['Day']

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import mplfinance as mpf
            import pandas as pd
            import numpy as np
            # Try pandas_ta first, fallback to ta library
            try:
                import pandas_ta
                self._use_pandas_ta = True
            except ImportError:
                import ta
                self._use_pandas_ta = False
        except ImportError as e:
            logger.warning(f"Required library not available: {e}")
            return {'chart_paths': []}

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get technical data with OHLCV
        if technical_data and technical_data.get('timeframes'):
            for timeframe, tf_data in technical_data.get('timeframes', {}).items():
                try:
                    chart_path = await self._generate_multipanel_chart(
                        ticker, timeframe, output_path, timestamp,
                        include_heiken_ashi
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    logger.warning(f"Chart generation error for {timeframe}: {e}")

        # Fallback to basic charts if no technical data
        if not chart_paths and data.get('price_history'):
            prices = data.get('price_history', [])
            volumes = data.get('volume_history', [])

            if prices and len(prices) >= 20:
                basic_chart = self._generate_basic_chart(
                    ticker, prices, volumes, output_path, timestamp
                )
                if basic_chart:
                    chart_paths.append(basic_chart)

        logger.info(f"📊 Generated {len(chart_paths)} enhanced charts for {ticker}")
        self._broadcast("charts_generated", {'ticker': ticker, 'count': len(chart_paths)})

        return {'chart_paths': chart_paths}

    async def _generate_multipanel_chart(
        self,
        ticker: str,
        timeframe: str,
        output_path: Path,
        timestamp: str,
        include_heiken_ashi: bool = False
    ) -> Optional[str]:
        """Generate a multi-panel chart for a specific timeframe."""
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        import pandas as pd
        import numpy as np
        # Note: ta library is imported in _add_chart_indicators methods

        # Load data from NSE
        data_path = "/var/www/sites/personal/stock_market/common/Data/NSE/"
        timeframe_dirs = {
            '15minute': '15minuteData',
            '30minute': '30minuteData',
            '60minute': '60minuteData',
            'day': 'DayData',
            'week': 'WeekData',
        }

        tf_dir = timeframe_dirs.get(timeframe.lower())
        if not tf_dir:
            return None

        data_dir = Path(data_path) / tf_dir
        pattern = f"*-{ticker}-*.csv.gz"
        files = sorted(data_dir.glob(pattern))

        if not files:
            pattern = f"*{ticker}*.csv.gz"
            files = sorted(data_dir.glob(pattern))

        if not files:
            return None

        # Load recent data
        recent_files = files[-2:] if timeframe.lower() == 'day' else files[-1:]
        dfs = []
        for f in recent_files:
            try:
                df = pd.read_csv(f, compression='gzip', on_bad_lines='skip')
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.lower()

        # Remove duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]

        # Prepare OHLCV data - handle timezone aware dates
        date_col = df['date'].copy()
        if date_col.dtype == 'object':
            date_col = pd.to_datetime(date_col, errors='coerce', utc=True)
            if date_col.dt.tz is not None:
                date_col = date_col.dt.tz_localize(None)
        else:
            date_col = pd.to_datetime(date_col, errors='coerce')

        df['date'] = date_col
        df = df.dropna(subset=['date', 'open', 'high', 'low', 'close'])
        df = df.set_index('date')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename columns for mplfinance
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })

        # Limit data points
        num_candles = 100 if timeframe.lower() in ['15minute', '30minute', '60minute'] else 150
        df = df.tail(num_candles)

        if len(df) < 30:
            return None

        # Heiken-Ashi conversion if requested
        if include_heiken_ashi:
            df = self._to_heiken_ashi(df)

        # Add indicators
        df = self._add_chart_indicators(df)

        # Build addplots for multi-panel chart
        apds = []
        current_panel = 0

        # Panel 0: Price with overlays (handled by mplfinance main chart)
        # Add moving averages
        if 'wma_9' in df.columns:
            apds.append(mpf.make_addplot(df['wma_9'], color='green', panel=current_panel, width=0.8))
        if 'wma_21' in df.columns:
            apds.append(mpf.make_addplot(df['wma_21'], color='orange', panel=current_panel, width=0.8))
        if 'wma_55' in df.columns:
            apds.append(mpf.make_addplot(df['wma_55'], color='red', panel=current_panel, width=0.8))

        # SuperTrend
        supert_col = [c for c in df.columns if c.startswith('SUPERT_')]
        if supert_col:
            apds.append(mpf.make_addplot(df[supert_col[0]], color='purple', panel=current_panel,
                                         linestyle='--', width=0.7))

        # LTP line
        ltp = df['Close'].iloc[-1]
        apds.append(mpf.make_addplot([ltp] * len(df), color='blue', panel=current_panel,
                                     alpha=0.3, width=0.5))

        # Panel 1: Volume/PVT (auto-added by mplfinance with volume=True)
        if 'pvt' in df.columns:
            current_panel = 2  # Volume is panel 1
            apds.append(mpf.make_addplot(df['pvt'], color='green', panel=current_panel))
            if 'pvt_9' in df.columns:
                apds.append(mpf.make_addplot(df['pvt_9'], color='orange', panel=current_panel))
            if 'pvt_21' in df.columns:
                apds.append(mpf.make_addplot(df['pvt_21'], color='red', panel=current_panel))
            current_panel += 1
        else:
            current_panel = 2

        # Panel 2: RSI with 45/55 fill zones
        if 'rsi' in df.columns:
            apds.append(mpf.make_addplot([45] * len(df), color='red', panel=current_panel,
                                         alpha=0.25, ylim=(0, 100)))
            apds.append(mpf.make_addplot([55] * len(df), color='green', panel=current_panel,
                                         alpha=0.25, ylim=(0, 100)))
            apds.append(mpf.make_addplot(df['rsi'], color='purple', panel=current_panel,
                                         ylim=(0, 100)))
            current_panel += 1

        # Panel 3: MACD with histogram
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            apds.append(mpf.make_addplot(df['macd'], color='green', panel=current_panel))
            apds.append(mpf.make_addplot(df['macd_signal'], color='red', panel=current_panel))
            if 'macd_hist' in df.columns:
                colors = ['green' if v >= 0 else 'red' for v in df['macd_hist']]
                apds.append(mpf.make_addplot(df['macd_hist'], type='bar', color=colors,
                                             panel=current_panel, alpha=0.5))

        # Generate chart
        try:
            last_date = df.index[-1].strftime('%Y-%m-%d %H:%M' if 'minute' in timeframe.lower() else '%Y-%m-%d')
            title = f"{ticker} ({timeframe}) - {last_date} | LTP: ₹{ltp:,.2f}"

            chart_file = output_path / f'{ticker}_{timeframe}_{timestamp}.png'

            fig, axlist = mpf.plot(
                df,
                type='candle' if not include_heiken_ashi else 'candle',
                style=self.STYLE,
                volume=True,
                title=title,
                figsize=self.FIGURE_SIZE,
                addplot=apds if apds else None,
                tight_layout=True,
                returnfig=True
            )

            # Add RSI fill zones if RSI panel exists
            if 'rsi' in df.columns:
                rsi_panel_idx = len(axlist) - 2  # Second to last panel
                if rsi_panel_idx >= 0 and rsi_panel_idx < len(axlist):
                    ax = axlist[rsi_panel_idx]
                    x = range(len(df))
                    rsi_values = df['rsi'].values
                    ax.fill_between(x, 55, rsi_values, where=(rsi_values >= 55),
                                   color='green', alpha=0.3, interpolate=True)
                    ax.fill_between(x, 45, rsi_values, where=(rsi_values <= 45),
                                   color='red', alpha=0.3, interpolate=True)
                    ax.fill_between(x, 45, 55, color='grey', alpha=0.1)

            fig.savefig(chart_file, dpi=self.DPI, bbox_inches='tight')
            plt.close(fig)

            return str(chart_file)

        except Exception as e:
            logger.error(f"mplfinance chart error: {e}")
            plt.close('all')
            return None

    def _add_chart_indicators(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add indicators needed for charting."""
        try:
            if getattr(self, '_use_pandas_ta', True):
                return self._add_chart_indicators_pandas_ta(df)
            else:
                return self._add_chart_indicators_ta(df)
        except Exception as e:
            logger.debug(f"Chart indicator error: {e}")
            return df

    def _add_chart_indicators_pandas_ta(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add chart indicators using pandas_ta."""
        import pandas_ta as ta

        try:
            df['wma_9'] = ta.wma(df['Low'], length=9)
            df['wma_21'] = ta.wma(df['Close'], length=21)
            df['wma_55'] = ta.wma(df['High'], length=55)
            df['rsi'] = ta.rsi(df['Close'], length=14)

            macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd_result is not None:
                df['macd'] = macd_result['MACD_12_26_9']
                df['macd_signal'] = macd_result['MACDs_12_26_9']
                df['macd_hist'] = macd_result['MACDh_12_26_9']

            df['pvt'] = ta.pvt(df['Close'], df['Volume'])
            df['pvt_9'] = ta.wma(df['pvt'], length=9)
            df['pvt_21'] = ta.wma(df['pvt'], length=21)

            st_result = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
            if st_result is not None:
                df = df.join(st_result)
        except Exception as e:
            logger.debug(f"pandas_ta chart indicator error: {e}")

        return df

    def _add_chart_indicators_ta(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add chart indicators using ta library."""
        import ta
        import numpy as np

        try:
            # WMA (using EMA as approximation since ta lib doesn't have WMA)
            df['wma_9'] = ta.trend.ema_indicator(df['Low'], window=9)
            df['wma_21'] = ta.trend.ema_indicator(df['Close'], window=21)
            df['wma_55'] = ta.trend.ema_indicator(df['High'], window=55)

            # RSI
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14)

            # MACD
            macd_indicator = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()

            # PVT (Price Volume Trend) - manual calculation
            pv_change = (df['Close'].diff() / df['Close'].shift(1)) * df['Volume']
            df['pvt'] = pv_change.cumsum()
            df['pvt_9'] = ta.trend.ema_indicator(df['pvt'], window=9)
            df['pvt_21'] = ta.trend.ema_indicator(df['pvt'], window=21)

            # SuperTrend - manual calculation
            atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=10)
            hl2 = (df['High'] + df['Low']) / 2
            multiplier = 3
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            supertrend = np.zeros(len(df))
            supertrend[0] = upper_band.iloc[0] if not np.isnan(upper_band.iloc[0]) else df['Close'].iloc[0]

            for i in range(1, len(df)):
                if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
                    supertrend[i] = supertrend[i-1]
                elif df['Close'].iloc[i] > supertrend[i-1]:
                    supertrend[i] = max(lower_band.iloc[i], supertrend[i-1]) if supertrend[i-1] < upper_band.iloc[i-1] else lower_band.iloc[i]
                else:
                    supertrend[i] = min(upper_band.iloc[i], supertrend[i-1]) if supertrend[i-1] > lower_band.iloc[i-1] else upper_band.iloc[i]

            df['SUPERT_10_3.0'] = supertrend

        except Exception as e:
            logger.debug(f"ta library chart indicator error: {e}")

        return df

    def _to_heiken_ashi(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Convert OHLC to Heiken-Ashi candles."""
        ha_df = df.copy()

        ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        ha_df['HA_Open'] = 0.0
        ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2

        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (
                ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]
            ) / 2

        ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)

        # Replace OHLC with HA values
        ha_df['Open'] = ha_df['HA_Open']
        ha_df['High'] = ha_df['HA_High']
        ha_df['Low'] = ha_df['HA_Low']
        ha_df['Close'] = ha_df['HA_Close']

        return ha_df.drop(columns=['HA_Open', 'HA_High', 'HA_Low', 'HA_Close'])

    def _generate_basic_chart(
        self,
        ticker: str,
        prices: List[float],
        volumes: List[float],
        output_path: Path,
        timestamp: str
    ) -> Optional[str]:
        """Generate basic price chart as fallback."""
        import matplotlib.pyplot as plt

        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

            x = range(len(prices))

            # Price chart
            axes[0].plot(x, prices, label='Price', linewidth=2, color='#2196F3')

            if len(prices) >= 20:
                ma20 = [sum(prices[i:i+20])/20 for i in range(len(prices)-19)]
                axes[0].plot(x[19:], ma20, label='MA20', color='#FF9800', linestyle='--')

            if len(prices) >= 50:
                ma50 = [sum(prices[i:i+50])/50 for i in range(len(prices)-49)]
                axes[0].plot(x[49:], ma50, label='MA50', color='#4CAF50', linestyle='--')

            axes[0].set_title(f'{ticker} - Price Chart', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Price (₹)')
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)

            # Volume chart
            if volumes and len(volumes) == len(prices):
                colors = ['#4CAF50' if i > 0 and prices[i] >= prices[i-1] else '#F44336'
                         for i in range(len(prices))]
                axes[1].bar(x, volumes, color=colors, alpha=0.7)
                axes[1].set_ylabel('Volume')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            chart_file = output_path / f'{ticker}_basic_{timestamp}.png'
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()

            return str(chart_file)

        except Exception as e:
            logger.error(f"Basic chart error: {e}")
            return None


class ScreenerAgent(BaseAgent):
    """
    Screener.in Agent for Indian company fundamentals.

    Extracts:
    - Financial ratios (P/E, P/B, ROCE, ROE, Debt/Equity)
    - Quarterly results
    - Shareholding pattern (Promoter, FII, DII, Public)
    - Peer companies
    """

    BASE_URL = "https://www.screener.in/company/"

    async def fetch(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental data from Screener.in."""
        result = {
            'ticker': ticker,
            'ratios': {},
            'shareholding': {},
            'quarterly_results': [],
            'peers': [],
            'success': False
        }

        try:
            import aiohttp
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("aiohttp or beautifulsoup4 not available for Screener scraping")
            return result

        url = f"{self.BASE_URL}{ticker}/"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        logger.warning(f"Screener.in returned {response.status} for {ticker}")
                        return result

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Extract ratios
                    result['ratios'] = self._get_ratios(soup)

                    # Extract shareholding
                    result['shareholding'] = self._get_shareholding(soup)

                    # Extract quarterly results
                    result['quarterly_results'] = self._get_quarterly_results(soup)

                    # Extract peers
                    result['peers'] = self._get_peers(soup)

                    result['success'] = True

        except asyncio.TimeoutError:
            logger.warning(f"Screener.in timeout for {ticker}")
        except Exception as e:
            logger.warning(f"Screener.in scraping error for {ticker}: {e}")

        self._broadcast("screener_fetch_complete", {
            'ticker': ticker,
            'success': result['success'],
            'has_ratios': bool(result['ratios'])
        })

        return result

    def _get_ratios(self, soup) -> Dict[str, Any]:
        """Extract financial ratios from Screener page."""
        ratios = {}

        try:
            # Find the ratios list (usually in a ul.flex-list)
            ratio_lists = soup.find_all('ul', class_='flex-list')

            for ul in ratio_lists:
                items = ul.find_all('li')
                for item in items:
                    name_elem = item.find('span', class_='name')
                    value_elem = item.find('span', class_='number')

                    if name_elem and value_elem:
                        name = name_elem.get_text(strip=True).lower()
                        value = value_elem.get_text(strip=True)

                        # Clean and parse value
                        value = value.replace(',', '').replace('%', '').replace('₹', '').strip()

                        # Map common ratios
                        if 'p/e' in name or 'pe' in name:
                            ratios['pe_ratio'] = self._parse_float(value)
                        elif 'p/b' in name or 'pb' in name:
                            ratios['pb_ratio'] = self._parse_float(value)
                        elif 'roce' in name:
                            ratios['roce'] = self._parse_float(value)
                        elif 'roe' in name:
                            ratios['roe'] = self._parse_float(value)
                        elif 'debt' in name and 'equity' in name:
                            ratios['debt_equity'] = self._parse_float(value)
                        elif 'market cap' in name:
                            ratios['market_cap'] = self._parse_float(value)
                        elif 'current price' in name or 'stock' in name:
                            ratios['current_price'] = self._parse_float(value)
                        elif 'book value' in name:
                            ratios['book_value'] = self._parse_float(value)
                        elif 'dividend yield' in name:
                            ratios['dividend_yield'] = self._parse_float(value)

            # Also try to find data from top ratios section
            top_ratios = soup.find('div', id='top-ratios')
            if top_ratios:
                spans = top_ratios.find_all('span')
                for i in range(0, len(spans) - 1, 2):
                    name = spans[i].get_text(strip=True).lower()
                    value = spans[i+1].get_text(strip=True)
                    value = value.replace(',', '').replace('%', '').replace('₹', '').strip()

                    if 'pe' in name and 'pe_ratio' not in ratios:
                        ratios['pe_ratio'] = self._parse_float(value)
                    elif 'roce' in name and 'roce' not in ratios:
                        ratios['roce'] = self._parse_float(value)

        except Exception as e:
            logger.debug(f"Ratio extraction error: {e}")

        return ratios

    def _get_shareholding(self, soup) -> Dict[str, Any]:
        """Extract shareholding pattern."""
        shareholding = {}

        try:
            # Find shareholding table
            sh_section = soup.find('section', id='shareholding')
            if not sh_section:
                return shareholding

            table = sh_section.find('table')
            if not table:
                return shareholding

            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    name = cells[0].get_text(strip=True).lower()
                    # Get the latest value (usually last column with data)
                    values = [c.get_text(strip=True) for c in cells[1:]]
                    latest_value = None
                    for v in reversed(values):
                        if v and v != '-':
                            latest_value = v.replace('%', '').strip()
                            break

                    if latest_value:
                        if 'promoter' in name:
                            shareholding['promoter'] = self._parse_float(latest_value)
                        elif 'fii' in name or 'foreign' in name:
                            shareholding['fii'] = self._parse_float(latest_value)
                        elif 'dii' in name or 'domestic' in name:
                            shareholding['dii'] = self._parse_float(latest_value)
                        elif 'public' in name:
                            shareholding['public'] = self._parse_float(latest_value)

        except Exception as e:
            logger.debug(f"Shareholding extraction error: {e}")

        return shareholding

    def _get_quarterly_results(self, soup) -> List[Dict[str, Any]]:
        """Extract quarterly results."""
        results = []

        try:
            # Find quarterly results section
            qr_section = soup.find('section', id='quarters')
            if not qr_section:
                return results

            table = qr_section.find('table')
            if not table:
                return results

            # Get headers (quarters)
            headers = []
            header_row = table.find('thead')
            if header_row:
                ths = header_row.find_all('th')
                headers = [th.get_text(strip=True) for th in ths]

            # Get data rows
            tbody = table.find('tbody')
            if not tbody:
                return results

            rows = tbody.find_all('tr')
            for row in rows[:5]:  # Limit to recent quarters
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    metric = cells[0].get_text(strip=True)
                    values = [self._parse_float(c.get_text(strip=True).replace(',', ''))
                              for c in cells[1:]]

                    if metric.lower() in ['sales', 'revenue', 'net profit', 'operating profit', 'eps']:
                        for i, val in enumerate(values[:4]):
                            if len(results) <= i:
                                results.append({'quarter': headers[i+1] if i+1 < len(headers) else f'Q{i+1}'})
                            results[i][metric.lower().replace(' ', '_')] = val

        except Exception as e:
            logger.debug(f"Quarterly results extraction error: {e}")

        return results[:4]  # Return last 4 quarters

    def _get_peers(self, soup) -> List[str]:
        """Extract peer companies."""
        peers = []

        try:
            # Find peers section
            peer_section = soup.find('section', id='peers')
            if not peer_section:
                return peers

            # Find peer links
            links = peer_section.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                if '/company/' in href:
                    # Extract ticker from URL
                    parts = href.strip('/').split('/')
                    if len(parts) >= 2:
                        peer_ticker = parts[-1].upper()
                        if peer_ticker and peer_ticker not in peers:
                            peers.append(peer_ticker)

        except Exception as e:
            logger.debug(f"Peers extraction error: {e}")

        return peers[:10]

    def _parse_float(self, value: str) -> Optional[float]:
        """Safely parse a float value."""
        if not value:
            return None
        try:
            # Handle Cr (crore) and L (lakh) suffixes
            value = value.strip()
            multiplier = 1
            if value.endswith('Cr'):
                value = value[:-2]
                multiplier = 10000000  # 1 crore
            elif value.endswith('L'):
                value = value[:-1]
                multiplier = 100000  # 1 lakh

            return float(value) * multiplier
        except (ValueError, TypeError):
            return None


class SocialSentimentAgent(BaseAgent):
    """
    Social Sentiment Agent for multi-source sentiment analysis.

    Aggregates sentiment from:
    - Web search news
    - Financial forums
    - Analyst discussions
    """

    def __init__(self, memory=None, context=None, bus=None, llm_module=None):
        super().__init__(memory, context, bus)
        self._llm = llm_module

    async def analyze(self, company: str, news_text: str, forum_text: str = "") -> Dict[str, Any]:
        """
        Analyze sentiment from multiple sources.

        Args:
            company: Company name
            news_text: News headlines and snippets
            forum_text: Forum discussions (optional)

        Returns:
            Dict with sentiment score, label, themes, and drivers
        """
        result = {
            'overall_sentiment': 0.0,
            'sentiment_label': 'NEUTRAL',
            'key_themes': [],
            'sentiment_drivers': {
                'positive': [],
                'negative': []
            }
        }

        if not news_text and not forum_text:
            return result

        try:
            if self._llm:
                # Use LLM for sophisticated sentiment analysis
                llm_result = self._llm(
                    company=company,
                    news_text=news_text[:3000] if news_text else "",
                    forum_text=forum_text[:1500] if forum_text else ""
                )

                # Parse sentiment score
                try:
                    score = float(llm_result.overall_sentiment)
                    result['overall_sentiment'] = max(-1, min(1, score))
                except (ValueError, TypeError):
                    result['overall_sentiment'] = 0.0

                # Parse sentiment label
                label = str(llm_result.sentiment_label).upper().strip()
                if label in ['BEARISH', 'NEUTRAL', 'BULLISH']:
                    result['sentiment_label'] = label
                elif result['overall_sentiment'] > 0.3:
                    result['sentiment_label'] = 'BULLISH'
                elif result['overall_sentiment'] < -0.3:
                    result['sentiment_label'] = 'BEARISH'

                # Parse themes
                if llm_result.key_themes:
                    result['key_themes'] = [t.strip() for t in str(llm_result.key_themes).split('|')][:5]

                # Parse sentiment drivers
                if llm_result.sentiment_drivers:
                    drivers_str = str(llm_result.sentiment_drivers)
                    if '||' in drivers_str:
                        parts = drivers_str.split('||')
                        if len(parts) >= 2:
                            result['sentiment_drivers']['positive'] = [d.strip() for d in parts[0].split('|') if d.strip()][:5]
                            result['sentiment_drivers']['negative'] = [d.strip() for d in parts[1].split('|') if d.strip()][:5]
            else:
                # Fallback to rule-based analysis
                result = self._rule_based_analysis(news_text, forum_text)

        except Exception as e:
            logger.warning(f"Social sentiment analysis error: {e}")
            result = self._rule_based_analysis(news_text, forum_text)

        self._broadcast("social_sentiment_complete", {
            'company': company,
            'sentiment': result['sentiment_label'],
            'score': result['overall_sentiment']
        })

        return result

    def _rule_based_analysis(self, news_text: str, forum_text: str) -> Dict[str, Any]:
        """Simple keyword-based sentiment analysis fallback."""
        combined_text = f"{news_text} {forum_text}".lower()

        positive_keywords = {
            'growth': 1, 'profit': 1, 'beat': 1.5, 'surge': 1.5, 'strong': 1,
            'buy': 1, 'upgrade': 1.5, 'record': 1.5, 'success': 1, 'bullish': 1.5,
            'outperform': 1.5, 'expansion': 1, 'innovation': 1, 'dividend': 1
        }

        negative_keywords = {
            'loss': -1, 'decline': -1, 'miss': -1.5, 'fall': -1, 'weak': -1,
            'sell': -1, 'downgrade': -1.5, 'concern': -1, 'risk': -0.5, 'bearish': -1.5,
            'underperform': -1.5, 'debt': -0.5, 'lawsuit': -1, 'investigation': -1
        }

        score = 0
        positive_drivers = []
        negative_drivers = []

        for word, weight in positive_keywords.items():
            count = combined_text.count(word)
            if count > 0:
                score += weight * min(count, 3)
                positive_drivers.append(word)

        for word, weight in negative_keywords.items():
            count = combined_text.count(word)
            if count > 0:
                score += weight * min(count, 3)
                negative_drivers.append(word)

        # Normalize score to -1 to 1 range
        max_possible = 15  # Rough estimate
        normalized_score = max(-1, min(1, score / max_possible))

        if normalized_score > 0.2:
            label = 'BULLISH'
        elif normalized_score < -0.2:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'

        return {
            'overall_sentiment': normalized_score,
            'sentiment_label': label,
            'key_themes': positive_drivers[:3] + negative_drivers[:2],
            'sentiment_drivers': {
                'positive': positive_drivers[:5],
                'negative': negative_drivers[:5]
            }
        }


class ReportGeneratorAgent(BaseAgent):
    """Generates final report and handles output."""

    async def generate(
        self,
        ticker: str,
        data: Dict[str, Any],
        analysis: Dict[str, Any],
        sentiment: Dict[str, Any],
        peers: Dict[str, Any],
        chart_paths: List[str],
        output_dir: str,
        send_telegram: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive report."""
        try:
            # Use enhanced research tool for professional report
            from ..skills.research.enhanced_research import enhanced_stock_research_tool

            params = {
                'ticker': ticker,
                'company_name': data.get('company_name', ticker),
                'exchange': data.get('exchange', 'NSE'),
                'target_price': data.get('target_mean_price'),
                'rating': analysis.get('rating', 'HOLD'),
                'output_dir': output_dir,
                'send_telegram': send_telegram,
            }

            result = await enhanced_stock_research_tool(params)

            return {
                'md_path': result.get('md_path', ''),
                'pdf_path': result.get('pdf_path', ''),
                'telegram_sent': result.get('telegram_sent', False),
                'success': result.get('success', False)
            }

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'success': False, 'error': str(e)}


# =============================================================================
# CONVENIENCE FUNCTIONS
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
