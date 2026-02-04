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

            # Sentiment analysis (if enabled)
            if self.config.include_sentiment and web_data.get('news_text'):
                analysis_tasks.append(
                    self._sentiment_analyzer.analyze(
                        merged_data.get('company_name', ticker),
                        web_data.get('news_text', '')
                    )
                )
            else:
                analysis_tasks.append(asyncio.coroutine(lambda: {'sentiment_score': 0, 'sentiment_label': 'NEUTRAL'})())

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
                analysis_tasks.append(asyncio.coroutine(lambda: {'overall_sentiment': 0, 'sentiment_label': 'NEUTRAL', 'sentiment_drivers': {}})())

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
                analysis_tasks.append(asyncio.coroutine(lambda: {'peers': [], 'comparison': {}})())

            # LLM analysis
            if self.config.use_llm_analysis:
                analysis_tasks.append(
                    self._llm_analyzer.analyze(merged_data, web_data.get('news_text', ''))
                )
            else:
                analysis_tasks.append(asyncio.coroutine(lambda: self._rule_based_analysis(merged_data))())

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
