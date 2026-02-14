"""Research Swarm - Main swarm orchestrator."""

import asyncio
import logging
import os
import re
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import dspy

from ..base import DomainSwarm, AgentTeam
from ..base.domain_swarm import PhaseExecutor
from ..base_swarm import AgentRole, register_swarm
from ..swarm_signatures import ResearchSwarmSignature

from .types import RatingType, ResearchConfig, ResearchResult, TopicResearchResult
from .signatures import (
    StockAnalysisSignature,
    SentimentAnalysisSignature,
    PeerSelectionSignature,
    SocialSentimentSignature,
    TechnicalSignalsSignature,
)
from .agents import (
    BaseSwarmAgent, DataFetcherAgent, WebSearchAgent,
    SentimentAgent, LLMAnalysisAgent, PeerComparisonAgent,
    ChartGeneratorAgent, TechnicalAnalysisAgent,
    EnhancedChartGeneratorAgent, ScreenerAgent,
    SocialSentimentAgent, ReportGeneratorAgent,
)

logger = logging.getLogger(__name__)

@register_swarm("research")
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

    def _init_shared_resources(self) -> None:
        """Initialize shared swarm resources."""
        if self._initialized:
            return

        # Auto-configure DSPy LM if needed
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            # Try direct Anthropic API first (fastest, no subprocess)
            try:
                from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
                if is_api_key_available():
                    lm = DirectAnthropicLM(model="haiku", max_tokens=8192)
                    dspy.configure(lm=lm)
                    logger.info(" Auto-configured DSPy with DirectAnthropicLM")
                else:
                    raise ValueError("No API key")
            except Exception:
                # Fallback to Claude CLI
                try:
                    from Jotty.core.integration.direct_claude_cli_lm import DirectClaudeCLI
                    lm = DirectClaudeCLI()
                    dspy.configure(lm=lm)
                    logger.info(" Auto-configured DSPy with DirectClaudeCLI")
                except Exception as e:
                    logger.warning(f"Could not configure DSPy LM: {e}")

        # Initialize shared resources
        try:
            from Jotty.core.agents.dag_agents import SwarmResources
            from Jotty.core.foundation.data_structures import SwarmConfig

            jotty_config = SwarmConfig()
            resources = SwarmResources.get_instance(jotty_config)

            self._memory = resources.memory
            self._context = resources.context
            self._bus = resources.bus

            logger.info(" Shared swarm resources initialized")
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

        Uses _safe_execute_domain to handle try/except, timing, and
        post-execution learning automatically via PhaseExecutor.

        Args:
            query: Company name or ticker
            ticker: Explicit ticker symbol
            exchange: Exchange (NSE, BSE, NYSE)
            send_telegram: Override Telegram setting

        Returns:
            ResearchResult with all data and analysis
        """
        # Parse inputs before entering executor
        ticker = ticker or self._extract_ticker(query)
        exchange = exchange or self.config.exchange
        send_tg = send_telegram if send_telegram is not None else self.config.send_telegram

        logger.info(f"Research Swarm starting for {ticker} ({exchange})")
        logger.info(f"   Config: LLM={self.config.use_llm_analysis}, Peers={self.config.include_peers}, Sentiment={self.config.include_sentiment}")

        # Default tools used across research phases
        default_tools = [
            'data_fetch', 'web_search', 'sentiment_analysis',
            'llm_analysis', 'peer_comparison', 'chart_generation',
            'report_generation', 'technical_analysis', 'screener',
        ]

        async def _execute_fn(executor: PhaseExecutor) -> ResearchResult:
            return await self._execute_phases(executor, ticker, exchange, send_tg)

        def _output_data_fn(result: ResearchResult) -> Dict[str, Any]:
            return {
                'ticker': result.ticker,
                'rating': result.rating,
                'confidence': str(result.rating_confidence),
                'sentiment': result.sentiment_label,
                'success': str(result.success),
            }

        def _input_data_fn() -> Dict[str, Any]:
            return {'ticker': ticker, 'exchange': exchange, 'query': query}

        # Use _safe_execute_domain for try/except + learning boilerplate.
        # Since ResearchResult is not a SwarmResult subclass, we wrap with
        # custom error handling.
        executor = self._phase_executor()
        try:
            result = await _execute_fn(executor)

            # Record post-execution learning (success path)
            exec_time = executor.elapsed()
            await self._post_execute_learning(
                success=result.success if hasattr(result, 'success') else True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(default_tools),
                task_type='research',
                output_data=_output_data_fn(result),
                input_data=_input_data_fn(),
            )
            self._learning_recorded = True
            return result

        except Exception as e:
            logger.error(f"Research swarm error: {e}")
            traceback.print_exc()
            exec_time = executor.elapsed()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(default_tools),
                task_type='research',
            )
            self._learning_recorded = True
            return ResearchResult(
                success=False,
                ticker=ticker,
                company_name=ticker,
                error=str(e),
                execution_time=exec_time,
            )

    async def _execute_phases(
        self,
        executor: PhaseExecutor,
        ticker: str,
        exchange: str,
        send_tg: bool,
    ) -> ResearchResult:
        """
        Domain-specific phase logic using PhaseExecutor for tracing.

        Phases:
            1. Parallel Data Collection (DataFetcher, WebSearch, Screener, Technical)
            2. Parallel Analysis (Sentiment, SocialSentiment, Peers, LLM)
            3. Chart Generation
            4. Report Generation

        Args:
            executor: PhaseExecutor for phase tracing and timing
            ticker: Stock ticker symbol
            exchange: Exchange (NSE, BSE, NYSE)
            send_tg: Whether to send result via Telegram

        Returns:
            ResearchResult with all data and analysis
        """
        # =================================================================
        # PHASE 1: PARALLEL DATA COLLECTION
        # =================================================================
        if self.config.parallel_fetch:
            # Build parallel task list for data collection
            fetch_task_list = [
                ("DataFetcher", AgentRole.ACTOR,
                 self._data_fetcher.fetch(ticker, exchange),
                 ['data_fetch']),
                ("WebSearch", AgentRole.ACTOR,
                 self._web_searcher.search(ticker, self.config.max_web_results),
                 ['web_search']),
            ]
            if self.config.use_screener:
                fetch_task_list.append(
                    ("Screener", AgentRole.ACTOR,
                     self._screener_agent.fetch(ticker),
                     ['screener'])
                )
            if self.config.include_technical:
                fetch_task_list.append(
                    ("TechnicalAnalyzer", AgentRole.ACTOR,
                     self._technical_analyzer.analyze(ticker, self.config.technical_timeframes),
                     ['technical_analysis'])
                )

            fetch_results = await executor.run_parallel(
                1, "Parallel Data Collection", fetch_task_list,
            )

            financial_data = fetch_results[0] if not isinstance(fetch_results[0], dict) or 'error' not in fetch_results[0] else {}
            web_data = fetch_results[1] if not isinstance(fetch_results[1], dict) or 'error' not in fetch_results[1] else {}

            # Extract optional results by position
            idx = 2
            screener_data = {}
            if self.config.use_screener:
                r = fetch_results[idx]
                screener_data = r if not (isinstance(r, dict) and 'error' in r) else {}
                idx += 1

            technical_data = {}
            if self.config.include_technical and idx < len(fetch_results):
                r = fetch_results[idx]
                technical_data = r if not (isinstance(r, dict) and 'error' in r) else {}
        else:
            # Sequential fetching with individual phase tracing
            financial_data = await executor.run_phase(
                1, "Data Fetching", "DataFetcher", AgentRole.ACTOR,
                self._data_fetcher.fetch(ticker, exchange),
                input_data={'ticker': ticker, 'exchange': exchange},
                tools_used=['data_fetch'],
            )
            web_data = await executor.run_phase(
                1, "Web Search", "WebSearch", AgentRole.ACTOR,
                self._web_searcher.search(ticker, self.config.max_web_results),
                input_data={'ticker': ticker},
                tools_used=['web_search'],
            )
            screener_data = {}
            if self.config.use_screener:
                screener_data = await executor.run_phase(
                    1, "Screener Fetch", "Screener", AgentRole.ACTOR,
                    self._screener_agent.fetch(ticker),
                    input_data={'ticker': ticker},
                    tools_used=['screener'],
                )
            technical_data = {}
            if self.config.include_technical:
                technical_data = await executor.run_phase(
                    1, "Technical Analysis", "TechnicalAnalyzer", AgentRole.ACTOR,
                    self._technical_analyzer.analyze(ticker, self.config.technical_timeframes),
                    input_data={'ticker': ticker},
                    tools_used=['technical_analysis'],
                )

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
        # Helper for creating async results (Python 3.11+ compatible)
        async def async_result(value):
            return value

        # Build analysis task list
        analysis_task_list = []

        # Sentiment analysis
        if self.config.include_sentiment and web_data.get('news_text'):
            analysis_task_list.append(
                ("Sentiment", AgentRole.ACTOR,
                 self._sentiment_analyzer.analyze(
                     merged_data.get('company_name', ticker),
                     web_data.get('news_text', '')
                 ),
                 ['sentiment_analysis'])
            )
        else:
            analysis_task_list.append(
                ("Sentiment", AgentRole.ACTOR,
                 async_result({'sentiment_score': 0, 'sentiment_label': 'NEUTRAL'}),
                 ['sentiment_analysis'])
            )

        # Social Sentiment analysis
        if self.config.include_social_sentiment and web_data.get('news_text'):
            analysis_task_list.append(
                ("SocialSentiment", AgentRole.ACTOR,
                 self._social_sentiment_agent.analyze(
                     merged_data.get('company_name', ticker),
                     web_data.get('news_text', ''),
                     ''  # Forum text - can be extended to scrape forums
                 ),
                 ['social_sentiment'])
            )
        else:
            analysis_task_list.append(
                ("SocialSentiment", AgentRole.ACTOR,
                 async_result({'overall_sentiment': 0, 'sentiment_label': 'NEUTRAL', 'sentiment_drivers': {}}),
                 ['social_sentiment'])
            )

        # Peer comparison
        if self.config.include_peers:
            analysis_task_list.append(
                ("PeerComparator", AgentRole.ACTOR,
                 self._peer_comparator.compare(
                     ticker,
                     merged_data.get('sector', ''),
                     merged_data.get('industry', ''),
                     exchange
                 ),
                 ['peer_comparison'])
            )
        else:
            analysis_task_list.append(
                ("PeerComparator", AgentRole.ACTOR,
                 async_result({'peers': [], 'comparison': {}}),
                 ['peer_comparison'])
            )

        # LLM analysis
        if self.config.use_llm_analysis:
            analysis_task_list.append(
                ("LLMAnalyzer", AgentRole.ACTOR,
                 self._llm_analyzer.analyze(merged_data, web_data.get('news_text', '')),
                 ['llm_analysis'])
            )
        else:
            analysis_task_list.append(
                ("LLMAnalyzer", AgentRole.ACTOR,
                 async_result(self._rule_based_analysis(merged_data)),
                 ['llm_analysis'])
            )

        analysis_results = await executor.run_parallel(
            2, "Parallel Analysis", analysis_task_list,
        )

        sentiment_result = analysis_results[0] if not (isinstance(analysis_results[0], dict) and 'error' in analysis_results[0]) else {}
        social_sentiment_result = analysis_results[1] if not (isinstance(analysis_results[1], dict) and 'error' in analysis_results[1]) else {}
        peer_result = analysis_results[2] if not (isinstance(analysis_results[2], dict) and 'error' in analysis_results[2]) else {}
        llm_result = analysis_results[3] if not (isinstance(analysis_results[3], dict) and 'error' in analysis_results[3]) else {}

        # =================================================================
        # PHASE 3: CHART GENERATION (if enabled)
        # =================================================================
        chart_paths = []
        if self.config.include_charts:
            chart_result = await executor.run_phase(
                3, "Chart Generation", "ChartGenerator", AgentRole.ACTOR,
                self._chart_generator.generate(
                    ticker,
                    merged_data,
                    self.config.output_dir,
                    timeframes=self.config.technical_timeframes,
                    technical_data=technical_data,
                    include_heiken_ashi=self.config.include_heiken_ashi
                ),
                input_data={'ticker': ticker},
                tools_used=['chart_generation'],
            )
            chart_paths = chart_result.get('chart_paths', []) if isinstance(chart_result, dict) else []

        # =================================================================
        # PHASE 4: REPORT GENERATION + OUTPUT
        # =================================================================
        report_result = await executor.run_phase(
            4, "Report Generation", "ReportGenerator", AgentRole.ACTOR,
            self._report_generator.generate(
                ticker=ticker,
                data=merged_data,
                analysis=llm_result,
                sentiment=sentiment_result,
                peers=peer_result,
                chart_paths=chart_paths,
                output_dir=self.config.output_dir,
                send_telegram=send_tg
            ),
            input_data={'ticker': ticker},
            tools_used=['report_generation'],
        )

        # =================================================================
        # BUILD RESULT
        # =================================================================
        exec_time = executor.elapsed()

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
            # Enhanced fields
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

        logger.info(f"Research complete: {ticker} = {result.rating} ({result.current_price:,.2f})")
        logger.info(f"   Sentiment: {result.sentiment_label} ({result.sentiment_score:+.2f})")
        logger.info(f"   Time: {exec_time:.1f}s | Telegram: {result.telegram_sent}")

        return result

    async def research_topic(
        self,
        topic: str,
        instruction: str,
        sub_queries: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        max_web_results: int = 25,
        send_telegram: Optional[bool] = None,
    ) -> TopicResearchResult:
        """
        Research a general topic (non-stock) using ResearchSwarm's WebSearch + LLM synthesis.

        Uses WebSearchAgent.search_topic(), topic synthesis, then document-converter and
        telegram-sender skills (same as stock research reports).

        Args:
            topic: Main topic (e.g. "dog welfare and safety").
            instruction: What to produce (e.g. "5th grade summary, 3 sections: rabies, diseases, bite first aid").
            sub_queries: Optional list of search queries (defaults to topic + instruction-derived phrases).
            output_dir: Where to write the markdown report (default: config.output_dir).
            max_web_results: Max web results to fetch.
            send_telegram: If True, send PDF to Telegram via telegram-sender skill (default: from config.send_telegram).

        Returns:
            TopicResearchResult with summary, md_path, pdf_path, telegram_sent, etc.
        """
        start = time.time()
        out_dir = output_dir or getattr(self.config, 'output_dir', os.path.expanduser('~/jotty/reports'))
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Ensure agents (e.g. _web_searcher) are created
        if not getattr(self, '_agents_initialized', False):
            self._init_agents()

        try:
            # Phase 1: Web search (topic mode)
            logger.info(f"Topic research: WebSearch for '{topic}'")
            if sub_queries is None:
                sub_queries = [
                    topic,
                    f"{topic} do's and don'ts",
                    f"{topic} articles links",
                ]
            web_data = await self._web_searcher.search_topic(
                topic, sub_queries=sub_queries, max_results=max_web_results
            )
            web_text = web_data.get('news_text', '') or ''
            news_count = web_data.get('news_count', 0)
            news_items = web_data.get('news_items', [])
            data_sources = [n.get('url', '') for n in news_items if n.get('url')]

            if not web_text.strip():
                logger.warning("Topic research: no web results")
                return TopicResearchResult(
                    success=False,
                    topic=topic,
                    error="No web search results",
                    execution_time=time.time() - start,
                )

            # Phase 2: LLM synthesis (topic summary)
            logger.info("Topic research: synthesizing summary")
            from .signatures import TopicSynthesisSignature

            synthesizer = dspy.Predict(TopicSynthesisSignature)
            # Truncate for context window
            web_text_trimmed = web_text[:12000] if len(web_text) > 12000 else web_text
            result = synthesizer(
                topic=topic,
                instruction=instruction,
                web_text=web_text_trimmed,
            )
            summary = (result.summary or "").strip()

            if not summary:
                return TopicResearchResult(
                    success=False,
                    topic=topic,
                    error="LLM synthesis produced empty summary",
                    news_count=news_count,
                    execution_time=time.time() - start,
                )

            # Phase 3: Write markdown report
            safe_topic = re.sub(r'[^\w\s-]', '', topic)[:50].strip() or "topic"
            safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
            md_path = os.path.join(out_dir, f"research_{safe_topic}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {topic}\n\n")
                f.write(summary)
            logger.info(f"Topic research: wrote {md_path}")

            # Phase 4: PDF via document-converter skill (same as stock research)
            pdf_path = ""
            registry = None
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
                registry = get_skills_registry()
                registry.init()
                doc_skill = registry.get_skill('document-converter')
                if doc_skill and doc_skill.tools:
                    convert_tool = doc_skill.tools.get('convert_to_pdf_tool')
                    if convert_tool:
                        pdf_path = os.path.join(out_dir, f"research_{safe_topic}.pdf")
                        pdf_result = convert_tool({
                            'input_file': md_path,
                            'output_file': pdf_path,
                            'title': topic,
                            'page_size': 'a4',
                        })
                        if pdf_result.get('success') and os.path.exists(pdf_result.get('output_path', '')):
                            pdf_path = pdf_result.get('output_path', pdf_path)
                            logger.info(f"Topic research: PDF generated {pdf_path}")
                        else:
                            pdf_path = ""
                            logger.warning(f"Topic research: PDF conversion failed: {pdf_result.get('error', 'unknown')}")
            except Exception as pdf_err:
                logger.warning(f"Topic research: PDF generation skipped: {pdf_err}")

            # Phase 5: Send to Telegram via telegram-sender skill (same as stock research)
            telegram_sent = False
            send_tg = send_telegram if send_telegram is not None else getattr(self.config, 'send_telegram', True)
            if send_tg and pdf_path and registry:
                try:
                    telegram_skill = registry.get_skill('telegram-sender')
                    if telegram_skill and telegram_skill.tools:
                        send_tool = telegram_skill.tools.get('send_telegram_file_tool')
                        if send_tool:
                            import inspect
                            caption = f" {topic} – Research Swarm (5th grade summary)"
                            params = {'file_path': pdf_path, 'caption': caption}
                            if inspect.iscoroutinefunction(send_tool):
                                tg_result = await send_tool(params)
                            else:
                                tg_result = send_tool(params)
                            telegram_sent = tg_result.get('success', False)
                            if telegram_sent:
                                logger.info("Topic research: sent to Telegram")
                            else:
                                logger.warning(f"Topic research: Telegram send failed: {tg_result.get('error')}")
                except Exception as tg_err:
                    logger.warning(f"Topic research: Telegram send skipped: {tg_err}")

            return TopicResearchResult(
                success=True,
                topic=topic,
                summary=summary,
                md_path=md_path,
                pdf_path=pdf_path,
                telegram_sent=telegram_sent,
                news_count=news_count,
                data_sources=data_sources[:20],
                execution_time=time.time() - start,
            )

        except Exception as e:
            logger.exception("Topic research failed")
            return TopicResearchResult(
                success=False,
                topic=topic,
                error=str(e),
                execution_time=time.time() - start,
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

# Set AGENT_TEAM and SWARM_SIGNATURE now that all agent classes are defined
ResearchSwarm.SWARM_SIGNATURE = ResearchSwarmSignature
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
