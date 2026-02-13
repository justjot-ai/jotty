"""Research Swarm - Types, config, and data classes."""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..swarm_types import SwarmBaseConfig

logger = logging.getLogger(__name__)

class RatingType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class ResearchConfig(SwarmBaseConfig):
    """Configuration for research swarm.

    Extends SwarmBaseConfig to gain: version, enable_self_improvement,
    enable_learning, parallel_execution, max_retries, timeout_seconds,
    gold_standard_path, improvement_threshold, gold_standard_max_version.
    """
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
    # Enhanced options
    include_technical: bool = True
    technical_timeframes: List[str] = field(default_factory=lambda: ["60minute", "Day"])
    use_screener: bool = True
    include_social_sentiment: bool = True
    include_heiken_ashi: bool = False
    nse_data_path: str = "/var/www/sites/personal/stock_market/common/Data/NSE/"

    def __post_init__(self):
        self.name = "ResearchSwarm"
        self.domain = "research"
        self.output_dir = os.path.expanduser('~/jotty/reports')


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


@dataclass
class TopicResearchResult:
    """Result from ResearchSwarm.research_topic() for general-topic research."""
    success: bool
    topic: str
    summary: str = ""
    md_path: str = ""
    pdf_path: str = ""
    pptx_path: str = ""
    telegram_sent: bool = False
    news_count: int = 0
    data_sources: List[str] = field(default_factory=list)
    error: str = ""
    execution_time: float = 0.0


# =============================================================================
# DSPy SIGNATURES FOR LLM-POWERED ANALYSIS
# =============================================================================

