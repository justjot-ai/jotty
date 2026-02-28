"""Topic Research Swarm — Types, config, and data classes.

A generic, topic-agnostic research swarm that can deeply research ANY topic
through multi-source web research, synthesis, comparative analysis, and
structured report generation.

Codified from the Paytm Merchant Yield Engine / PhonePe DRHP research pattern:
  Phase 0: Topic decomposition
  Phase 1: Parallel multi-source web research
  Phase 2: Data synthesis + fact-checking
  Phase 3: Comparative analysis (if applicable)
  Phase 4: Report compilation (Markdown / PPTX)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from .._base.swarm_types import SwarmBaseConfig, SwarmResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class ResearchDepth(Enum):
    """How deep to go."""

    QUICK = "quick"  # 1 search per sub-topic, no comparison
    STANDARD = "standard"  # 2-3 searches, light synthesis
    DEEP = "deep"  # 5+ searches, full synthesis + fact-check
    EXHAUSTIVE = "exhaustive"  # 10+ searches, comparison, PPTX output


class OutputFormat(Enum):
    """Desired output format."""

    MARKDOWN = "markdown"
    PPTX = "pptx"
    PDF = "pdf"
    JSON = "json"


class ResearchMode(Enum):
    """Type of research."""

    SINGLE_TOPIC = "single_topic"  # Research one topic deeply
    COMPARISON = "comparison"  # Compare 2+ entities head-to-head
    LANDSCAPE = "landscape"  # Map an entire domain/market
    DEEP_DIVE = "deep_dive"  # Exhaustive analysis of one entity


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TopicResearchConfig(SwarmBaseConfig):
    """Configuration for topic research swarm.

    Extends SwarmBaseConfig to gain: version, enable_self_improvement,
    enable_learning, parallel_execution, max_retries, timeout_seconds,
    improvement_threshold.
    """

    # Research parameters
    depth: ResearchDepth = ResearchDepth.DEEP
    mode: ResearchMode = ResearchMode.SINGLE_TOPIC
    output_format: OutputFormat = OutputFormat.MARKDOWN
    max_sub_topics: int = 8
    max_web_results_per_query: int = 10
    max_total_sources: int = 50
    include_comparison: bool = False
    comparison_entities: List[str] = field(default_factory=list)
    comparison_dimensions: List[str] = field(default_factory=list)

    # Output options
    send_telegram: bool = False
    generate_pptx: bool = False
    target_report_pages: int = 15

    # Quality
    enable_fact_checking: bool = True
    min_sources_per_claim: int = 2
    require_recent_data: bool = True
    max_data_age_days: int = 90

    def __post_init__(self) -> None:
        self.name = "TopicResearchSwarm"
        self.domain = "topic_research"
        self.output_dir = os.path.expanduser("~/jotty/research")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SourceReference:
    """A single source used in research."""

    url: str = ""
    title: str = ""
    snippet: str = ""
    date: str = ""
    reliability_score: float = 0.5  # 0-1


@dataclass
class ResearchFinding:
    """A single finding from research."""

    claim: str = ""
    evidence: str = ""
    confidence: float = 0.0  # 0-1
    sources: List[SourceReference] = field(default_factory=list)
    sub_topic: str = ""


@dataclass
class ComparisonDimension:
    """One dimension of a head-to-head comparison."""

    dimension: str = ""
    entity_scores: Dict[str, str] = field(default_factory=dict)  # entity → value
    winner: str = ""
    analysis: str = ""


@dataclass
class TopicResearchResult(SwarmResult):
    """Result from topic research swarm.

    Inherits from SwarmResult: success, swarm_name, domain, output,
    execution_time, error, agent_traces, metadata.
    """

    # Research identity
    topic: str = ""
    sub_topics: List[str] = field(default_factory=list)
    mode: str = "single_topic"
    depth: str = "deep"

    # Findings
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    detailed_findings: List[ResearchFinding] = field(default_factory=list)

    # Comparison (if mode=comparison)
    comparison_entities: List[str] = field(default_factory=list)
    comparison_matrix: List[ComparisonDimension] = field(default_factory=list)
    comparison_verdict: str = ""

    # Sources
    sources: List[SourceReference] = field(default_factory=list)
    total_sources_consulted: int = 0
    unique_domains: int = 0

    # Outputs
    report_markdown: str = ""
    md_path: str = ""
    pptx_path: str = ""
    pdf_path: str = ""
    telegram_sent: bool = False

    # Quality
    fact_check_score: float = 0.0  # 0-1
    completeness_score: float = 0.0  # 0-1
    data_recency_score: float = 0.0  # 0-1

    # Agent contributions
    agent_contributions: Dict[str, float] = field(default_factory=dict)
