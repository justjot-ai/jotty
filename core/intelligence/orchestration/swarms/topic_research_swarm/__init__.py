"""Topic Research Swarm — Generic, topic-agnostic research.

Codified from the Paytm MYE / PhonePe DRHP research workflow into a
reusable swarm that can deeply research ANY topic through:
  Phase 0: Topic decomposition into sub-topics
  Phase 1: Parallel multi-source web research
  Phase 2: Data synthesis + structured findings
  Phase 3: Comparative analysis (if applicable)
  Phase 4: Fact checking
  Phase 5: Report compilation (Markdown / PPTX)

Quick usage::

    from Jotty.core.intelligence.orchestration.swarms.topic_research_swarm import (
        topic_research,
    )

    # Simple research
    result = await topic_research("Impact of AI on healthcare")

    # Comparison
    result = await topic_research(
        "PhonePe vs Paytm",
        mode="comparison",
        entities=["PhonePe", "Paytm"],
        dimensions=["Revenue", "Market Share", "Profitability", "Moat"],
    )
"""

from .agents import (
    ComparativeAnalystAgent,
    DataSynthesizerAgent,
    FactCheckerAgent,
    ReportCompilerAgent,
    TopicDecomposerAgent,
    WebResearcherAgent,
)
from .pptx_generator import generate_research_pptx
from .swarm import TopicResearchSwarm, topic_research, topic_research_sync
from .types import (
    ComparisonDimension,
    OutputFormat,
    ResearchDepth,
    ResearchFinding,
    ResearchMode,
    SourceReference,
    TopicResearchConfig,
    TopicResearchResult,
)

__all__ = [
    # Swarm
    "TopicResearchSwarm",
    "topic_research",
    "topic_research_sync",
    # Config & Result
    "TopicResearchConfig",
    "TopicResearchResult",
    "ResearchDepth",
    "ResearchMode",
    "OutputFormat",
    # Data classes
    "ResearchFinding",
    "SourceReference",
    "ComparisonDimension",
    # PPTX
    "generate_research_pptx",
    # Agents
    "TopicDecomposerAgent",
    "WebResearcherAgent",
    "DataSynthesizerAgent",
    "ComparativeAnalystAgent",
    "ReportCompilerAgent",
    "FactCheckerAgent",
]
