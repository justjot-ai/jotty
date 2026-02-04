"""
Jotty World-Class Swarm Templates
==================================

Production-grade multi-agent swarms with:
- LLM-powered analysis (DSPy Chain of Thought)
- Parallel agent execution
- Shared resources (memory, context, bus)
- Learning from past research

Available Swarms:
- ResearchSwarm: Comprehensive stock research with sentiment, peers, charts

Usage:
    from core.swarms import ResearchSwarm, research

    # Full-featured
    swarm = ResearchSwarm()
    result = await swarm.research("Paytm", send_telegram=True)

    # One-liner
    result = await research("RELIANCE")

    # Sync version
    from core.swarms import research_sync
    result = research_sync("TCS")
"""

from .research_swarm import (
    # Main swarm
    ResearchSwarm,
    ResearchConfig,
    ResearchResult,
    RatingType,

    # Convenience functions
    research,
    research_sync,

    # Individual agents (for customization)
    BaseAgent,
    DataFetcherAgent,
    WebSearchAgent,
    SentimentAgent,
    LLMAnalysisAgent,
    PeerComparisonAgent,
    ChartGeneratorAgent,
    ReportGeneratorAgent,

    # DSPy Signatures (for extension)
    StockAnalysisSignature,
    SentimentAnalysisSignature,
    PeerSelectionSignature,
)

__all__ = [
    # Main swarm
    "ResearchSwarm",
    "ResearchConfig",
    "ResearchResult",
    "RatingType",

    # Convenience functions
    "research",
    "research_sync",

    # Agents
    "BaseAgent",
    "DataFetcherAgent",
    "WebSearchAgent",
    "SentimentAgent",
    "LLMAnalysisAgent",
    "PeerComparisonAgent",
    "ChartGeneratorAgent",
    "ReportGeneratorAgent",

    # DSPy Signatures
    "StockAnalysisSignature",
    "SentimentAnalysisSignature",
    "PeerSelectionSignature",
]
