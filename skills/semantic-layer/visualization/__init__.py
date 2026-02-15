"""
Visualization Layer

Automatic visualization generation using LIDA (Microsoft's LLM-powered visualization library).

Uses Claude CLI by default via core.llm (no API keys required).
Supports fallback to Anthropic API, Gemini, or OpenAI.

Integrates seamlessly with:
- SemanticLayer for NL-to-SQL-to-Viz pipelines
- ConnectorX for fast DataFrame loading
- MongoDB aggregation results
- Skills system for composite workflows
- Core LLM module for unified provider access

Architecture:
    Question → SemanticLayer → DataFrame → VisualizationLayer → Chart
                                    ↓
                              LIDA Manager
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                Summarize      Generate Goals   Visualize
                    │
                    ▼
              Claude CLI (core.llm)
"""

from .dashboard_planner import (
    ChartPlan,
    ChartWithInsight,
    CompleteDashboard,
    DashboardPlan,
    DashboardPlanner,
)
from .data_source import (
    DataFrameSource,
    DataSource,
    DataSourceFactory,
    MongoDBSource,
    SemanticDataSource,
)
from .layer import ChartResult, DataSummary, VisualizationGoal, VisualizationLayer
from .llm_provider import ClaudeLLMTextGenerator, get_lida_text_generator
from .renderers import (
    AltairRenderer,
    ChartRenderer,
    HTMLRenderer,
    MatplotlibRenderer,
    PlotlyRenderer,
    RendererFactory,
)
from .skill import DashboardSkill, DataAnalysisSkill, VisualizationSkill

__all__ = [
    # Core Layer
    "VisualizationLayer",
    "ChartResult",
    "VisualizationGoal",
    "DataSummary",
    # Data Sources
    "DataSource",
    "SemanticDataSource",
    "DataFrameSource",
    "MongoDBSource",
    "DataSourceFactory",
    # Renderers
    "ChartRenderer",
    "HTMLRenderer",
    "MatplotlibRenderer",
    "AltairRenderer",
    "PlotlyRenderer",
    "RendererFactory",
    # Skills
    "VisualizationSkill",
    "DataAnalysisSkill",
    "DashboardSkill",
    # LLM Provider
    "ClaudeLLMTextGenerator",
    "get_lida_text_generator",
    # Dashboard Planner
    "DashboardPlanner",
    "DashboardPlan",
    "ChartPlan",
    "ChartWithInsight",
    "CompleteDashboard",
]
