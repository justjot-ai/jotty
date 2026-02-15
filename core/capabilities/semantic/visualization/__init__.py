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
from .layer import (
    VisualizationLayer,
    ChartResult,
    VisualizationGoal,
    DataSummary,
)
from .data_source import (
    DataSource,
    SemanticDataSource,
    DataFrameSource,
    MongoDBSource,
    DataSourceFactory,
)
from .renderers import (
    ChartRenderer,
    HTMLRenderer,
    MatplotlibRenderer,
    AltairRenderer,
    PlotlyRenderer,
    RendererFactory,
)
from .skill import (
    VisualizationSkill,
    DataAnalysisSkill,
    DashboardSkill,
)
from .llm_provider import (
    ClaudeLLMTextGenerator,
    get_lida_text_generator,
)
from .dashboard_planner import (
    DashboardPlanner,
    DashboardPlan,
    ChartPlan,
    ChartWithInsight,
    CompleteDashboard,
)

__all__ = [
    # Core Layer
    'VisualizationLayer',
    'ChartResult',
    'VisualizationGoal',
    'DataSummary',

    # Data Sources
    'DataSource',
    'SemanticDataSource',
    'DataFrameSource',
    'MongoDBSource',
    'DataSourceFactory',

    # Renderers
    'ChartRenderer',
    'HTMLRenderer',
    'MatplotlibRenderer',
    'AltairRenderer',
    'PlotlyRenderer',
    'RendererFactory',

    # Skills
    'VisualizationSkill',
    'DataAnalysisSkill',
    'DashboardSkill',

    # LLM Provider
    'ClaudeLLMTextGenerator',
    'get_lida_text_generator',

    # Dashboard Planner
    'DashboardPlanner',
    'DashboardPlan',
    'ChartPlan',
    'ChartWithInsight',
    'CompleteDashboard',
]
