"""
JustJot Idea Builder

Orchestrates the creation of complete JustJot visualization ideas
with multiple sections (chart, data, code, insights).

This is the main interface for LIDA-JustJot integration.

Example:
    from core.semantic.visualization import VisualizationLayer
    from core.semantic.visualization.justjot import JustJotIdeaBuilder

    viz = VisualizationLayer.from_dataframe(df)
    builder = JustJotIdeaBuilder(viz)

    # Create a complete idea
    idea = builder.create_visualization_idea(
        question="Analyze sales by region",
        include_data=True,
        include_code=True,
        include_insights=True
    )

    # The idea can be saved via MCP
    print(idea.to_dict())
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .section_types import DEFAULT_COLORS, JustJotIdea, JustJotSection, get_section_types_context
from .transformers import ChartTransformer, SectionTransformer

logger = logging.getLogger(__name__)


@dataclass
class VisualizationIdeaConfig:
    """Configuration for visualization idea creation."""

    # Section options
    include_data: bool = True
    include_chart: bool = True
    include_code: bool = True
    include_insights: bool = True
    include_executive_summary: bool = True

    # Chart options
    interactive: bool = True  # Use interactive HTML for Plotly/Altair

    # Data options
    max_data_rows: int = 100

    # LLM options
    llm_provider: str = "claude-cli"
    llm_model: str = ""  # "" → resolved from config_defaults

    # Styling
    colors: List[str] = None

    def __post_init__(self) -> None:
        if not self.llm_model:
            from Jotty.core.infrastructure.foundation.config_defaults import DEFAULT_MODEL_ALIAS

            self.llm_model = DEFAULT_MODEL_ALIAS


class JustJotIdeaBuilder:
    """
    Build complete JustJot visualization ideas from LIDA outputs.

    This class orchestrates the transformation of LIDA visualizations
    into JustJot ideas with multiple sections.

    Features:
    - Automatic chart generation with LLM
    - Data table inclusion
    - Code section with visualization code
    - LLM-generated insights
    - Executive summary for dashboards
    """

    def __init__(self, viz_layer: Any, config: VisualizationIdeaConfig = None) -> None:
        """
        Initialize the idea builder.

        Args:
            viz_layer: VisualizationLayer instance
            config: Configuration options
        """
        self.viz_layer = viz_layer
        self.config = config or VisualizationIdeaConfig()

        # Initialize transformers
        colors = self.config.colors or DEFAULT_COLORS
        self.section_transformer = SectionTransformer()
        self.chart_transformer = ChartTransformer(colors=colors)

        # Lazy import LLM
        self._llm_generate = None

    @property
    def llm_generate(self) -> Any:
        """Lazy import of core.llm.generate."""
        if self._llm_generate is None:
            from core.llm import generate

            self._llm_generate = generate
        return self._llm_generate

    def create_visualization_idea(
        self,
        question: str,
        title: str = None,
        description: str = None,
        tags: List[str] = None,
        **kwargs: Any,
    ) -> JustJotIdea:
        """
        Create a complete visualization idea from a natural language question.

        Args:
            question: Natural language question (e.g., "Show sales by region")
            title: Idea title (auto-generated if not provided)
            description: Idea description
            tags: Tags for the idea
            **kwargs: Override config options

        Returns:
            JustJotIdea with all sections

        Example:
            idea = builder.create_visualization_idea(
                "Analyze revenue by product category",
                tags=["analytics", "revenue"]
            )
        """
        # Merge kwargs with config
        include_data = kwargs.get("include_data", self.config.include_data)
        include_chart = kwargs.get("include_chart", self.config.include_chart)
        include_code = kwargs.get("include_code", self.config.include_code)
        include_insights = kwargs.get("include_insights", self.config.include_insights)

        # Get the data
        df = self.viz_layer.data_source.query("").to_dataframe()

        # Generate visualization
        logger.info(f"Generating visualization for: {question}")
        library = kwargs.get("library", "plotly" if self.config.interactive else "matplotlib")

        charts = self.viz_layer.visualize(question=question, library=library, n=1)

        if not charts or not charts[0].success:
            error = charts[0].error if charts else "No charts generated"
            logger.error(f"Visualization failed: {error}")
            # Return idea with error message
            return self._create_error_idea(question, str(error), tags)

        chart = charts[0]

        # Generate title if not provided
        if not title:
            title = self._generate_title(question, chart)

        # Create the idea
        idea = JustJotIdea(
            title=title,
            description=description or f"AI-generated visualization for: {question}",
            tags=tags or ["lida", "visualization", "ai-generated"],
            template_name="Blank",
            status="Draft",
        )

        # Add sections based on config
        sections_added = []

        # 1. Data section
        if include_data:
            data_section = self._create_data_section(df)
            idea.add_section(data_section)
            sections_added.append("data")

        # 2. Chart section
        if include_chart:
            chart_section = self._create_chart_section(chart, df, question)
            idea.add_section(chart_section)
            sections_added.append("chart")

        # 3. Code section
        if include_code and chart.code:
            code_section = self._create_code_section(chart)
            idea.add_section(code_section)
            sections_added.append("code")

        # 4. Insights section
        if include_insights:
            insight_section = self._create_insight_section(question, chart, df)
            idea.add_section(insight_section)
            sections_added.append("insights")

        logger.info(f"Created idea with sections: {sections_added}")
        return idea

    def create_dashboard_idea(
        self, user_request: str, num_charts: int = 4, title: str = None, tags: List[str] = None
    ) -> JustJotIdea:
        """
        Create a multi-chart dashboard idea.

        Args:
            user_request: User's analysis request
            num_charts: Number of charts to generate
            title: Dashboard title
            tags: Tags for the idea

        Returns:
            JustJotIdea with multiple chart sections
        """
        # Import dashboard planner
        from ..dashboard_planner import DashboardPlanner

        df = self.viz_layer.data_source.query("").to_dataframe()

        # Create dashboard
        library = "plotly" if self.config.interactive else "matplotlib"
        planner = DashboardPlanner(
            self.viz_layer,
            library=library,
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )

        dashboard = planner.create_dashboard(
            user_request,
            num_charts=num_charts,
            include_insights=True,
            include_summary=self.config.include_executive_summary,
        )

        if not dashboard.success:
            return self._create_error_idea(user_request, dashboard.error, tags)

        # Create idea
        idea = JustJotIdea(
            title=title or dashboard.plan.title,
            description=dashboard.plan.description,
            tags=tags or ["dashboard", "lida", "ai-generated"],
            template_name="Blank",
            status="Draft",
        )

        # Add executive summary
        if dashboard.executive_summary:
            summary_section = self.section_transformer.transform_text(
                dashboard.executive_summary, "Executive Summary"
            )
            idea.add_section(summary_section)

        # Add data section
        if self.config.include_data:
            data_section = self._create_data_section(df)
            idea.add_section(data_section)

        # Add each chart with its insight
        for i, cwi in enumerate(dashboard.charts):
            if cwi.chart and cwi.chart.success:
                # Chart section
                chart_section = self._create_chart_section(
                    cwi.chart, df, cwi.title, interactive_html=cwi.interactive_html
                )
                idea.add_section(chart_section)

                # Insight for this chart
                if cwi.insight:
                    insight_section = self.section_transformer.transform_text(
                        cwi.insight, f"Insight: {cwi.title}"
                    )
                    idea.add_section(insight_section)

        return idea

    def _create_data_section(self, df: Any) -> JustJotSection:
        """Create data table section."""
        return self.section_transformer.transform_dataframe(
            df, section_type="data-table", title="Source Data", max_rows=self.config.max_data_rows
        )

    def _create_chart_section(
        self, chart: Any, df: Any, title: str, interactive_html: str = None
    ) -> JustJotSection:
        """Create chart section based on config."""
        # If we have interactive HTML and config wants it, use HTML section
        if interactive_html and self.config.interactive:
            return self.section_transformer.transform_html(interactive_html, title)

        # Use ChartTransformer for LIDA code analysis
        return self.chart_transformer.transform(chart, df, title=title)

    def _create_code_section(self, chart: Any) -> JustJotSection:
        """Create code section."""
        code = getattr(chart, "code", "") or ""
        return self.section_transformer.transform_code(code, "Visualization Code")

    def _create_insight_section(self, question: str, chart: Any, df: Any) -> JustJotSection:
        """Generate and create insight section using LLM."""
        # Get data summary
        data_summary = df.describe().to_string()

        prompt = f"""Analyze this visualization and provide key insights.

QUESTION: {question}

DATA SUMMARY:
{data_summary}

GENERATED CODE:
{chart.code[:500] if chart.code else 'No code available'}

Provide 2-3 sentences of actionable insights:
1. What is the main finding from this visualization?
2. Are there any notable patterns or outliers?
3. What business implication does this suggest?

Keep it concise and professional. No markdown formatting."""

        response = self.llm_generate(
            prompt=prompt,
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            timeout=60,
        )

        insight_text = response.text if response.success else "Insight generation failed."

        return self.section_transformer.transform_text(insight_text, "Key Insights")

    def _generate_title(self, question: str, chart: Any) -> str:
        """Generate a concise title for the idea."""
        # Try to extract from chart code
        if chart.code:
            import re

            title_match = re.search(r'title\s*=\s*[\'"]([^\'"]+)[\'"]', chart.code)
            if title_match:
                return title_match.group(1)

        # Generate from question
        words = question.split()[:6]
        return " ".join(words).title()

    def _create_error_idea(self, question: str, error: str, tags: List[str] = None) -> JustJotIdea:
        """Create an idea with error message."""
        idea = JustJotIdea(
            title=f"Failed: {question[:50]}",
            description=f"Visualization generation failed: {error}",
            tags=tags or ["error", "lida"],
            status="Draft",
        )

        error_section = self.section_transformer.transform_text(
            f"## Error\n\nVisualization generation failed:\n\n```\n{error}\n```", "Error Details"
        )
        idea.add_section(error_section)

        return idea

    def to_justjot_mcp_format(self, idea: JustJotIdea) -> Dict:
        """
        Convert idea to format expected by JustJot MCP create_idea tool.

        Returns dict ready for mcp__justjot__create_idea.
        """
        return {
            "title": idea.title,
            "description": idea.description,
            "tags": idea.tags,
            "templateName": idea.template_name,
            "status": idea.status,
            "sections": [
                {
                    "title": s.title,
                    "type": s.type.value if hasattr(s.type, "value") else s.type,
                    "content": s.content,
                }
                for s in idea.sections
            ],
        }


# ============================================
# Convenience Functions
# ============================================


def create_quick_visualization_idea(
    df: Any, question: str, title: str = None, interactive: bool = True
) -> JustJotIdea:
    """
    Quick function to create a visualization idea from DataFrame.

    Args:
        df: pandas DataFrame
        question: Natural language question
        title: Idea title (optional)
        interactive: Use interactive Plotly charts

    Returns:
        JustJotIdea ready for MCP
    """
    from ..layer import VisualizationLayer

    viz = VisualizationLayer.from_dataframe(df)
    config = VisualizationIdeaConfig(interactive=interactive)
    builder = JustJotIdeaBuilder(viz, config)

    return builder.create_visualization_idea(question, title=title)


def create_quick_dashboard_idea(
    df: Any, user_request: str, num_charts: int = 4, title: str = None
) -> JustJotIdea:
    """
    Quick function to create a dashboard idea from DataFrame.

    Args:
        df: pandas DataFrame
        user_request: Analysis request
        num_charts: Number of charts
        title: Dashboard title

    Returns:
        JustJotIdea with multiple charts
    """
    from ..layer import VisualizationLayer

    viz = VisualizationLayer.from_dataframe(df)
    config = VisualizationIdeaConfig(interactive=True, include_executive_summary=True)
    builder = JustJotIdeaBuilder(viz, config)

    return builder.create_dashboard_idea(user_request, num_charts=num_charts, title=title)
