"""
Creative Dashboard Planner

Uses Claude to intelligently plan and generate comprehensive dashboards
with multiple complementary charts and LLM-generated insights.

Features:
- Analyzes user question and data to plan optimal chart combinations
- Generates multiple complementary visualizations
- Creates LLM summaries and insights for each chart
- Produces cohesive, story-telling dashboards
"""
import logging
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ChartPlan:
    """Plan for a single chart."""
    title: str
    question: str
    chart_type: str
    rationale: str
    columns: List[str] = field(default_factory=list)
    aggregation: str = None
    priority: int = 1


@dataclass
class DashboardPlan:
    """Complete dashboard plan."""
    title: str
    description: str
    charts: List[ChartPlan]
    layout: str = "grid"  # grid, vertical, horizontal
    theme: str = "dark"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChartWithInsight:
    """Chart with LLM-generated insight."""
    chart: Any  # ChartResult
    insight: str
    title: str
    data_summary: str = None
    library: str = "matplotlib"
    interactive_html: str = None  # For Plotly/Altair interactive charts


@dataclass
class CompleteDashboard:
    """Complete dashboard with all charts and insights."""
    plan: DashboardPlan
    charts: List[ChartWithInsight]
    executive_summary: str
    html: str = None
    success: bool = True
    error: str = None


class DashboardPlanner:
    """
    Intelligent dashboard planner that uses Claude to create
    comprehensive, insightful dashboards.

    Usage:
        from core.semantic.visualization import VisualizationLayer
        from core.semantic.visualization.dashboard_planner import DashboardPlanner

        viz = VisualizationLayer.from_dataframe(df)
        planner = DashboardPlanner(viz)

        dashboard = planner.create_dashboard(
            "Analyze sales performance across regions and products"
        )
    """

    def __init__(
        self,
        viz_layer,
        provider: str = "claude-cli",
        model: str = "",
        max_charts: int = 6,
        library: str = "plotly",
        **kwargs
    ):
        """
        Initialize dashboard planner.

        Args:
            viz_layer: VisualizationLayer instance
            provider: LLM provider
            model: LLM model
            max_charts: Maximum charts per dashboard
            library: Chart library (plotly, matplotlib, altair, seaborn)
            **kwargs: Additional LLM options
        """
        from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS
        self.viz_layer = viz_layer
        self.provider = provider
        self.model = model or DEFAULT_MODEL_ALIAS
        self.max_charts = max_charts
        self.library = library
        self.kwargs = kwargs

        # Lazy import
        self._llm_generate = None

    @property
    def llm_generate(self):
        """Lazy import of core.llm.generate."""
        if self._llm_generate is None:
            from core.llm import generate
            self._llm_generate = generate
        return self._llm_generate

    def _get_data_context(self) -> str:
        """Get data context for planning."""
        df = self.viz_layer.data_source.query("").to_dataframe()

        context_parts = [
            f"Dataset: {len(df)} rows, {len(df.columns)} columns",
            f"\nColumns:",
        ]

        for col in df.columns:
            dtype = str(df[col].dtype)
            if df[col].dtype in ['int64', 'float64']:
                stats = f"min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
            elif df[col].dtype == 'object':
                unique = df[col].nunique()
                samples = df[col].unique()[:5].tolist()
                stats = f"unique={unique}, samples={samples}"
            else:
                stats = f"dtype={dtype}"
            context_parts.append(f"  - {col}: {stats}")

        # Sample data
        context_parts.append(f"\nSample data (first 3 rows):")
        context_parts.append(df.head(3).to_string())

        return "\n".join(context_parts)

    def plan_dashboard(
        self,
        user_request: str,
        num_charts: int = 4
    ) -> DashboardPlan:
        """
        Use Claude to plan a comprehensive dashboard.

        Args:
            user_request: User's analysis request
            num_charts: Target number of charts

        Returns:
            DashboardPlan with chart specifications
        """
        num_charts = min(num_charts, self.max_charts)
        data_context = self._get_data_context()

        prompt = f"""You are a data visualization expert. Plan a comprehensive dashboard based on the user's request.

USER REQUEST: {user_request}

DATA CONTEXT:
{data_context}

Create a dashboard plan with {num_charts} complementary charts that tell a complete story.
Each chart should reveal different insights and together they should provide comprehensive analysis.

Respond with a JSON object in this exact format:
{{
    "title": "Dashboard title",
    "description": "Brief dashboard description",
    "charts": [
        {{
            "title": "Chart 1 Title",
            "question": "Natural language question for this chart",
            "chart_type": "bar|line|pie|scatter|heatmap|histogram",
            "rationale": "Why this chart is important",
            "columns": ["col1", "col2"],
            "aggregation": "sum|mean|count|none",
            "priority": 1
        }},
        ...
    ],
    "layout": "grid",
    "theme": "dark"
}}

IMPORTANT:
- Use only columns that exist in the data
- Each chart should answer a different aspect of the user's question
- Order charts by importance (priority 1 = most important)
- Include a mix of chart types for visual variety
- Make questions specific and actionable

Respond ONLY with the JSON, no other text."""

        from Jotty.core.foundation.config_defaults import LLM_TIMEOUT_SECONDS
        response = self.llm_generate(
            prompt=prompt,
            provider=self.provider,
            model=self.model,
            timeout=LLM_TIMEOUT_SECONDS,
            **self.kwargs
        )

        if not response.success:
            raise RuntimeError(f"Planning failed: {response.error}")

        # Parse JSON response
        try:
            # Clean up response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            plan_data = json.loads(text)

            charts = [
                ChartPlan(
                    title=c.get("title", f"Chart {i+1}"),
                    question=c.get("question", ""),
                    chart_type=c.get("chart_type", "bar"),
                    rationale=c.get("rationale", ""),
                    columns=c.get("columns", []),
                    aggregation=c.get("aggregation"),
                    priority=c.get("priority", i+1)
                )
                for i, c in enumerate(plan_data.get("charts", []))
            ]

            return DashboardPlan(
                title=plan_data.get("title", "Dashboard"),
                description=plan_data.get("description", ""),
                charts=charts,
                layout=plan_data.get("layout", "grid"),
                theme=plan_data.get("theme", "dark")
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.error(f"Response was: {response.text[:500]}")
            raise RuntimeError(f"Failed to parse dashboard plan: {e}")

    def generate_chart_insight(
        self,
        chart,
        chart_plan: ChartPlan,
        df
    ) -> str:
        """Generate LLM insight for a chart."""
        # Get relevant data summary
        if chart_plan.columns:
            relevant_cols = [c for c in chart_plan.columns if c in df.columns]
            if relevant_cols:
                data_summary = df[relevant_cols].describe().to_string()
            else:
                data_summary = "Data summary not available"
        else:
            data_summary = df.describe().to_string()

        prompt = f"""Analyze this chart and provide a brief, insightful summary.

CHART: {chart_plan.title}
QUESTION: {chart_plan.question}
CHART TYPE: {chart_plan.chart_type}

DATA SUMMARY:
{data_summary}

Provide a 2-3 sentence insight that:
1. States the key finding from this visualization
2. Highlights any notable patterns or outliers
3. Suggests a potential business implication

Keep it concise and actionable. Do not use markdown formatting."""

        response = self.llm_generate(
            prompt=prompt,
            provider=self.provider,
            model=self.model,
            timeout=60,
            **self.kwargs
        )

        if response.success:
            return response.text.strip()
        return "Insight generation failed."

    def generate_executive_summary(
        self,
        plan: DashboardPlan,
        insights: List[str]
    ) -> str:
        """Generate executive summary for the entire dashboard."""
        insights_text = "\n".join([f"- {insight}" for insight in insights])

        prompt = f"""Create a brief executive summary for this dashboard.

DASHBOARD: {plan.title}
PURPOSE: {plan.description}

KEY INSIGHTS FROM CHARTS:
{insights_text}

Write a 3-4 sentence executive summary that:
1. States the main purpose of this analysis
2. Highlights the 2-3 most important findings
3. Provides a high-level recommendation or conclusion

Keep it professional and actionable. Do not use markdown formatting."""

        response = self.llm_generate(
            prompt=prompt,
            provider=self.provider,
            model=self.model,
            timeout=60,
            **self.kwargs
        )

        if response.success:
            return response.text.strip()
        return "Executive summary generation failed."

    def create_dashboard(
        self,
        user_request: str,
        num_charts: int = 4,
        include_insights: bool = True,
        include_summary: bool = True
    ) -> CompleteDashboard:
        """
        Create a complete dashboard with charts and insights.

        Args:
            user_request: User's analysis request
            num_charts: Number of charts to generate
            include_insights: Generate insights for each chart
            include_summary: Generate executive summary

        Returns:
            CompleteDashboard with all components
        """
        try:
            # Step 1: Plan the dashboard
            logger.info(f"Planning dashboard for: {user_request}")
            plan = self.plan_dashboard(user_request, num_charts)
            logger.info(f"Planned {len(plan.charts)} charts")

            # Step 2: Generate each chart
            charts_with_insights = []
            df = self.viz_layer.data_source.query("").to_dataframe()

            for i, chart_plan in enumerate(plan.charts):
                logger.info(f"Generating chart {i+1}/{len(plan.charts)}: {chart_plan.title}")

                try:
                    # Generate visualization with configured library
                    chart_results = self.viz_layer.visualize(
                        question=chart_plan.question,
                        library=self.library,
                        n=1
                    )

                    chart = chart_results[0] if chart_results else None

                    # Generate insight
                    insight = ""
                    if include_insights and chart and chart.success:
                        insight = self.generate_chart_insight(chart, chart_plan, df)

                    # Generate interactive HTML for Plotly/Altair
                    interactive_html = None
                    if chart and chart.success and chart.code and self.library in ["plotly", "altair"]:
                        interactive_html = self._generate_interactive_html(chart, chart_plan.title)

                    charts_with_insights.append(ChartWithInsight(
                        chart=chart,
                        insight=insight,
                        title=chart_plan.title,
                        library=self.library,
                        interactive_html=interactive_html
                    ))

                except Exception as e:
                    logger.error(f"Failed to generate chart {chart_plan.title}: {e}")
                    charts_with_insights.append(ChartWithInsight(
                        chart=None,
                        insight=f"Chart generation failed: {str(e)}",
                        title=chart_plan.title,
                        library=self.library
                    ))

            # Step 3: Generate executive summary
            executive_summary = ""
            if include_summary:
                insights = [c.insight for c in charts_with_insights if c.insight]
                executive_summary = self.generate_executive_summary(plan, insights)

            # Step 4: Render HTML dashboard
            html = self._render_dashboard_html(plan, charts_with_insights, executive_summary)

            return CompleteDashboard(
                plan=plan,
                charts=charts_with_insights,
                executive_summary=executive_summary,
                html=html,
                success=True
            )

        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return CompleteDashboard(
                plan=None,
                charts=[],
                executive_summary="",
                html=None,
                success=False,
                error=str(e)
            )

    def _generate_interactive_html(self, chart, title: str) -> Optional[str]:
        """
        Generate interactive HTML from chart code for Plotly/Altair.

        For Plotly: Executes the code and extracts the figure as HTML
        For Altair: Converts to Vega-Lite spec and embeds
        """
        if not chart.code:
            return None

        try:
            if self.library == "plotly":
                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd
                import numpy as np

                # Get the dataframe
                df = self.viz_layer.data_source.query("").to_dataframe()

                # Create execution namespace with all imports
                exec_namespace = {
                    'px': px,
                    'go': go,
                    'pd': pd,
                    'np': np,
                    'numpy': np,
                    'pandas': pd,
                    'df': df,
                    'data': df,
                    '__builtins__': __builtins__
                }

                # Execute the chart code
                exec(chart.code, exec_namespace)

                # Find the figure object - LIDA typically creates 'chart' variable
                fig = exec_namespace.get('chart') or exec_namespace.get('fig') or exec_namespace.get('figure')

                if fig and hasattr(fig, 'to_html'):
                    # Generate HTML with full Plotly JS included
                    return fig.to_html(
                        full_html=False,
                        include_plotlyjs='cdn',
                        config={'responsive': True, 'displayModeBar': True}
                    )

            elif self.library == "altair":
                import altair as alt
                import pandas as pd
                import numpy as np

                df = self.viz_layer.data_source.query("").to_dataframe()

                exec_namespace = {
                    'alt': alt,
                    'pd': pd,
                    'np': np,
                    'numpy': np,
                    'pandas': pd,
                    'df': df,
                    'data': df,
                    '__builtins__': __builtins__
                }

                exec(chart.code, exec_namespace)

                # Find the chart object
                alt_chart = exec_namespace.get('chart') or exec_namespace.get('fig')

                if alt_chart and hasattr(alt_chart, 'to_html'):
                    return alt_chart.to_html()

        except Exception as e:
            logger.warning(f"Failed to generate interactive HTML for {title}: {e}")
            return None

        return None

    def _render_dashboard_html(
        self,
        plan: DashboardPlan,
        charts: List[ChartWithInsight],
        executive_summary: str
    ) -> str:
        """Render complete dashboard as HTML."""
        import base64

        # Theme colors
        if plan.theme == "dark":
            bg_color = "#1a1a2e"
            card_bg = "#16213e"
            text_color = "#eaeaea"
            accent_color = "#4ecca3"
            border_color = "#0f3460"
        else:
            bg_color = "#f8f9fa"
            card_bg = "#ffffff"
            text_color = "#333333"
            accent_color = "#4CAF50"
            border_color = "#dee2e6"

        # Build chart cards
        chart_cards = []
        for cwi in charts:
            # Prefer interactive HTML for Plotly/Altair
            if cwi.interactive_html:
                card = f"""
                <div class="chart-card">
                    <h3 class="chart-title">{cwi.title}</h3>
                    <div class="chart-interactive">{cwi.interactive_html}</div>
                    <div class="chart-insight">
                        <strong>Insight:</strong> {cwi.insight}
                    </div>
                </div>
                """
                chart_cards.append(card)
            elif cwi.chart and cwi.chart.success and cwi.chart.raster:
                # Fallback to static image
                if isinstance(cwi.chart.raster, bytes):
                    b64 = base64.b64encode(cwi.chart.raster).decode('utf-8')
                else:
                    b64 = cwi.chart.raster

                card = f"""
                <div class="chart-card">
                    <h3 class="chart-title">{cwi.title}</h3>
                    <img class="chart-image" src="data:image/png;base64,{b64}" alt="{cwi.title}"/>
                    <div class="chart-insight">
                        <strong>Insight:</strong> {cwi.insight}
                    </div>
                </div>
                """
                chart_cards.append(card)

        # Determine grid columns
        num_charts = len(chart_cards)
        if num_charts <= 2:
            grid_cols = num_charts
        elif num_charts <= 4:
            grid_cols = 2
        else:
            grid_cols = 3

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{plan.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: {bg_color};
            color: {text_color};
            padding: 20px;
            min-height: 100vh;
        }}
        .dashboard-header {{
            text-align: center;
            padding: 30px 20px;
            margin-bottom: 30px;
            background: linear-gradient(135deg, {card_bg} 0%, {border_color} 100%);
            border-radius: 12px;
            border: 1px solid {border_color};
        }}
        .dashboard-title {{
            font-size: 2.5em;
            font-weight: 700;
            color: {accent_color};
            margin-bottom: 10px;
        }}
        .dashboard-description {{
            font-size: 1.1em;
            opacity: 0.8;
            max-width: 800px;
            margin: 0 auto;
        }}
        .executive-summary {{
            background: {card_bg};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 4px solid {accent_color};
            border: 1px solid {border_color};
        }}
        .executive-summary h2 {{
            color: {accent_color};
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .executive-summary p {{
            line-height: 1.7;
            font-size: 1.05em;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat({grid_cols}, 1fr);
            gap: 25px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .chart-card {{
            background: {card_bg};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid {border_color};
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
            min-height: 450px;
        }}
        .chart-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .chart-title {{
            font-size: 1.1em;
            font-weight: 600;
            color: {accent_color};
            margin-bottom: 15px;
            padding: 10px 5px;
            border-bottom: 1px solid {border_color};
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
            min-height: 50px;
            line-height: 1.4;
        }}
        .chart-image {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            margin-bottom: 15px;
            flex-shrink: 0;
        }}
        .chart-interactive {{
            width: 100%;
            min-height: 300px;
            margin-bottom: 15px;
            border-radius: 8px;
            overflow: hidden;
        }}
        .chart-interactive .plotly-graph-div {{
            width: 100% !important;
        }}
        .chart-insight {{
            font-size: 0.95em;
            line-height: 1.6;
            padding: 15px;
            background: rgba(78, 204, 163, 0.1);
            border-radius: 8px;
            border-left: 3px solid {accent_color};
            margin-top: auto;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            margin-top: 40px;
            opacity: 0.6;
            font-size: 0.9em;
        }}
        @media (max-width: 1200px) {{
            .charts-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .dashboard-title {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <header class="dashboard-header">
        <h1 class="dashboard-title">{plan.title}</h1>
        <p class="dashboard-description">{plan.description}</p>
    </header>

    <section class="executive-summary">
        <h2>Executive Summary</h2>
        <p>{executive_summary}</p>
    </section>

    <main class="charts-grid">
        {''.join(chart_cards)}
    </main>

    <footer class="footer">
        Generated by Jotty Visualization Layer | Powered by Claude + LIDA | {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </footer>
</body>
</html>"""

        return html


__all__ = [
    'DashboardPlanner',
    'DashboardPlan',
    'ChartPlan',
    'ChartWithInsight',
    'CompleteDashboard',
]
