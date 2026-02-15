"""
Visualization Layer

Main orchestrator that integrates LIDA with our data pipeline.
Provides a unified interface for:
- Automatic visualization generation from NL questions
- Goal-based chart creation
- Multi-chart dashboards
- Chart editing and explanation

Example:
    from core.semantic import SemanticLayer
    from core.semantic.visualization import VisualizationLayer

    # Create from SemanticLayer
    semantic = SemanticLayer.from_database(db_type='postgresql', ...)
    viz = VisualizationLayer.from_semantic_layer(semantic)

    # Generate visualizations
    charts = viz.visualize("Show revenue by region for last 30 days")

    # Or from DataFrame
    viz = VisualizationLayer.from_dataframe(df)
    charts = viz.visualize("Distribution of sales by category")
"""
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

from .data_source import DataSource, DataSourceFactory, DataSourceResult

logger = logging.getLogger(__name__)


class ChartLibrary(Enum):
    """Supported visualization libraries."""
    MATPLOTLIB = "matplotlib"
    SEABORN = "seaborn"
    ALTAIR = "altair"
    PLOTLY = "plotly"
    GGPLOT = "ggplot"


@dataclass
class DataSummary:
    """Summary of data characteristics for LIDA."""
    name: str
    file_path: str = None
    n_rows: int = 0
    n_columns: int = 0
    columns: List[Dict[str, Any]] = field(default_factory=list)
    sample_data: List[Dict] = field(default_factory=list)
    raw_summary: Dict = field(default_factory=dict)

    @classmethod
    def from_lida(cls, lida_summary: Dict) -> 'DataSummary':
        """Create from LIDA summary dict."""
        return cls(
            name=lida_summary.get('name', 'data'),
            file_path=lida_summary.get('file_name'),
            n_rows=lida_summary.get('n_rows', 0),
            n_columns=lida_summary.get('n_columns', 0),
            columns=lida_summary.get('fields', []),
            raw_summary=lida_summary
        )


@dataclass
class VisualizationGoal:
    """A visualization goal/objective."""
    question: str
    visualization_type: str = None
    rationale: str = None
    index: int = 0
    raw_goal: Any = None

    @classmethod
    def from_lida(cls, lida_goal: Any, index: int = 0) -> 'VisualizationGoal':
        """Create from LIDA Goal object."""
        return cls(
            question=getattr(lida_goal, 'question', str(lida_goal)),
            visualization_type=getattr(lida_goal, 'visualization', None),
            rationale=getattr(lida_goal, 'rationale', None),
            index=index,
            raw_goal=lida_goal
        )


@dataclass
class ChartResult:
    """Result of chart generation."""
    success: bool
    code: str = None
    raster: bytes = None  # PNG image bytes
    svg: str = None  # SVG string
    spec: Dict = None  # Vega/Altair spec
    error: str = None
    goal: VisualizationGoal = None
    library: ChartLibrary = None
    raw_chart: Any = None

    def save_image(self, path: str) -> None:
        """Save chart image to file."""
        if self.raster:
            # Handle both bytes and base64 string
            data = self.raster
            if isinstance(data, str):
                import base64
                data = base64.b64decode(data)
            with open(path, 'wb') as f:
                f.write(data)
        elif self.svg:
            with open(path, 'w') as f:
                f.write(self.svg)

    def to_base64(self) -> str:
        """Get base64 encoded image for web display."""
        import base64
        if self.raster:
            return base64.b64encode(self.raster).decode('utf-8')
        return None

    def display(self) -> None:
        """Display chart in Jupyter/IPython."""
        try:
            from IPython.display import display, Image, SVG
            if self.raster:
                display(Image(self.raster))
            elif self.svg:
                display(SVG(self.svg))
        except ImportError:
            logger.warning("IPython not available for display")


class VisualizationLayer:
    """
    Main visualization layer that orchestrates LIDA integration.

    Provides:
    - Automatic chart generation from NL questions
    - Goal exploration for data analysis
    - Multiple library support (matplotlib, altair, plotly)
    - Integration with SemanticLayer and ConnectorX

    Uses Claude CLI by default (no API keys required).
    """

    def __init__(self, data_source: DataSource, llm_provider: str = 'claude-cli', llm_model: str = '', default_library: ChartLibrary = ChartLibrary.MATPLOTLIB, **llm_kwargs: Any) -> None:
        """
        Initialize visualization layer.

        Args:
            data_source: DataSource instance for data access
            llm_provider: LLM provider (claude-cli, anthropic, gemini, openai)
            llm_model: Model name (sonnet, opus, haiku for Claude)
            default_library: Default chart library
            **llm_kwargs: Additional LLM configuration
        """
        from Jotty.core.infrastructure.foundation.config_defaults import DEFAULT_MODEL_ALIAS
        self.data_source = data_source
        self.llm_provider = llm_provider
        self.llm_model = llm_model or DEFAULT_MODEL_ALIAS
        self.default_library = default_library
        self._llm_kwargs = llm_kwargs

        self._manager = None
        self._summary_cache: Dict[str, DataSummary] = {}

    @classmethod
    def from_semantic_layer(cls, semantic_layer: Any, llm_provider: str = 'claude-cli', llm_model: str = '', **kwargs: Any) -> 'VisualizationLayer':
        """
        Create VisualizationLayer from SemanticLayer.

        Args:
            semantic_layer: SemanticLayer instance
            llm_provider: LLM provider (claude-cli, anthropic, gemini, openai)
            llm_model: Model name (sonnet, opus, haiku for Claude)
            **kwargs: Additional options

        Returns:
            VisualizationLayer instance
        """
        data_source = DataSourceFactory.create(semantic_layer)
        return cls(data_source, llm_provider=llm_provider, llm_model=llm_model, **kwargs)

    @classmethod
    def from_dataframe(cls, dataframe: Any, name: str = 'data', llm_provider: str = 'claude-cli', llm_model: str = '', **kwargs: Any) -> 'VisualizationLayer':
        """
        Create VisualizationLayer from DataFrame.

        Args:
            dataframe: pandas/polars/arrow DataFrame
            name: Name for the data
            llm_provider: LLM provider (claude-cli, anthropic, gemini, openai)
            llm_model: Model name (sonnet, opus, haiku for Claude)
            **kwargs: Additional options

        Returns:
            VisualizationLayer instance
        """
        data_source = DataSourceFactory.create(dataframe, name=name)
        return cls(data_source, llm_provider=llm_provider, llm_model=llm_model, **kwargs)

    @classmethod
    def from_csv(cls, path: str, llm_provider: str = 'claude-cli', llm_model: str = '', **kwargs: Any) -> 'VisualizationLayer':
        """
        Create VisualizationLayer from CSV file.

        Args:
            path: Path to CSV file
            llm_provider: LLM provider (claude-cli, anthropic, gemini, openai)
            llm_model: Model name (sonnet, opus, haiku for Claude)
            **kwargs: Additional options

        Returns:
            VisualizationLayer instance
        """
        data_source = DataSourceFactory.create(path)
        return cls(data_source, llm_provider=llm_provider, llm_model=llm_model, **kwargs)

    @property
    def manager(self) -> Any:
        """Get or create LIDA Manager."""
        if self._manager is None:
            from lida import Manager

            # Use Claude LLM provider from core.llm
            from .llm_provider import ClaudeLLMTextGenerator

            text_gen = ClaudeLLMTextGenerator(
                provider=self.llm_provider,
                model=self.llm_model,
                **self._llm_kwargs
            )
            self._manager = Manager(text_gen=text_gen)

        return self._manager

    def summarize(self, data: Any = None, question: str = None, **kwargs: Any) -> DataSummary:
        """
        Summarize data for visualization context.

        Args:
            data: Optional data override (DataFrame)
            question: Optional NL question to query data first
            **kwargs: Additional options

        Returns:
            DataSummary with data characteristics
        """
        # Get data
        if question:
            result = self.data_source.query(question)
            if not result.success:
                raise ValueError(f"Query failed: {result.error}")
            df = result.to_dataframe()
        elif data is not None:
            import pandas as pd
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        else:
            result = self.data_source.query("")
            df = result.to_dataframe()

        # Convert to CSV for LIDA
        csv_path = self.data_source.to_csv_file(df)

        try:
            # Get LIDA summary
            lida_summary = self.manager.summarize(csv_path, **kwargs)
            summary = DataSummary.from_lida(lida_summary)
            summary.file_path = csv_path
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise

    def goals(self, summary: DataSummary = None, n: int = 5, question: str = None, persona: str = None, **kwargs: Any) -> List[VisualizationGoal]:
        """
        Generate visualization goals for data exploration.

        Args:
            summary: Data summary (auto-generated if not provided)
            n: Number of goals to generate
            question: Optional NL question to focus goals
            persona: Optional persona for goal generation (e.g., "data analyst")
            **kwargs: Additional options

        Returns:
            List of VisualizationGoal objects
        """
        if summary is None:
            summary = self.summarize(question=question)

        try:
            lida_goals = self.manager.goals(
                summary.raw_summary,
                n=n,
                persona=persona,
                **kwargs
            )
            return [VisualizationGoal.from_lida(g, i) for i, g in enumerate(lida_goals)]
        except Exception as e:
            logger.error(f"Goal generation failed: {e}")
            raise

    def visualize(self, question: str = None, goal: Union[VisualizationGoal, str] = None, summary: DataSummary = None, library: Union[ChartLibrary, str] = None, n: int = 1, return_code: bool = True, **kwargs: Any) -> List[ChartResult]:
        """
        Generate visualizations from NL question or goal.

        This is the main method for creating charts. It can:
        1. Take an NL question, query data, and generate charts
        2. Use a predefined goal for specific visualization
        3. Explore data with multiple chart suggestions

        Args:
            question: Natural language question (e.g., "Show sales by region")
            goal: Specific visualization goal (VisualizationGoal or string)
            summary: Pre-computed data summary
            library: Chart library to use
            n: Number of chart variations to generate
            return_code: Include code in result
            **kwargs: Additional options

        Returns:
            List of ChartResult objects

        Example:
            # From NL question
            charts = viz.visualize("Show monthly revenue trend")

            # With specific goal
            goal = viz.goals()[0]
            charts = viz.visualize(goal=goal)

            # Using Altair
            charts = viz.visualize("Sales by category", library="altair")
        """
        # Resolve library
        if library is None:
            library = self.default_library
        elif isinstance(library, str):
            library = ChartLibrary(library.lower())

        # Get data summary
        if summary is None:
            summary = self.summarize(question=question)

        # Convert goal
        if goal is None and question:
            # Use question as goal
            lida_goal = question
        elif isinstance(goal, VisualizationGoal):
            lida_goal = goal.raw_goal or goal.question
        elif isinstance(goal, str):
            lida_goal = goal
        else:
            # Generate goals and use first one
            goals = self.goals(summary, n=1)
            if not goals:
                raise ValueError("Could not generate visualization goals")
            lida_goal = goals[0].raw_goal or goals[0].question

        try:
            # Generate visualization
            # Note: LIDA's visualize returns one chart per call
            # For multiple charts, we call it multiple times or use different goals
            from llmx.datamodel import TextGenerationConfig as LLMXConfig

            charts = []
            for _ in range(n):
                chart_list = self.manager.visualize(
                    summary=summary.raw_summary,
                    goal=lida_goal,
                    textgen_config=LLMXConfig(temperature=0.2),  # Slight variation
                    library=library.value,
                    return_error=True,
                    **kwargs
                )
                if chart_list:
                    charts.extend(chart_list if isinstance(chart_list, list) else [chart_list])

            results = []
            for chart in charts:
                code = getattr(chart, 'code', None)
                raster = getattr(chart, 'raster', None)
                error = getattr(chart, 'error', None)

                # For interactive libraries (plotly, altair), code is sufficient
                # They don't need raster images - we'll render them interactively
                is_interactive = library in [ChartLibrary.PLOTLY, ChartLibrary.ALTAIR]

                # Consider success if:
                # 1. No error at all, OR
                # 2. Has code AND it's an interactive library (raster failure is OK)
                is_success = (not error) or (code and is_interactive)

                if is_success:
                    results.append(ChartResult(
                        success=True,
                        code=code if return_code else None,
                        raster=raster,
                        spec=getattr(chart, 'spec', None),
                        goal=VisualizationGoal(question=str(lida_goal)),
                        library=library,
                        raw_chart=chart
                    ))
                else:
                    results.append(ChartResult(
                        success=False,
                        code=code if return_code else None,  # Still include code if available
                        error=error,
                        goal=VisualizationGoal(question=str(lida_goal)),
                        library=library
                    ))

            return results

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return [ChartResult(
                success=False,
                error=str(e),
                goal=VisualizationGoal(question=str(lida_goal)) if lida_goal else None,
                library=library
            )]

    def explain(self, chart: ChartResult, **kwargs: Any) -> str:
        """
        Get explanation for a chart.

        Args:
            chart: ChartResult to explain
            **kwargs: Additional options

        Returns:
            Explanation string
        """
        if not chart.raw_chart:
            return "Cannot explain chart without raw chart data"

        try:
            explanations = self.manager.explain(
                code=chart.code,
                **kwargs
            )
            return explanations[0] if explanations else "No explanation available"
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return f"Explanation failed: {e}"

    def edit(self, chart: ChartResult, instructions: str, **kwargs: Any) -> ChartResult:
        """
        Edit a chart based on instructions.

        Args:
            chart: ChartResult to edit
            instructions: Edit instructions (e.g., "change colors to blue")
            **kwargs: Additional options

        Returns:
            New ChartResult with edited chart
        """
        if not chart.raw_chart:
            return ChartResult(success=False, error="Cannot edit without raw chart")

        try:
            edited = self.manager.edit(
                code=chart.code,
                instructions=instructions,
                **kwargs
            )

            if edited:
                return ChartResult(
                    success=True,
                    code=getattr(edited[0], 'code', None),
                    raster=getattr(edited[0], 'raster', None),
                    goal=chart.goal,
                    library=chart.library,
                    raw_chart=edited[0]
                )

            return ChartResult(success=False, error="Edit returned no results")

        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return ChartResult(success=False, error=str(e))

    def dashboard(self, questions: List[str] = None, n_goals: int = 4, library: Union[ChartLibrary, str] = None, **kwargs: Any) -> List[ChartResult]:
        """
        Generate a multi-chart dashboard.

        Args:
            questions: List of NL questions for charts
            n_goals: Number of auto-generated goals if no questions
            library: Chart library to use
            **kwargs: Additional options

        Returns:
            List of ChartResult objects for dashboard
        """
        charts = []

        if questions:
            for q in questions:
                chart_results = self.visualize(question=q, library=library, n=1, **kwargs)
                charts.extend(chart_results)
        else:
            # Auto-generate goals
            summary = self.summarize()
            goals = self.goals(summary, n=n_goals)
            for goal in goals:
                chart_results = self.visualize(
                    goal=goal,
                    summary=summary,
                    library=library,
                    n=1,
                    **kwargs
                )
                charts.extend(chart_results)

        return charts

    def query_and_visualize(self, question: str, library: Union[ChartLibrary, str] = None, n: int = 1, **kwargs: Any) -> Dict[str, Any]:
        """
        Complete pipeline: Query data and visualize in one call.

        This combines SemanticLayer query with LIDA visualization.

        Args:
            question: Natural language question
            library: Chart library
            n: Number of chart variations
            **kwargs: Additional options

        Returns:
            Dictionary with query results and charts
        """
        # Query data
        query_result = self.data_source.query(question)

        if not query_result.success:
            return {
                'success': False,
                'error': query_result.error,
                'query': question
            }

        # Visualize
        charts = self.visualize(
            question=question,
            library=library,
            n=n,
            **kwargs
        )

        return {
            'success': True,
            'query': question,
            'generated_query': query_result.generated_query,
            'data': query_result.data,
            'row_count': len(query_result.to_dataframe()),
            'charts': charts,
            'metadata': query_result.metadata
        }


__all__ = [
    'VisualizationLayer',
    'ChartResult',
    'VisualizationGoal',
    'DataSummary',
    'ChartLibrary',
]
