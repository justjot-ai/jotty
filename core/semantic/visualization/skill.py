"""
Visualization Skills

Skill wrappers for the visualization layer that integrate with
the skills/composite skills system.

These skills can be:
- Used standalone for visualization tasks
- Composed with other skills (e.g., DataAnalysisSkill + VisualizationSkill)
- Chained in pipelines (Query → Transform → Visualize → Export)

DRY Principle: Skills wrap the core VisualizationLayer, adding
standardized interfaces without duplicating logic.
"""
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """Skill execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some steps succeeded


@dataclass
class SkillResult:
    """Result from skill execution."""
    success: bool
    status: SkillStatus = SkillStatus.SUCCESS
    data: Any = None
    error: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'status': self.status.value,
            'data': self.data if not callable(self.data) else str(self.data),
            'error': self.error,
            'metadata': self.metadata
        }


class BaseSkill(ABC):
    """
    Abstract base class for all skills.

    Skills are composable units of functionality that can be:
    - Executed standalone
    - Chained together in pipelines
    - Combined into composite skills
    """

    def __init__(self, name: str = None, description: str = None):
        """
        Initialize skill.

        Args:
            name: Optional skill name (defaults to class name)
            description: Optional skill description
        """
        self.name = name or self.__class__.__name__
        self.description = description or self._get_default_description()
        self._hooks: Dict[str, List[Callable]] = {
            'before_execute': [],
            'after_execute': [],
            'on_error': [],
        }

    def _get_default_description(self) -> str:
        """Get default description from class docstring."""
        return (self.__class__.__doc__ or "No description available").strip().split('\n')[0]

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill.

        Args:
            **kwargs: Skill-specific parameters

        Returns:
            SkillResult with execution outcome
        """
        pass

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook callback for an event."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _run_hooks(self, event: str, **kwargs):
        """Run all hooks for an event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(**kwargs)
            except Exception as e:
                logger.warning(f"Hook {hook.__name__} failed: {e}")


class VisualizationSkill(BaseSkill):
    """
    Skill for generating visualizations from natural language.

    Can be used standalone or composed with other skills:

    Example standalone:
        skill = VisualizationSkill(semantic_layer)
        result = skill.execute(question="Show sales by region")

    Example in pipeline:
        pipeline = Pipeline([
            DataQuerySkill(semantic_layer),
            TransformSkill(aggregations=['sum', 'mean']),
            VisualizationSkill(library='altair'),
            ExportSkill(format='html')
        ])
        result = pipeline.execute(question="Monthly revenue trend")
    """

    def __init__(
        self,
        source=None,  # SemanticLayer, DataFrame, or VisualizationLayer
        library: str = "matplotlib",
        llm_provider: str = "claude-cli",
        default_n_charts: int = 1,
        include_code: bool = False,
        **kwargs
    ):
        """
        Initialize visualization skill.

        Args:
            source: Data source (SemanticLayer, DataFrame, VisualizationLayer)
            library: Default chart library
            llm_provider: LLM provider for LIDA
            default_n_charts: Default number of chart variations
            include_code: Include chart code in results
            **kwargs: Additional options
        """
        super().__init__(name="VisualizationSkill")
        self.library = library
        self.llm_provider = llm_provider
        self.default_n_charts = default_n_charts
        self.include_code = include_code
        self._extra_kwargs = kwargs

        # Create or store VisualizationLayer
        self._viz_layer = None
        if source is not None:
            self._init_viz_layer(source)

    def _init_viz_layer(self, source):
        """Initialize VisualizationLayer from source."""
        from .layer import VisualizationLayer

        if isinstance(source, VisualizationLayer):
            self._viz_layer = source
        else:
            self._viz_layer = VisualizationLayer.from_semantic_layer(
                source,
                llm_provider=self.llm_provider,
                **self._extra_kwargs
            ) if hasattr(source, 'query') else VisualizationLayer.from_dataframe(
                source,
                llm_provider=self.llm_provider,
                **self._extra_kwargs
            )

    @property
    def viz_layer(self):
        """Get visualization layer."""
        if self._viz_layer is None:
            raise ValueError("No data source configured. Pass source to __init__ or execute().")
        return self._viz_layer

    def execute(
        self,
        question: str = None,
        data=None,
        goal=None,
        library: str = None,
        n: int = None,
        output_format: str = "result",  # result, html, base64
        **kwargs
    ) -> SkillResult:
        """
        Execute visualization skill.

        Args:
            question: Natural language question for visualization
            data: Optional data override (DataFrame)
            goal: Specific visualization goal
            library: Chart library override
            n: Number of chart variations
            output_format: Output format (result, html, base64)
            **kwargs: Additional options

        Returns:
            SkillResult with charts
        """
        self._run_hooks('before_execute', question=question, data=data)

        try:
            # Update viz layer if new data provided
            if data is not None:
                from .layer import VisualizationLayer
                self._viz_layer = VisualizationLayer.from_dataframe(
                    data,
                    llm_provider=self.llm_provider,
                    **self._extra_kwargs
                )

            # Generate visualization
            charts = self.viz_layer.visualize(
                question=question,
                goal=goal,
                library=library or self.library,
                n=n or self.default_n_charts,
                return_code=self.include_code,
                **kwargs
            )

            # Format output
            if output_format == 'html':
                from .renderers import HTMLRenderer
                renderer = HTMLRenderer(include_code=self.include_code)
                render_result = renderer.render_multiple(charts)
                output = render_result.output
            elif output_format == 'base64':
                output = [c.to_base64() for c in charts if c.success]
            else:
                output = charts

            result = SkillResult(
                success=any(c.success for c in charts),
                status=SkillStatus.SUCCESS if all(c.success for c in charts) else SkillStatus.PARTIAL,
                data=output,
                metadata={
                    'question': question,
                    'library': library or self.library,
                    'chart_count': len(charts),
                    'successful_charts': sum(1 for c in charts if c.success),
                }
            )

            self._run_hooks('after_execute', result=result)
            return result

        except Exception as e:
            logger.error(f"VisualizationSkill failed: {e}")
            self._run_hooks('on_error', error=e)
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=str(e)
            )

    def explore(self, n_goals: int = 5, **kwargs) -> SkillResult:
        """
        Generate exploration goals for data.

        Args:
            n_goals: Number of goals to generate
            **kwargs: Additional options

        Returns:
            SkillResult with visualization goals
        """
        try:
            goals = self.viz_layer.goals(n=n_goals, **kwargs)
            return SkillResult(
                success=True,
                data=goals,
                metadata={'goal_count': len(goals)}
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e), status=SkillStatus.FAILED)


class DataAnalysisSkill(BaseSkill):
    """
    Skill for comprehensive data analysis with visualizations.

    Combines data querying, summarization, and visualization.

    Example:
        skill = DataAnalysisSkill(semantic_layer)
        result = skill.execute(
            question="Analyze sales performance by region",
            include_stats=True,
            n_charts=3
        )
    """

    def __init__(
        self,
        source=None,
        llm_provider: str = "claude-cli",
        **kwargs
    ):
        """
        Initialize data analysis skill.

        Args:
            source: Data source (SemanticLayer, DataFrame)
            llm_provider: LLM provider
            **kwargs: Additional options
        """
        super().__init__(name="DataAnalysisSkill")
        self.llm_provider = llm_provider
        self._viz_skill = VisualizationSkill(
            source=source,
            llm_provider=llm_provider,
            **kwargs
        )
        self._source = source

    def execute(
        self,
        question: str,
        include_stats: bool = True,
        include_summary: bool = True,
        n_charts: int = 3,
        **kwargs
    ) -> SkillResult:
        """
        Execute comprehensive data analysis.

        Args:
            question: Analysis question
            include_stats: Include statistical summary
            include_summary: Include data summary
            n_charts: Number of charts to generate
            **kwargs: Additional options

        Returns:
            SkillResult with analysis results
        """
        self._run_hooks('before_execute', question=question)

        try:
            results = {
                'question': question,
                'query_result': None,
                'summary': None,
                'statistics': None,
                'charts': None,
                'insights': []
            }

            # Query data
            from .data_source import DataSourceFactory
            data_source = DataSourceFactory.create(self._source)
            query_result = data_source.query(question)

            if not query_result.success:
                return SkillResult(
                    success=False,
                    status=SkillStatus.FAILED,
                    error=query_result.error,
                    data=results
                )

            df = query_result.to_dataframe()
            results['query_result'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'generated_query': query_result.generated_query
            }

            # Statistical summary
            if include_stats and len(df) > 0:
                results['statistics'] = {
                    'describe': df.describe().to_dict(),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'null_counts': df.isnull().sum().to_dict()
                }

            # Data summary via LIDA
            if include_summary:
                try:
                    summary = self._viz_skill.viz_layer.summarize(data=df)
                    results['summary'] = {
                        'n_rows': summary.n_rows,
                        'n_columns': summary.n_columns,
                        'columns': summary.columns
                    }
                except Exception as e:
                    logger.warning(f"Summary generation failed: {e}")

            # Generate visualizations
            viz_result = self._viz_skill.execute(
                question=question,
                data=df,
                n=n_charts,
                **kwargs
            )

            if viz_result.success:
                results['charts'] = viz_result.data
                results['insights'] = [
                    c.goal.question for c in viz_result.data
                    if hasattr(c, 'goal') and c.goal
                ]

            self._run_hooks('after_execute', results=results)

            return SkillResult(
                success=True,
                status=SkillStatus.SUCCESS,
                data=results,
                metadata={
                    'row_count': len(df),
                    'chart_count': n_charts,
                    'has_stats': include_stats,
                }
            )

        except Exception as e:
            logger.error(f"DataAnalysisSkill failed: {e}")
            self._run_hooks('on_error', error=e)
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=str(e)
            )


class DashboardSkill(BaseSkill):
    """
    Skill for generating multi-chart dashboards.

    Creates cohesive dashboards from multiple questions or auto-generated goals.

    Example:
        skill = DashboardSkill(semantic_layer)
        result = skill.execute(
            questions=[
                "Revenue by region",
                "Monthly sales trend",
                "Top products by quantity"
            ],
            title="Sales Dashboard",
            columns=2
        )
    """

    def __init__(
        self,
        source=None,
        llm_provider: str = "claude-cli",
        default_library: str = "matplotlib",
        **kwargs
    ):
        """
        Initialize dashboard skill.

        Args:
            source: Data source
            llm_provider: LLM provider
            default_library: Default chart library
            **kwargs: Additional options
        """
        super().__init__(name="DashboardSkill")
        self.default_library = default_library
        self._viz_skill = VisualizationSkill(
            source=source,
            library=default_library,
            llm_provider=llm_provider,
            **kwargs
        )

    def execute(
        self,
        questions: List[str] = None,
        n_auto_charts: int = 4,
        title: str = "Dashboard",
        columns: int = 2,
        output_format: str = "html",
        theme: str = "light",
        **kwargs
    ) -> SkillResult:
        """
        Generate a dashboard.

        Args:
            questions: List of NL questions for charts
            n_auto_charts: Number of auto-generated charts if no questions
            title: Dashboard title
            columns: Number of columns in layout
            output_format: Output format (html, charts, base64)
            theme: Color theme (light/dark)
            **kwargs: Additional options

        Returns:
            SkillResult with dashboard
        """
        self._run_hooks('before_execute', questions=questions, title=title)

        try:
            charts = self._viz_skill.viz_layer.dashboard(
                questions=questions,
                n_goals=n_auto_charts,
                library=self.default_library,
                **kwargs
            )

            # Render based on format
            if output_format == 'html':
                from .renderers import HTMLRenderer
                renderer = HTMLRenderer(title=title, theme=theme)
                render_result = renderer.render_multiple(charts, columns=columns)
                output = render_result.output
            elif output_format == 'base64':
                output = [c.to_base64() for c in charts if c.success and c.raster]
            else:
                output = charts

            successful = sum(1 for c in charts if c.success)

            return SkillResult(
                success=successful > 0,
                status=SkillStatus.SUCCESS if successful == len(charts) else SkillStatus.PARTIAL,
                data=output,
                metadata={
                    'title': title,
                    'total_charts': len(charts),
                    'successful_charts': successful,
                    'questions': questions,
                    'columns': columns,
                    'output_format': output_format
                }
            )

        except Exception as e:
            logger.error(f"DashboardSkill failed: {e}")
            self._run_hooks('on_error', error=e)
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=str(e)
            )


class CompositeSkill(BaseSkill):
    """
    Combines multiple skills into a pipeline.

    Allows chaining skills where output of one feeds into the next.

    Example:
        pipeline = CompositeSkill([
            ('query', DataQuerySkill(semantic_layer)),
            ('transform', TransformSkill()),
            ('visualize', VisualizationSkill(library='altair')),
        ])
        result = pipeline.execute(question="Monthly sales trend")
    """

    def __init__(self, skills: List[tuple], name: str = "CompositeSkill"):
        """
        Initialize composite skill.

        Args:
            skills: List of (name, skill) tuples
            name: Name for the composite skill
        """
        super().__init__(name=name)
        self.skills = skills

    def execute(self, **kwargs) -> SkillResult:
        """
        Execute all skills in sequence.

        The output of each skill is passed to the next.

        Returns:
            SkillResult with final output and intermediate results
        """
        self._run_hooks('before_execute', **kwargs)

        results = {}
        current_data = kwargs

        for skill_name, skill in self.skills:
            try:
                logger.debug(f"Executing skill: {skill_name}")

                # Execute skill with current data
                result = skill.execute(**current_data)
                results[skill_name] = result.to_dict()

                if not result.success:
                    return SkillResult(
                        success=False,
                        status=SkillStatus.PARTIAL,
                        data=results,
                        error=f"Skill '{skill_name}' failed: {result.error}",
                        metadata={'failed_at': skill_name}
                    )

                # Pass result data to next skill
                if isinstance(result.data, dict):
                    current_data.update(result.data)
                else:
                    current_data['data'] = result.data

            except Exception as e:
                logger.error(f"Skill '{skill_name}' raised exception: {e}")
                self._run_hooks('on_error', error=e, skill=skill_name)
                return SkillResult(
                    success=False,
                    status=SkillStatus.FAILED,
                    data=results,
                    error=str(e),
                    metadata={'failed_at': skill_name}
                )

        self._run_hooks('after_execute', results=results)

        return SkillResult(
            success=True,
            status=SkillStatus.SUCCESS,
            data=results,
            metadata={'skills_executed': [s[0] for s in self.skills]}
        )


__all__ = [
    'BaseSkill',
    'SkillResult',
    'SkillStatus',
    'VisualizationSkill',
    'DataAnalysisSkill',
    'DashboardSkill',
    'CompositeSkill',
]
