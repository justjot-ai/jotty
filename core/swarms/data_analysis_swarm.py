"""
Data Analysis Swarm - World-Class Data Science & Analytics
==========================================================

Production-grade swarm for:
- Exploratory data analysis (EDA)
- Statistical analysis
- Data visualization
- ML model recommendations
- Insight generation
- Report creation

Agents:
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA ANALYSIS SWARM                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Data         │  │   EDA          │  │  Statistical   │            │
│  │   Profiler     │  │   Agent        │  │   Agent        │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │  Visualization │  │   ML           │  │   Insight      │            │
│  │    Agent       │  │   Recommender  │  │   Generator    │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     REPORT GENERATOR                             │   │
│  │   Creates comprehensive data analysis reports with visuals       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.data_analysis_swarm import DataAnalysisSwarm, analyze_data

    # Full swarm
    swarm = DataAnalysisSwarm()
    result = await swarm.analyze(data, target_column="price")

    # One-liner
    result = await analyze_data(df, question="What drives customer churn?")

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace
)
from .base import DomainSwarm, AgentTeam
from ..agents.base import DomainAgent, DomainAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class AnalysisType(Enum):
    EDA = "eda"
    STATISTICAL = "statistical"
    PREDICTIVE = "predictive"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"


class DataType(Enum):
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


class VisualizationType(Enum):
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    TIME_SERIES = "time_series"
    COMPARISON = "comparison"
    COMPOSITION = "composition"
    RELATIONSHIP = "relationship"


@dataclass
class DataAnalysisConfig(SwarmConfig):
    """Configuration for DataAnalysisSwarm."""
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [
        AnalysisType.EDA, AnalysisType.STATISTICAL
    ])
    include_visualizations: bool = True
    include_ml_recommendations: bool = True
    include_insights: bool = True
    confidence_level: float = 0.95
    max_categories: int = 20
    outlier_method: str = "iqr"
    missing_threshold: float = 0.5

    def __post_init__(self):
        self.name = "DataAnalysisSwarm"
        self.domain = "data_analysis"


@dataclass
class DataProfile:
    """Data profiling results."""
    row_count: int
    column_count: int
    columns: Dict[str, Dict[str, Any]]
    missing_summary: Dict[str, float]
    duplicates: int
    memory_usage: str
    data_types: Dict[str, str]


@dataclass
class StatisticalResult:
    """Statistical analysis results."""
    descriptive_stats: Dict[str, Dict[str, float]]
    correlations: Dict[str, Dict[str, float]]
    distributions: Dict[str, str]
    outliers: Dict[str, List[Any]]
    hypothesis_tests: List[Dict[str, Any]]


@dataclass
class Insight:
    """A data insight."""
    title: str
    description: str
    evidence: str
    impact: str
    confidence: float
    category: str


@dataclass
class MLRecommendation:
    """ML model recommendation."""
    task_type: str
    recommended_models: List[str]
    feature_importance: Dict[str, float]
    preprocessing_steps: List[str]
    expected_performance: str


@dataclass
class Visualization:
    """Visualization specification."""
    viz_type: VisualizationType
    title: str
    description: str
    columns: List[str]
    code: str


@dataclass
class AnalysisResult(SwarmResult):
    """Result from DataAnalysisSwarm."""
    profile: Optional[DataProfile] = None
    statistics: Optional[StatisticalResult] = None
    insights: List[Insight] = field(default_factory=list)
    ml_recommendations: Optional[MLRecommendation] = None
    visualizations: List[Visualization] = field(default_factory=list)
    summary: str = ""
    data_quality_score: float = 0.0


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class DataProfilingSignature(dspy.Signature):
    """Profile a dataset.

    You are a DATA PROFILER. Analyze the dataset to identify:
    1. Basic statistics (count, mean, std, etc.)
    2. Data types and their distributions
    3. Missing values and patterns
    4. Unique values and cardinality
    5. Potential data quality issues

    Be thorough in profiling.
    """
    data_summary: str = dspy.InputField(desc="JSON summary of data structure")
    sample_data: str = dspy.InputField(desc="Sample rows from the dataset")
    column_info: str = dspy.InputField(desc="Column names and inferred types")

    profile_summary: str = dspy.OutputField(desc="Overall data profile summary")
    column_profiles: str = dspy.OutputField(desc="JSON of column-level profiles")
    quality_issues: str = dspy.OutputField(desc="Data quality issues found, separated by |")
    recommendations: str = dspy.OutputField(desc="Data cleaning recommendations, separated by |")


class EDASignature(dspy.Signature):
    """Perform exploratory data analysis.

    You are an EDA EXPERT. Explore the data to discover:
    1. Patterns and trends
    2. Relationships between variables
    3. Anomalies and outliers
    4. Group differences
    5. Temporal patterns (if applicable)

    Think like a detective finding clues in data.
    """
    data_profile: str = dspy.InputField(desc="Data profile summary")
    statistics: str = dspy.InputField(desc="Descriptive statistics")
    question: str = dspy.InputField(desc="Analysis question or objective")

    patterns: str = dspy.OutputField(desc="Patterns discovered, separated by |")
    relationships: str = dspy.OutputField(desc="Key relationships found, separated by |")
    anomalies: str = dspy.OutputField(desc="Anomalies identified, separated by |")
    exploration_summary: str = dspy.OutputField(desc="Summary of exploration findings")


class StatisticalAnalysisSignature(dspy.Signature):
    """Perform statistical analysis on provided data.

    You are a STATISTICIAN. ALWAYS provide structured analysis output.

    IMPORTANT: If data is provided as text (raw_text field), extract the numerical
    values and perform analysis on them. Never ask for more information - analyze
    what is given.

    Conduct rigorous analysis:
    1. Descriptive statistics (mean, median, std, min, max, etc.)
    2. Distribution analysis (skewness, normality)
    3. Correlation analysis (relationships between variables)
    4. Hypothesis testing (trends, comparisons)
    5. Confidence intervals

    Apply appropriate tests for the data types. Always output valid JSON/structured data.
    """
    data_summary: str = dspy.InputField(desc="Data summary JSON - may include raw_text field with text data to extract values from")
    columns_of_interest: str = dspy.InputField(desc="Columns to analyze (or 'text' for text data)")
    analysis_goals: str = dspy.InputField(desc="Statistical analysis goals")

    reasoning: str = dspy.OutputField(desc="Brief reasoning about the analysis approach")
    descriptive: str = dspy.OutputField(desc="Descriptive statistics as JSON object with keys like mean, median, std, min, max, sum, count")
    distributions: str = dspy.OutputField(desc="Distribution analysis: normal/skewed/uniform and key observations")
    correlations: str = dspy.OutputField(desc="Correlation summary: positive/negative/none and relationships found")
    hypothesis_tests: str = dspy.OutputField(desc="Statistical tests results, separated by | (e.g., 'trend: increasing | best: Q4 | growth: 15%')")


class InsightGenerationSignature(dspy.Signature):
    """Generate actionable insights.

    You are an INSIGHT ANALYST. Generate insights that are:
    1. Actionable - can drive decisions
    2. Evidence-based - supported by data
    3. Specific - clear and concrete
    4. Impactful - matter to the business
    5. Novel - not obvious

    Quality over quantity.
    """
    analysis_results: str = dspy.InputField(desc="All analysis results")
    business_context: str = dspy.InputField(desc="Business context if available")
    question: str = dspy.InputField(desc="Original analysis question")

    insights: str = dspy.OutputField(desc="JSON list of insights with title, description, evidence, impact")
    key_findings: str = dspy.OutputField(desc="Top 3 key findings, separated by |")
    recommendations: str = dspy.OutputField(desc="Action recommendations, separated by |")


class MLRecommendationSignature(dspy.Signature):
    """Recommend ML approaches.

    You are an ML ADVISOR. Recommend:
    1. Appropriate ML task type
    2. Suitable algorithms
    3. Feature engineering steps
    4. Preprocessing requirements
    5. Evaluation metrics

    Consider data characteristics and business goals.
    """
    data_profile: str = dspy.InputField(desc="Data profile")
    target_column: str = dspy.InputField(desc="Target variable if applicable")
    goal: str = dspy.InputField(desc="ML goal")

    task_type: str = dspy.OutputField(desc="ML task type: classification, regression, clustering, etc.")
    recommended_models: str = dspy.OutputField(desc="Recommended models, separated by |")
    feature_engineering: str = dspy.OutputField(desc="Feature engineering suggestions, separated by |")
    preprocessing: str = dspy.OutputField(desc="Preprocessing steps, separated by |")
    evaluation_metrics: str = dspy.OutputField(desc="Recommended metrics, separated by |")


class VisualizationSignature(dspy.Signature):
    """Recommend visualizations.

    You are a DATA VISUALIZATION EXPERT. Suggest:
    1. Appropriate chart types
    2. Key visualizations for insights
    3. Interactive features
    4. Color schemes
    5. Accessibility considerations

    Visualize to illuminate, not complicate.
    """
    data_summary: str = dspy.InputField(desc="Data summary")
    insights: str = dspy.InputField(desc="Insights to visualize")
    audience: str = dspy.InputField(desc="Target audience")

    visualizations: str = dspy.OutputField(desc="JSON list of visualization specs with type, columns, description")
    chart_recommendations: str = dspy.OutputField(desc="Chart type recommendations, separated by |")
    design_tips: str = dspy.OutputField(desc="Design tips, separated by |")


# =============================================================================
# AGENTS
# =============================================================================

class BaseDataAgent(DomainAgent):
    """Base class for data analysis agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, signature=None):
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass


class DataProfilerAgent(BaseDataAgent):
    """Profiles datasets."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=DataProfilingSignature)
        self._profiler = dspy.ChainOfThought(DataProfilingSignature)
        self.learned_context = learned_context

    async def profile(
        self,
        data_summary: Dict[str, Any],
        sample_data: List[Dict],
        column_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """Profile the dataset."""
        try:
            context_suffix = f"\n\n[Learned Context]: {self.learned_context}" if self.learned_context else ""
            result = self._profiler(
                data_summary=json.dumps(data_summary) + context_suffix,
                sample_data=json.dumps(sample_data[:10]),
                column_info=json.dumps(column_info)
            )

            try:
                column_profiles = json.loads(result.column_profiles)
            except:
                column_profiles = {}

            quality_issues = [q.strip() for q in str(result.quality_issues).split('|') if q.strip()]
            recommendations = [r.strip() for r in str(result.recommendations).split('|') if r.strip()]

            self._broadcast("data_profiled", {
                'columns': len(column_info),
                'issues': len(quality_issues)
            })

            return {
                'profile_summary': str(result.profile_summary),
                'column_profiles': column_profiles,
                'quality_issues': quality_issues,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Data profiling failed: {e}")
            return {'error': str(e)}


class EDAAgent(BaseDataAgent):
    """Performs exploratory data analysis."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=EDASignature)
        self._explorer = dspy.ChainOfThought(EDASignature)
        self.learned_context = learned_context

    async def explore(
        self,
        data_profile: str,
        statistics: str,
        question: str = "Explore the data"
    ) -> Dict[str, Any]:
        """Perform EDA."""
        try:
            context_suffix = f"\n\n[Learned Context]: {self.learned_context}" if self.learned_context else ""
            result = self._explorer(
                data_profile=data_profile,
                statistics=statistics,
                question=question + context_suffix
            )

            patterns = [p.strip() for p in str(result.patterns).split('|') if p.strip()]
            relationships = [r.strip() for r in str(result.relationships).split('|') if r.strip()]
            anomalies = [a.strip() for a in str(result.anomalies).split('|') if a.strip()]

            self._broadcast("eda_completed", {
                'patterns': len(patterns),
                'relationships': len(relationships)
            })

            return {
                'patterns': patterns,
                'relationships': relationships,
                'anomalies': anomalies,
                'exploration_summary': str(result.exploration_summary)
            }

        except Exception as e:
            logger.error(f"EDA failed: {e}")
            return {'error': str(e)}


class StatisticalAgent(BaseDataAgent):
    """Performs statistical analysis."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=StatisticalAnalysisSignature)
        self._analyst = dspy.ChainOfThought(StatisticalAnalysisSignature)
        self.learned_context = learned_context

    async def analyze(
        self,
        data_summary: str,
        columns: List[str],
        goals: str = "Comprehensive statistical analysis"
    ) -> Dict[str, Any]:
        """Perform statistical analysis."""
        try:
            context_suffix = f"\n\n[Learned Context]: {self.learned_context}" if self.learned_context else ""
            result = self._analyst(
                data_summary=data_summary,
                columns_of_interest=json.dumps(columns) if columns else '["text"]',
                analysis_goals=goals + context_suffix
            )

            try:
                descriptive = json.loads(result.descriptive)
            except:
                # If not valid JSON, create a simple dict
                descriptive = {'raw': str(result.descriptive)}

            hypothesis_tests = [h.strip() for h in str(result.hypothesis_tests).split('|') if h.strip()]

            self._broadcast("statistics_completed", {
                'tests': len(hypothesis_tests)
            })

            return {
                'reasoning': str(getattr(result, 'reasoning', '')),
                'descriptive': descriptive,
                'distributions': str(result.distributions),
                'correlations': str(result.correlations),
                'hypothesis_tests': hypothesis_tests
            }

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e), 'descriptive': {}, 'distributions': '', 'correlations': '', 'hypothesis_tests': []}


class InsightAgent(BaseDataAgent):
    """Generates insights."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=InsightGenerationSignature)
        self._generator = dspy.ChainOfThought(InsightGenerationSignature)
        self.learned_context = learned_context

    async def generate(
        self,
        analysis_results: str,
        business_context: str = "",
        question: str = ""
    ) -> List[Insight]:
        """Generate insights."""
        try:
            context_suffix = f"\n\n[Learned Context]: {self.learned_context}" if self.learned_context else ""
            result = self._generator(
                analysis_results=analysis_results + context_suffix,
                business_context=business_context or "General analysis",
                question=question or "What are the key insights?"
            )

            try:
                insights_data = json.loads(result.insights)
            except:
                insights_data = []

            insights = []
            for ins in insights_data:
                insights.append(Insight(
                    title=ins.get('title', ''),
                    description=ins.get('description', ''),
                    evidence=ins.get('evidence', ''),
                    impact=ins.get('impact', ''),
                    confidence=ins.get('confidence', 0.7),
                    category=ins.get('category', 'general')
                ))

            self._broadcast("insights_generated", {
                'insight_count': len(insights)
            })

            return insights

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []


class MLRecommenderAgent(BaseDataAgent):
    """Recommends ML approaches."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=MLRecommendationSignature)
        self._recommender = dspy.ChainOfThought(MLRecommendationSignature)
        self.learned_context = learned_context

    async def recommend(
        self,
        data_profile: str,
        target_column: str = "",
        goal: str = ""
    ) -> MLRecommendation:
        """Recommend ML approach."""
        try:
            context_suffix = f"\n\n[Learned Context]: {self.learned_context}" if self.learned_context else ""
            result = self._recommender(
                data_profile=data_profile,
                target_column=target_column or "Not specified",
                goal=(goal or "Build predictive model") + context_suffix
            )

            models = [m.strip() for m in str(result.recommended_models).split('|') if m.strip()]
            preprocessing = [p.strip() for p in str(result.preprocessing).split('|') if p.strip()]

            self._broadcast("ml_recommended", {
                'task_type': str(result.task_type),
                'model_count': len(models)
            })

            return MLRecommendation(
                task_type=str(result.task_type),
                recommended_models=models,
                feature_importance={},
                preprocessing_steps=preprocessing,
                expected_performance="Depends on data quality and feature engineering"
            )

        except Exception as e:
            logger.error(f"ML recommendation failed: {e}")
            return MLRecommendation(
                task_type="unknown",
                recommended_models=[],
                feature_importance={},
                preprocessing_steps=[],
                expected_performance="Error in recommendation"
            )


class VisualizationAgent(BaseDataAgent):
    """Recommends visualizations."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, signature=VisualizationSignature)
        self._visualizer = dspy.ChainOfThought(VisualizationSignature)
        self.learned_context = learned_context

    async def recommend(
        self,
        data_summary: str,
        insights: str,
        audience: str = "analysts"
    ) -> List[Visualization]:
        """Recommend visualizations."""
        try:
            context_suffix = f"\n\n[Learned Context]: {self.learned_context}" if self.learned_context else ""
            result = self._visualizer(
                data_summary=data_summary + context_suffix,
                insights=insights,
                audience=audience
            )

            try:
                viz_data = json.loads(result.visualizations)
            except:
                viz_data = []

            visualizations = []
            for viz in viz_data:
                viz_type = viz.get('type', 'distribution')
                try:
                    viz_enum = VisualizationType(viz_type)
                except:
                    viz_enum = VisualizationType.DISTRIBUTION

                visualizations.append(Visualization(
                    viz_type=viz_enum,
                    title=viz.get('title', ''),
                    description=viz.get('description', ''),
                    columns=viz.get('columns', []),
                    code=viz.get('code', '')
                ))

            self._broadcast("visualizations_recommended", {
                'viz_count': len(visualizations)
            })

            return visualizations

        except Exception as e:
            logger.error(f"Visualization recommendation failed: {e}")
            return []


# =============================================================================
# DATA ANALYSIS SWARM
# =============================================================================

@register_swarm("data_analysis")
class DataAnalysisSwarm(DomainSwarm):
    """
    World-Class Data Analysis Swarm.

    Provides comprehensive data analysis with:
    - Data profiling
    - EDA
    - Statistical analysis
    - Insight generation
    - ML recommendations
    - Visualization suggestions
    """

    # Declarative agent team - auto-initialized by DomainSwarm
    AGENT_TEAM = AgentTeam.define(
        (DataProfilerAgent, "DataProfiler", "_profiler"),
        (EDAAgent, "EDA", "_eda_agent"),
        (StatisticalAgent, "Statistical", "_statistical_agent"),
        (InsightAgent, "Insight", "_insight_agent"),
        (MLRecommenderAgent, "MLRecommender", "_ml_recommender"),
        (VisualizationAgent, "Visualization", "_visualization_agent"),
    )

    def __init__(self, config: DataAnalysisConfig = None):
        super().__init__(config or DataAnalysisConfig())

    async def _execute_domain(
        self,
        data: Any,
        **kwargs
    ) -> AnalysisResult:
        """Execute data analysis (called by DomainSwarm.execute())."""
        return await self.analyze(data, **kwargs)

    async def analyze(
        self,
        data: Any,
        question: str = "",
        target_column: str = "",
        business_context: str = ""
    ) -> AnalysisResult:
        """
        Perform comprehensive data analysis.

        Args:
            data: DataFrame or dict representation of data
            question: Analysis question
            target_column: Target variable for prediction
            business_context: Business context for insights

        Returns:
            AnalysisResult with complete analysis
        """
        start_time = datetime.now()

        # Note: Pre-execution learning and agent init handled by DomainSwarm.execute()

        logger.info(f"📊 DataAnalysisSwarm starting...")

        # Convert data to analyzable format
        if isinstance(data, dict):
            data_summary = data
            sample_data = data.get('sample', [])
            column_info = data.get('columns', {})
        elif isinstance(data, str):
            # Handle text/string data - pass directly for LLM analysis
            # The LLM can extract structured data from natural language
            data_summary = {
                'type': 'text_data',
                'description': 'Data provided as text/natural language',
                'raw_text': data[:2000],  # Limit to avoid context overflow
                'shape': [1, 1],  # Treat as single text block
            }
            sample_data = [{'text': data[:500]}]
            column_info = {'text': 'string'}
            logger.info("📝 Received text data - will extract structured info via LLM")
        else:
            # Assume pandas DataFrame
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data_summary = {
                        'shape': list(data.shape),
                        'columns': list(data.columns),
                        'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                        'missing': data.isnull().sum().to_dict()
                    }
                    sample_data = data.head(10).to_dict('records')
                    column_info = {col: str(dtype) for col, dtype in data.dtypes.items()}
                else:
                    # Fallback: convert to string representation
                    data_str = str(data)
                    data_summary = {
                        'type': 'unknown_converted',
                        'description': f'Data of type {type(data).__name__} converted to text',
                        'raw_text': data_str[:2000],
                        'shape': [1, 1],
                    }
                    sample_data = [{'text': data_str[:500]}]
                    column_info = {'text': 'string'}
            except Exception as e:
                # Last resort: treat as text
                data_str = str(data)
                data_summary = {
                    'type': 'fallback_text',
                    'description': f'Could not process data: {e}',
                    'raw_text': data_str[:2000],
                    'shape': [1, 1],
                }
                sample_data = [{'text': data_str[:500]}]
                column_info = {'text': 'string'}

        config = self.config

        try:
            # =================================================================
            # PHASE 1: DATA PROFILING
            # =================================================================
            logger.info("📋 Phase 1: Profiling data...")

            profile_result = await self._profiler.profile(
                data_summary,
                sample_data,
                column_info
            )

            profile = DataProfile(
                row_count=data_summary.get('shape', [0, 0])[0],
                column_count=data_summary.get('shape', [0, 0])[1] if len(data_summary.get('shape', [])) > 1 else 0,
                columns=profile_result.get('column_profiles', {}),
                missing_summary=data_summary.get('missing', {}),
                duplicates=0,
                memory_usage="Unknown",
                data_types=column_info
            )

            self._trace_phase("DataProfiler", AgentRole.ACTOR,
                {'rows': profile.row_count, 'cols': profile.column_count},
                {'quality_issues': len(profile_result.get('quality_issues', []))},
                success=True, phase_start=start_time, tools_used=['data_profile'])

            # =================================================================
            # PHASE 2: STATISTICAL ANALYSIS & EDA (parallel)
            # =================================================================
            logger.info("📈 Phase 2: Statistical analysis & EDA...")

            stats_task = self._statistical_agent.analyze(
                json.dumps(data_summary),
                list(column_info.keys()),
                "Comprehensive analysis"
            )
            eda_task = self._eda_agent.explore(
                profile_result.get('profile_summary', ''),
                json.dumps(data_summary),
                question or "Explore the data"
            )

            stats_result, eda_result = await asyncio.gather(
                stats_task, eda_task, return_exceptions=True
            )

            if isinstance(stats_result, Exception):
                stats_result = {'error': str(stats_result)}
            if isinstance(eda_result, Exception):
                eda_result = {'error': str(eda_result)}

            statistics = StatisticalResult(
                descriptive_stats=stats_result.get('descriptive', {}),
                correlations={},
                distributions={},
                outliers={},
                hypothesis_tests=[{'test': t} for t in stats_result.get('hypothesis_tests', [])]
            )

            phase2_start = datetime.now()
            self._trace_phase("Statistical", AgentRole.EXPERT,
                {'columns': len(column_info)},
                {'has_error': 'error' in stats_result},
                success='error' not in stats_result, phase_start=start_time, tools_used=['stats_analyze'])
            self._trace_phase("EDA", AgentRole.ACTOR,
                {'question': question or 'explore'},
                {'has_error': 'error' in eda_result},
                success='error' not in eda_result, phase_start=start_time, tools_used=['eda_explore'])

            # =================================================================
            # PHASE 3: INSIGHTS & ML RECOMMENDATIONS (parallel)
            # =================================================================
            logger.info("💡 Phase 3: Generating insights & ML recommendations...")

            analysis_summary = json.dumps({
                'profile': profile_result,
                'eda': eda_result,
                'statistics': stats_result
            })

            insight_task = self._insight_agent.generate(
                analysis_summary,
                business_context,
                question
            )

            ml_task = None
            if config.include_ml_recommendations and target_column:
                ml_task = self._ml_recommender.recommend(
                    profile_result.get('profile_summary', ''),
                    target_column,
                    f"Predict {target_column}"
                )

            if ml_task:
                insights, ml_rec = await asyncio.gather(
                    insight_task, ml_task, return_exceptions=True
                )
            else:
                insights = await insight_task
                ml_rec = None

            if isinstance(insights, Exception):
                insights = []
            if isinstance(ml_rec, Exception):
                ml_rec = None

            self._trace_phase("Insight", AgentRole.EXPERT,
                {'has_context': bool(business_context)},
                {'insights_count': len(insights) if isinstance(insights, list) else 0},
                success=True, phase_start=phase2_start, tools_used=['insight_generate'])
            self._trace_phase("MLRecommender", AgentRole.PLANNER,
                {'target_column': target_column or 'none'},
                {'has_recommendation': ml_rec is not None},
                success=True, phase_start=phase2_start, tools_used=['ml_recommend'])

            # =================================================================
            # PHASE 4: VISUALIZATION RECOMMENDATIONS
            # =================================================================
            visualizations = []
            if config.include_visualizations:
                logger.info("📊 Phase 4: Recommending visualizations...")

                insights_summary = json.dumps([
                    {'title': i.title, 'description': i.description}
                    for i in (insights if isinstance(insights, list) else [])
                ])

                visualizations = await self._visualization_agent.recommend(
                    json.dumps(data_summary),
                    insights_summary,
                    "analysts"
                )

            self._trace_phase("Visualization", AgentRole.ACTOR,
                {'include_viz': config.include_visualizations},
                {'viz_count': len(visualizations)},
                success=True, phase_start=phase2_start, tools_used=['viz_recommend'])

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()

            # Calculate quality score
            missing_ratio = sum(data_summary.get('missing', {}).values()) / (
                profile.row_count * profile.column_count
            ) if profile.row_count * profile.column_count > 0 else 0
            quality_score = max(0, 1 - missing_ratio - len(profile_result.get('quality_issues', [])) * 0.1)

            # Build summary
            summary = f"""
            Data Analysis Summary:
            - Rows: {profile.row_count:,}
            - Columns: {profile.column_count}
            - Quality Score: {quality_score:.1%}
            - Insights Generated: {len(insights) if isinstance(insights, list) else 0}
            - Visualizations Recommended: {len(visualizations)}
            """

            result = AnalysisResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={'rows': profile.row_count, 'columns': profile.column_count},
                execution_time=exec_time,
                profile=profile,
                statistics=statistics,
                insights=insights if isinstance(insights, list) else [],
                ml_recommendations=ml_rec,
                visualizations=visualizations,
                summary=summary.strip(),
                data_quality_score=quality_score
            )

            logger.info(f"✅ DataAnalysisSwarm complete: {len(insights) if isinstance(insights, list) else 0} insights generated")

            # Post-execution learning
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['data_profile', 'eda_analyze', 'insight_generate']),
                task_type='data_analysis',
                output_data={
                    'rows': profile.row_count,
                    'columns': profile.column_count,
                    'data_quality_score': quality_score,
                    'insights_count': len(insights) if isinstance(insights, list) else 0,
                    'visualizations_count': len(visualizations),
                    'has_ml_recommendation': ml_rec is not None,
                    'hypothesis_tests_count': len(statistics.hypothesis_tests),
                    'quality_issues_count': len(profile_result.get('quality_issues', [])),
                },
                input_data={
                    'question': question,
                    'target_column': target_column,
                    'business_context': business_context,
                    'row_count': data_summary.get('shape', [0, 0])[0],
                    'column_count': data_summary.get('shape', [0, 0])[1] if len(data_summary.get('shape', [])) > 1 else 0,
                    'column_names': list(column_info.keys()),
                }
            )

            return result

        except Exception as e:
            logger.error(f"❌ DataAnalysisSwarm error: {e}")
            import traceback
            traceback.print_exc()
            exec_time = (datetime.now() - start_time).total_seconds()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(['data_profile']),
                task_type='data_analysis'
            )
            return AnalysisResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def analyze_data(data: Any, **kwargs) -> AnalysisResult:
    """
    One-liner data analysis.

    Usage:
        from core.swarms.data_analysis_swarm import analyze_data
        result = await analyze_data(df, question="What drives sales?")
    """
    swarm = DataAnalysisSwarm()
    return await swarm.analyze(data, **kwargs)


def analyze_data_sync(data: Any, **kwargs) -> AnalysisResult:
    """Synchronous data analysis."""
    return asyncio.run(analyze_data(data, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataAnalysisSwarm',
    'DataAnalysisConfig',
    'AnalysisResult',
    'DataProfile',
    'StatisticalResult',
    'Insight',
    'MLRecommendation',
    'Visualization',
    'AnalysisType',
    'DataType',
    'VisualizationType',
    'analyze_data',
    'analyze_data_sync',
    # Agents
    'DataProfilerAgent',
    'EDAAgent',
    'StatisticalAgent',
    'InsightAgent',
    'MLRecommenderAgent',
    'VisualizationAgent',
]
