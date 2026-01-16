"""
Team Templates - Factory Functions for Multi-Agent Teams
==========================================================

Phase 8: Team coordination patterns using MultiAgentsOrchestrator.

Teams are MultiAgentsOrchestrator instances coordinating multiple
SingleAgentOrchestrator agents (expert or non-expert).

This module provides factory functions for common team patterns.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from .conductor import MultiAgentsOrchestrator
from ..foundation import AgentConfig as ActorConfig, JottyConfig
from ..experts.expert_templates import (
    create_mermaid_expert,
    create_plantuml_expert,
    create_sql_expert,
    create_latex_math_expert
)
from .single_agent_orchestrator import SingleAgentOrchestrator
import dspy


# =============================================================================
# DIAGRAM GENERATION TEAM
# =============================================================================

def create_diagram_team(
    config: JottyConfig = None,
    metadata_provider: Any = None,
    include_plantuml: bool = True,
    include_mermaid: bool = True
) -> MultiAgentsOrchestrator:
    """
    Create a team specialized in diagram generation.

    Team Members:
    - Mermaid Expert: Flowcharts, sequence diagrams, etc.
    - PlantUML Expert: UML class diagrams, activity diagrams, etc.

    Args:
        config: Jotty configuration
        metadata_provider: Metadata provider for tool discovery
        include_plantuml: Include PlantUML expert (default: True)
        include_mermaid: Include Mermaid expert (default: True)

    Returns:
        MultiAgentsOrchestrator coordinating diagram experts

    Example:
        >>> team = create_diagram_team(config=config, metadata_provider=provider)
        >>> result = await team.run(
        ...     goal="Generate sequence diagram for user authentication and class diagram for user model"
        ... )
    """
    config = config or JottyConfig()
    actors = []

    if include_mermaid:
        actors.append(ActorConfig(
            name="MermaidExpert",
            agent=create_mermaid_expert(config=config),
            enable_architect=True,
            enable_auditor=True,
            tools=[],
            description="Expert in Mermaid diagram generation (flowcharts, sequence, gantt, etc.)"
        ))

    if include_plantuml:
        actors.append(ActorConfig(
            name="PlantUMLExpert",
            agent=create_plantuml_expert(config=config),
            enable_architect=True,
            enable_auditor=True,
            tools=[],
            description="Expert in PlantUML diagram generation (UML class, activity, state, etc.)"
        ))

    if not actors:
        raise ValueError("At least one diagram expert must be enabled")

    return MultiAgentsOrchestrator(
        actors=actors,
        metadata_provider=metadata_provider,
        config=config
    )


# =============================================================================
# SQL ANALYTICS TEAM
# =============================================================================

def create_sql_analytics_team(
    config: JottyConfig = None,
    metadata_provider: Any = None,
    sql_dialect: str = "postgresql"
) -> MultiAgentsOrchestrator:
    """
    Create a team specialized in SQL analytics.

    Team Members:
    - SQL Expert: Query generation and optimization
    - Data Analyst: Result interpretation (non-expert)
    - Visualization Expert: Chart/diagram creation

    Args:
        config: Jotty configuration
        metadata_provider: Metadata provider for tool discovery
        sql_dialect: SQL dialect for the expert

    Returns:
        MultiAgentsOrchestrator coordinating SQL analytics team

    Example:
        >>> team = create_sql_analytics_team(config=config, metadata_provider=provider)
        >>> result = await team.run(
        ...     goal="Analyze Q4 2023 revenue by region and create visualization"
        ... )
    """
    config = config or JottyConfig()

    # SQL Expert
    sql_expert = ActorConfig(
        name="SQLExpert",
        agent=create_sql_expert(config=config, dialect=sql_dialect),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description=f"Expert in {sql_dialect} query generation and optimization"
    )

    # Data Analyst (non-expert, regular agent)
    class DataAnalysisSignature(dspy.Signature):
        """Analyze data and provide insights."""
        data: str = dspy.InputField(desc="Query results or dataset")
        question: str = dspy.InputField(desc="Analysis question")
        insights: str = dspy.OutputField(desc="Key insights and patterns")
        recommendations: str = dspy.OutputField(desc="Actionable recommendations")

    data_analyst = ActorConfig(
        name="DataAnalyst",
        agent=SingleAgentOrchestrator(
            agent=dspy.ChainOfThought(DataAnalysisSignature),
            architect_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "analyst" / "planning.md")
            ],
            auditor_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "analyst" / "validation.md")
            ],
            architect_tools=[],
            auditor_tools=[],
            config=config
        ),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="Data analyst providing insights and recommendations"
    )

    # Visualization Expert (Mermaid for charts)
    viz_expert = ActorConfig(
        name="VisualizationExpert",
        agent=create_mermaid_expert(config=config),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="Expert in data visualization using Mermaid charts"
    )

    return MultiAgentsOrchestrator(
        actors=[sql_expert, data_analyst, viz_expert],
        metadata_provider=metadata_provider,
        config=config
    )


# =============================================================================
# DOCUMENTATION TEAM
# =============================================================================

def create_documentation_team(
    config: JottyConfig = None,
    metadata_provider: Any = None,
    include_latex: bool = True,
    include_diagrams: bool = True
) -> MultiAgentsOrchestrator:
    """
    Create a team specialized in technical documentation.

    Team Members:
    - Technical Writer: Main documentation (non-expert)
    - LaTeX Math Expert: Mathematical notation
    - Diagram Expert: Visual documentation
    - Code Documenter: Code examples and API docs (non-expert)

    Args:
        config: Jotty configuration
        metadata_provider: Metadata provider for tool discovery
        include_latex: Include LaTeX math expert
        include_diagrams: Include diagram expert

    Returns:
        MultiAgentsOrchestrator coordinating documentation team

    Example:
        >>> team = create_documentation_team(config=config, metadata_provider=provider)
        >>> result = await team.run(
        ...     goal="Document the authentication system with code examples and diagrams"
        ... )
    """
    config = config or JottyConfig()
    actors = []

    # Technical Writer (non-expert)
    class TechnicalWritingSignature(dspy.Signature):
        """Write clear technical documentation."""
        topic: str = dspy.InputField(desc="Topic to document")
        audience: str = dspy.InputField(desc="Target audience (developers, users, etc.)")
        documentation: str = dspy.OutputField(desc="Clear, structured documentation")

    tech_writer = ActorConfig(
        name="TechnicalWriter",
        agent=SingleAgentOrchestrator(
            agent=dspy.ChainOfThought(TechnicalWritingSignature),
            architect_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "writer" / "planning.md")
            ],
            auditor_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "writer" / "validation.md")
            ],
            architect_tools=[],
            auditor_tools=[],
            config=config
        ),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="Technical writer for clear documentation"
    )
    actors.append(tech_writer)

    # LaTeX Math Expert
    if include_latex:
        latex_expert = ActorConfig(
            name="LatexMathExpert",
            agent=create_latex_math_expert(config=config),
            enable_architect=True,
            enable_auditor=True,
            tools=[],
            description="Expert in LaTeX mathematical notation"
        )
        actors.append(latex_expert)

    # Diagram Expert
    if include_diagrams:
        diagram_expert = ActorConfig(
            name="DiagramExpert",
            agent=create_mermaid_expert(config=config),
            enable_architect=True,
            enable_auditor=True,
            tools=[],
            description="Expert in diagram generation for documentation"
        )
        actors.append(diagram_expert)

    return MultiAgentsOrchestrator(
        actors=actors,
        metadata_provider=metadata_provider,
        config=config
    )


# =============================================================================
# DATA SCIENCE TEAM
# =============================================================================

def create_data_science_team(
    config: JottyConfig = None,
    metadata_provider: Any = None,
    sql_dialect: str = "postgresql"
) -> MultiAgentsOrchestrator:
    """
    Create a complete data science team.

    Team Members:
    - SQL Expert: Data extraction
    - Data Analyst: Statistical analysis (non-expert)
    - ML Engineer: Model recommendations (non-expert)
    - Visualization Expert: Charts and dashboards
    - Technical Writer: Report generation (non-expert)

    Args:
        config: Jotty configuration
        metadata_provider: Metadata provider for tool discovery
        sql_dialect: SQL dialect

    Returns:
        MultiAgentsOrchestrator coordinating data science team

    Example:
        >>> team = create_data_science_team(config=config, metadata_provider=provider)
        >>> result = await team.run(
        ...     goal="Complete analysis: extract data, analyze patterns, recommend ML approach, create visualizations, generate report"
        ... )
    """
    config = config or JottyConfig()

    # SQL Expert
    sql_expert = ActorConfig(
        name="SQLExpert",
        agent=create_sql_expert(config=config, dialect=sql_dialect),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="Expert in data extraction via SQL"
    )

    # Data Analyst
    class DataAnalysisSignature(dspy.Signature):
        """Perform statistical data analysis."""
        data: str = dspy.InputField(desc="Dataset to analyze")
        analysis_type: str = dspy.InputField(desc="Type of analysis (descriptive, inferential, etc.)")
        analysis_results: str = dspy.OutputField(desc="Statistical analysis results")
        insights: str = dspy.OutputField(desc="Key insights and patterns")

    data_analyst = ActorConfig(
        name="DataAnalyst",
        agent=SingleAgentOrchestrator(
            agent=dspy.ChainOfThought(DataAnalysisSignature),
            architect_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "analyst" / "planning.md")
            ],
            auditor_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "analyst" / "validation.md")
            ],
            architect_tools=[],
            auditor_tools=[],
            config=config
        ),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="Data analyst for statistical analysis"
    )

    # ML Engineer
    class MLEngineeringSignature(dspy.Signature):
        """Recommend ML approaches and model architectures."""
        problem: str = dspy.InputField(desc="Problem description and data characteristics")
        requirements: str = dspy.InputField(desc="Requirements (accuracy, interpretability, etc.)")
        ml_recommendation: str = dspy.OutputField(desc="Recommended ML approach and models")
        implementation_plan: str = dspy.OutputField(desc="Implementation steps")

    ml_engineer = ActorConfig(
        name="MLEngineer",
        agent=SingleAgentOrchestrator(
            agent=dspy.ChainOfThought(MLEngineeringSignature),
            architect_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "ml" / "planning.md")
            ],
            auditor_prompts=[
                str(Path(__file__).parent.parent / "prompts" / "ml" / "validation.md")
            ],
            architect_tools=[],
            auditor_tools=[],
            config=config
        ),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="ML engineer for model recommendations"
    )

    # Visualization Expert
    viz_expert = ActorConfig(
        name="VisualizationExpert",
        agent=create_mermaid_expert(config=config),
        enable_architect=True,
        enable_auditor=True,
        tools=[],
        description="Expert in data visualization"
    )

    return MultiAgentsOrchestrator(
        actors=[sql_expert, data_analyst, ml_engineer, viz_expert],
        metadata_provider=metadata_provider,
        config=config
    )


# =============================================================================
# CUSTOM TEAM FACTORY
# =============================================================================

def create_custom_team(
    actors: List[ActorConfig],
    config: JottyConfig = None,
    metadata_provider: Any = None,
    **kwargs
) -> MultiAgentsOrchestrator:
    """
    Create a custom team with any combination of agents.

    Args:
        actors: List of ActorConfig instances
        config: Jotty configuration
        metadata_provider: Metadata provider for tool discovery
        **kwargs: Additional kwargs passed to MultiAgentsOrchestrator

    Returns:
        MultiAgentsOrchestrator coordinating custom team

    Example:
        >>> team = create_custom_team(
        ...     actors=[
        ...         ActorConfig(name="Expert1", agent=create_mermaid_expert(config)),
        ...         ActorConfig(name="Expert2", agent=create_sql_expert(config)),
        ...         ActorConfig(name="Agent3", agent=my_regular_agent)
        ...     ],
        ...     config=config,
        ...     metadata_provider=provider
        ... )
    """
    config = config or JottyConfig()

    return MultiAgentsOrchestrator(
        actors=actors,
        metadata_provider=metadata_provider,
        config=config,
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'create_diagram_team',
    'create_sql_analytics_team',
    'create_documentation_team',
    'create_data_science_team',
    'create_custom_team',
]
