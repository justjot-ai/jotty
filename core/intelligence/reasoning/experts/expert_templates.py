"""
Expert Templates - Factory Functions for Domain Experts
========================================================

Factory functions for creating domain expert instances.

Expert agents are BaseExpert subclasses with:
- Domain-specific DSPy signatures
- Gold standard training data
- Domain validators
- Improvement persistence via SwarmMemory

This module provides convenience factory functions for common expert types.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import dspy

from Jotty.core.infrastructure.foundation import SwarmConfig

# =============================================================================
# MERMAID EXPERT
# =============================================================================


def create_mermaid_expert(
    config: SwarmConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    memory: Any = None,
    improvements: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    Create Mermaid diagram generation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, uses defaults)
        memory: Optional SwarmMemory instance
        improvements: Optional list of learned improvements

    Returns:
        MermaidExpertAgent instance

    Example:
        >>> expert = create_mermaid_expert()
        >>> result = await expert.generate(task="Generate sequence diagram for user login")
    """
    from .expert_agent import ExpertAgentConfig
    from .mermaid_expert import MermaidExpertAgent

    expert_config = ExpertAgentConfig(
        name="mermaid_expert",
        domain="mermaid",
        description="Expert agent for generating perfect Mermaid diagrams",
        training_gold_standards=gold_standards,
        max_training_iterations=5,
        min_validation_score=1.0,
        use_memory_storage=memory is not None,
    )

    return MermaidExpertAgent(
        config=expert_config,
        memory=memory,
        improvements=improvements,
    )


# =============================================================================
# PLANTUML EXPERT
# =============================================================================


def create_plantuml_expert(
    config: SwarmConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    memory: Any = None,
    improvements: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    Create PlantUML diagram generation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, uses defaults)
        memory: Optional SwarmMemory instance
        improvements: Optional list of learned improvements

    Returns:
        PlantUMLExpertAgent instance

    Example:
        >>> expert = create_plantuml_expert()
        >>> result = await expert.generate(task="Generate UML class diagram for e-commerce")
    """
    from .expert_agent import ExpertAgentConfig
    from .plantuml_expert import PlantUMLExpertAgent

    expert_config = ExpertAgentConfig(
        name="plantuml_expert",
        domain="plantuml",
        description="Expert agent for generating perfect PlantUML diagrams",
        training_gold_standards=gold_standards,
        max_training_iterations=5,
        min_validation_score=1.0,
        use_memory_storage=memory is not None,
    )

    return PlantUMLExpertAgent(
        config=expert_config,
        memory=memory,
        improvements=improvements,
    )


# =============================================================================
# SQL EXPERT
# =============================================================================


def create_sql_expert(
    config: SwarmConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    dialect: str = "postgresql",
    memory: Any = None,
    improvements: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    Create SQL query generation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional)
        dialect: SQL dialect (postgresql, mysql, sqlite, etc.)
        memory: Optional SwarmMemory instance
        improvements: Optional list of learned improvements

    Returns:
        ExpertAgent configured for SQL generation

    Example:
        >>> expert = create_sql_expert(dialect="postgresql")
        >>> result = await expert.generate(
        ...     task="Get top 10 customers by revenue in Q4 2023",
        ...     context={"schema": database_schema}
        ... )
    """
    from .expert_agent import ExpertAgent, ExpertAgentConfig

    # SQL signature
    class SQLGenerationSignature(dspy.Signature):
        """Generate valid SQL query for the given question and schema."""

        task: str = dspy.InputField(desc="Natural language question")
        learned_improvements: str = dspy.InputField(desc="Previously learned patterns", default="")
        sql_query: str = dspy.OutputField(desc="Valid SQL query")

    def create_sql_agent(improvements: Any = None) -> Any:
        return dspy.ChainOfThought(SQLGenerationSignature)

    def create_sql_teacher() -> Any:
        class SQLTeacherSignature(dspy.Signature):
            """Provide the correct SQL query."""

            task: str = dspy.InputField(desc="Task description")
            gold_standard: str = dspy.InputField(desc="The correct SQL query")
            student_output: str = dspy.InputField(desc="What the student generated")
            output: str = dspy.OutputField(desc="The correct SQL query")

        return dspy.Predict(SQLTeacherSignature)

    expert_config = ExpertAgentConfig(
        name=f"sql_{dialect}_expert",
        domain=f"sql_{dialect}",
        description=f"Expert agent for generating valid {dialect} SQL queries",
        training_gold_standards=gold_standards or [],
        max_training_iterations=5,
        min_validation_score=1.0,
        agent_module=create_sql_agent,
        teacher_module=create_sql_teacher,
        use_memory_storage=memory is not None,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ExpertAgent(config=expert_config, memory=memory)


# =============================================================================
# LATEX MATH EXPERT
# =============================================================================


def create_latex_math_expert(
    config: SwarmConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    memory: Any = None,
    improvements: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    Create LaTeX mathematical notation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, uses defaults)
        memory: Optional SwarmMemory instance
        improvements: Optional list of learned improvements

    Returns:
        MathLaTeXExpertAgent instance

    Example:
        >>> expert = create_latex_math_expert()
        >>> result = await expert.generate(task="Express the quadratic formula in LaTeX")
    """
    from .expert_agent import ExpertAgentConfig
    from .math_latex_expert import MathLaTeXExpertAgent

    expert_config = ExpertAgentConfig(
        name="math_latex_expert",
        domain="math_latex",
        description="Expert agent for generating perfect Math LaTeX expressions",
        training_gold_standards=gold_standards,
        max_training_iterations=5,
        min_validation_score=1.0,
        use_memory_storage=memory is not None,
    )

    return MathLaTeXExpertAgent(
        config=expert_config,
        memory=memory,
        improvements=improvements,
    )


# =============================================================================
# CUSTOM EXPERT FACTORY
# =============================================================================


def create_custom_expert(
    domain: str,
    agent: Any = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    description: str = "",
    memory: Any = None,
    improvements: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a custom expert for any domain.

    This is a flexible factory for creating domain experts that don't have
    pre-built templates.

    Args:
        domain: Domain name (e.g., "json", "yaml", "regex")
        agent: Optional DSPy module/agent (callable that returns an agent)
        gold_standards: Gold standard examples
        description: Human-readable description
        memory: Optional SwarmMemory instance
        improvements: Optional list of learned improvements
        **kwargs: Additional kwargs passed to ExpertAgentConfig

    Returns:
        ExpertAgent configured for the custom domain

    Example:
        >>> expert = create_custom_expert(
        ...     domain="json",
        ...     description="Expert for generating valid JSON",
        ...     gold_standards=json_examples,
        ... )
    """
    from .expert_agent import ExpertAgent, ExpertAgentConfig

    expert_config = ExpertAgentConfig(
        name=f"{domain}_expert",
        domain=domain,
        description=description or f"Expert agent for {domain}",
        training_gold_standards=gold_standards or [],
        max_training_iterations=kwargs.get("max_training_iterations", 5),
        min_validation_score=kwargs.get("min_validation_score", 1.0),
        agent_module=agent,
        teacher_module=kwargs.get("teacher_module"),
        evaluation_function=kwargs.get("evaluation_function"),
        use_memory_storage=memory is not None,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ExpertAgent(config=expert_config, memory=memory)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "create_mermaid_expert",
    "create_plantuml_expert",
    "create_sql_expert",
    "create_latex_math_expert",
    "create_custom_expert",
]
