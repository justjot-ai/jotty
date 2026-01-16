"""
Expert Templates - Factory Functions for Domain Experts
========================================================

Phase 8: Expert system integration with SingleAgentOrchestrator.

Expert agents are just SingleAgentOrchestrator instances with:
- enable_gold_standard_learning=True
- domain-specific prompts
- gold standard examples
- domain validators

This module provides factory functions for common expert types.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import after defining functions to avoid circular imports
from ..orchestration import SingleAgentOrchestrator
from ..foundation import JottyConfig
import dspy


# =============================================================================
# MERMAID EXPERT
# =============================================================================

def create_mermaid_expert(
    config: JottyConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None
) -> SingleAgentOrchestrator:
    """
    Create Mermaid diagram generation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, loads defaults)

    Returns:
        SingleAgentOrchestrator configured as Mermaid expert

    Example:
        >>> expert = create_mermaid_expert(config=JottyConfig())
        >>> result = await expert.arun(
        ...     question="Generate sequence diagram for user login"
        ... )
    """
    from .domain_validators import MermaidValidator
    from .training_data_loader import TrainingDataLoader

    config = config or JottyConfig()

    # Load default gold standards if not provided
    if gold_standards is None:
        try:
            loader = TrainingDataLoader(domain="mermaid")
            gold_standards = loader.load_from_github_repo(
                repo_url="https://github.com/mermaid-js/mermaid",
                path="packages/mermaid/src/diagrams/",
                file_pattern="*.spec.js",
                max_files=50
            )
            logger.info(f"Loaded {len(gold_standards)} Mermaid gold standards from GitHub")
        except Exception as e:
            logger.warning(f"Failed to load Mermaid gold standards: {e}")
            gold_standards = []

    # Create validator
    validator = MermaidValidator()

    # Mermaid signature
    class MermaidSignature(dspy.Signature):
        """Generate valid Mermaid diagram code."""
        description: str = dspy.InputField(desc="Description of the diagram to generate")
        diagram_type: str = dspy.InputField(desc="Type of diagram (flowchart, sequence, class, etc.)")
        mermaid_code: str = dspy.OutputField(desc="Valid Mermaid diagram code")

    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(MermaidSignature),
        architect_prompts=[
            str(Path(__file__).parent / "prompts" / "mermaid" / "planning.md"),
            str(Path(__file__).parent / "prompts" / "mermaid" / "diagram_types.md")
        ],
        auditor_prompts=[
            str(Path(__file__).parent / "prompts" / "mermaid" / "validation.md"),
            str(Path(__file__).parent / "prompts" / "mermaid" / "syntax_check.md")
        ],
        architect_tools=[],
        auditor_tools=[],
        config=config,

        # 🎓 Expert features (Phase 8)
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=validator.get_test_cases() if hasattr(validator, 'get_test_cases') else [],
        domain="mermaid",
        domain_validator=validator.validate,
        max_training_iterations=5,
        min_validation_score=1.0
    )


# =============================================================================
# PLANTUML EXPERT
# =============================================================================

def create_plantuml_expert(
    config: JottyConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None
) -> SingleAgentOrchestrator:
    """
    Create PlantUML diagram generation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, loads defaults)

    Returns:
        SingleAgentOrchestrator configured as PlantUML expert

    Example:
        >>> expert = create_plantuml_expert(config=JottyConfig())
        >>> result = await expert.arun(
        ...     question="Generate UML class diagram for e-commerce system"
        ... )
    """
    from .domain_validators import PlantUMLValidator
    from .training_data_loader import TrainingDataLoader

    config = config or JottyConfig()

    # Load default gold standards if not provided
    if gold_standards is None:
        try:
            loader = TrainingDataLoader(domain="plantuml")
            gold_standards = loader.load_from_github_repo(
                repo_url="https://github.com/plantuml/plantuml",
                path="test/",
                file_pattern="*.puml",
                max_files=50
            )
            logger.info(f"Loaded {len(gold_standards)} PlantUML gold standards from GitHub")
        except Exception as e:
            logger.warning(f"Failed to load PlantUML gold standards: {e}")
            gold_standards = []

    # Create validator
    validator = PlantUMLValidator()

    # PlantUML signature
    class PlantUMLSignature(dspy.Signature):
        """Generate valid PlantUML diagram code."""
        description: str = dspy.InputField(desc="Description of the UML diagram to generate")
        diagram_type: str = dspy.InputField(desc="Type of UML diagram (class, sequence, activity, etc.)")
        plantuml_code: str = dspy.OutputField(desc="Valid PlantUML code")

    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(PlantUMLSignature),
        architect_prompts=[
            str(Path(__file__).parent / "prompts" / "plantuml" / "planning.md"),
            str(Path(__file__).parent / "prompts" / "plantuml" / "uml_types.md")
        ],
        auditor_prompts=[
            str(Path(__file__).parent / "prompts" / "plantuml" / "validation.md"),
            str(Path(__file__).parent / "prompts" / "plantuml" / "syntax_check.md")
        ],
        architect_tools=[],
        auditor_tools=[],
        config=config,

        # 🎓 Expert features (Phase 8)
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=validator.get_test_cases() if hasattr(validator, 'get_test_cases') else [],
        domain="plantuml",
        domain_validator=validator.validate,
        max_training_iterations=5,
        min_validation_score=1.0
    )


# =============================================================================
# SQL EXPERT
# =============================================================================

def create_sql_expert(
    config: JottyConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None,
    dialect: str = "postgresql"
) -> SingleAgentOrchestrator:
    """
    Create SQL query generation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, loads defaults)
        dialect: SQL dialect (postgresql, mysql, sqlite, etc.)

    Returns:
        SingleAgentOrchestrator configured as SQL expert

    Example:
        >>> expert = create_sql_expert(config=JottyConfig(), dialect="postgresql")
        >>> result = await expert.arun(
        ...     question="Get top 10 customers by revenue in Q4 2023",
        ...     schema=database_schema
        ... )
    """
    from .domain_validators import SQLValidator

    config = config or JottyConfig()

    # Load default gold standards if not provided
    if gold_standards is None:
        # In a real implementation, load from SQL examples database
        # For now, use empty list (can be populated later)
        gold_standards = []
        logger.info("SQL expert: No gold standards provided, using empty list")

    # Create validator
    validator = SQLValidator(dialect=dialect)

    # SQL signature
    class SQLSignature(dspy.Signature):
        """Generate valid SQL query."""
        question: str = dspy.InputField(desc="Natural language question")
        schema: str = dspy.InputField(desc="Database schema (tables, columns, relationships)")
        sql_query: str = dspy.OutputField(desc="Valid SQL query")
        explanation: str = dspy.OutputField(desc="Explanation of the query logic")

    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(SQLSignature),
        architect_prompts=[
            str(Path(__file__).parent / "prompts" / "sql" / "planning.md"),
            str(Path(__file__).parent / "prompts" / "sql" / f"{dialect}_dialect.md")
        ],
        auditor_prompts=[
            str(Path(__file__).parent / "prompts" / "sql" / "validation.md"),
            str(Path(__file__).parent / "prompts" / "sql" / "syntax_check.md")
        ],
        architect_tools=[],
        auditor_tools=[],
        config=config,

        # 🎓 Expert features (Phase 8)
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=validator.get_test_cases() if hasattr(validator, 'get_test_cases') else [],
        domain=f"sql_{dialect}",
        domain_validator=validator.validate,
        max_training_iterations=5,
        min_validation_score=1.0
    )


# =============================================================================
# LATEX MATH EXPERT
# =============================================================================

def create_latex_math_expert(
    config: JottyConfig = None,
    gold_standards: Optional[List[Dict[str, Any]]] = None
) -> SingleAgentOrchestrator:
    """
    Create LaTeX mathematical notation expert.

    Args:
        config: Jotty configuration (optional, uses defaults)
        gold_standards: Custom gold standard examples (optional, loads defaults)

    Returns:
        SingleAgentOrchestrator configured as LaTeX math expert

    Example:
        >>> expert = create_latex_math_expert(config=JottyConfig())
        >>> result = await expert.arun(
        ...     question="Express the quadratic formula in LaTeX"
        ... )
    """
    from .domain_validators import LatexValidator

    config = config or JottyConfig()

    # Load default gold standards if not provided
    if gold_standards is None:
        gold_standards = []
        logger.info("LaTeX math expert: No gold standards provided, using empty list")

    # Create validator
    validator = LatexValidator()

    # LaTeX signature
    class LatexMathSignature(dspy.Signature):
        """Generate valid LaTeX mathematical notation."""
        description: str = dspy.InputField(desc="Description of the mathematical expression")
        latex_code: str = dspy.OutputField(desc="Valid LaTeX math code")

    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(LatexMathSignature),
        architect_prompts=[
            str(Path(__file__).parent / "prompts" / "latex" / "planning.md"),
            str(Path(__file__).parent / "prompts" / "latex" / "math_symbols.md")
        ],
        auditor_prompts=[
            str(Path(__file__).parent / "prompts" / "latex" / "validation.md"),
            str(Path(__file__).parent / "prompts" / "latex" / "syntax_check.md")
        ],
        architect_tools=[],
        auditor_tools=[],
        config=config,

        # 🎓 Expert features (Phase 8)
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=validator.get_test_cases() if hasattr(validator, 'get_test_cases') else [],
        domain="latex_math",
        domain_validator=validator.validate,
        max_training_iterations=5,
        min_validation_score=1.0
    )


# =============================================================================
# CUSTOM EXPERT FACTORY
# =============================================================================

def create_custom_expert(
    domain: str,
    agent: dspy.Module,
    architect_prompts: List[str],
    auditor_prompts: List[str],
    gold_standards: List[Dict[str, Any]],
    domain_validator: Any,
    config: JottyConfig = None,
    **kwargs
) -> SingleAgentOrchestrator:
    """
    Create a custom expert for any domain.

    This is a flexible factory for creating domain experts that don't have
    pre-built templates.

    Args:
        domain: Domain name (e.g., "json", "yaml", "regex")
        agent: DSPy module/agent
        architect_prompts: List of architect prompt file paths
        auditor_prompts: List of auditor prompt file paths
        gold_standards: Gold standard examples
        domain_validator: Validation function
        config: Jotty configuration (optional)
        **kwargs: Additional kwargs passed to SingleAgentOrchestrator

    Returns:
        SingleAgentOrchestrator configured as custom expert

    Example:
        >>> expert = create_custom_expert(
        ...     domain="json",
        ...     agent=dspy.ChainOfThought("input -> json_output"),
        ...     architect_prompts=["prompts/json/planning.md"],
        ...     auditor_prompts=["prompts/json/validation.md"],
        ...     gold_standards=json_examples,
        ...     domain_validator=json_validator.validate
        ... )
    """
    config = config or JottyConfig()

    return SingleAgentOrchestrator(
        agent=agent,
        architect_prompts=architect_prompts,
        auditor_prompts=auditor_prompts,
        architect_tools=kwargs.get('architect_tools', []),
        auditor_tools=kwargs.get('auditor_tools', []),
        config=config,

        # 🎓 Expert features (Phase 8)
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=kwargs.get('validation_cases', []),
        domain=domain,
        domain_validator=domain_validator,
        max_training_iterations=kwargs.get('max_training_iterations', 5),
        min_validation_score=kwargs.get('min_validation_score', 1.0)
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'create_mermaid_expert',
    'create_plantuml_expert',
    'create_sql_expert',
    'create_latex_math_expert',
    'create_custom_expert',
]
