"""
Expert Agents Module

Provides specialized, pre-trained agents that use OptimizationPipeline
to ensure reliable, correct outputs for specific domains.

🆕 Phase 8: Expert templates - Use SingleAgentOrchestrator with gold standard learning
Recommended: Use expert_templates factory functions instead of ExpertAgent class
"""

# 🆕 Phase 8: Expert Templates (Recommended)
from .expert_templates import (
    create_mermaid_expert,
    create_plantuml_expert,
    create_sql_expert,
    create_latex_math_expert,
    create_custom_expert
)

# ⚠️ Deprecated: Old ExpertAgent class (backward compatibility)
from .expert_agent import ExpertAgent, ExpertAgentConfig
from .mermaid_expert import MermaidExpertAgent
from .pipeline_expert import PipelineExpertAgent
from .plantuml_expert import PlantUMLExpertAgent
from .math_latex_expert import MathLaTeXExpertAgent
from .expert_registry import ExpertRegistry
from .memory_integration import (
    store_improvement_to_memory,
    retrieve_improvements_from_memory,
    retrieve_synthesized_improvements,
    retrieve_synthesized_improvements_async,
    consolidate_improvements,
    run_improvement_consolidation_cycle,
    sync_improvements_to_memory
)

__all__ = [
    # Phase 8: Expert Templates (Recommended)
    "create_mermaid_expert",
    "create_plantuml_expert",
    "create_sql_expert",
    "create_latex_math_expert",
    "create_custom_expert",

    # Deprecated: Old ExpertAgent classes
    "ExpertAgent",
    "ExpertAgentConfig",
    "MermaidExpertAgent",
    "PipelineExpertAgent",
    "PlantUMLExpertAgent",
    "MathLaTeXExpertAgent",
    "ExpertRegistry",

    # Memory integration utilities
    "store_improvement_to_memory",
    "retrieve_improvements_from_memory",
    "retrieve_synthesized_improvements",
    "retrieve_synthesized_improvements_async",
    "consolidate_improvements",
    "run_improvement_consolidation_cycle",
    "sync_improvements_to_memory"
]
