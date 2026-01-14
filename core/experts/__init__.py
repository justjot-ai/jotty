"""
Expert Agents Module

Provides specialized, pre-trained agents that use OptimizationPipeline
to ensure reliable, correct outputs for specific domains.
"""

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
    "ExpertAgent",
    "ExpertAgentConfig",
    "MermaidExpertAgent",
    "PipelineExpertAgent",
    "PlantUMLExpertAgent",
    "MathLaTeXExpertAgent",
    "ExpertRegistry",
    "store_improvement_to_memory",
    "retrieve_improvements_from_memory",
    "retrieve_synthesized_improvements",
    "retrieve_synthesized_improvements_async",
    "consolidate_improvements",
    "run_improvement_consolidation_cycle",
    "sync_improvements_to_memory"
]
