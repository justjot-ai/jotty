"""
Agents Module — All Jotty agents: concrete, domain, and expert.

Contains:
- Concrete agents (AutoAgent, ChatAssistant, etc.) — high-level autonomous agents
- Domain agents (MermaidAgent, BackendAgent, etc.) — BaseAgent + capability mixins
- Expert agents (ExpertAgent, MermaidExpertAgent) — legacy OptimizationPipeline agents
- Expert templates (create_mermaid_expert, etc.) — factory functions
"""

# Concrete Agents (from modes/agent/agents/)
from .auto_agent import AutoAgent
from .autonomous_agent import AutonomousAgent, AutonomousAgentConfig

# Domain Agents (BaseAgent + LearningCapability + ValidationCapability)
from .backend_agent import BackendAgent
from .chat_assistant import ChatAssistant
from .composite_agent import CompositeAgent
from .designer_agent import DesignerAgent
from .domain_agent import DomainAgent

# ExpertAgent classes (legacy)
from .expert_agent import ExpertAgent, ExpertAgentConfig
from .expert_registry import ExpertRegistry

# Expert Templates (Recommended)
from .expert_templates import (
    create_custom_expert,
    create_latex_math_expert,
    create_mermaid_expert,
    create_plantuml_expert,
    create_sql_expert,
)
from .frontend_agent import FrontendAgent
from .latex_agent import LaTeXAgent
from .math_latex_expert import MathLaTeXExpertAgent
from .memory_integration import (
    consolidate_improvements,
    retrieve_improvements_from_memory,
    retrieve_synthesized_improvements,
    retrieve_synthesized_improvements_async,
    run_improvement_consolidation_cycle,
    store_improvement_to_memory,
    sync_improvements_to_memory,
)
from .mermaid_agent import MermaidAgent
from .mermaid_expert import MermaidExpertAgent
from .meta_agent import MetaAgent
from .model_chat_agent import ModelChatAgent
from .pipeline_agent import PipelineAgent
from .pipeline_expert import PipelineExpertAgent
from .plantuml_agent import PlantUMLAgent
from .plantuml_expert import PlantUMLExpertAgent
from .qa_agent import QAAgent
from .skill_based_agent import SkillBasedAgent
from .swarm_agent import SwarmLearningAgent
from .task_breakdown_agent import TaskBreakdownAgent
from .todo_creator_agent import TodoCreatorAgent
from .ux_researcher_agent import UXResearcherAgent
from .validation_agent import ValidationAgent

__all__ = [
    # Concrete Agents
    "AutoAgent",
    "AutonomousAgent",
    "AutonomousAgentConfig",
    "ChatAssistant",
    "CompositeAgent",
    "DomainAgent",
    "MetaAgent",
    "ModelChatAgent",
    "SkillBasedAgent",
    "SwarmLearningAgent",
    "TaskBreakdownAgent",
    "TodoCreatorAgent",
    "ValidationAgent",
    # Domain Agents
    "MermaidAgent",
    "PlantUMLAgent",
    "LaTeXAgent",
    "BackendAgent",
    "FrontendAgent",
    "DesignerAgent",
    "PipelineAgent",
    "QAAgent",
    "UXResearcherAgent",
    # Expert Templates (Recommended)
    "create_mermaid_expert",
    "create_plantuml_expert",
    "create_sql_expert",
    "create_latex_math_expert",
    "create_custom_expert",
    # Legacy ExpertAgent classes
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
    "sync_improvements_to_memory",
]
