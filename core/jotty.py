"""
JOTTY: SYnergistic Neural Agent Processing & Self-organizing Execution
=========================================================================

Brain-Inspired Orchestration for LLM Agent Swarms

This is the main entry point for the JOTTY framework (evolved from Jotty).

JOTTY Terminology:
- Conductor = Main orchestrator (was Conductor)
- JottyCore = Core execution engine 
- Architect = Pre-execution planner 
- Auditor = Post-execution validator 
- Cortex = Hierarchical memory (was HierarchicalMemory)
- Axon = Agent communication channel (was AgentSlack)
- Roadmap = Task planning with state tracking (was MarkovianTODO)

Scientific Foundations:
- Reinforcement Learning: TD(λ), Q-Learning, Policy Exploration
- Game Theory: Shapley Values, Nash Equilibrium, Coalition Formation
- Neuroscience: Hippocampal Memory, Sharp Wave Ripple, Cortical Hierarchy
- Information Theory: Shannon Entropy, Surprise, Compression
- Multi-Agent Systems: MARL, Swarm Coordination, Predictive Cooperation

# ✅ GENERIC: No domain-specific logic (no SQL, no domain-specific, no tool-specific code)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# JOTTY CORE IMPORTS (mapping new names to existing implementations)
# =============================================================================

# Conductor = Main Orchestrator
from .orchestration.conductor import Conductor

# JottyCore = Core Execution Engine
from .orchestration.jotty_core import JottyCore

# Configuration
from .foundation.data_structures import SwarmConfig, JottyConfig  # JottyConfig = backward compat
from .foundation.agent_config import AgentSpec, AgentConfig  # AgentConfig = backward compat

# Architect = Pre-execution Planner 
# Auditor = Post-execution Validator 
from .agents.inspector import InspectorAgent as InspectorAgent
from .agents.inspector import MultiRoundValidator as IterativeAuditor

# Cortex = Hierarchical Memory
from .memory.cortex import HierarchicalMemory as Cortex

# Axon = Agent Communication
from .agents.axon import SmartAgentSlack as Axon

# Roadmap = Markovian TODO
from .orchestration.roadmap import MarkovianTODO as Roadmap
from .orchestration.roadmap import SubtaskState as Checkpoint

# Optimization Pipeline
from .orchestration.optimization_pipeline import (
    OptimizationPipeline,
    OptimizationConfig,
    IterationResult,
    create_optimization_pipeline
)

# Learning Components
from .learning.learning import TDLambdaLearner as TemporalLearner
from .learning.q_learning import LLMQPredictor as RewardLearner

# Credit Assignment (Game Theory)
from .learning.algorithmic_credit import ShapleyValueEstimator as ContributionEstimator
from .learning.algorithmic_credit import DifferenceRewardEstimator as ImpactEstimator

# Context Management
from .context.global_context_guard import GlobalContextGuard as ContextSentinel
from .context.context_manager import SmartContextManager as Focus
from .context.chunker import AgenticChunker as Segmenter
from .context.compressor import AgenticCompressor as Distiller

# Data Flow
from .data.io_manager import IOManager as Datastream
from .persistence.shared_context import SharedContext as Blackboard
from .data.data_registry import DataRegistry as Catalog

# Persistence
from .persistence.persistence import Vault as Vault
from .persistence.session_manager import SessionManager as Chronicle

# Predictive Cooperation
from .learning.predictive_cooperation import (
    CooperationPrinciples,
    NashBargainingSolver,
    CooperationReasoner,
    PredictiveCooperativeAgent
)

# Context Gradient (LLM-based Learning)
from .context.context_gradient import (
    ContextGradient,
    ContextApplier,
    ContextUpdate
)

# =============================================================================
# JOTTY CONVENIENCE FUNCTIONS
# =============================================================================

def create_conductor(
    agents: List[AgentConfig],
    config: Optional[JottyConfig] = None,
    metadata_provider: Any = None,
    **kwargs
) -> Conductor:
    """
    Create a new JOTTY Conductor (orchestrator) for agent swarms.
    
    Args:
        agents: List of AgentConfig defining the agents in the swarm
        config: JottyConfig with framework settings
        metadata_provider: Optional metadata provider instance
        **kwargs: Additional arguments passed to Conductor
    
    Returns:
        Conductor instance ready to orchestrate the swarm
    
    Example:
        ```python
        from jotty import create_conductor, AgentConfig
        
        agents = [
            AgentConfig(
                name="MyAgent",
                agent=my_dspy_module,
                architect_prompts=["Plan the execution"],
                auditor_prompts=["Validate the output"]
            )
        ]
        
        conductor = create_conductor(agents)
        result = conductor.run(goal="Do something")
        ```
    """
    if config is None:
        config = JottyConfig()
    
    return Conductor(
        actors=agents,  # Still uses 'actors' internally for now
        config=config,
        metadata_provider=metadata_provider,
        **kwargs
    )


def create_cortex(config: Optional[JottyConfig] = None) -> Cortex:
    """
    Create a new Cortex (hierarchical memory) instance.
    
    The Cortex manages multi-level memory similar to the brain's cortex:
    - Episodic: Recent experiences
    - Semantic: General knowledge
    - Procedural: How to do things
    - Causal: Why things work
    """
    if config is None:
        config = JottyConfig()
    return Cortex(config)


def create_axon() -> Axon:
    """
    Create a new Axon (agent communication channel).
    
    The Axon enables neural-inspired messaging between agents,
    with automatic format transformation and context compression.
    """
    return Axon()


def create_roadmap(goal: str) -> Roadmap:
    """
    Create a new Roadmap (Markovian TODO) for task planning.
    
    The Roadmap tracks tasks with state, enables checkpointing,
    and supports dynamic updates during execution.
    """
    roadmap = Roadmap(main_goal=goal)
    return roadmap


# =============================================================================
# JOTTY VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__codename__ = "JOTTY"
__description__ = "Brain-Inspired Orchestration for LLM Agent Swarms"

# =============================================================================
# USE CASE LAYER (NEW - Production-Grade Multi-Agent System)
# =============================================================================

# Use Cases
from .use_cases import (
    BaseUseCase,
    ChatUseCase,
    WorkflowUseCase,
    UseCaseResult,
    UseCaseConfig,
)

# Chat Components
from .use_cases.chat import (
    ChatContext,
    ChatMessage,
    ChatExecutor,
    ChatOrchestrator,
)

# Workflow Components
from .use_cases.workflow import (
    WorkflowContext,
    WorkflowExecutor,
    WorkflowOrchestrator,
)

# API Layer (NEW)
from .api import (
    JottyAPI,
    ChatAPI,
    WorkflowAPI,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # JOTTY Core
    "Conductor",
    "JottyCore",
    "SwarmConfig",
    "JottyConfig",  # Backward compatibility
    "AgentSpec",
    "AgentConfig",  # Backward compatibility
    "InspectorAgent",
    "IterativeAuditor",
    
    # Memory & State
    "Cortex",
    "Axon",
    "Roadmap",
    "Checkpoint",
    
    # Learning
    "TemporalLearner",
    "RewardLearner",
    "ContributionEstimator",
    "ImpactEstimator",
    
    # Context Management
    "ContextSentinel",
    "Focus",
    "Segmenter",
    "Distiller",
    
    # Data Flow
    "Datastream",
    "Blackboard",
    "Catalog",
    "Vault",
    "Chronicle",
    
    # Cooperation
    "CooperationPrinciples",
    "NashBargainingSolver",
    "CooperationReasoner",
    "PredictiveCooperativeAgent",
    "ContextGradient",
    "ContextApplier",
    "ContextUpdate",
    
    # Optimization Pipeline
    "OptimizationPipeline",
    "OptimizationConfig",
    "IterationResult",
    "create_optimization_pipeline",
    
    # Use Cases (NEW)
    "BaseUseCase",
    "ChatUseCase",
    "WorkflowUseCase",
    "UseCaseResult",
    "UseCaseConfig",
    "ChatContext",
    "ChatMessage",
    "WorkflowContext",
    
    # API Layer (NEW)
    "JottyAPI",
    "ChatAPI",
    "WorkflowAPI",
    
    # Convenience functions
    "create_conductor",
    "create_cortex",
    "create_axon",
    "create_roadmap",
    
    # Version info
    "__version__",
    "__codename__",
    "__description__",
]

