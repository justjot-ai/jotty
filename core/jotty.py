"""
JOTTY Core - Alias Facade for Framework Components
====================================================

Maps brain-inspired terminology to actual implementations:
- Orchestrator: Main orchestrator
- Orchestrator: Alias for Orchestrator
- Cortex: SwarmMemory
- Axon: SmartAgentSlack (agent communication)
- Roadmap: SwarmTaskBoard (task planning)
"""

import logging
from typing import List, Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# JOTTY CORE IMPORTS (mapping new names to existing implementations)
# =============================================================================

# Orchestrator = Main Orchestrator (V2)
from .orchestration import Orchestrator


# Configuration
from .foundation.data_structures import SwarmConfig, SwarmLearningConfig
from .foundation.agent_config import AgentConfig

# Architect = Pre-execution Planner 
# Auditor = Post-execution Validator 
from .agents.inspector import ValidatorAgent as ValidatorAgent
from .agents.inspector import MultiRoundValidator as IterativeAuditor

# Cortex = Hierarchical Memory
from .memory.cortex import SwarmMemory as Cortex

# Axon = Agent Communication
from .agents.axon import SmartAgentSlack as Axon

# Roadmap = Markovian Task List (V2)
from .orchestration import SwarmTaskBoard as Roadmap
from .orchestration import SubtaskState as Checkpoint

# Optimization Pipeline (V2)
from .orchestration import (
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
from .context.chunker import ContextChunker as Segmenter
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

def create_swarm_manager(agents: List[AgentConfig], config: Optional[SwarmConfig] = None, metadata_provider: Any = None, **kwargs: Any) -> Orchestrator:
    """
    Create a new Orchestrator (orchestrator) for agent swarms.
    
    Args:
        agents: List of AgentConfig defining the agents in the swarm
        config: SwarmConfig with framework settings
        metadata_provider: Optional metadata provider instance
        **kwargs: Additional arguments passed to Orchestrator
    
    Returns:
        Orchestrator instance ready to orchestrate the swarm
    
    Example:
        ```python
        from core import create_swarm_manager, AgentConfig

        agents = [
            AgentConfig(
                name="MyAgent",
                agent=my_dspy_module,
                architect_prompts=["Plan the execution"],
                auditor_prompts=["Validate the output"]
            )
        ]

        manager = create_swarm_manager(agents)
        result = manager.run(goal="Do something")
        ```
    """
    if config is None:
        config = SwarmConfig()

    return Orchestrator(
        agents=agents,
        config=config,
        **kwargs
    )




def create_cortex(config: Optional[SwarmConfig] = None) -> Cortex:
    """
    Create a new Cortex (hierarchical memory) instance.
    
    The Cortex manages multi-level memory similar to the brain's cortex:
    - Episodic: Recent experiences
    - Semantic: General knowledge
    - Procedural: How to do things
    - Causal: Why things work
    """
    if config is None:
        config = SwarmConfig()
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
    Create a new Roadmap (Markovian Task List) for task planning.
    
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
)

# Workflow Components
from .use_cases.workflow import (
    WorkflowContext,
)

# API Layer (NEW)
from .api import (
    JottyAPI,
    ChatAPI,
    WorkflowAPI,
)

# Server Layer (NEW - Minimal Client Integration)
try:
    from .server import (
        JottyHTTPServer,
        JottyServer,
        JottyServerConfig,
        AuthMiddleware,
        LoggingMiddleware,
        ErrorMiddleware,
        SSEFormatter,
        useChatFormatter,
    )
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # JOTTY Core
    "Orchestrator",
    "Orchestrator",
    "SwarmConfig",
    "AgentConfig",
    "ValidatorAgent",
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
    
    # Server Layer (NEW - Minimal Client Integration)
    "JottyHTTPServer",
    "JottyServer",
    "JottyServerConfig",
    "AuthMiddleware",
    "LoggingMiddleware",
    "ErrorMiddleware",
    "SSEFormatter",
    "useChatFormatter",
    
    # Convenience functions
    "create_swarm_manager",
    "create_cortex",
    "create_axon",
    "create_roadmap",
    
    # Version info
    "__version__",
    "__codename__",
    "__description__",
]

