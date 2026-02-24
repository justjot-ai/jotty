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
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# JOTTY CORE IMPORTS (mapping new names to existing implementations)
# =============================================================================

# Axon = Agent Communication
from .infrastructure.context.chunker import ContextChunker as Segmenter
from .infrastructure.context.compressor import AgenticCompressor as Distiller

# Context Gradient (LLM-based Learning)
from .infrastructure.context.context_gradient import (
    ContextApplier,
    ContextGradient,
    ContextUpdate,
)
from .infrastructure.context.context_manager import SmartContextManager as Focus
from .infrastructure.data.data_registry import DataRegistry as Catalog

# Data Flow
from .infrastructure.data.io_manager import IOManager as Datastream
from .infrastructure.foundation.agent_config import AgentConfig

# Configuration
from .infrastructure.foundation.data_structures import SwarmLearningConfig as SwarmConfig

# Persistence
from .infrastructure.persistence.persistence import Vault as Vault
from .infrastructure.persistence.session_manager import (
    SessionManager as Chronicle,
)
from .infrastructure.persistence.shared_context import (
    SharedContext as Blackboard,
)

# Context Management
from .infrastructure.utils.algorithmic_foundations import (
    GlobalContextGuard as ContextSentinel,
)

# Credit Assignment (Game Theory)
from .intelligence.learning.algorithmic_credit import (
    DifferenceRewardEstimator as ImpactEstimator,
)
from .intelligence.learning.algorithmic_credit import ShapleyValueEstimator as ContributionEstimator
from .intelligence.learning.q_learning import LLMQPredictor as RewardLearner

# Learning Components
from .intelligence.learning.td_lambda import TDLambdaLearner as TemporalLearner

# Cortex = Hierarchical Memory
from .intelligence.memory.cortex import SwarmMemory as Cortex

# Optimization Pipeline (V2)
# Roadmap = Markovian Task List (V2)
# Orchestrator = Main Orchestrator (V2)
from .intelligence.orchestration.core.swarm_manager import Orchestrator
from .intelligence.orchestration.learning.optimization_pipeline import (
    IterationResult,
    OptimizationConfig,
    OptimizationPipeline,
    create_optimization_pipeline,
)
from .intelligence.orchestration.state.swarm_roadmap import SubtaskState as Checkpoint
from .intelligence.orchestration.state.swarm_roadmap import SwarmTaskBoard as Roadmap
from .intelligence.reasoning.tools.axon import SmartAgentSlack as Axon

# Architect = Pre-execution Planner
# Auditor = Post-execution Validator
from .intelligence.reasoning.tools.inspector import (
    MultiRoundValidator as IterativeAuditor,
)
from .intelligence.reasoning.tools.inspector import ValidatorAgent as ValidatorAgent

# =============================================================================
# JOTTY CONVENIENCE FUNCTIONS
# =============================================================================


def create_swarm_manager(
    agents: List[AgentConfig],
    config: Optional[SwarmConfig] = None,
    metadata_provider: Any = None,
    **kwargs: Any,
) -> Orchestrator:
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

    return Orchestrator(agents=agents, config=config, **kwargs)


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
    return Cortex(config)  # type: ignore[arg-type, call-arg]


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
    roadmap = Roadmap(main_goal=goal)  # type: ignore[call-arg]
    return roadmap


# =============================================================================
# JOTTY VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__codename__ = "JOTTY"
__description__ = "Brain-Inspired Orchestration for LLM Agent Swarms"

# =============================================================================
# API LAYER — Thin facades over Orchestrator.run() and Orchestrator.chat()
# =============================================================================

from .interface.api import ChatAPI, JottyAPI, WorkflowAPI

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # JOTTY Core
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
    "ContextGradient",
    "ContextApplier",
    "ContextUpdate",
    # Optimization Pipeline
    "OptimizationPipeline",
    "OptimizationConfig",
    "IterationResult",
    "create_optimization_pipeline",
    # API Layer
    "JottyAPI",
    "ChatAPI",
    "WorkflowAPI",
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
