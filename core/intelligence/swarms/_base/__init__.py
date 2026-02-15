"""
Swarm infrastructure — base classes, types, evaluation, and learning.

Re-exports all public names so callers can use::

    from core.intelligence.swarms._base import SwarmLearning, SwarmResult
"""

from ._coordination_mixin import SwarmCoordinationMixin
from ._knowledge_mixin import SwarmKnowledgeMixin
from ._learning_mixin import SwarmLearningMixin
from .evaluation import EvaluationHistory, GoldStandardDB, ImprovementHistory
from .improvement_agents import (
    ActorAgent,
    AuditorAgent,
    CollapsedEvaluator,
    CollapsedExecutor,
    ExpertAgent,
    LearnerAgent,
    PlannerAgent,
    ReviewerAgent,
)
from .pattern_selector import PatternSelector
from .registry import SwarmRegistry, register_swarm
from .stage_config import StageConfig, StageResult, topological_sort, validate_stages
from .swarm_learning import SwarmLearning
from .swarm_types import (
    AgentRole,
    Evaluation,
    EvaluationResult,
    ExecutionTrace,
    GoldStandard,
    ImprovementSuggestion,
    ImprovementType,
    SwarmAgentConfig,
    SwarmBaseConfig,
    SwarmConfig,
    SwarmResult,
    _safe_join,
    _safe_num,
    _split_field,
)

__all__ = [
    # Core
    "SwarmLearning",
    "SwarmCoordinationMixin",
    "SwarmKnowledgeMixin",
    "SwarmLearningMixin",
    # Types
    "AgentRole",
    "Evaluation",
    "EvaluationResult",
    "ExecutionTrace",
    "GoldStandard",
    "ImprovementSuggestion",
    "ImprovementType",
    "SwarmAgentConfig",
    "SwarmBaseConfig",
    "SwarmConfig",
    "SwarmResult",
    "_safe_join",
    "_safe_num",
    "_split_field",
    # Evaluation & improvement
    "EvaluationHistory",
    "GoldStandardDB",
    "ImprovementHistory",
    "ActorAgent",
    "AuditorAgent",
    "CollapsedEvaluator",
    "CollapsedExecutor",
    "ExpertAgent",
    "LearnerAgent",
    "PlannerAgent",
    "ReviewerAgent",
    # Configuration
    "PatternSelector",
    "StageConfig",
    "StageResult",
    "SwarmRegistry",
    "register_swarm",
    "topological_sort",
    "validate_stages",
]
