"""
Swarm Types and Data Classes
==============================

Core enums and data classes used throughout the swarm infrastructure:
- AgentRole, EvaluationResult, ImprovementType (enums)
- GoldStandard, Evaluation, ImprovementSuggestion (evaluation types)
- AgentConfig, ExecutionTrace (agent types)
- SwarmConfig, SwarmResult (swarm types)

Extracted from base_swarm.py for modularity.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class AgentRole(Enum):
    """Roles in the self-improving loop."""
    EXPERT = "expert"          # Evaluates against gold standard
    REVIEWER = "reviewer"      # Reviews agent performance
    PLANNER = "planner"        # Plans task execution
    ACTOR = "actor"            # Executes tasks
    ORCHESTRATOR = "orchestrator"  # Coordinates agents
    AUDITOR = "auditor"            # Verifies outputs (Byzantine + quality audit)
    LEARNER = "learner"            # Extracts gold standards from successful runs


class EvaluationResult(Enum):
    """Evaluation outcomes."""
    EXCELLENT = "excellent"    # Exceeds gold standard
    GOOD = "good"              # Meets gold standard
    ACCEPTABLE = "acceptable"  # Minor issues
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILED = "failed"


class ImprovementType(Enum):
    """Types of improvements suggested."""
    PROMPT_REFINEMENT = "prompt_refinement"
    PARAMETER_TUNING = "parameter_tuning"
    WORKFLOW_CHANGE = "workflow_change"
    AGENT_REPLACEMENT = "agent_replacement"
    TRAINING_DATA = "training_data"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GoldStandard:
    """Gold standard for evaluation."""
    id: str
    domain: str
    task_type: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    evaluation_criteria: Dict[str, float]  # criterion -> weight
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1


@dataclass
class Evaluation:
    """Result of expert evaluation."""
    gold_standard_id: str
    actual_output: Dict[str, Any]
    scores: Dict[str, float]  # criterion -> score (0-1)
    overall_score: float
    result: EvaluationResult
    feedback: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementSuggestion:
    """Suggestion from reviewer for improvement."""
    agent_role: AgentRole
    improvement_type: ImprovementType
    description: str
    priority: int  # 1-5, 5 being highest
    expected_impact: float  # 0-1
    implementation_details: Dict[str, Any]
    based_on_evaluations: List[str]  # evaluation IDs


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    role: AgentRole
    name: str
    model: str = "sonnet"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class ExecutionTrace:
    """Trace of agent execution for learning."""
    agent_name: str
    agent_role: AgentRole
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SwarmConfig:
    """Base configuration for all swarms."""
    name: str = "BaseSwarm"
    domain: str = "general"
    version: str = "1.0.0"
    enable_self_improvement: bool = True
    enable_learning: bool = True
    parallel_execution: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    gold_standard_path: Optional[str] = None
    improvement_threshold: float = 0.7  # Below this triggers improvement
    gold_standard_max_version: int = 3  # Max gold standard versions before capping
    output_dir: str = field(default_factory=lambda: str(Path.home() / "jotty" / "swarm_outputs"))


@dataclass
class SwarmResult:
    """Base result from any swarm."""
    success: bool
    swarm_name: str
    domain: str
    output: Dict[str, Any]
    execution_time: float
    agent_traces: List[ExecutionTrace] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    improvements: List[ImprovementSuggestion] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    'AgentRole',
    'EvaluationResult',
    'ImprovementType',
    'GoldStandard',
    'Evaluation',
    'ImprovementSuggestion',
    'AgentConfig',
    'ExecutionTrace',
    'SwarmConfig',
    'SwarmResult',
]
