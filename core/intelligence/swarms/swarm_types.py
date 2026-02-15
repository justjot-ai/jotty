"""
Swarm Types and Data Classes
==============================

Core enums and data classes used throughout the swarm infrastructure:
- AgentRole, EvaluationResult, ImprovementType (enums)
- GoldStandard, Evaluation, ImprovementSuggestion (evaluation types)
- SwarmAgentConfig, ExecutionTrace (agent types)
- SwarmBaseConfig, SwarmResult (swarm types)

Extracted from base_swarm.py for modularity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class AgentRole(Enum):
    """Roles in the self-improving loop."""

    EXPERT = "expert"  # Evaluates against gold standard
    REVIEWER = "reviewer"  # Reviews agent performance
    PLANNER = "planner"  # Plans task execution
    ACTOR = "actor"  # Executes tasks
    ORCHESTRATOR = "orchestrator"  # Coordinates agents
    AUDITOR = "auditor"  # Verifies outputs (Byzantine + quality audit)
    LEARNER = "learner"  # Extracts gold standards from successful runs


class EvaluationResult(Enum):
    """Evaluation outcomes."""

    EXCELLENT = "excellent"  # Exceeds gold standard
    GOOD = "good"  # Meets gold standard
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
class SwarmAgentConfig:
    """Configuration for an agent.

    Sentinel defaults (0 / 0.0 / '') resolved from config_defaults in __post_init__.
    """

    role: AgentRole
    name: str
    model: str = ""  # "" → DEFAULT_MODEL_ALIAS
    temperature: float = 0.0  # 0.0 → LLM_TEMPERATURE
    max_tokens: int = 0  # 0 → LLM_MAX_OUTPUT_TOKENS
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        from Jotty.core.infrastructure.foundation.config_defaults import DEFAULTS

        if not self.model:
            self.model = DEFAULTS.DEFAULT_MODEL_ALIAS
        if self.temperature == 0.0:
            self.temperature = DEFAULTS.LLM_TEMPERATURE
        if self.max_tokens <= 0:
            self.max_tokens = DEFAULTS.LLM_MAX_OUTPUT_TOKENS


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
    """Base configuration for all swarms.

    All swarm-specific configs should inherit from this class.

    ⚠️ RENAMED: Previously called 'SwarmBaseConfig' (now deprecated).
    This is the main swarm configuration with basic metadata (name, domain, version, etc.).

    For learning/RL configuration, use 'SwarmLearningConfig' from foundation.data_structures.

    If you see import errors, use: from ..swarm_types import SwarmConfig
    """

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

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate max_retries
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}\n"
                f"💡 Suggestion: Set max_retries to 3 (recommended) or 0 to disable retries"
            )

        # Validate timeout
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}\n"
                f"💡 Suggestion: Set timeout_seconds to 300 (5 minutes) or adjust based on task complexity"
            )

        # Validate improvement_threshold
        if not 0.0 <= self.improvement_threshold <= 1.0:
            raise ValueError(
                f"improvement_threshold must be between 0.0 and 1.0, got {self.improvement_threshold}\n"
                f"💡 Suggestion: Use 0.7 (70% quality threshold)"
            )

        # Ensure output directory exists
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)


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


# =============================================================================
# DEFENSIVE UTILITIES — safe LLM output parsing
# =============================================================================


def _split_field(value: Any, sep: Any = "|") -> List:
    """Safely split a DSPy output field into a list of strings.
    Handles: str (pipe-split), list (coerce items to str), dict (flatten), None.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [
            item.get("name", str(item)) if isinstance(item, dict) else str(item).strip()
            for item in value
            if item is not None
        ]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items() if v]
    return [s.strip() for s in str(value).split(sep) if s.strip()]


def _safe_join(items: Any, sep: Any = ", ") -> str:
    """Safely join items into a string, coercing non-string elements."""
    if not items:
        return ""
    if isinstance(items, str):
        return items
    return sep.join(str(item) for item in items)


def _safe_num(value: Any, default: Any = 0) -> Any:
    """Extract a number from LLM output. Returns default for non-numeric."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    if isinstance(value, dict):
        return default
    return default


# Deprecated alias
SwarmBaseConfig = SwarmConfig


__all__ = [
    "AgentRole",
    "EvaluationResult",
    "ImprovementType",
    "GoldStandard",
    "Evaluation",
    "ImprovementSuggestion",
    "SwarmAgentConfig",
    "ExecutionTrace",
    "SwarmConfig",
    "SwarmBaseConfig",  # Deprecated alias for SwarmConfig
    "SwarmResult",
    "_split_field",
    "_safe_join",
    "_safe_num",
]


# =============================================================================
# BACKWARD COMPATIBILITY WITH HELPFUL ERROR
# =============================================================================


def __getattr__(name: str) -> Any:
    """Intercept attempts to import deprecated names and provide helpful errors."""
    if name == "SwarmBaseConfig":
        import warnings

        warnings.warn(
            "\n" + "=" * 80 + "\n"
            "⚠️  DEPRECATED: 'SwarmBaseConfig' has been renamed to 'SwarmConfig'\n\n"
            "Fix your code:\n"
            "  ❌ from ..swarm_types import SwarmBaseConfig\n"
            "  ✅ from ..swarm_types import SwarmConfig\n\n"
            "  ❌ class MyConfig(SwarmBaseConfig):\n"
            "  ✅ class MyConfig(SwarmConfig):\n\n"
            "See: Jotty/CLAUDE.md - Legacy Imports section\n" + "=" * 80,
            DeprecationWarning,
            stacklevel=2,
        )
        # Return the correct class (backward compatible)
        return SwarmConfig

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
