"""
V3 Execution Types
==================

Core types for the tiered execution system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import IntEnum
from datetime import datetime


class ExecutionTier(IntEnum):
    """Execution tiers - progressive complexity."""
    DIRECT = 1      # Single LLM call with tools
    AGENTIC = 2     # Planning + multi-step orchestration
    LEARNING = 3    # Memory + validation
    RESEARCH = 4    # Domain swarm execution
    AUTONOMOUS = 5  # Sandbox, coalition, curriculum, full V2 features


@dataclass
class ExecutionConfig:
    """
    Configuration for tiered execution.

    Each tier has its own config section. Only relevant fields
    are used based on the selected tier.
    """
    # Core
    tier: Optional[ExecutionTier] = None  # None = auto-detect

    # Tier 1 (DIRECT) - no config needed

    # Tier 2 (AGENTIC)
    max_planning_depth: int = 3
    enable_parallel_execution: bool = True
    max_concurrent_steps: int = 3

    # Tier 3 (LEARNING)
    memory_backend: str = "json"  # json | redis | postgres
    memory_ttl_hours: int = 24
    enable_validation: bool = True
    validation_retries: int = 1
    track_success_rate: bool = True

    # Tier 4 (RESEARCH) - domain swarm execution
    enable_td_lambda: bool = False
    enable_hierarchical_memory: bool = False
    enable_multi_round_validation: bool = False
    memory_levels: int = 2  # episodic + semantic only
    enable_swarm_intelligence: bool = False

    # Tier 4/5 options
    swarm_name: Optional[str] = None        # e.g. "coding", "research", "testing"
    paradigm: Optional[str] = None          # "relay", "debate", "refinement"
    enable_sandbox: bool = False            # Tier 5: sandbox execution
    enable_coalition: bool = False          # Tier 5: coalition formation
    trust_level: str = "standard"           # Tier 5: "standard", "elevated", "restricted"

    # Shared across all tiers
    timeout_seconds: int = 300
    max_retries: int = 3
    provider: Optional[str] = None  # None = auto-detect
    model: Optional[str] = None  # None = use provider default
    temperature: float = 0.7
    max_tokens: int = 4000

    def to_v2_config(self) -> Dict[str, Any]:
        """Convert V3 config to V2 JottyConfig format."""
        return {
            'enable_learning': self.enable_td_lambda,
            'enable_memory': self.enable_hierarchical_memory,
            'enable_validation': self.enable_multi_round_validation,
            'timeout_seconds': self.timeout_seconds,
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


@dataclass
class ExecutionStep:
    """A single step in multi-step execution."""
    step_num: int
    description: str
    skill: Optional[str] = None
    depends_on: List[int] = field(default_factory=list)
    can_parallelize: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None

    @property
    def is_complete(self) -> bool:
        return self.result is not None or self.error is not None


@dataclass
class ExecutionPlan:
    """Multi-step execution plan."""
    goal: str
    steps: List[ExecutionStep]
    estimated_cost: float = 0.0
    estimated_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def parallelizable_steps(self) -> int:
        return sum(1 for s in self.steps if s.can_parallelize)


@dataclass
class ValidationResult:
    """Result of output validation."""
    success: bool
    confidence: float  # 0-1
    feedback: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryContext:
    """Retrieved memory context for task."""
    entries: List[Dict[str, Any]]
    relevance_scores: List[float]
    total_retrieved: int
    retrieval_time_ms: float


@dataclass
class ExecutionResult:
    """
    Result of task execution at any tier.

    Contains tier-specific metadata but common interface.
    """
    # Core fields (all tiers)
    output: Any
    tier: ExecutionTier
    success: bool = True
    error: Optional[str] = None

    # Performance metrics
    llm_calls: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Tier 2+ (AGENTIC)
    plan: Optional[ExecutionPlan] = None
    steps: List[ExecutionStep] = field(default_factory=list)

    # Tier 3+ (LEARNING)
    validation: Optional[ValidationResult] = None
    used_memory: bool = False
    memory_context: Optional[MemoryContext] = None
    success_rate: Optional[float] = None  # Historical success rate for similar tasks

    # Tier 4/5 - swarm and autonomous data
    v2_episode: Optional[Any] = None  # EpisodeResult from V2
    learning_data: Dict[str, Any] = field(default_factory=dict)
    swarm_name: Optional[str] = None
    paradigm_used: Optional[str] = None
    sandbox_log: Optional[str] = None

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Any] = None  # Trace from observability.tracing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'output': str(self.output),
            'tier': self.tier.name,
            'success': self.success,
            'error': self.error,
            'llm_calls': self.llm_calls,
            'latency_ms': self.latency_ms,
            'cost_usd': self.cost_usd,
            'used_memory': self.used_memory,
            'steps': len(self.steps),
            'trace_id': self.trace.trace_id if self.trace else None,
            'timestamp': self.started_at.isoformat(),
        }

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] Tier {self.tier.value} | {self.llm_calls} calls | {self.latency_ms:.0f}ms | ${self.cost_usd:.4f}"
