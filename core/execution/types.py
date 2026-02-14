"""
Execution Types
===============

Core types for the tiered execution system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import IntEnum, Enum
from datetime import datetime
from collections import deque
import threading
import time
import logging

logger = logging.getLogger(__name__)


class ExecutionTier(IntEnum):
    """Execution tiers - progressive complexity."""
    DIRECT = 1      # Single LLM call with tools
    AGENTIC = 2     # Planning + multi-step orchestration
    LEARNING = 3    # Memory + validation
    RESEARCH = 4    # Domain swarm execution
    AUTONOMOUS = 5  # Sandbox, coalition, curriculum, full features


class StreamEventType(Enum):
    """Types of events emitted during streaming execution."""
    STATUS = "status"            # Phase change (planning, executing, validating...)
    STEP_COMPLETE = "step_complete"  # An execution step finished
    PARTIAL_OUTPUT = "partial_output"  # Partial result from a step
    TOKEN = "token"              # Individual token from LLM (Tier 1 streaming)
    RESULT = "result"            # Final ExecutionResult
    ERROR = "error"              # Execution error


@dataclass
class StreamEvent:
    """A single event emitted during streaming execution."""
    type: StreamEventType
    data: Any
    tier: Optional['ExecutionTier'] = None
    timestamp: datetime = field(default_factory=datetime.now)


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

    def to_swarm_config(self) -> Dict[str, Any]:
        """Convert execution config to SwarmConfig-compatible dict.

        Only includes fields that exist on SwarmConfig.
        Provider/model/temperature are NOT SwarmConfig fields — they're
        set via DSPy LM configuration or passed directly to LLMProvider.
        """
        return {
            'enable_validation': self.enable_validation,
            'enable_multi_round': self.enable_multi_round_validation,
            'enable_rl': self.enable_td_lambda,
            'enable_causal_learning': self.enable_td_lambda,
            'enable_agent_communication': True,
            'llm_timeout_seconds': self.timeout_seconds,
            'max_eval_retries': self.validation_retries,
            'validation_mode': 'full' if self.enable_multi_round_validation else 'none',
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
class TierValidationResult:
    """Result of output validation."""
    success: bool
    confidence: float  # 0-1
    feedback: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

# Backward-compat alias
ValidationResult = TierValidationResult


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
    validation: Optional[TierValidationResult] = None
    used_memory: bool = False
    memory_context: Optional[MemoryContext] = None
    success_rate: Optional[float] = None  # Historical success rate for similar tasks

    # Tier 4/5 - swarm and autonomous data
    episode: Optional[Any] = None  # EpisodeResult from swarm execution
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
        status = "OK" if self.success else "FAIL"
        return f"[{status}] Tier {self.tier.value} | {self.llm_calls} calls | {self.latency_ms:.0f}ms | ${self.cost_usd:.4f}"


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

class ErrorType(Enum):
    """Classifies the source/nature of an error for recovery decisions.

    Enables proper recovery strategy:
    - INFRASTRUCTURE → retry with backoff
    - LOGIC → fix the approach (wrong selector, bad syntax)
    - DATA → investigate data source (empty results, malformed)
    - ENVIRONMENT → detect and adapt (proxy, SSL, firewall)
    """
    NONE = "none"
    INFRASTRUCTURE = "infrastructure"
    LOGIC = "logic"
    DATA = "data"
    ENVIRONMENT = "environment"

    @classmethod
    def classify(cls, error_msg: str) -> "ErrorType":
        """Auto-classify an error message into an ErrorType.

        Order matters: more specific patterns checked first to avoid
        ambiguous matches (e.g. 'element not found' is LOGIC, not DATA).
        """
        lower = error_msg.lower()

        # ENVIRONMENT: very specific indicators — check first
        env_keywords = ['ssl', 'certificate', 'proxy', 'firewall', 'zscaler',
                        'bluecoat', 'tls', 'handshake']
        if any(kw in lower for kw in env_keywords):
            return cls.ENVIRONMENT

        # LOGIC: check before DATA since 'element not found' is logic
        logic_keywords = ['selector', 'element not found', 'syntax', 'type error',
                          'attribute error', 'missing required', 'invalid argument',
                          'missing parameter']
        if any(kw in lower for kw in logic_keywords):
            return cls.LOGIC

        # INFRASTRUCTURE: network/service issues
        infra_keywords = ['timeout', 'connection', 'rate limit', 'server error',
                          '503', '502', '504', '429', 'unavailable', 'network']
        if any(kw in lower for kw in infra_keywords):
            return cls.INFRASTRUCTURE

        # DATA: content/result issues
        data_keywords = ['empty result', 'no results', 'malformed',
                         'parse error', 'invalid json', 'decode']
        if any(kw in lower for kw in data_keywords):
            return cls.DATA

        return cls.INFRASTRUCTURE  # Default: treat as infra (retryable)


class ValidationStatus(Enum):
    """Outcome of a validation or completion check."""
    PASS = "pass"
    FAIL = "fail"
    EXTERNAL_ERROR = "external_error"
    ENQUIRY = "enquiry"


@dataclass
class ValidationVerdict:
    """Structured validation result with error classification and recovery hints.

    Used by Inspector completion review and step-level validation to provide
    actionable information for retry/recovery decisions.
    """
    status: ValidationStatus
    error_type: ErrorType = ErrorType.NONE
    reason: str = ""
    issues: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    retryable: bool = False

    @property
    def is_pass(self) -> bool:
        return self.status == ValidationStatus.PASS

    @classmethod
    def ok(cls, reason: str = "", confidence: float = 1.0) -> "ValidationVerdict":
        """Convenience constructor for a passing verdict."""
        return cls(status=ValidationStatus.PASS, reason=reason,
                   confidence=confidence, error_type=ErrorType.NONE)

    @classmethod
    def from_error(cls, error_msg: str) -> "ValidationVerdict":
        """Construct a failing verdict from an error message with auto-classification."""
        error_type = ErrorType.classify(error_msg)
        retryable = error_type in (ErrorType.INFRASTRUCTURE, ErrorType.ENVIRONMENT)
        return cls(
            status=ValidationStatus.FAIL,
            error_type=error_type,
            reason=error_msg,
            issues=[error_msg],
            retryable=retryable,
        )


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation — calls pass through
    OPEN = "open"            # Failing — calls rejected immediately
    HALF_OPEN = "half_open"  # Testing — one probe call allowed


class CircuitBreaker:
    """Thread-safe circuit breaker preventing cascading failures.

    State machine: CLOSED → (failures >= threshold) → OPEN → (cooldown expires) →
    HALF_OPEN → (success) → CLOSED  |  (failure) → OPEN

    Usage:
        breaker = CircuitBreaker("llm", failure_threshold=5, cooldown_seconds=60)
        if breaker.allow_request():
            try:
                result = call_llm(...)
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(self, name: str, failure_threshold: int = 5, cooldown_seconds: float = 60.0) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            return True  # Allow one probe
        return False  # OPEN — reject

    def record_success(self) -> None:
        """Record a successful call — resets breaker to CLOSED."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call — may trip breaker to OPEN."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN after "
                    f"{self._failure_count} failures"
                )

    def reset(self) -> None:
        """Manually reset to CLOSED."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED


# Global circuit breakers — shared across all agents
LLM_CIRCUIT_BREAKER = CircuitBreaker("llm", failure_threshold=5, cooldown_seconds=60)
TOOL_CIRCUIT_BREAKER = CircuitBreaker("tool", failure_threshold=3, cooldown_seconds=30)


# =============================================================================
# ADAPTIVE TIMEOUT
# =============================================================================

class AdaptiveTimeout:
    """Track observed latencies and compute adaptive timeouts per operation.

    Uses P95 of last N observations with configurable floor/ceiling.
    Falls back to default_seconds when no observations exist.

    Usage:
        timeout = AdaptiveTimeout()
        t = timeout.get("llm_call")       # Returns adaptive or default
        timeout.record("llm_call", 2.3)   # Record observed latency
    """

    def __init__(self, default_seconds: float = 30.0, min_seconds: float = 5.0, max_seconds: float = 300.0, window_size: int = 50) -> None:
        self.default_seconds = default_seconds
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.window_size = window_size
        self._observations: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, operation: str, latency_seconds: float) -> None:
        """Record an observed latency for an operation."""
        with self._lock:
            if operation not in self._observations:
                self._observations[operation] = deque(maxlen=self.window_size)
            self._observations[operation].append(latency_seconds)

    def get(self, operation: str, multiplier: float = 2.0) -> float:
        """Get adaptive timeout = P95 * multiplier, bounded by [min, max].

        Args:
            operation: Operation name (e.g. "llm_call", "web_search")
            multiplier: Safety multiplier applied to P95 (default 2x)
        """
        with self._lock:
            obs = self._observations.get(operation)
            if not obs or len(obs) < 3:
                return self.default_seconds
            sorted_obs = sorted(obs)
            p95_idx = int(len(sorted_obs) * 0.95)
            p95 = sorted_obs[min(p95_idx, len(sorted_obs) - 1)]
        timeout = p95 * multiplier
        return max(self.min_seconds, min(self.max_seconds, timeout))


# Global adaptive timeout tracker
ADAPTIVE_TIMEOUT = AdaptiveTimeout()


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

@dataclass
class DeadLetter:
    """A failed operation stored for later retry."""
    operation: str
    args: Dict[str, Any]
    error: str
    error_type: ErrorType
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3


class DeadLetterQueue:
    """Thread-safe queue for failed operations that can be retried later.

    Usage:
        dlq = DeadLetterQueue()
        dlq.enqueue("web_search", {"query": "..."}, "timeout", ErrorType.INFRASTRUCTURE)
        retryable = dlq.get_retryable()  # Get items eligible for retry
        dlq.mark_resolved(letter)         # Remove after successful retry
    """

    def __init__(self, max_size: int = 100) -> None:
        self._queue: List[DeadLetter] = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def enqueue(self, operation: str, args: Dict[str, Any], error: str,
                error_type: ErrorType = ErrorType.INFRASTRUCTURE) -> DeadLetter:
        """Add a failed operation to the queue."""
        letter = DeadLetter(
            operation=operation, args=args, error=error, error_type=error_type
        )
        with self._lock:
            if len(self._queue) >= self._max_size:
                self._queue.pop(0)  # Drop oldest
            self._queue.append(letter)
        logger.debug(f"DLQ: enqueued {operation} ({error_type.value}): {error[:80]}")
        return letter

    def get_retryable(self) -> List[DeadLetter]:
        """Get all items eligible for retry (retry_count < max_retries)."""
        with self._lock:
            return [l for l in self._queue if l.retry_count < l.max_retries]

    def mark_resolved(self, letter: DeadLetter) -> None:
        """Remove a successfully retried item."""
        with self._lock:
            if letter in self._queue:
                self._queue.remove(letter)

    def retry_all(self, executor_fn: Callable[[DeadLetter], bool]) -> int:
        """Attempt to retry all retryable items. Returns count of successes."""
        retryable = self.get_retryable()
        successes = 0
        for letter in retryable:
            letter.retry_count += 1
            try:
                if executor_fn(letter):
                    self.mark_resolved(letter)
                    successes += 1
            except Exception:
                pass  # Will be retried next cycle
        return successes

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


# Global DLQ
DEAD_LETTER_QUEUE = DeadLetterQueue()


# =============================================================================
# TIMEOUT WARNING
# =============================================================================

class TimeoutWarning:
    """Track elapsed time and generate warnings at threshold percentages.

    Agents inject these warnings into LLM context so models can wrap up
    gracefully instead of being abruptly cut off.

    Usage:
        tw = TimeoutWarning(timeout_seconds=120)
        tw.start()
        ...
        warning = tw.check()  # Returns warning string or None
    """

    THRESHOLDS = [
        (0.80, "80% of time budget used. Start wrapping up your current work."),
        (0.95, "95% of time budget used. Finalize output immediately."),
    ]

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._start_time: Optional[float] = None
        self._triggered: set = set()

    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.time()
        self._triggered.clear()

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        return max(0.0, self.timeout_seconds - self.elapsed)

    @property
    def fraction_used(self) -> float:
        if self.timeout_seconds <= 0:
            return 1.0
        return min(1.0, self.elapsed / self.timeout_seconds)

    def check(self) -> Optional[str]:
        """Check if any warning threshold has been crossed. Returns message or None."""
        if self._start_time is None:
            return None
        frac = self.fraction_used
        for threshold, message in self.THRESHOLDS:
            if frac >= threshold and threshold not in self._triggered:
                self._triggered.add(threshold)
                remaining = self.remaining
                return f"[TIMEOUT WARNING] {message} ({remaining:.0f}s remaining)"
        return None

    @property
    def is_expired(self) -> bool:
        return self.elapsed >= self.timeout_seconds
