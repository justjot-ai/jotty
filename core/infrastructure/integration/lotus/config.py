"""
LOTUS Configuration - Single Source of Truth for Optimization Settings

DRY Principle: All optimization parameters centralized here.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class ModelTier(Enum):
    """Model tiers for cascade selection."""

    FAST = auto()  # Haiku-class: fastest, cheapest
    BALANCED = auto()  # Sonnet-class: balanced cost/quality
    POWERFUL = auto()  # Opus-class: highest quality, most expensive


@dataclass
class CascadeThresholds:
    """
    Thresholds for model cascade decisions.

    LOTUS insight: Items with proxy confidence > tau_pos → accept proxy result
                   Items with proxy confidence < tau_neg → reject by proxy
                   Items in between → route to oracle (more expensive model)
    """

    tau_pos: float = 0.85  # Positive threshold (accept proxy if confidence >= this)
    tau_neg: float = 0.15  # Negative threshold (reject by proxy if confidence <= this)

    # Statistical guarantees
    recall_target: float = 0.95  # Target recall rate
    precision_target: float = 0.90  # Target precision rate
    failure_probability: float = 0.05  # Statistical confidence margin

    def validate(self) -> bool:
        """Validate thresholds are sensible."""
        return (
            0 <= self.tau_neg < self.tau_pos <= 1
            and 0 < self.recall_target <= 1
            and 0 < self.precision_target <= 1
        )


@dataclass
class CacheConfig:
    """Semantic cache configuration."""

    enabled: bool = True
    max_entries: int = 10000
    ttl_seconds: int = 3600  # 1 hour default TTL

    # Semantic similarity threshold for cache hits
    similarity_threshold: float = 0.95

    # Use embeddings for semantic matching (vs exact hash)
    use_semantic_matching: bool = True

    # Cache storage backend
    backend: str = "memory"  # "memory", "redis", "sqlite"

    # Persistence
    persist_to_disk: bool = False
    persistence_path: Optional[str] = None


@dataclass
class BatchConfig:
    """Batch execution configuration."""

    enabled: bool = True
    max_batch_size: int = 20
    max_wait_ms: int = 100  # Max wait time before flushing partial batch

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000

    # Parallel execution
    max_parallel_batches: int = 5


@dataclass
class LotusConfig:
    """
    Master configuration for LOTUS optimization layer.

    DRY: Single source of truth for all optimization settings.
    """

    # Model Configuration (reuses existing pricing from cost_tracker)
    models: Dict[ModelTier, str] = field(
        default_factory=lambda: {
            ModelTier.FAST: "claude-3-5-haiku-latest",  # $0.25/$1.25 per 1M
            ModelTier.BALANCED: "claude-sonnet-4",  # $3/$15 per 1M
            ModelTier.POWERFUL: "claude-opus-4",  # $15/$75 per 1M
        }
    )

    # Cost per 1M tokens (input, output) - DRY: sync with cost_tracker.py
    model_costs: Dict[str, tuple] = field(
        default_factory=lambda: {
            "claude-3-5-haiku-latest": (0.25, 1.25),
            "claude-sonnet-4": (3.0, 15.0),
            "claude-opus-4": (15.0, 75.0),
        }
    )

    # Cascade configuration per operation type
    cascade_thresholds: Dict[str, CascadeThresholds] = field(
        default_factory=lambda: {
            "filter": CascadeThresholds(tau_pos=0.90, tau_neg=0.10),
            "classify": CascadeThresholds(tau_pos=0.85, tau_neg=0.15),
            "extract": CascadeThresholds(tau_pos=0.80, tau_neg=0.20),
            "map": CascadeThresholds(tau_pos=0.85, tau_neg=0.15),
            "default": CascadeThresholds(),
        }
    )

    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Batch configuration
    batch: BatchConfig = field(default_factory=BatchConfig)

    # Adaptive validation
    validation_skip_threshold: float = 0.95  # Skip if historical success rate >= this
    validation_sample_rate: float = 0.10  # Sample rate for skipped validations

    # Cost awareness
    enable_cost_estimation: bool = True
    max_budget_usd: Optional[float] = None  # Optional budget cap

    # Logging
    log_optimizations: bool = True
    log_cache_hits: bool = False  # Verbose logging

    def get_cascade_thresholds(self, operation: str) -> CascadeThresholds:
        """Get thresholds for an operation type."""
        return self.cascade_thresholds.get(operation, self.cascade_thresholds["default"])

    def get_model(self, tier: ModelTier) -> str:
        """Get model name for a tier."""
        return self.models.get(tier, self.models[ModelTier.BALANCED])

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model call."""
        costs = self.model_costs.get(model, (3.0, 15.0))  # Default to Sonnet
        input_cost = (input_tokens / 1_000_000) * costs[0]
        output_cost = (output_tokens / 1_000_000) * costs[1]
        return input_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "models": {k.name: v for k, v in self.models.items()},
            "cache": {
                "enabled": self.cache.enabled,
                "max_entries": self.cache.max_entries,
                "ttl_seconds": self.cache.ttl_seconds,
            },
            "batch": {
                "enabled": self.batch.enabled,
                "max_batch_size": self.batch.max_batch_size,
            },
            "validation_skip_threshold": self.validation_skip_threshold,
        }
