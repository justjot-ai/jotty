"""
LOTUS-Inspired Optimization Layer for Jotty v2
===============================================

Semantic Operators and Query Optimization for LLM-based Data Processing.

Inspired by: https://github.com/lotus-data/lotus
Paper: "Semantic Operators: Declarative AI-based Transformations" (arXiv:2407.11418)

Key Components (DRY: Single responsibility, composable):
- ModelCascade: Tiered model selection (Haiku→Sonnet→Opus) based on confidence
- SemanticCache: Memoize semantic operations with content fingerprinting
- BatchExecutor: Batch LLM calls for throughput optimization
- AdaptiveValidator: Learn when to skip validation based on history
- SemanticOperators: Declarative sem_filter, sem_map, sem_join operations

Cost Reduction Strategy:
- 70% queries resolved by Haiku (60x cheaper than Opus)
- 25% queries resolved by Sonnet (5x cheaper than Opus)
- 5% queries fall through to Opus (complex cases only)
- Result: ~10x cost reduction with minimal quality loss

Usage:
    from Jotty.core.infrastructure.integration.lotus import (
        ModelCascade,
        SemanticCache,
        BatchExecutor,
        AdaptiveValidator,
        LotusConfig,
    )

    # Initialize with config
    config = LotusConfig()
    cascade = ModelCascade(config)
    cache = SemanticCache(config)

    # Execute with optimizations
    result = await cascade.execute(operation, items)
"""

from .adaptive_validator import AdaptiveValidator, ValidationDecision
from .batch_executor import BatchConfig, BatchExecutor, BatchResult
from .config import CacheConfig, CascadeThresholds, LotusConfig
from .model_cascade import CascadeResult, ModelCascade, ModelTier
from .optimizer import LotusOptimizer
from .semantic_cache import CacheEntry, CacheStats, SemanticCache
from .semantic_operators import (
    SemanticDataFrame,
    SemanticOperator,
    SemExtract,
    SemFilter,
    SemMap,
    SemTopK,
)

__all__ = [
    # Config
    "LotusConfig",
    "CascadeThresholds",
    "CacheConfig",
    # Model Cascade
    "ModelCascade",
    "CascadeResult",
    "ModelTier",
    # Semantic Cache
    "SemanticCache",
    "CacheEntry",
    "CacheStats",
    # Batch Executor
    "BatchExecutor",
    "BatchResult",
    "BatchConfig",
    # Adaptive Validator
    "AdaptiveValidator",
    "ValidationDecision",
    # Semantic Operators
    "SemanticOperator",
    "SemFilter",
    "SemMap",
    "SemExtract",
    "SemTopK",
    "SemanticDataFrame",
    # Unified Optimizer
    "LotusOptimizer",
]
