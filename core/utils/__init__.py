"""
Utils Layer - Utility Functions & Helpers
=========================================

Utility functions, timeouts, parsers, logging, algorithms.

Modules:
--------
- algorithmic_foundations: Core algorithms
- context_logger: Enhanced logging
- timeouts: Timeout management
- trajectory_parser: Parse agent trajectories
"""

from .algorithmic_foundations import (
    AgentContribution,
    Coalition,
    ShapleyValueEstimator,
    DifferenceRewardEstimator,
    AlgorithmicCreditAssigner,
    InformationWeightedMemory,
    SurpriseEstimator,
    InformationTheoreticStorage,
    MutualInformationRetriever,
    OverflowDetector,
    ContextOverflowInfo,
    GlobalContextGuard,
    UniversalContextGuard,
    patch_dspy_with_guard,
    unpatch_dspy,
    ContentChunk,
    ProcessedContent,
    RelevanceEstimator,
    ContentGate,
    ContextAwareDocumentProcessor,
    with_content_gate,
    AlgorithmicReVal,
)
from .context_logger import (
    EnhancedLogger,
    ContextRequirements,
    TokenBudgetManager,
    SemanticFilter,
)
from .timeouts import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    TimeoutError,
    timeout,
    async_timeout,
    FailedOperation,
    DeadLetterQueue,
    AdaptiveTimeout,
)
from .trajectory_parser import (
    TaggedAttempt,
    TrajectoryParser,
    create_parser,
)
from .tokenizer import (
    SmartTokenizer,
    get_tokenizer,
    count_tokens,
    estimate_tokens,
)
from .llm_cache import (
    LLMCallCache,
    CachedResponse,
    CacheStats,
    get_cache,
)
from .budget_tracker import (
    BudgetTracker,
    BudgetConfig,
    BudgetUsage,
    BudgetScope,
    BudgetExceededError,
    get_budget_tracker,
)

__all__ = [
    # algorithmic_foundations
    'AgentContribution',
    'Coalition',
    'ShapleyValueEstimator',
    'DifferenceRewardEstimator',
    'AlgorithmicCreditAssigner',
    'InformationWeightedMemory',
    'SurpriseEstimator',
    'InformationTheoreticStorage',
    'MutualInformationRetriever',
    'OverflowDetector',
    'ContextOverflowInfo',
    'GlobalContextGuard',
    'UniversalContextGuard',
    'patch_dspy_with_guard',
    'unpatch_dspy',
    'ContentChunk',
    'ProcessedContent',
    'RelevanceEstimator',
    'ContentGate',
    'ContextAwareDocumentProcessor',
    'with_content_gate',
    'AlgorithmicReVal',
    # context_logger
    'EnhancedLogger',
    'ContextRequirements',
    'TokenBudgetManager',
    'SemanticFilter',
    # timeouts
    'CircuitState',
    'CircuitBreakerConfig',
    'CircuitBreaker',
    'CircuitOpenError',
    'TimeoutError',
    'timeout',
    'async_timeout',
    'FailedOperation',
    'DeadLetterQueue',
    'AdaptiveTimeout',
    # trajectory_parser
    'TaggedAttempt',
    'TrajectoryParser',
    'create_parser',
    # tokenizer
    'SmartTokenizer',
    'get_tokenizer',
    'count_tokens',
    'estimate_tokens',
    # llm_cache
    'LLMCallCache',
    'CachedResponse',
    'CacheStats',
    'get_cache',
    # budget_tracker
    'BudgetTracker',
    'BudgetConfig',
    'BudgetUsage',
    'BudgetScope',
    'BudgetExceededError',
    'get_budget_tracker',
]
