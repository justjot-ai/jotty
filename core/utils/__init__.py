"""
Utils Layer - Utility Functions & Helpers
=========================================

Utility functions, timeouts, parsers, logging, algorithms.
All imports are lazy to avoid pulling in heavy dependencies (DSPy, etc.) at module load time.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # algorithmic_foundations
    "AgentContribution": ".algorithmic_foundations",
    "Coalition": ".algorithmic_foundations",
    "ShapleyValueEstimator": ".algorithmic_foundations",
    "DifferenceRewardEstimator": ".algorithmic_foundations",
    "AlgorithmicCreditAssigner": ".algorithmic_foundations",
    "InformationWeightedMemory": ".algorithmic_foundations",
    "SurpriseEstimator": ".algorithmic_foundations",
    "InformationTheoreticStorage": ".algorithmic_foundations",
    "MutualInformationRetriever": ".algorithmic_foundations",
    "OverflowDetector": ".algorithmic_foundations",
    "ContextOverflowInfo": ".algorithmic_foundations",
    "GlobalContextGuard": ".algorithmic_foundations",
    "UniversalContextGuard": ".algorithmic_foundations",
    "patch_dspy_with_guard": ".algorithmic_foundations",
    "unpatch_dspy": ".algorithmic_foundations",
    "ContentChunk": ".algorithmic_foundations",
    "ProcessedContent": ".algorithmic_foundations",
    "RelevanceEstimator": ".algorithmic_foundations",
    "ContentGate": ".algorithmic_foundations",
    "ContextAwareDocumentProcessor": ".algorithmic_foundations",
    "with_content_gate": ".algorithmic_foundations",
    "AlgorithmicReVal": ".algorithmic_foundations",
    # context_logger
    "EnhancedLogger": ".context_logger",
    "ContextRequirements": ".context_logger",
    "TokenBudgetManager": ".context_logger",
    "SemanticFilter": ".context_logger",
    # timeouts
    "CircuitState": ".timeouts",
    "CircuitBreakerConfig": ".timeouts",
    "CircuitBreaker": ".timeouts",
    "CircuitOpenError": ".timeouts",
    "TimeoutError": ".timeouts",
    "timeout": ".timeouts",
    "async_timeout": ".timeouts",
    "FailedOperation": ".timeouts",
    "DeadLetterQueue": ".timeouts",
    "AdaptiveTimeout": ".timeouts",
    # trajectory_parser
    "TaggedAttempt": ".trajectory_parser",
    "TrajectoryParser": ".trajectory_parser",
    "create_parser": ".trajectory_parser",
    # tokenizer
    "SmartTokenizer": ".tokenizer",
    "get_tokenizer": ".tokenizer",
    "count_tokens": ".tokenizer",
    "estimate_tokens": ".tokenizer",
    # llm_cache
    "LLMCallCache": ".llm_cache",
    "CachedResponse": ".llm_cache",
    "CacheStats": ".llm_cache",
    "get_cache": ".llm_cache",
    # budget_tracker
    "BudgetTracker": ".budget_tracker",
    "BudgetConfig": ".budget_tracker",
    "BudgetUsage": ".budget_tracker",
    "BudgetScope": ".budget_tracker",
    "BudgetExceededError": ".budget_tracker",
    "get_budget_tracker": ".budget_tracker",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
