"""
Optimization Module

Provides optimizations for LLM calls and context management.
Based on OAgents efficiency principles.

Note: Context compression has been consolidated into
``core.infrastructure.context``. The old ``context_compressor`` module
has been removed.
"""

from .prompt_optimizer import LLMCache, OptimizationResult, PromptOptimizer

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "LLMCache",
]
