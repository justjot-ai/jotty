"""
Optimization Module

Provides optimizations for LLM calls and context management.
Based on OAgents efficiency principles.
"""

from .context_compressor import CompressionResult, ContextCompressor, ContextManager
from .prompt_optimizer import LLMCache, OptimizationResult, PromptOptimizer

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "LLMCache",
    "ContextCompressor",
    "CompressionResult",
    "ContextManager",
]
