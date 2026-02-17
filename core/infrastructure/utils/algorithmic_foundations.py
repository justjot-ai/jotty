"""
Algorithmic Foundations — Re-export Shim
=========================================

Canonical location: ``core.intelligence.learning.algorithmic_foundations``
This module re-exports for backward compatibility.
"""

from Jotty.core.intelligence.learning.algorithmic_foundations import (
    AgentContribution,
    AlgorithmicCreditAssigner,
    AlgorithmicReVal,
    Coalition,
    ContentChunk,
    ContentGate,
    ContextOverflowInfo,
    DifferenceRewardEstimator,
    GlobalContextGuard,
    InformationTheoreticStorage,
    InformationWeightedMemory,
    MutualInformationRetriever,
    OverflowDetector,
    ProcessedContent,
    RelevanceEstimator,
    ShapleyValueEstimator,
    SortingAlgorithms,
    SurpriseEstimator,
    patch_dspy_with_guard,
    unpatch_dspy,
    with_content_gate,
)

__all__ = [
    "AgentContribution",
    "Coalition",
    "ShapleyValueEstimator",
    "DifferenceRewardEstimator",
    "AlgorithmicCreditAssigner",
    "InformationWeightedMemory",
    "SurpriseEstimator",
    "InformationTheoreticStorage",
    "MutualInformationRetriever",
    "OverflowDetector",
    "ContextOverflowInfo",
    "GlobalContextGuard",
    "patch_dspy_with_guard",
    "unpatch_dspy",
    "ContentChunk",
    "ProcessedContent",
    "RelevanceEstimator",
    "ContentGate",
    "with_content_gate",
    "SortingAlgorithms",
    "AlgorithmicReVal",
]
