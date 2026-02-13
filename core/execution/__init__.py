"""
Jotty V3 - Tiered Execution System
===================================

Progressive complexity model:
- Tier 1 (DIRECT): Single LLM call - Fast path
- Tier 2 (AGENTIC): Planning + orchestration - Default
- Tier 3 (LEARNING): Memory + validation - Production
- Tier 4 (RESEARCH): Full V2 features - Advanced

All tiers share:
- UnifiedRegistry (skills + UI)
- LLM Provider abstraction
- Basic error handling

No V2 code is modified - V2 becomes Tier 4.
"""

from .executor import UnifiedExecutor, ExecutionConfig, ExecutionTier, ExecutionResult
from .tier_detector import TierDetector

__all__ = [
    'UnifiedExecutor',
    'ExecutionConfig',
    'ExecutionTier',
    'ExecutionResult',
    'TierDetector',
]
