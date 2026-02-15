"""
Jotty - Tiered Execution System
===================================

Progressive complexity model:
- Tier 1 (DIRECT): Single LLM call - Fast path
- Tier 2 (AGENTIC): Planning + orchestration - Default
- Tier 3 (LEARNING): Memory + validation - Production
- Tier 4 (RESEARCH): Domain swarm execution - Advanced
- Tier 5 (AUTONOMOUS): Sandbox + coalition + full features

All tiers share:
- UnifiedRegistry (skills + UI)
- LLM Provider abstraction
- Basic error handling
"""

from .executor import TierExecutor, ExecutionConfig, ExecutionTier, ExecutionResult
from .types import StreamEvent, StreamEventType
from .tier_detector import TierDetector

__all__ = [
    'TierExecutor',
    'ExecutionConfig',
    'ExecutionTier',
    'ExecutionResult',
    'StreamEvent',
    'StreamEventType',
    'TierDetector',
]
