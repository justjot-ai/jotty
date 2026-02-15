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

from .executor import ExecutionConfig, ExecutionResult, ExecutionTier, TierExecutor
from .tier_detector import TierDetector
from .types import StreamEvent, StreamEventType

__all__ = [
    "TierExecutor",
    "ExecutionConfig",
    "ExecutionTier",
    "ExecutionResult",
    "StreamEvent",
    "StreamEventType",
    "TierDetector",
]
