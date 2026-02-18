"""
Re-export shim — canonical location is now:
    Jotty.core.intelligence.orchestration.intent.intent_parser
"""

from Jotty.core.intelligence.orchestration.intent.intent_parser import (
    IntentParser,
    TaskGraph,
)

__all__ = ["IntentParser", "TaskGraph"]
