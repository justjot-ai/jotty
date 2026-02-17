"""
Agent Facades for SDK
======================

Provides agent classes to SDK via interface layer.
SDK should import from here, not directly from core.intelligence.reasoning.

Architecture:
    SDK → core/interface/api/agents.py → core/intelligence/reasoning/agents/

This ensures:
- SDK respects layer boundaries
- Core can refactor agents without breaking SDK
- Clean separation of concerns
"""

from Jotty.core.intelligence.reasoning.agents.auto_agent import AutoAgent  # type: ignore[import]
from Jotty.core.intelligence.reasoning.agents.chat_assistant import (  # type: ignore[import]
    ChatAssistant,
    create_chat_assistant,
)

__all__ = [
    "ChatAssistant",
    "create_chat_assistant",
    "AutoAgent",
]
