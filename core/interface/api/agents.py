"""
Agent Facades for SDK
======================

Provides agent classes to SDK via interface layer.
SDK should import from here, not directly from core.modes.

Architecture:
    SDK → core/interface/api/agents.py → core/modes/agent/

This ensures:
- SDK respects layer boundaries
- Core can refactor agents without breaking SDK
- Clean separation of concerns
"""

from Jotty.core.modes.agent.agents.chat_assistant import ChatAssistant, create_chat_assistant
from Jotty.core.modes.agent.agents.auto_agent import AutoAgent

__all__ = [
    'ChatAssistant',
    'create_chat_assistant',
    'AutoAgent',
]
