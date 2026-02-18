"""
Communication — inter-agent message bus and smart data routing.

Canonical home for SmartAgentSlack (intelligent agent communication layer).
Moved from reasoning/tools/axon.py (swarm communication, not agent tooling).
"""

from .axon import AgentCapabilities, FormatRegistry, Message, MessageBus, SmartAgentSlack

__all__ = [
    "SmartAgentSlack",
    "MessageBus",
    "FormatRegistry",
    "AgentCapabilities",
    "Message",
]
