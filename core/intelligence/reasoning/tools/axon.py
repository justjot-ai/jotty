"""
Re-export shim — canonical location is now:
    Jotty.core.intelligence.orchestration.communication.axon
"""

from Jotty.core.intelligence.orchestration.communication.axon import (
    AgentCapabilities,
    FormatRegistry,
    Message,
    MessageBus,
    SmartAgentSlack,
)

__all__ = [
    "SmartAgentSlack",
    "MessageBus",
    "FormatRegistry",
    "AgentCapabilities",
    "Message",
]
