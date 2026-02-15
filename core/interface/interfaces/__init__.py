"""
Jotty Interfaces Package
========================

Unified message and interface types for multi-frontend support.
Enables CLI, Telegram, and Web UI to share the same backend.
"""

from .host_provider import CLIHost, Host, HostProvider, NullHost
from .message import (
    Attachment,
    EventType,
    InterfaceType,
    InternalEvent,
    JottyMessage,
    MessageAdapter,
)

__all__ = [
    "JottyMessage",
    "InterfaceType",
    "Attachment",
    "InternalEvent",
    "EventType",
    "MessageAdapter",
    "HostProvider",
    "Host",
    "CLIHost",
    "NullHost",
]
