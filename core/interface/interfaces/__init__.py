"""
Jotty Interfaces Package
========================

Unified message and interface types for multi-frontend support.
Enables CLI, Telegram, and Web UI to share the same backend.
"""

from .message import JottyMessage, InterfaceType, Attachment, InternalEvent, EventType, MessageAdapter
from .host_provider import HostProvider, Host, CLIHost, NullHost

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
