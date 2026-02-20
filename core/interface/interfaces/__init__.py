"""
Jotty Interfaces Package
========================

Unified message and interface types for all channels:
CLI, Web, Telegram, WhatsApp, Slack, Discord, Signal,
X/Twitter, LinkedIn, Mastodon, Bluesky, Reddit, Matrix,
Teams, Google Chat, iMessage, and custom channels.

InterfaceType is an alias for ChannelType (single source of truth).
"""

from .host_provider import CLIHost, Host, HostProvider, NullHost
from .message import (
    Attachment,
    EventType,
    InterfaceType,
    InternalEvent,
    JottyMessage,
    MessageAdapter,
    SessionKey,
)

__all__ = [
    "JottyMessage",
    "InterfaceType",
    "SessionKey",
    "Attachment",
    "InternalEvent",
    "EventType",
    "MessageAdapter",
    "HostProvider",
    "Host",
    "CLIHost",
    "NullHost",
]
