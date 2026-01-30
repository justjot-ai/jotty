"""
Jotty Interfaces Package
========================

Unified message and interface types for multi-frontend support.
Enables CLI, Telegram, and Web UI to share the same backend.
"""

from .message import JottyMessage, InterfaceType, Attachment

__all__ = [
    "JottyMessage",
    "InterfaceType",
    "Attachment",
]
