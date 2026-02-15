"""
Shared UI Components for Jotty
===============================

Platform-agnostic chat UI components that work across:
- Terminal (CLI/TUI)
- Telegram
- WhatsApp
- Web (PWA)
- Tauri (Desktop/Mobile)

Architecture:
- Abstract base classes define interface
- Platform-specific renderers implement rendering
- Unified message model and state management
- Event-driven updates

Usage:
    from apps.shared import ChatInterface, Message, ChatState
    from apps.shared.renderers import TerminalRenderer

    # Create chat interface with terminal renderer
    chat = ChatInterface(renderer=TerminalRenderer())
    await chat.send_message("Hello, Jotty!")
"""

from .events import EventProcessor, EventQueue
from .interface import ChatInterface, InputHandler, MessageRenderer, StatusRenderer
from .models import Attachment, Error, Message, Status
from .state import ChatState, ChatStateMachine

__all__ = [
    # Models
    "Message",
    "Attachment",
    "Status",
    "Error",
    # State
    "ChatState",
    "ChatStateMachine",
    # Interface
    "ChatInterface",
    "MessageRenderer",
    "StatusRenderer",
    "InputHandler",
    # Events
    "EventProcessor",
    "EventQueue",
]
