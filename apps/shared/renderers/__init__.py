"""
Platform-Specific Renderers
============================

Implementations of abstract interfaces for each platform.

Available renderers:
- TerminalRenderer: Rich terminal UI (for CLI/TUI)
- TelegramRenderer: Telegram MarkdownV2 formatting
- WebRenderer: React components (for PWA/Tauri)
"""

from .telegram_renderer import TelegramMessageRenderer, TelegramStatusRenderer
from .terminal import TerminalInputHandler, TerminalMessageRenderer, TerminalStatusRenderer

# Note: web.tsx is TypeScript/React, not Python - see apps/shared/renderers/web.tsx

__all__ = [
    # Terminal
    "TerminalMessageRenderer",
    "TerminalStatusRenderer",
    "TerminalInputHandler",
    # Telegram
    "TelegramMessageRenderer",
    "TelegramStatusRenderer",
]
