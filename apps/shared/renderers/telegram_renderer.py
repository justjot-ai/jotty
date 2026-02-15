"""
Telegram Renderer
=================

Telegram MarkdownV2 formatter for bot messages.
"""

import os
import re
import sys
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ..interface import MessageRenderer, StatusRenderer
from ..models import Error, Message, Status


class TelegramMessageRenderer(MessageRenderer):
    """
    Telegram message renderer.

    Converts messages to Telegram MarkdownV2 format and handles:
    - Message splitting (4096 char limit)
    - Special character escaping
    - Code block formatting
    """

    MAX_MESSAGE_LENGTH = 4000  # Leave buffer for formatting

    def __init__(self, send_callback=None):
        """
        Initialize Telegram renderer.

        Args:
            send_callback: Function to send message to Telegram
        """
        self._send = send_callback or print  # Fallback to print for testing

    def render_text(self, text: str) -> Any:
        """Render plain text."""
        escaped = self._escape_markdown(text)
        self._send_message(escaped)

    def render_markdown(self, markdown: str) -> Any:
        """Render markdown content."""
        converted = self._convert_markdown(markdown)
        messages = self._split_message(converted)
        for msg in messages:
            self._send_message(msg)

    def render_code(self, code: str, language: str = "python") -> Any:
        """Render code block."""
        # Telegram code blocks: ```language\ncode\n```
        formatted = f"```{language}\n{code}\n```"
        self._send_message(formatted)

    def render_message(self, message: Message) -> Any:
        """Render complete message."""
        # Get status icon and role
        icon = message.get_status_icon()
        role_emoji = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}
        role_icon = role_emoji.get(message.role, "•")

        # Build message
        parts = []

        # Header
        if icon:
            parts.append(f"{icon} ")
        parts.append(f"{role_icon} *{message.role.capitalize()}*")

        # Progress
        progress_text = message.get_progress_text()
        if progress_text:
            parts.append(f" \\[{progress_text}\\]")

        parts.append("\n\n")

        # Content
        # Handle both enum and string format values
        format_val = message.format.value if hasattr(message.format, "value") else message.format
        if format_val == "markdown":
            content = self._convert_markdown(message.content)
        else:
            content = self._escape_markdown(message.content)

        parts.append(content)

        # Attachments
        if message.attachments:
            parts.append(f"\n\n📎 {len(message.attachments)} attachment(s)")

        full_message = "".join(parts)
        messages = self._split_message(full_message)

        for msg in messages:
            self._send_message(msg)

    def render_message_list(self, messages: List[Message]) -> Any:
        """Render list of messages."""
        for message in messages:
            if not message.hidden:
                self.render_message(message)

    def update_streaming_message(self, message: Message, chunk: str) -> Any:
        """Update streaming message (edit previous message)."""
        # For Telegram, we'd edit the last message
        # This requires message_id tracking
        converted = self._convert_markdown(message.content)
        messages = self._split_message(converted)

        # Send only the last part (or edit if possible)
        if messages:
            self._send_message(messages[-1], edit=True)

    def clear_display(self) -> None:
        """Clear display (not applicable for Telegram)."""
        pass

    def _send_message(self, text: str, edit: bool = False) -> None:
        """Send message via callback."""
        # Check if send callback accepts edit parameter
        import inspect

        sig = inspect.signature(self._send)
        if "edit" in sig.parameters:
            self._send(text, edit=edit)
        else:
            self._send(text)

    def _convert_markdown(self, text: str) -> str:
        """
        Convert standard markdown to Telegram MarkdownV2.

        Escapes special characters outside code blocks.
        """
        parts = []
        last_end = 0

        # Find code blocks (``` ... ```) and inline code (` ... `)
        code_pattern = r"```[\s\S]*?```|`[^`]+`"

        for match in re.finditer(code_pattern, text):
            # Process text before code block
            before = text[last_end : match.start()]
            parts.append(self._escape_markdown(before))

            # Keep code block as-is
            parts.append(match.group())
            last_end = match.end()

        # Process remaining text
        parts.append(self._escape_markdown(text[last_end:]))

        return "".join(parts)

    def _escape_markdown(self, text: str) -> str:
        """
        Escape special characters for Telegram MarkdownV2.

        Special chars: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        # All MarkdownV2 special characters that need escaping
        # Note: Backtick ` is handled separately in code blocks
        special_chars = [
            "_",
            "*",
            "[",
            "]",
            "(",
            ")",
            "~",
            "`",
            ">",
            "#",
            "+",
            "-",
            "=",
            "|",
            "{",
            "}",
            ".",
            "!",
            "\\",
        ]

        # Simple approach: escape all special characters
        result = text
        for char in special_chars:
            result = result.replace(char, "\\" + char)

        return result

    def _split_message(self, text: str) -> List[str]:
        """
        Split message into chunks respecting Telegram limits.

        Tries to split at paragraph boundaries.
        """
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return [text]

        messages = []
        remaining = text

        while remaining:
            if len(remaining) <= self.MAX_MESSAGE_LENGTH:
                messages.append(remaining)
                break

            # Find good split point
            split_at = self.MAX_MESSAGE_LENGTH

            # Try to split at paragraph
            para_break = remaining[:split_at].rfind("\n\n")
            if para_break > self.MAX_MESSAGE_LENGTH // 2:
                split_at = para_break + 2

            # Try to split at line
            elif "\n" in remaining[:split_at]:
                line_break = remaining[:split_at].rfind("\n")
                if line_break > self.MAX_MESSAGE_LENGTH // 2:
                    split_at = line_break + 1

            # Try to split at sentence
            elif ". " in remaining[:split_at]:
                sentence_break = remaining[:split_at].rfind(". ")
                if sentence_break > self.MAX_MESSAGE_LENGTH // 2:
                    split_at = sentence_break + 2

            messages.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        return messages


class TelegramStatusRenderer(StatusRenderer):
    """
    Telegram status renderer.

    Displays status using emojis and text.
    """

    def __init__(self, send_callback=None):
        """
        Initialize status renderer.

        Args:
            send_callback: Function to send status message
        """
        self._send = send_callback or print
        self._last_status_message_id: Optional[int] = None

    def render_status(self, status: Status) -> Any:
        """Render status indicator."""
        emoji_map = {
            "idle": "✅",
            "thinking": "🤔",
            "planning": "📋",
            "executing_skill": "🔧",
            "executing_agent": "🤖",
            "coordinating_swarm": "🐝",
            "streaming": "💬",
            "transcribing": "🎤",
            "synthesizing": "🔊",
            "waiting_input": "⏳",
            "validating": "🔍",
            "learning": "📚",
            "error": "❌",
        }

        icon = emoji_map.get(status.state, "⚙️")
        message = f"{icon} {status.message or status.state.capitalize()}"

        if status.progress is not None:
            message += f" \\[{int(status.progress * 100)}%\\]"

        self._send(message)

    def render_progress(self, progress: float, message: Optional[str] = None) -> Any:
        """Render progress."""
        # Create progress bar using blocks
        bars = int(progress * 10)
        bar = "▓" * bars + "░" * (10 - bars)
        text = f"⏳ {message or 'Progress'}: {bar} {int(progress * 100)}%"
        self._send(text)

    def render_thinking(self, message: Optional[str] = None) -> Any:
        """Render thinking indicator."""
        text = f"🤔 {message or 'Thinking...'}"
        self._send(text)

    def render_error(self, error: Error) -> Any:
        """Render error message."""
        # Escape markdown
        escaped_msg = TelegramMessageRenderer()._escape_markdown(error.message)
        text = f"❌ *Error*: {escaped_msg}"

        if not error.recoverable:
            text += "\n\n⚠️ _This error is not recoverable_"

        self._send(text)

    def update_status(self, status: Status) -> None:
        """Update existing status."""
        self.render_status(status)

    def clear_status(self) -> None:
        """Clear status (not applicable for Telegram)."""
        pass
