"""
Telegram Output Renderer
========================

Formats Jotty output for Telegram display.
Handles markdown conversion, message splitting, and formatting.
"""

import html
import re
from typing import List, Tuple


class TelegramRenderer:
    """
    Renders Jotty output for Telegram.

    Handles:
    - Markdown to Telegram MarkdownV2 conversion
    - Message splitting (Telegram 4096 char limit)
    - Code block formatting
    - Checklist formatting
    """

    MAX_MESSAGE_LENGTH = 4000  # Leave buffer for formatting
    MAX_CAPTION_LENGTH = 1000

    @classmethod
    def render(cls, content: str, format_type: str = "markdown") -> List[str]:
        """
        Render content for Telegram.

        Args:
            content: Raw content from TierExecutor
            format_type: Output format hint

        Returns:
            List of message strings (split if needed)
        """
        if not content:
            return ["No content to display."]

        # Convert to Telegram-friendly format
        formatted = cls._convert_markdown(content)

        # Split into chunks
        return cls._split_message(formatted)

    @classmethod
    def _convert_markdown(cls, text: str) -> str:
        """
        Convert standard markdown to Telegram MarkdownV2.

        Telegram MarkdownV2 requires escaping special characters.
        """
        # First, escape special characters outside of code blocks
        # We need to preserve code blocks

        parts = []
        last_end = 0

        # Find code blocks (``` ... ```) and inline code (` ... `)
        code_pattern = r"```[\s\S]*?```|`[^`]+`"

        for match in re.finditer(code_pattern, text):
            # Process text before code block
            before = text[last_end : match.start()]
            parts.append(cls._escape_markdown(before))

            # Keep code block as-is (but format for Telegram)
            code = match.group()
            if code.startswith("```"):
                # Multi-line code block
                parts.append(code)
            else:
                # Inline code
                parts.append(code)

            last_end = match.end()

        # Process remaining text
        parts.append(cls._escape_markdown(text[last_end:]))

        return "".join(parts)

    @classmethod
    def _escape_markdown(cls, text: str) -> str:
        """
        Escape special characters for Telegram MarkdownV2.

        Special chars: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        # Characters that need escaping in MarkdownV2
        special_chars = r"_*[]()~>#+-=|{}.!"

        # Don't escape inside URLs
        # Simple approach: escape character by character
        result = []

        i = 0
        while i < len(text):
            char = text[i]

            # Check for URLs (don't escape inside)
            if text[i : i + 4] in ("http", "www."):
                # Find end of URL
                url_match = re.match(r"https?://[^\s\)]+|\www\.[^\s\)]+", text[i:])
                if url_match:
                    result.append(url_match.group())
                    i += len(url_match.group())
                    continue

            # Escape special characters
            if char in special_chars:
                result.append("\\" + char)
            else:
                result.append(char)

            i += 1

        return "".join(result)

    @classmethod
    def _split_message(cls, text: str) -> List[str]:
        """
        Split message into chunks respecting Telegram limits.

        Tries to split at paragraph boundaries.
        """
        if len(text) <= cls.MAX_MESSAGE_LENGTH:
            return [text]

        messages = []
        remaining = text

        while remaining:
            if len(remaining) <= cls.MAX_MESSAGE_LENGTH:
                messages.append(remaining)
                break

            # Find good split point
            split_at = cls.MAX_MESSAGE_LENGTH

            # Try to split at paragraph
            para_break = remaining[:split_at].rfind("\n\n")
            if para_break > cls.MAX_MESSAGE_LENGTH // 2:
                split_at = para_break + 2

            # Try to split at line
            elif "\n" in remaining[:split_at]:
                line_break = remaining[:split_at].rfind("\n")
                if line_break > cls.MAX_MESSAGE_LENGTH // 2:
                    split_at = line_break + 1

            # Try to split at sentence
            elif ". " in remaining[:split_at]:
                sentence_break = remaining[:split_at].rfind(". ")
                if sentence_break > cls.MAX_MESSAGE_LENGTH // 2:
                    split_at = sentence_break + 2

            messages.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        return messages

    @classmethod
    def format_status(cls, stage: str, detail: str = "") -> str:
        """Format status update for Telegram."""
        emoji_map = {
            "analyzing": "🔍",
            "searching": "🌐",
            "reading": "📖",
            "generating": "✨",
            "generated": "✅",
            "saving": "💾",
            "saved": "📁",
            "sending": "📤",
            "decision": "🤔",
            "error": "❌",
        }

        emoji = emoji_map.get(stage.lower(), "⚙️")
        msg = f"{emoji} {stage.capitalize()}"

        if detail:
            msg += f": {detail}"

        return msg

    @classmethod
    def format_error(cls, error: str) -> str:
        """Format error message for Telegram."""
        return f"❌ Error: {cls._escape_markdown(error)}"

    @classmethod
    def format_help(cls) -> str:
        """Format help message for Telegram."""
        return """🤖 *Jotty Bot Commands*

/start \\- Start the bot
/help \\- Show this help message
/status \\- Show bot status
/history \\- Show conversation history
/clear \\- Clear conversation history
/session \\- Show session info

*Usage:*
Simply send any message to process it with Jotty\\.

*Examples:*
• Search for AI news
• Create a checklist for project management
• Summarize this article: \\[url\\]
• Explain how transformers work

Your session persists across messages\\."""

    @classmethod
    def format_session_info(cls, session_data: dict) -> str:
        """Format session info for Telegram."""
        lines = [
            "📊 *Session Info*",
            "",
            f"ID: `{session_data.get('session_id', 'unknown')}`",
            f"Messages: {session_data.get('message_count', 0)}",
            f"Created: {session_data.get('created_at', 'unknown')[:10]}",
        ]

        interface_summary = session_data.get("interface_summary", {})
        if interface_summary:
            lines.append("")
            lines.append("*Messages by interface:*")
            for iface, count in interface_summary.items():
                lines.append(f"  {iface}: {count}")

        return "\n".join(lines)

    @classmethod
    def format_history(cls, messages: List[dict], limit: int = 10) -> str:
        """Format conversation history for Telegram."""
        if not messages:
            return "📭 No conversation history\\."

        lines = ["📜 *Recent History*", ""]

        for msg in messages[-limit:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            interface = msg.get("interface", "cli")

            role_emoji = "👤" if role == "user" else "🤖"
            interface_tag = f"\\[{interface}\\]" if interface != "telegram" else ""

            # Truncate and escape
            content = cls._escape_markdown(content)
            if len(content) > 100:
                content = content[:97] + "\\.\\.\\."

            lines.append(f"{role_emoji} {interface_tag} {content}")

        return "\n".join(lines)
