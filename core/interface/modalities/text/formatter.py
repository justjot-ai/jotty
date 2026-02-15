"""
Text Output Formatter
=====================

Formats text output for different platforms (markdown, emoji, etc.).
"""

from typing import Any, Dict, Optional


class TextFormatter:
    """Format text output for different platforms."""

    def __init__(self, platform: str = "generic"):
        """
        Initialize formatter for specific platform.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
        """
        self.platform = platform

    def format(self, content: str, **kwargs) -> str:
        """
        Format text output for platform.

        Args:
            content: Text content to format
            **kwargs: Platform-specific formatting options

        Returns:
            Formatted text ready for platform
        """
        # Platform-specific formatting
        if self.platform == "telegram":
            return self._format_telegram(content, **kwargs)
        elif self.platform == "whatsapp":
            return self._format_whatsapp(content, **kwargs)
        elif self.platform == "cli":
            return self._format_cli(content, **kwargs)
        elif self.platform == "web":
            return self._format_web(content, **kwargs)
        else:
            return content

    def _format_telegram(self, content: str, **kwargs) -> str:
        """Format for Telegram (supports markdown)."""
        # Telegram supports markdown
        return content

    def _format_whatsapp(self, content: str, **kwargs) -> str:
        """Format for WhatsApp (limited markdown)."""
        # WhatsApp has limited markdown support
        return content

    def _format_cli(self, content: str, **kwargs) -> str:
        """Format for CLI (plain text with ANSI colors)."""
        # CLI can use ANSI color codes
        return content

    def _format_web(self, content: str, **kwargs) -> str:
        """Format for Web (HTML or markdown)."""
        # Web can use HTML
        return content


def format_output(content: str, platform: str = "generic", **kwargs) -> str:
    """
    Convenience function to format text output.

    Args:
        content: Text content to format
        platform: Platform name
        **kwargs: Platform-specific formatting options

    Returns:
        Formatted text
    """
    formatter = TextFormatter(platform)
    return formatter.format(content, **kwargs)
