"""
Text Modality - Text Input/Output Handling
===========================================

Handles text-based communication across all platforms.

## Responsibilities

- Parse text input from different platforms
- Format text output for different platforms
- Handle text encoding/decoding
- Apply platform-specific formatting (markdown, emoji, etc.)
"""

from .formatter import TextFormatter, format_output
from .parser import TextParser, parse_input

__all__ = [
    "TextModality",
    "TextParser",
    "TextFormatter",
    "parse_input",
    "format_output",
]


class TextModality:
    """
    Text modality handler.

    Provides unified interface for text input/output across platforms.
    """

    def __init__(self, platform: str = "generic"):
        """
        Initialize text modality.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
        """
        self.platform = platform
        self.parser = TextParser(platform)
        self.formatter = TextFormatter(platform)

    def parse(self, raw_input: str, **kwargs) -> dict:
        """Parse text input from platform."""
        return self.parser.parse(raw_input, **kwargs)

    def format(self, content: str, **kwargs) -> str:
        """Format text output for platform."""
        return self.formatter.format(content, **kwargs)
