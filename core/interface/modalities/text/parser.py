"""
Text Input Parser
=================

Parses text input from different platforms into a unified format.
"""

from typing import Any, Dict, Optional


class TextParser:
    """Parse text input from different platforms."""

    def __init__(self, platform: str = "generic"):
        """
        Initialize parser for specific platform.

        Args:
            platform: Platform name (telegram, whatsapp, cli, web)
        """
        self.platform = platform

    def parse(self, raw_input: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse text input into unified format.

        Args:
            raw_input: Raw text from platform
            metadata: Optional platform-specific metadata

        Returns:
            Parsed input with standardized structure
        """
        return {
            "text": raw_input,
            "platform": self.platform,
            "metadata": metadata or {},
            "modality": "text",
        }


def parse_input(raw_input: str, platform: str = "generic", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to parse text input.

    Args:
        raw_input: Raw text from platform
        platform: Platform name
        **kwargs: Additional metadata

    Returns:
        Parsed input dictionary
    """
    parser = TextParser(platform)
    return parser.parse(raw_input, metadata=kwargs)
