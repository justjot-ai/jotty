"""CLI UI Components."""

from .progress import ProgressManager, SwarmDashboard, SwarmState
from .renderer import (
    DesktopNotifier,
    FooterHints,
    MarkdownStreamRenderer,
    REPLState,
    RichRenderer,
    ShimmerEffect,
)
from .tables import TableRenderer
from .themes import ColorDistance, Theme, get_theme, validate_palette_contrast

__all__ = [
    "RichRenderer",
    "ShimmerEffect",
    "MarkdownStreamRenderer",
    "FooterHints",
    "REPLState",
    "DesktopNotifier",
    "ProgressManager",
    "SwarmState",
    "SwarmDashboard",
    "TableRenderer",
    "Theme",
    "get_theme",
    "ColorDistance",
    "validate_palette_contrast",
]
