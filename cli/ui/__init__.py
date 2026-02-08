"""CLI UI Components."""

from .renderer import (
    RichRenderer,
    ShimmerEffect,
    MarkdownStreamRenderer,
    FooterHints,
    REPLState,
    DesktopNotifier,
)
from .progress import ProgressManager, SwarmState, SwarmDashboard
from .tables import TableRenderer
from .themes import Theme, get_theme, ColorDistance, validate_palette_contrast

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
