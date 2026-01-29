"""
Color Themes for Jotty CLI
==========================

Rich-based color themes for terminal output.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Theme:
    """Color theme for CLI output."""

    # Core colors
    primary: str = "cyan"
    secondary: str = "green"
    accent: str = "magenta"
    warning: str = "yellow"
    error: str = "red"
    success: str = "green"
    info: str = "blue"
    muted: str = "dim"

    # Component colors
    prompt: str = "bold cyan"
    input: str = "white"
    command: str = "bold green"
    argument: str = "yellow"
    flag: str = "magenta"

    # Status colors
    running: str = "cyan"
    completed: str = "green"
    failed: str = "red"
    pending: str = "yellow"

    # Agent colors
    agent_name: str = "bold magenta"
    skill_name: str = "bold blue"
    tool_name: str = "bold cyan"

    # Output colors
    output_key: str = "cyan"
    output_value: str = "white"
    code: str = "green"
    path: str = "blue underline"

    # Table colors
    table_header: str = "bold cyan"
    table_border: str = "dim"
    table_row_odd: str = "white"
    table_row_even: str = "dim white"


# Built-in themes
THEMES: Dict[str, Theme] = {
    "default": Theme(),

    "dark": Theme(
        primary="bright_cyan",
        secondary="bright_green",
        accent="bright_magenta",
        prompt="bold bright_cyan",
        table_header="bold bright_white",
    ),

    "light": Theme(
        primary="blue",
        secondary="green",
        accent="purple",
        muted="grey50",
        prompt="bold blue",
        table_border="grey70",
    ),

    "minimal": Theme(
        primary="white",
        secondary="white",
        accent="white",
        warning="white",
        error="bold white",
        success="white",
        info="white",
        muted="dim",
        prompt="bold white",
        table_header="bold",
    ),

    "matrix": Theme(
        primary="green",
        secondary="bright_green",
        accent="green",
        warning="yellow",
        error="red",
        success="bright_green",
        info="green",
        prompt="bold bright_green",
        command="green",
        code="bright_green",
    ),

    "ocean": Theme(
        primary="cyan",
        secondary="blue",
        accent="bright_blue",
        prompt="bold cyan",
        agent_name="bold bright_blue",
        skill_name="bold cyan",
    ),
}


def get_theme(name: str = "default") -> Theme:
    """
    Get theme by name.

    Args:
        name: Theme name (default, dark, light, minimal, matrix, ocean)

    Returns:
        Theme instance
    """
    return THEMES.get(name, THEMES["default"])


def list_themes() -> list:
    """List available theme names."""
    return list(THEMES.keys())
