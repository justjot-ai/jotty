"""
Color Themes for Jotty CLI
==========================

Rich-based color themes for terminal output.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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

    "muted": Theme(
        primary="rgb(200,200,200)",
        secondary="rgb(120,150,120)",
        accent="rgb(150,130,170)",
        warning="rgb(180,130,90)",
        error="rgb(180,90,90)",
        success="rgb(120,150,120)",
        info="rgb(130,150,180)",
        muted="rgb(120,120,120)",
        prompt="bold rgb(200,200,200)",
        input="rgb(200,200,200)",
        command="bold rgb(120,150,120)",
        argument="rgb(180,130,90)",
        flag="rgb(150,130,170)",
        running="rgb(130,150,180)",
        completed="rgb(120,150,120)",
        failed="rgb(180,90,90)",
        pending="rgb(180,130,90)",
        agent_name="bold rgb(150,130,170)",
        skill_name="bold rgb(130,150,180)",
        tool_name="bold rgb(130,150,180)",
        output_key="rgb(130,150,180)",
        output_value="rgb(200,200,200)",
        code="rgb(120,150,120)",
        path="rgb(130,150,180) underline",
        table_header="bold rgb(200,200,200)",
        table_border="rgb(80,80,80)",
        table_row_odd="rgb(200,200,200)",
        table_row_even="rgb(160,160,160)",
    ),
}


class ColorDistance:
    """CIE76 perceptual color distance utility for palette validation."""

    @staticmethod
    def srgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert sRGB (0-255) to CIE XYZ."""
        # Linearize sRGB
        def linearize(c: int) -> float:
            c_norm = c / 255.0
            if c_norm <= 0.04045:
                return c_norm / 12.92
            return ((c_norm + 0.055) / 1.055) ** 2.4

        rl, gl, bl = linearize(r), linearize(g), linearize(b)

        # sRGB -> XYZ (D65 illuminant)
        x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
        y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
        z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
        return x, y, z

    @staticmethod
    def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert CIE XYZ to CIE L*a*b*."""
        # D65 reference white
        xn, yn, zn = 0.95047, 1.00000, 1.08883

        def f(t: float) -> float:
            delta = 6.0 / 29.0
            if t > delta ** 3:
                return t ** (1.0 / 3.0)
            return t / (3.0 * delta ** 2) + 4.0 / 29.0

        fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
        l_star = 116.0 * fy - 16.0
        a_star = 500.0 * (fx - fy)
        b_star = 200.0 * (fy - fz)
        return l_star, a_star, b_star

    @classmethod
    def rgb_to_lab(cls, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert sRGB (0-255) to CIE L*a*b*."""
        x, y, z = cls.srgb_to_xyz(r, g, b)
        return cls.xyz_to_lab(x, y, z)

    @classmethod
    def cie76(
        cls,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
    ) -> float:
        """
        Compute CIE76 perceptual distance between two sRGB colors.

        Args:
            color1: (R, G, B) tuple, 0-255
            color2: (R, G, B) tuple, 0-255

        Returns:
            Delta-E value (0 = identical, >40 = very different)
        """
        l1, a1, b1 = cls.rgb_to_lab(*color1)
        l2, a2, b2 = cls.rgb_to_lab(*color2)
        return math.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)


def validate_palette_contrast(
    colors: Dict[str, Tuple[int, int, int]],
    min_distance: float = 20.0,
) -> Dict[str, float]:
    """
    Validate that palette colors are perceptually distinguishable.

    Args:
        colors: Named colors as {name: (R, G, B)} dict
        min_distance: Minimum CIE76 distance between any pair

    Returns:
        Dict of pair names to distances that are below min_distance.
        Empty dict means all pairs pass.
    """
    failures = {}
    names = list(colors.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = ColorDistance.cie76(colors[names[i]], colors[names[j]])
            if dist < min_distance:
                pair_key = f"{names[i]}-{names[j]}"
                failures[pair_key] = round(dist, 2)
    return failures


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
