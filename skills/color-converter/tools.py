"""Color Converter Skill — convert between color formats."""
import re
import colorsys
from typing import Dict, Any, Tuple, Optional

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("color-converter")

NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "white": (255, 255, 255), "black": (0, 0, 0), "yellow": (255, 255, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "orange": (255, 165, 0),
    "purple": (128, 0, 128), "pink": (255, 192, 203), "brown": (165, 42, 42),
    "gray": (128, 128, 128), "grey": (128, 128, 128), "navy": (0, 0, 128),
    "teal": (0, 128, 128), "coral": (255, 127, 80), "salmon": (250, 128, 114),
    "gold": (255, 215, 0), "silver": (192, 192, 192), "lime": (0, 255, 0),
    "indigo": (75, 0, 130), "violet": (238, 130, 238), "maroon": (128, 0, 0),
}


def _parse_color(color: str) -> Tuple[int, int, int]:
    color = color.strip().lower()
    if color in NAMED_COLORS:
        return NAMED_COLORS[color]
    hex_match = re.match(r"^#?([0-9a-f]{6})$", color)
    if hex_match:
        h = hex_match.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb_match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color)
    if rgb_match:
        return int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
    raise ValueError(f"Cannot parse color: {color}")


@tool_wrapper(required_params=["color"])
def convert_color_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between color formats."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        r, g, b = _parse_color(params["color"])
    except ValueError as e:
        return tool_error(str(e))

    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    hex_val = f"#{r:02x}{g:02x}{b:02x}"

    return tool_response(
        hex=hex_val,
        rgb={"r": r, "g": g, "b": b},
        hsl={"h": round(h * 360), "s": round(s * 100), "l": round(l * 100)},
        css_rgb=f"rgb({r}, {g}, {b})",
        css_hsl=f"hsl({round(h * 360)}, {round(s * 100)}%, {round(l * 100)}%)",
    )


@tool_wrapper(required_params=["color"])
def complementary_color_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get complementary and analogous colors."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        r, g, b = _parse_color(params["color"])
    except ValueError as e:
        return tool_error(str(e))

    comp = (255 - r, 255 - g, 255 - b)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    analogous = []
    for offset in [-30, 30]:
        ah = ((h * 360 + offset) % 360) / 360
        ar, ag, ab = colorsys.hls_to_rgb(ah, l, s)
        analogous.append(f"#{int(ar*255):02x}{int(ag*255):02x}{int(ab*255):02x}")

    return tool_response(
        original=f"#{r:02x}{g:02x}{b:02x}",
        complementary=f"#{comp[0]:02x}{comp[1]:02x}{comp[2]:02x}",
        analogous=analogous,
    )


__all__ = ["convert_color_tool", "complementary_color_tool"]
