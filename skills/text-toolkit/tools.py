"""
Text Toolkit — Unified text processing and manipulation skill.

Consolidates: string-case-converter, slug-generator, text-statistics,
word-frequency-analyzer, regex-tester, regex-builder, color-converter.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("text-toolkit")

# =============================================================================
# CONSTANTS
# =============================================================================

_UNICODE_REPLACEMENTS = {
    "\u00e4": "ae",
    "\u00f6": "oe",
    "\u00fc": "ue",
    "\u00c4": "Ae",
    "\u00d6": "Oe",
    "\u00dc": "Ue",
    "\u00df": "ss",
    "\u00e9": "e",
    "\u00e8": "e",
    "\u00e0": "a",
    "\u00e2": "a",
    "\u00f4": "o",
    "\u00e7": "c",
    "\u00f1": "n",
    "\u00ee": "i",
    "\u00f9": "u",
}

_NAMED_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "brown": (139, 69, 19),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "coral": (255, 127, 80),
    "salmon": (250, 128, 114),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "indigo": (75, 0, 130),
}


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _split_words(text: str) -> list:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return re.split(r"[\s_\-]+", text.strip())


def _transliterate(text: str) -> str:
    for src, dst in _UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word or len(word) <= 2:
        return max(1, 0 if not word else 1)
    word = re.sub(r"(?:es|ed|e)$", "", word) or word
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def _parse_regex_flags(flags_str: str) -> int:
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
    flags = 0
    for c in flags_str.lower():
        if c in flag_map:
            flags |= flag_map[c]
    return flags


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple:
    r_, g_, b_ = r / 255, g / 255, b / 255
    mx, mn = max(r_, g_, b_), min(r_, g_, b_)
    l = (mx + mn) / 2
    if mx == mn:
        h = s = 0.0
    else:
        d = mx - mn
        s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r_:
            h = (g_ - b_) / d + (6 if g_ < b_ else 0)
        elif mx == g_:
            h = (b_ - r_) / d + 2
        else:
            h = (r_ - g_) / d + 4
        h /= 6
    return round(h * 360), round(s * 100), round(l * 100)


def _parse_color(color: str) -> tuple | None:
    color = color.strip().lower()
    if color in _NAMED_COLORS:
        return _NAMED_COLORS[color]
    hex_match = re.match(r"^#?([0-9a-f]{6})$", color)
    if hex_match:
        h = hex_match.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb_match = re.match(r"^(\d{1,3})\s*[,\s]\s*(\d{1,3})\s*[,\s]\s*(\d{1,3})$", color)
    if rgb_match:
        return int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
    return None


# =============================================================================
# CASE CONVERSION
# =============================================================================


@tool_wrapper(required_params=["text", "to_case"])
def convert_case_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string between naming conventions (camelCase, snake_case, PascalCase, etc.)."""
    status.set_callback(params.pop("_status_callback", None))
    words = _split_words(params["text"])
    target = params["to_case"].lower().replace(" ", "").replace("_", "")
    cases = {
        "camelcase": lambda w: w[0].lower() + "".join(x.capitalize() for x in w[1:]),
        "pascalcase": lambda w: "".join(x.capitalize() for x in w),
        "snakecase": lambda w: "_".join(x.lower() for x in w),
        "snake": lambda w: "_".join(x.lower() for x in w),
        "kebabcase": lambda w: "-".join(x.lower() for x in w),
        "kebab": lambda w: "-".join(x.lower() for x in w),
        "uppercase": lambda w: "_".join(x.upper() for x in w),
        "upper": lambda w: "_".join(x.upper() for x in w),
        "screamingsnake": lambda w: "_".join(x.upper() for x in w),
        "titlecase": lambda w: " ".join(x.capitalize() for x in w),
        "title": lambda w: " ".join(x.capitalize() for x in w),
        "lowercase": lambda w: " ".join(x.lower() for x in w),
        "lower": lambda w: " ".join(x.lower() for x in w),
        "dotcase": lambda w: ".".join(x.lower() for x in w),
        "dot": lambda w: ".".join(x.lower() for x in w),
    }
    converter = cases.get(target)
    if not converter:
        return tool_error(
            f"Unknown case: {params['to_case']}. Use: camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE, Title Case, dot.case"
        )
    return tool_response(result=converter(words), from_words=words, to_case=params["to_case"])


# =============================================================================
# SLUG GENERATION
# =============================================================================


@tool_wrapper(required_params=["text"])
def slugify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a URL-friendly slug with unicode transliteration."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    sep = params.get("separator", "-")
    max_len = params.get("max_length", 200)
    slug = _transliterate(text)
    if params.get("lowercase", True):
        slug = slug.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", sep, slug).strip(sep)
    if max_len and len(slug) > max_len:
        slug = slug[:max_len].rstrip(sep)
    return tool_response(slug=slug, original=text, length=len(slug))


# =============================================================================
# TEXT STATISTICS
# =============================================================================


@tool_wrapper(required_params=["text"])
def analyze_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Text statistics: word count, char count, sentences, reading time, Flesch-Kincaid grade."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    words = re.findall(r"\b[a-zA-Z0-9']+\b", text)
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    total_syllables = sum(_count_syllables(w) for w in words)
    fk = (
        round(
            0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59, 2
        )
        if word_count > 0
        else 0.0
    )
    return tool_response(
        word_count=word_count,
        char_count=len(text),
        char_count_no_spaces=len(text.replace(" ", "")),
        sentence_count=len(sentences),
        reading_time_minutes=round(word_count / 238, 2),
        flesch_kincaid_grade=fk,
        syllable_count=total_syllables,
    )


# =============================================================================
# WORD FREQUENCY
# =============================================================================


@tool_wrapper(required_params=["text"])
def word_frequency_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze word frequency with top-N results."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    top_n = int(params.get("top_n", 20))
    min_length = int(params.get("min_length", 1))
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    filtered = [w for w in words if len(w) >= min_length]
    counter = Counter(filtered)
    top_words = (
        [
            {"word": w, "count": c, "percentage": round(c / len(filtered) * 100, 2)}
            for w, c in counter.most_common(top_n)
        ]
        if filtered
        else []
    )
    return tool_response(
        top_words=top_words,
        total_words=len(filtered),
        unique_words=len(counter),
        vocabulary_richness=round(len(counter) / max(len(filtered), 1), 4),
    )


# =============================================================================
# REGEX TOOLS
# =============================================================================


@tool_wrapper(required_params=["pattern", "text"])
def regex_match_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Test a regex pattern and return all matches with positions and groups."""
    status.set_callback(params.pop("_status_callback", None))
    flags = _parse_regex_flags(params.get("flags", ""))
    try:
        compiled = re.compile(params["pattern"], flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")
    matches = []
    for m in compiled.finditer(params["text"]):
        info: Dict[str, Any] = {
            "match": m.group(),
            "start": m.start(),
            "end": m.end(),
            "groups": list(m.groups()),
        }
        if m.groupdict():
            info["named_groups"] = m.groupdict()
        matches.append(info)
    return tool_response(matches=matches, count=len(matches), pattern=params["pattern"])


@tool_wrapper(required_params=["pattern", "text"])
def regex_replace_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace regex matches in text."""
    status.set_callback(params.pop("_status_callback", None))
    flags = _parse_regex_flags(params.get("flags", ""))
    try:
        result, n = re.subn(
            params["pattern"],
            params.get("replacement", ""),
            params["text"],
            count=int(params.get("count", 0)),
            flags=flags,
        )
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")
    return tool_response(result=result, replacements=n)


@tool_wrapper(required_params=["pattern", "text"])
def regex_split_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Split text by regex pattern."""
    status.set_callback(params.pop("_status_callback", None))
    flags = _parse_regex_flags(params.get("flags", ""))
    try:
        parts = re.split(params["pattern"], params["text"], flags=flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")
    return tool_response(parts=parts, count=len(parts))


# =============================================================================
# COLOR CONVERSION
# =============================================================================


@tool_wrapper(required_params=["color"])
def color_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert colors between hex, RGB, HSL, and named color formats."""
    status.set_callback(params.pop("_status_callback", None))
    rgb = _parse_color(params["color"])
    if rgb is None:
        return tool_error(
            f"Cannot parse color: {params['color']}. Use hex (#FF5733), RGB (255,87,51), or named (red)."
        )
    r, g, b = rgb
    if not all(0 <= v <= 255 for v in (r, g, b)):
        return tool_error("RGB values must be between 0 and 255")
    h, s, l = _rgb_to_hsl(r, g, b)
    hex_val = f"#{r:02x}{g:02x}{b:02x}"
    return tool_response(
        hex=hex_val,
        rgb={"r": r, "g": g, "b": b},
        hsl={"h": h, "s": s, "l": l},
        css_hex=hex_val,
        css_rgb=f"rgb({r}, {g}, {b})",
        css_hsl=f"hsl({h}, {s}%, {l}%)",
    )


__all__ = [
    "convert_case_tool",
    "slugify_tool",
    "analyze_text_tool",
    "word_frequency_tool",
    "regex_match_tool",
    "regex_replace_tool",
    "regex_split_tool",
    "color_convert_tool",
]
