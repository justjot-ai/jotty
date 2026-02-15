"""Regex Builder Skill — build, test, and explain regex patterns."""
import re
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("regex-builder")

_PRESETS = {
    "email": {"pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "description": "Email address"},
    "url": {"pattern": r'https?://[^\s<>"]+', "description": "HTTP/HTTPS URL"},
    "phone": {"pattern": r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "description": "US phone number"},
    "ipv4": {"pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "description": "IPv4 address"},
    "date_iso": {"pattern": r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])", "description": "ISO date (YYYY-MM-DD)"},
    "uuid": {"pattern": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "description": "UUID v4"},
    "hex_color": {"pattern": r"#(?:[0-9a-fA-F]{3}){1,2}\b", "description": "Hex color code"},
    "ip": {"pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "description": "IPv4 address"},
    "date": {"pattern": r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])", "description": "ISO date"},
}

_EXPLAIN = {
    "\\d": "digit (0-9)", "\\w": "word char (a-z, A-Z, 0-9, _)", "\\s": "whitespace",
    "\\b": "word boundary", ".": "any character", "+": "one or more", "*": "zero or more",
    "?": "zero or one (optional)", "^": "start of string", "$": "end of string",
    "\\D": "non-digit", "\\W": "non-word char", "\\S": "non-whitespace",
}


@tool_wrapper(required_params=["action"])
def regex_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build, test, or explain regular expressions."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "preset":
        name = params.get("name", "")
        if not name:
            return tool_response(available=list(_PRESETS.keys()))
        preset = _PRESETS.get(name)
        if not preset:
            return tool_error(f"Unknown preset: {name}. Available: {list(_PRESETS.keys())}")
        return tool_response(name=name, **preset)

    if action == "test":
        pattern = params.get("pattern", "")
        text = params.get("text", "")
        if not pattern or not text:
            return tool_error("pattern and text required for test")
        try:
            matches = re.findall(pattern, text)
            full = bool(re.fullmatch(pattern, text))
            return tool_response(pattern=pattern, matches=matches, count=len(matches), full_match=full)
        except re.error as e:
            return tool_error(f"Invalid regex: {e}")

    if action == "explain":
        pattern = params.get("pattern", "")
        if not pattern:
            return tool_error("pattern required for explain")
        parts = []
        for token, desc in _EXPLAIN.items():
            if token in pattern:
                parts.append({"token": token, "meaning": desc})
        return tool_response(pattern=pattern, components=parts)

    return tool_error(f"Unknown action: {action}. Use: preset, test, explain")


__all__ = ["regex_tool"]
