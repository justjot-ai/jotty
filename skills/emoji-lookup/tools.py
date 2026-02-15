"""Emoji Lookup Skill — search emojis, get info, convert shortcodes."""
import unicodedata
import re
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("emoji-lookup")

_DB: Dict[str, Dict[str, Any]] = {
    ":smile:": {"emoji": "\U0001f604", "name": "smiling face with open mouth and smiling eyes", "category": "faces"},
    ":grinning:": {"emoji": "\U0001f600", "name": "grinning face", "category": "faces"},
    ":heart:": {"emoji": "\u2764\ufe0f", "name": "red heart", "category": "symbols"},
    ":thumbsup:": {"emoji": "\U0001f44d", "name": "thumbs up", "category": "hands"},
    ":thumbsdown:": {"emoji": "\U0001f44e", "name": "thumbs down", "category": "hands"},
    ":fire:": {"emoji": "\U0001f525", "name": "fire", "category": "nature"},
    ":star:": {"emoji": "\u2b50", "name": "star", "category": "symbols"},
    ":sun:": {"emoji": "\u2600\ufe0f", "name": "sun", "category": "nature"},
    ":moon:": {"emoji": "\U0001f319", "name": "crescent moon", "category": "nature"},
    ":rocket:": {"emoji": "\U0001f680", "name": "rocket", "category": "travel"},
    ":check:": {"emoji": "\u2705", "name": "check mark", "category": "symbols"},
    ":x:": {"emoji": "\u274c", "name": "cross mark", "category": "symbols"},
    ":warning:": {"emoji": "\u26a0\ufe0f", "name": "warning", "category": "symbols"},
    ":wave:": {"emoji": "\U0001f44b", "name": "waving hand", "category": "hands"},
    ":clap:": {"emoji": "\U0001f44f", "name": "clapping hands", "category": "hands"},
    ":cry:": {"emoji": "\U0001f622", "name": "crying face", "category": "faces"},
    ":laugh:": {"emoji": "\U0001f602", "name": "face with tears of joy", "category": "faces"},
    ":think:": {"emoji": "\U0001f914", "name": "thinking face", "category": "faces"},
    ":100:": {"emoji": "\U0001f4af", "name": "hundred points", "category": "symbols"},
    ":party:": {"emoji": "\U0001f389", "name": "party popper", "category": "objects"},
    ":globe:": {"emoji": "\U0001f30d", "name": "globe showing Europe-Africa", "category": "travel"},
    ":coffee:": {"emoji": "\u2615", "name": "hot beverage", "category": "food"},
    ":bug:": {"emoji": "\U0001f41b", "name": "bug", "category": "nature"},
    ":lock:": {"emoji": "\U0001f512", "name": "locked", "category": "objects"},
    ":key:": {"emoji": "\U0001f511", "name": "key", "category": "objects"},
}


@tool_wrapper(required_params=["action"])
def emoji_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search, get info, or convert emoji shortcodes."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "search":
        query = params.get("query", "").lower()
        if not query:
            return tool_error("query required for search")
        results = []
        for code, info in _DB.items():
            if query in code or query in info["name"]:
                results.append({"shortcode": code, **info})
        return tool_response(results=results, count=len(results))

    if action == "info":
        emoji = params.get("emoji", "")
        if not emoji:
            return tool_error("emoji parameter required")
        name = unicodedata.name(emoji[0], "unknown")
        cp = "+".join(f"U+{ord(c):04X}" for c in emoji)
        return tool_response(emoji=emoji, name=name.lower(), codepoints=cp)

    if action == "convert":
        shortcode = params.get("shortcode", "")
        if not shortcode:
            return tool_error("shortcode parameter required")
        if not shortcode.startswith(":"):
            shortcode = f":{shortcode}:"
        if not shortcode.endswith(":"):
            shortcode = f"{shortcode}:"
        entry = _DB.get(shortcode)
        if not entry:
            return tool_error(f"Unknown shortcode: {shortcode}")
        return tool_response(emoji=entry["emoji"], name=entry["name"], shortcode=shortcode)

    return tool_error(f"Unknown action: {action}. Use: search, info, convert")


__all__ = ["emoji_lookup_tool"]
