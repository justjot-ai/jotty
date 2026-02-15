"""Regex Tester Skill — test, match, and explain regex patterns."""

import re
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("regex-tester")


def _parse_flags(flags_str: str) -> int:
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
    flags = 0
    for c in flags_str.lower():
        if c in flag_map:
            flags |= flag_map[c]
    return flags


@tool_wrapper(required_params=["pattern", "text"])
def regex_match_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Test a regex pattern and return all matches."""
    status.set_callback(params.pop("_status_callback", None))
    pattern = params["pattern"]
    text = params["text"]
    flags = _parse_flags(params.get("flags", ""))

    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")

    matches = []
    for m in compiled.finditer(text):
        match_info = {
            "match": m.group(),
            "start": m.start(),
            "end": m.end(),
            "groups": list(m.groups()),
        }
        if m.groupdict():
            match_info["named_groups"] = m.groupdict()
        matches.append(match_info)

    return tool_response(matches=matches, count=len(matches), pattern=pattern)


@tool_wrapper(required_params=["pattern", "text"])
def regex_replace_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace regex matches in text."""
    status.set_callback(params.pop("_status_callback", None))
    replacement = params.get("replacement", "")
    flags = _parse_flags(params.get("flags", ""))
    count = int(params.get("count", 0))  # 0 = all

    try:
        result, n = re.subn(
            params["pattern"], replacement, params["text"], count=count, flags=flags
        )
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")

    return tool_response(result=result, replacements=n)


@tool_wrapper(required_params=["pattern", "text"])
def regex_split_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Split text by regex pattern."""
    status.set_callback(params.pop("_status_callback", None))
    flags = _parse_flags(params.get("flags", ""))
    try:
        parts = re.split(params["pattern"], params["text"], flags=flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")
    return tool_response(parts=parts, count=len(parts))


__all__ = ["regex_match_tool", "regex_replace_tool", "regex_split_tool"]
