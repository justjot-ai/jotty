"""Palindrome checker — check, find substrings, generate."""
import re
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("palindrome-checker")

_KNOWN = ["racecar", "level", "deified", "civic", "rotor", "kayak", "madam",
           "refer", "noon", "radar", "repaper", "rotator", "reviver", "sagas",
           "solos", "stats", "tenet", "wow", "deed", "peep"]


def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_pal(s: str) -> bool:
    return s == s[::-1]


def _find_palindromes(text: str, min_len: int = 2) -> List[str]:
    clean = _clean(text)
    n = len(clean)
    found: set = set()
    for i in range(n):
        # odd-length
        l, r = i, i
        while l >= 0 and r < n and clean[l] == clean[r]:
            if r - l + 1 >= min_len:
                found.add(clean[l:r + 1])
            l -= 1
            r += 1
        # even-length
        l, r = i, i + 1
        while l >= 0 and r < n and clean[l] == clean[r]:
            if r - l + 1 >= min_len:
                found.add(clean[l:r + 1])
            l -= 1
            r += 1
    return sorted(found, key=len, reverse=True)


@tool_wrapper(required_params=["text"])
def palindrome_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check palindrome, find palindromic substrings."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    min_len = int(params.get("min_length", 2))
    clean = _clean(text)
    is_palindrome = _is_pal(clean) if clean else False
    substrings = _find_palindromes(text, min_len)[:20]
    longest = substrings[0] if substrings else ""
    return tool_response(
        text=text, cleaned=clean, is_palindrome=is_palindrome,
        palindromic_substrings=substrings, longest_palindrome=longest,
        substring_count=len(substrings),
        examples=_KNOWN[:5],
    )


__all__ = ["palindrome_tool"]
