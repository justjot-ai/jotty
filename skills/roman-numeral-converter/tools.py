"""Roman Numeral Converter Skill — convert between Roman numerals and integers."""
import re
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("roman-numeral-converter")

_TO_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]
_FROM_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_VALID_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _int_to_roman(n: int) -> str:
    parts = []
    for value, numeral in _TO_ROMAN:
        while n >= value:
            parts.append(numeral)
            n -= value
    return "".join(parts)


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _FROM_ROMAN.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


@tool_wrapper(required_params=["action"])
def roman_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between Roman numerals and integers."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "to_roman":
        n = params.get("number")
        if n is None:
            return tool_error("number required")
        n = int(n)
        if n < 1 or n > 3999:
            return tool_error("Number must be 1-3999")
        return tool_response(number=n, roman=_int_to_roman(n))

    if action == "to_integer":
        roman = params.get("roman", "")
        if not roman:
            return tool_error("roman required")
        val = _roman_to_int(roman)
        return tool_response(roman=roman.upper(), number=val)

    if action == "validate":
        roman = params.get("roman", "")
        if not roman:
            return tool_error("roman required")
        valid = bool(_VALID_RE.match(roman.upper().strip()))
        return tool_response(roman=roman.upper(), valid=valid)

    return tool_error(f"Unknown action: {action}. Use: to_roman, to_integer, validate")


__all__ = ["roman_tool"]
