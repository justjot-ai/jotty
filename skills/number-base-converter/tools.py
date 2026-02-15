"""Number Base Converter Skill — arbitrary base conversion (2-36)."""
import string
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("number-base-converter")

_DIGITS = string.digits + string.ascii_lowercase  # 0-9a-z = 36 chars


def _to_decimal(number: str, base: int, digits: str = _DIGITS) -> int:
    number = number.strip().lower()
    result = 0
    for ch in number:
        val = digits.index(ch)
        if val >= base:
            raise ValueError(f"Digit '{ch}' invalid for base {base}")
        result = result * base + val
    return result


def _from_decimal(n: int, base: int, digits: str = _DIGITS) -> str:
    if n == 0:
        return digits[0]
    negative = n < 0
    n = abs(n)
    chars = []
    while n > 0:
        chars.append(digits[n % base])
        n //= base
    if negative:
        chars.append("-")
    return "".join(reversed(chars))


@tool_wrapper(required_params=["number", "from_base", "to_base"])
def base_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a number between arbitrary bases (2-36)."""
    status.set_callback(params.pop("_status_callback", None))
    number = str(params["number"])
    fb = int(params["from_base"])
    tb = int(params["to_base"])

    if not (2 <= fb <= 36) or not (2 <= tb <= 36):
        return tool_error("Bases must be between 2 and 36")

    digits = params.get("custom_digits", _DIGITS)
    decimal_val = _to_decimal(number, fb, digits)
    result = _from_decimal(decimal_val, tb, digits)
    return tool_response(
        original=number, from_base=fb, to_base=tb,
        result=result, decimal_value=decimal_val,
    )


__all__ = ["base_convert_tool"]
