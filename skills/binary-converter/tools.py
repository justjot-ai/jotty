"""Binary Converter Skill — convert bases and bitwise operations."""
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("binary-converter")

_BASES = {"binary": 2, "decimal": 10, "hex": 16, "octal": 8}
_PREFIX = {"binary": "0b", "hex": "0x", "octal": "0o", "decimal": ""}


def _to_int(value: str, base_name: str) -> int:
    b = _BASES.get(base_name)
    if b is None:
        raise ValueError(f"Unknown base: {base_name}")
    v = value.strip().lower().replace("0b", "").replace("0x", "").replace("0o", "")
    return int(v, b)


def _from_int(n: int, base_name: str) -> str:
    if base_name == "binary":
        return bin(n)
    if base_name == "hex":
        return hex(n)
    if base_name == "octal":
        return oct(n)
    return str(n)


@tool_wrapper(required_params=["action"])
def binary_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between number bases or perform bitwise operations."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "convert":
        value = params.get("value", "")
        fb = params.get("from_base", "decimal")
        tb = params.get("to_base", "binary")
        if not value:
            return tool_error("value required")
        n = _to_int(str(value), fb)
        result = _from_int(n, tb)
        return tool_response(original=value, from_base=fb, to_base=tb,
                             result=result, decimal_value=n)

    if action == "bitwise":
        op = params.get("op", "").upper()
        a = int(params.get("a", 0))
        if op == "NOT":
            return tool_response(op=op, a=a, result=~a, binary=bin(~a & 0xFFFFFFFF))
        b = int(params.get("b", 0))
        ops = {"AND": a & b, "OR": a | b, "XOR": a ^ b,
               "LSHIFT": a << b, "RSHIFT": a >> b}
        if op not in ops:
            return tool_error(f"Unknown op: {op}. Use: AND, OR, XOR, NOT, LSHIFT, RSHIFT")
        r = ops[op]
        return tool_response(op=op, a=a, b=b, result=r, binary=bin(r))

    return tool_error(f"Unknown action: {action}. Use: convert, bitwise")


__all__ = ["binary_convert_tool"]
