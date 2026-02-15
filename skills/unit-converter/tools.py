"""Unit converter — length, weight, temperature, volume, speed."""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("unit-converter")

_LENGTH = {
    "m": 1,
    "ft": 0.3048,
    "in": 0.0254,
    "km": 1000,
    "mi": 1609.344,
    "cm": 0.01,
    "mm": 0.001,
    "yd": 0.9144,
}
_WEIGHT = {"kg": 1, "lb": 0.453592, "oz": 0.0283495, "g": 0.001, "mg": 1e-6, "ton": 907.185}
_VOLUME = {
    "l": 1,
    "gal": 3.78541,
    "ml": 0.001,
    "cup": 0.236588,
    "pt": 0.473176,
    "qt": 0.946353,
    "fl_oz": 0.0295735,
}
_SPEED = {"km/h": 1, "mph": 1.60934, "m/s": 3.6, "kn": 1.852, "ft/s": 1.09728}


def _convert_table(val: float, f: str, t: str, tbl: dict) -> float | None:
    if f in tbl and t in tbl:
        return val * tbl[f] / tbl[t]
    return None


def _temp(val: float, f: str, t: str) -> float | None:
    aliases = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}
    fc, tc = aliases.get(f), aliases.get(t)
    if not fc or not tc:
        return None
    if fc == tc:
        return val
    to_c = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}
    from_c = {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}
    return from_c[tc](to_c[fc](val))


@tool_wrapper(required_params=["value", "from_unit", "to_unit"])
def convert_unit_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert value between units."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        val = float(params["value"])
    except (ValueError, TypeError):
        return tool_error("value must be a number")
    f = params["from_unit"].lower().strip()
    t = params["to_unit"].lower().strip()
    for tbl in [_LENGTH, _WEIGHT, _VOLUME, _SPEED]:
        r = _convert_table(val, f, t, tbl)
        if r is not None:
            return tool_response(result=round(r, 6), from_unit=f, to_unit=t, value=val)
    r = _temp(val, f, t)
    if r is not None:
        return tool_response(result=round(r, 6), from_unit=f, to_unit=t, value=val)
    return tool_error(f"Unsupported conversion: {f} -> {t}")


__all__ = ["convert_unit_tool"]
