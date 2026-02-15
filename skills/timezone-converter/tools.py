"""Timezone converter — common TZ names to UTC offsets."""

from datetime import datetime, timedelta
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("timezone-converter")

_TZ = {
    "UTC": 0,
    "GMT": 0,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "AKST": -9,
    "AKDT": -8,
    "HST": -10,
    "IST": 5.5,
    "JST": 9,
    "KST": 9,
    "CST_CN": 8,
    "SGT": 8,
    "HKT": 8,
    "AEST": 10,
    "AEDT": 11,
    "NZST": 12,
    "NZDT": 13,
    "CET": 1,
    "CEST": 2,
    "EET": 2,
    "EEST": 3,
    "WET": 0,
    "WEST": 1,
    "BRT": -3,
    "ART": -3,
    "GST": 4,
    "PKT": 5,
    "NPT": 5.75,
    "ICT": 7,
    "WIB": 7,
    "WITA": 8,
    "WIT": 9,
}

_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d",
    "%H:%M",
]


def _parse_dt(s: str) -> datetime | None:
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _get_offset(tz: str) -> float | None:
    up = tz.upper().strip()
    if up in _TZ:
        return _TZ[up]
    try:
        return float(tz)
    except ValueError:
        return None


@tool_wrapper(required_params=["datetime_str", "from_tz", "to_tz"])
def timezone_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime between timezones."""
    status.set_callback(params.pop("_status_callback", None))
    dt = _parse_dt(params["datetime_str"])
    if dt is None:
        return tool_error(
            f"Cannot parse datetime: {params['datetime_str']}. Use YYYY-MM-DD HH:MM format."
        )
    fo = _get_offset(params["from_tz"])
    to = _get_offset(params["to_tz"])
    if fo is None:
        return tool_error(
            f"Unknown timezone: {params['from_tz']}. Supported: {', '.join(sorted(_TZ))}"
        )
    if to is None:
        return tool_error(
            f"Unknown timezone: {params['to_tz']}. Supported: {', '.join(sorted(_TZ))}"
        )
    utc_dt = dt - timedelta(hours=fo)
    result_dt = utc_dt + timedelta(hours=to)
    return tool_response(
        original=dt.strftime("%Y-%m-%d %H:%M:%S"),
        converted=result_dt.strftime("%Y-%m-%d %H:%M:%S"),
        from_tz=params["from_tz"].upper(),
        to_tz=params["to_tz"].upper(),
        utc=utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
        offset_diff=to - fo,
    )


__all__ = ["timezone_convert_tool"]
