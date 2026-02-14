"""Date Calculator Skill — date arithmetic and formatting."""
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("date-calculator")

FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
           "%Y-%m-%dT%H:%M:%S%z", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
           "%Y%m%d", "%d-%b-%Y"]


def _parse_date(s: str) -> datetime:
    for fmt in FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}. Use YYYY-MM-DD format.")


@tool_wrapper(required_params=["date_a", "date_b"])
def date_diff_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate difference between two dates."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        a = _parse_date(params["date_a"])
        b = _parse_date(params["date_b"])
    except ValueError as e:
        return tool_error(str(e))

    delta = abs(b - a)
    days = delta.days
    return tool_response(days=days, weeks=round(days / 7, 1),
                         months=round(days / 30.44, 1), years=round(days / 365.25, 2))


@tool_wrapper(required_params=["date"])
def date_add_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add or subtract days/weeks/months from a date."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        dt = _parse_date(params["date"])
    except ValueError as e:
        return tool_error(str(e))

    days = int(params.get("days", 0))
    weeks = int(params.get("weeks", 0))
    result = dt + timedelta(days=days, weeks=weeks)

    return tool_response(result=result.strftime("%Y-%m-%d"), original=params["date"],
                         added_days=days + weeks * 7)


@tool_wrapper()
def now_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get current date and time in various formats."""
    status.set_callback(params.pop("_status_callback", None))
    tz_name = params.get("timezone", "UTC")
    now = datetime.now(timezone.utc)
    return tool_response(
        iso=now.isoformat(), date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"), timestamp=int(now.timestamp()),
        day_of_week=now.strftime("%A"), timezone="UTC",
    )


__all__ = ["date_diff_tool", "date_add_tool", "now_tool"]
