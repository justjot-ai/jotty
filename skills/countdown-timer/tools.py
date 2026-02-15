"""Countdown timer — days/hours/minutes to a target date."""

from datetime import datetime, timedelta
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("countdown-timer")

_HOLIDAYS = {
    "new year": "01-01",
    "valentine": "02-14",
    "st patrick": "03-17",
    "easter": "04-20",
    "mother": "05-11",
    "father": "06-15",
    "independence day": "07-04",
    "halloween": "10-31",
    "thanksgiving": "11-27",
    "christmas": "12-25",
    "new year eve": "12-31",
}
_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%dT%H:%M:%S",
]


def _parse_target(s: str) -> datetime | None:
    low = s.lower().strip()
    for name, md in _HOLIDAYS.items():
        if name in low:
            now = datetime.now()
            dt = datetime.strptime(f"{now.year}-{md}", "%Y-%m-%d")
            if dt < now:
                dt = dt.replace(year=now.year + 1)
            return dt
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


@tool_wrapper(required_params=["target_date"])
def countdown_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate time remaining until target date."""
    status.set_callback(params.pop("_status_callback", None))
    target = _parse_target(params["target_date"])
    if target is None:
        return tool_error(
            f"Cannot parse date: {params['target_date']}. Use YYYY-MM-DD or a holiday name."
        )
    now = datetime.now()
    delta = target - now
    total_sec = int(delta.total_seconds())
    is_past = total_sec < 0
    total_sec = abs(total_sec)
    days = total_sec // 86400
    hours = (total_sec % 86400) // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    weeks = days // 7
    return tool_response(
        target=target.strftime("%Y-%m-%d %H:%M:%S"),
        is_past=is_past,
        total_days=days,
        weeks=weeks,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        human=f"{'Passed ' if is_past else ''}{days}d {hours}h {minutes}m {seconds}s{'ago' if is_past else ''}",
    )


__all__ = ["countdown_tool"]
