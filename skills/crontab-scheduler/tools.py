"""Convert between human-readable schedules and cron expressions."""

import re
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("crontab-scheduler")

_PATTERNS = [
    (r"every\s+minute", "* * * * *"),
    (r"every\s+(\d+)\s+minutes?", "{0}/{1} * * * *"),
    (r"every\s+hour", "0 * * * *"),
    (r"every\s+(\d+)\s+hours?", "0 */{1} * * *"),
    (r"every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", "_daily"),
    (
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        "_weekly",
    ),
    (r"every\s+month\s+on\s+day\s+(\d{1,2})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", "_monthly"),
]
_DAYS = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 0,
}
_FIELD_NAMES = ["minute", "hour", "day of month", "month", "day of week"]


def _resolve_hour(h: str, m: str, ampm: str) -> tuple:
    hour = int(h)
    minute = int(m) if m else 0
    if ampm:
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
    return minute, hour


@tool_wrapper(required_params=["schedule"])
def schedule_to_cron(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a human-readable schedule to a cron expression."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["schedule"].strip().lower()

    # every N minutes
    m = re.match(r"every\s+(\d+)\s+minutes?", text)
    if m:
        return tool_response(cron=f"*/{m.group(1)} * * * *", description=text)
    if re.match(r"every\s+minute", text):
        return tool_response(cron="* * * * *", description=text)

    # every N hours
    m = re.match(r"every\s+(\d+)\s+hours?", text)
    if m:
        return tool_response(cron=f"0 */{m.group(1)} * * *", description=text)
    if re.match(r"every\s+hour", text):
        return tool_response(cron="0 * * * *", description=text)

    # every day at H:MM am/pm
    m = re.match(r"every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if m:
        minute, hour = _resolve_hour(m.group(1), m.group(2), m.group(3))
        return tool_response(cron=f"{minute} {hour} * * *", description=text)

    # every <weekday> at H
    m = re.match(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text,
    )
    if m:
        minute, hour = _resolve_hour(m.group(2), m.group(3), m.group(4))
        dow = _DAYS[m.group(1)]
        return tool_response(cron=f"{minute} {hour} * * {dow}", description=text)

    # every month on day N at H
    m = re.match(
        r"every\s+month\s+on\s+day\s+(\d{1,2})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text
    )
    if m:
        minute, hour = _resolve_hour(m.group(2), m.group(3), m.group(4))
        return tool_response(cron=f"{minute} {hour} {m.group(1)} * *", description=text)

    return tool_error(f"Could not parse schedule: {text}")


@tool_wrapper(required_params=["cron"])
def cron_to_human(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a cron expression to human-readable text."""
    status.set_callback(params.pop("_status_callback", None))
    parts = params["cron"].strip().split()
    if len(parts) != 5:
        return tool_error("Cron expression must have exactly 5 fields")
    mi, hr, dom, mon, dow = parts
    pieces = []
    if mi == "*" and hr == "*":
        pieces.append("every minute")
    elif mi.startswith("*/"):
        pieces.append(f"every {mi[2:]} minutes")
    elif hr.startswith("*/"):
        pieces.append(f"every {hr[2:]} hours at minute {mi}")
    else:
        h = int(hr) if hr != "*" else None
        m = int(mi) if mi != "*" else 0
        if h is not None:
            ampm = "AM" if h < 12 else "PM"
            display_h = h % 12 or 12
            pieces.append(f"at {display_h}:{m:02d} {ampm}")
    if dow != "*":
        day_names = {v: k for k, v in _DAYS.items()}
        pieces.append(f"on {day_names.get(int(dow), dow)}")
    if dom != "*":
        pieces.append(f"on day {dom} of the month")
    if mon != "*":
        pieces.append(f"in month {mon}")
    return tool_response(human=" ".join(pieces) if pieces else params["cron"], cron=params["cron"])


__all__ = ["schedule_to_cron", "cron_to_human"]
