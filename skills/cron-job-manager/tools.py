"""Cron Job Manager Skill — parse, explain, validate cron expressions."""
from datetime import datetime, timedelta
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("cron-job-manager")

FIELD_NAMES = ["minute", "hour", "day_of_month", "month", "day_of_week"]
FIELD_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DAY_NAMES = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}

COMMON = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0", "@daily": "0 0 * * *", "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}


def _explain_field(value: str, name: str) -> str:
    if value == "*":
        return f"every {name}"
    elif value.startswith("*/"):
        return f"every {value[2:]} {name}s"
    elif "," in value:
        return f"{name} {value}"
    elif "-" in value:
        parts = value.split("-")
        return f"{name} {parts[0]} through {parts[1]}"
    else:
        return f"{name} {value}"


@tool_wrapper(required_params=["expression"])
def explain_cron_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a cron expression in human-readable format."""
    status.set_callback(params.pop("_status_callback", None))
    expr = params["expression"].strip()
    expr = COMMON.get(expr, expr)
    parts = expr.split()

    if len(parts) not in (5, 6):
        return tool_error(f"Expected 5 or 6 fields, got {len(parts)}: {expr}")

    fields = parts[:5]
    explanations = [_explain_field(f, n) for f, n in zip(fields, FIELD_NAMES)]
    explanation = "Runs at " + ", ".join(explanations)

    return tool_response(expression=expr, explanation=explanation,
                         fields={n: f for n, f in zip(FIELD_NAMES, fields)})


@tool_wrapper(required_params=["expression"])
def validate_cron_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a cron expression."""
    status.set_callback(params.pop("_status_callback", None))
    expr = params["expression"].strip()
    expr = COMMON.get(expr, expr)
    parts = expr.split()

    if len(parts) not in (5, 6):
        return tool_response(valid=False, error=f"Expected 5 or 6 fields, got {len(parts)}")

    errors = []
    for i, (field, (lo, hi)) in enumerate(zip(parts[:5], FIELD_RANGES)):
        if field == "*":
            continue
        field_clean = field.replace("*/", "")
        for segment in field_clean.split(","):
            for part in segment.split("-"):
                try:
                    val = int(part)
                    if val < lo or val > hi:
                        errors.append(f"{FIELD_NAMES[i]}: {val} out of range [{lo}-{hi}]")
                except ValueError:
                    if part.lower() not in MONTH_NAMES and part.lower() not in DAY_NAMES:
                        errors.append(f"{FIELD_NAMES[i]}: invalid value '{part}'")

    return tool_response(valid=len(errors) == 0, errors=errors, expression=expr)


@tool_wrapper()
def build_cron_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a cron expression from human-readable description."""
    status.set_callback(params.pop("_status_callback", None))
    minute = params.get("minute", "*")
    hour = params.get("hour", "*")
    dom = params.get("day_of_month", "*")
    month = params.get("month", "*")
    dow = params.get("day_of_week", "*")

    expr = f"{minute} {hour} {dom} {month} {dow}"
    return tool_response(expression=expr)


__all__ = ["explain_cron_tool", "validate_cron_tool", "build_cron_tool"]
