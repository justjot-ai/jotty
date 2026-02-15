"""Tip calculator — amount, total, per-person split."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("tip-calculator")


@tool_wrapper(required_params=["bill_amount"])
def tip_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate tip amount, total, and per-person split."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        bill = float(params["bill_amount"])
    except (ValueError, TypeError):
        return tool_error("bill_amount must be a number")
    if bill < 0:
        return tool_error("bill_amount cannot be negative")
    pct = float(params.get("tip_percent", 18))
    people = int(params.get("num_people", 1))
    if people < 1:
        return tool_error("num_people must be at least 1")
    tip = round(bill * pct / 100, 2)
    total = round(bill + tip, 2)
    per_person = round(total / people, 2)
    suggestions = {}
    for p in [15, 18, 20, 25]:
        t = round(bill * p / 100, 2)
        suggestions[f"{p}%"] = {"tip": t, "total": round(bill + t, 2),
                                "per_person": round((bill + t) / people, 2)}
    return tool_response(
        bill_amount=bill, tip_percent=pct, tip_amount=tip,
        total=total, num_people=people, per_person=per_person,
        suggestions=suggestions,
    )


__all__ = ["tip_tool"]
