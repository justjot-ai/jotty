"""BMI calculator — value, category, healthy range."""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("bmi-calculator")


@tool_wrapper(required_params=["weight_kg", "height_m"])
def bmi_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate BMI from weight (kg) and height (m)."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        w = float(params["weight_kg"])
        h = float(params["height_m"])
    except (ValueError, TypeError):
        return tool_error("weight_kg and height_m must be numbers")
    if h <= 0 or w <= 0:
        return tool_error("height and weight must be positive")
    bmi = round(w / (h * h), 2)
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal weight"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    low = round(18.5 * h * h, 1)
    high = round(24.9 * h * h, 1)
    return tool_response(
        bmi=bmi,
        category=cat,
        healthy_weight_range_kg={"min": low, "max": high},
        weight_kg=w,
        height_m=h,
    )


__all__ = ["bmi_tool"]
