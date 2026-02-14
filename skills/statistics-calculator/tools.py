"""Statistics calculator — descriptive stats using stdlib statistics."""
import statistics as st
import math
from typing import Dict, Any, List
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("statistics-calculator")

def _percentile(data: List[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f, c = int(math.floor(k)), int(math.ceil(k))
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

@tool_wrapper(required_params=["numbers"])
def statistics_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute descriptive statistics: mean, median, mode, stdev, variance, percentiles."""
    status.set_callback(params.pop("_status_callback", None))
    nums = [float(x) for x in params["numbers"]]
    if not nums:
        return tool_error("numbers list is empty")
    op = params.get("operation", "summary").lower()
    try:
        if op == "mean":
            return tool_response(result=st.mean(nums))
        if op == "median":
            return tool_response(result=st.median(nums))
        if op == "mode":
            return tool_response(result=st.mode(nums))
        if op == "stdev":
            return tool_response(result=st.stdev(nums) if len(nums) > 1 else 0.0)
        if op == "variance":
            return tool_response(result=st.variance(nums) if len(nums) > 1 else 0.0)
        if op == "percentile":
            p = float(params.get("percentile", 50))
            return tool_response(result=_percentile(nums, p))
        if op == "quartiles":
            return tool_response(q1=_percentile(nums, 25), q2=_percentile(nums, 50), q3=_percentile(nums, 75))
        # summary
        result = {
            "count": len(nums), "mean": st.mean(nums), "median": st.median(nums),
            "min": min(nums), "max": max(nums),
            "stdev": st.stdev(nums) if len(nums) > 1 else 0.0,
            "variance": st.variance(nums) if len(nums) > 1 else 0.0,
            "q1": _percentile(nums, 25), "q3": _percentile(nums, 75),
        }
        return tool_response(**result)
    except Exception as e:
        return tool_error(str(e))

__all__ = ["statistics_tool"]
