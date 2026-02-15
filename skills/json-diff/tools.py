"""JSON Diff Skill — compare two JSON objects with path tracking."""
import json
from typing import Dict, Any, List, Tuple

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("json-diff")


def _diff(a: Any, b: Any, path: str = "$") -> Tuple[List, List, List]:
    added, removed, modified = [], [], []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            p = f"{path}.{k}"
            if k not in a:
                added.append({"path": p, "value": b[k]})
            elif k not in b:
                removed.append({"path": p, "value": a[k]})
            else:
                a2, r2, m2 = _diff(a[k], b[k], p)
                added.extend(a2); removed.extend(r2); modified.extend(m2)
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            p = f"{path}[{i}]"
            if i >= len(a):
                added.append({"path": p, "value": b[i]})
            elif i >= len(b):
                removed.append({"path": p, "value": a[i]})
            else:
                a2, r2, m2 = _diff(a[i], b[i], p)
                added.extend(a2); removed.extend(r2); modified.extend(m2)
    elif a != b:
        modified.append({"path": path, "old": a, "new": b})
    return added, removed, modified


def _parse(v: Any) -> Any:
    if isinstance(v, str):
        return json.loads(v)
    return v


@tool_wrapper(required_params=["a", "b"])
def json_diff_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two JSON objects and report differences."""
    status.set_callback(params.pop("_status_callback", None))
    a = _parse(params["a"])
    b = _parse(params["b"])
    added, removed, modified = _diff(a, b)
    return tool_response(
        added=added, removed=removed, modified=modified,
        total_changes=len(added) + len(removed) + len(modified),
    )


__all__ = ["json_diff_tool"]
