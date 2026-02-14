"""Semver Manager Skill — parse, compare, bump semantic versions."""
import re
from typing import Dict, Any, Tuple, Optional

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("semver-manager")

_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([\w.]+))?(?:\+([\w.]+))?$")


def _parse(v: str) -> Tuple[int, int, int, Optional[str], Optional[str]]:
    m = _RE.match(v.strip())
    if not m:
        raise ValueError(f"Invalid semver: {v}")
    return int(m[1]), int(m[2]), int(m[3]), m[4], m[5]


def _fmt(ma: int, mi: int, pa: int, pre: Optional[str] = None, bld: Optional[str] = None) -> str:
    s = f"{ma}.{mi}.{pa}"
    if pre:
        s += f"-{pre}"
    if bld:
        s += f"+{bld}"
    return s


def _cmp(a: str, b: str) -> int:
    a1, a2, a3, _, _ = _parse(a)
    b1, b2, b3, _, _ = _parse(b)
    for x, y in [(a1, b1), (a2, b2), (a3, b3)]:
        if x != y:
            return 1 if x > y else -1
    return 0


@tool_wrapper(required_params=["action", "version"])
def semver_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Manage semantic versions."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]
    ver = params["version"]

    if action == "parse":
        ma, mi, pa, pre, bld = _parse(ver)
        return tool_response(major=ma, minor=mi, patch=pa, prerelease=pre, build=bld)

    if action == "bump":
        ma, mi, pa, _, _ = _parse(ver)
        part = params.get("part", "patch")
        if part == "major":
            ma, mi, pa = ma + 1, 0, 0
        elif part == "minor":
            mi, pa = mi + 1, 0
        else:
            pa += 1
        return tool_response(original=ver, bumped=_fmt(ma, mi, pa), part=part)

    if action == "compare":
        other = params.get("other", "")
        if not other:
            return tool_error("other parameter required for compare")
        c = _cmp(ver, other)
        rel = "equal" if c == 0 else ("greater" if c > 0 else "less")
        return tool_response(version=ver, other=other, result=c, relation=rel)

    if action == "satisfies":
        constraint = params.get("constraint", "")
        if not constraint:
            return tool_error("constraint parameter required")
        m = re.match(r"^([><=!]+)\s*(.+)$", constraint.strip())
        if not m:
            return tool_error(f"Invalid constraint: {constraint}")
        op, cv = m[1], m[2]
        c = _cmp(ver, cv)
        ok = {">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0,
              "==": c == 0, "!=": c != 0, "=": c == 0}.get(op)
        if ok is None:
            return tool_error(f"Unknown operator: {op}")
        return tool_response(version=ver, constraint=constraint, satisfies=ok)

    return tool_error(f"Unknown action: {action}. Use: parse, bump, compare, satisfies")


__all__ = ["semver_tool"]
