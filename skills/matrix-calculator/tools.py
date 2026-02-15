"""Matrix calculator — add, multiply, transpose, determinant, inverse."""

from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("matrix-calculator")


def _det(m: List[List[float]]) -> float:
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    d = 0.0
    for c in range(n):
        sub = [[m[r][j] for j in range(n) if j != c] for r in range(1, n)]
        d += ((-1) ** c) * m[0][c] * _det(sub)
    return d


def _transpose(m: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*m)]


def _minor(m: List[List[float]], i: int, j: int) -> List[List[float]]:
    return [[m[r][c] for c in range(len(m)) if c != j] for r in range(len(m)) if r != i]


def _inverse(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    d = _det(m)
    if abs(d) < 1e-12:
        raise ValueError("Matrix is singular")
    if n == 1:
        return [[1.0 / d]]
    cofactors = [[(-1) ** (i + j) * _det(_minor(m, i, j)) for j in range(n)] for i in range(n)]
    adj = _transpose(cofactors)
    return [[adj[i][j] / d for j in range(n)] for i in range(n)]


@tool_wrapper(required_params=["operation"])
def matrix_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform matrix operations: add, subtract, multiply, transpose, determinant, inverse."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    a = params.get("matrix_a")
    b = params.get("matrix_b")
    if not a:
        return tool_error("matrix_a is required")
    try:
        if op == "transpose":
            return tool_response(result=_transpose(a))
        if op == "determinant":
            return tool_response(result=_det(a))
        if op == "inverse":
            return tool_response(result=_inverse(a))
        if op in ("add", "subtract"):
            if not b:
                return tool_error("matrix_b required for add/subtract")
            sign = 1 if op == "add" else -1
            result = [[a[i][j] + sign * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
            return tool_response(result=result)
        if op == "multiply":
            if not b:
                return tool_error("matrix_b required for multiply")
            ra, ca, cb = len(a), len(a[0]), len(b[0])
            result = [
                [sum(a[i][k] * b[k][j] for k in range(ca)) for j in range(cb)] for i in range(ra)
            ]
            return tool_response(result=result)
        return tool_error(
            f"Unknown operation: {op}. Use add/subtract/multiply/transpose/determinant/inverse"
        )
    except Exception as e:
        return tool_error(str(e))


__all__ = ["matrix_tool"]
