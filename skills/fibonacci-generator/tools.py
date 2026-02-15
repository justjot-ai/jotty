"""Fibonacci generator — sequence, membership check, golden ratio."""

import math
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("fibonacci-generator")


def _fib_seq(n: int) -> List[int]:
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def _is_fib(n: int) -> bool:
    if n < 0:
        return False

    def _is_perfect_square(x: int) -> bool:
        s = int(math.isqrt(x))
        return s * s == x

    return _is_perfect_square(5 * n * n + 4) or _is_perfect_square(5 * n * n - 4)


def _nth_fib(n: int) -> int:
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


@tool_wrapper(required_params=["operation"])
def fibonacci_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fibonacci operations: sequence, nth, is_fibonacci, golden_ratio."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    try:
        if op == "sequence":
            count = int(params.get("count", 10))
            if count > 1000:
                return tool_error("Count capped at 1000")
            seq = _fib_seq(count)
            return tool_response(sequence=seq, count=len(seq))
        if op == "nth":
            n = int(params.get("n", 1))
            return tool_response(n=n, fibonacci=_nth_fib(n))
        if op == "is_fibonacci":
            num = int(params.get("number", 0))
            return tool_response(number=num, is_fibonacci=_is_fib(num))
        if op == "golden_ratio":
            n = int(params.get("precision", 20))
            a, b = _nth_fib(n), _nth_fib(n - 1)
            ratio = a / b if b != 0 else float("inf")
            phi = (1 + math.sqrt(5)) / 2
            return tool_response(
                approximation=ratio, exact_phi=phi, terms_used=n, error=abs(ratio - phi)
            )
        return tool_error(f"Unknown op: {op}. Use sequence/nth/is_fibonacci/golden_ratio")
    except Exception as e:
        return tool_error(str(e))


__all__ = ["fibonacci_tool"]
