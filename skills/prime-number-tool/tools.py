"""Prime number tool — primality, sieve, factorization, nth prime."""

from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("prime-number-tool")


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _sieve(limit: int) -> List[int]:
    if limit < 2:
        return []
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i, v in enumerate(s) if v]


def _factorize(n: int) -> List[int]:
    if n < 2:
        return []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _nth_prime(n: int) -> int:
    if n < 1:
        return 2
    count, candidate = 0, 1
    while count < n:
        candidate += 1
        if _is_prime(candidate):
            count += 1
    return candidate


@tool_wrapper(required_params=["operation"])
def prime_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Prime operations: is_prime, sieve, factorize, nth_prime."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    try:
        if op == "is_prime":
            n = int(params.get("number", 0))
            return tool_response(number=n, is_prime=_is_prime(n))
        if op == "sieve":
            limit = int(params.get("limit", 100))
            if limit > 1_000_000:
                return tool_error("Limit capped at 1000000")
            return tool_response(primes=_sieve(limit), count=len(_sieve(limit)))
        if op == "factorize":
            n = int(params.get("number", 0))
            f = _factorize(n)
            return tool_response(number=n, factors=f, expression=" x ".join(map(str, f)))
        if op == "nth_prime":
            n = int(params.get("n", 1))
            if n > 10000:
                return tool_error("n capped at 10000")
            return tool_response(n=n, prime=_nth_prime(n))
        return tool_error(f"Unknown op: {op}. Use is_prime/sieve/factorize/nth_prime")
    except Exception as e:
        return tool_error(str(e))


__all__ = ["prime_tool"]
