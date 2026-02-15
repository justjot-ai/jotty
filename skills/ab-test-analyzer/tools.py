"""A/B Test Analyzer Skill — calculate significance using stdlib math."""

import math
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("ab-test-analyzer")


def _normal_cdf(x: float) -> float:
    """Approximate the cumulative distribution function of the standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _z_to_p(z: float) -> float:
    """Convert z-score to two-tailed p-value."""
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def _confidence_to_z(confidence: float) -> float:
    """Get z critical value for confidence level."""
    alpha = 1.0 - confidence
    # Newton's method approximation for inverse normal
    # Common values
    z_table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576, 0.999: 3.291}
    if confidence in z_table:
        return z_table[confidence]
    # Approximation for other values
    p = alpha / 2.0
    t = math.sqrt(-2.0 * math.log(p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )


@tool_wrapper(required_params=["visitors_a", "conversions_a", "visitors_b", "conversions_b"])
def ab_test_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze A/B test for statistical significance."""
    status.set_callback(params.pop("_status_callback", None))
    n_a = int(params["visitors_a"])
    c_a = int(params["conversions_a"])
    n_b = int(params["visitors_b"])
    c_b = int(params["conversions_b"])
    confidence = float(params.get("confidence_level", 0.95))

    if n_a <= 0 or n_b <= 0:
        return tool_error("Visitor counts must be positive")
    if c_a < 0 or c_b < 0:
        return tool_error("Conversion counts must be non-negative")
    if c_a > n_a or c_b > n_b:
        return tool_error("Conversions cannot exceed visitors")

    rate_a = c_a / n_a
    rate_b = c_b / n_b

    # Pooled proportion
    p_pool = (c_a + c_b) / (n_a + n_b)
    se = (
        math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        if p_pool > 0 and p_pool < 1
        else 0.0001
    )

    z_score = (rate_b - rate_a) / se if se > 0 else 0
    p_value = _z_to_p(z_score)
    z_crit = _confidence_to_z(confidence)
    significant = abs(z_score) > z_crit

    # Confidence interval for difference
    se_diff = math.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    diff = rate_b - rate_a
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0

    return tool_response(
        significant=significant,
        p_value=round(p_value, 6),
        z_score=round(z_score, 4),
        rate_a=round(rate_a, 6),
        rate_b=round(rate_b, 6),
        lift=round(lift, 2),
        confidence_interval={"lower": round(ci_lower, 6), "upper": round(ci_upper, 6)},
        confidence_level=confidence,
        recommendation=(
            "B is better"
            if significant and z_score > 0
            else "A is better" if significant and z_score < 0 else "No significant difference"
        ),
    )


@tool_wrapper(required_params=["baseline_rate", "minimum_effect"])
def sample_size_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate required sample size for an A/B test."""
    status.set_callback(params.pop("_status_callback", None))
    p1 = float(params["baseline_rate"])
    mde = float(params["minimum_effect"])  # relative effect
    confidence = float(params.get("confidence_level", 0.95))
    power = float(params.get("power", 0.80))

    if not (0 < p1 < 1):
        return tool_error("baseline_rate must be between 0 and 1")

    p2 = p1 * (1 + mde)
    if not (0 < p2 < 1):
        return tool_error("minimum_effect produces invalid rate")

    z_alpha = _confidence_to_z(confidence)
    z_beta = _confidence_to_z(0.5 + power / 2)

    p_avg = (p1 + p2) / 2
    n = (
        (
            z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        / (p2 - p1)
    ) ** 2
    n = math.ceil(n)

    return tool_response(
        sample_size_per_group=n,
        total_sample_size=n * 2,
        baseline_rate=p1,
        expected_rate=round(p2, 6),
        minimum_effect=mde,
        confidence_level=confidence,
        power=power,
    )


__all__ = ["ab_test_tool", "sample_size_tool"]
