"""Decision Matrix Builder Skill - weighted multi-criteria comparison."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("decision-matrix-builder")


@tool_wrapper(required_params=["options", "criteria", "scores"])
def build_decision_matrix_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a weighted decision matrix to compare options."""
    status.set_callback(params.pop("_status_callback", None))
    options = params["options"]
    criteria = params["criteria"]
    scores = params["scores"]

    if len(options) < 2:
        return tool_error("Need at least 2 options to compare")
    if not criteria:
        return tool_error("Need at least 1 criterion")

    # Validate
    crit_names = [c["name"] for c in criteria]
    total_weight = sum(c.get("weight", 1) for c in criteria)

    results = []
    for option in options:
        if option not in scores:
            return tool_error(f"Missing scores for option: {option}")
        weighted_total = 0
        raw_total = 0
        detail = {}
        for crit in criteria:
            name = crit["name"]
            weight = crit.get("weight", 1)
            score = scores[option].get(name, 0)
            if not (0 <= score <= 10):
                return tool_error(f"Score for {option}/{name} must be 0-10, got {score}")
            ws = round(score * weight, 2)
            weighted_total += ws
            raw_total += score
            detail[name] = {"score": score, "weight": weight, "weighted": ws}

        normalized = round(weighted_total / total_weight, 2) if total_weight else 0
        results.append({
            "option": option,
            "weighted_total": round(weighted_total, 2),
            "normalized_score": normalized,
            "raw_total": raw_total,
            "detail": detail,
        })

    # Sort by weighted total descending
    results.sort(key=lambda x: x["weighted_total"], reverse=True)
    winner = results[0]["option"]

    # Build text matrix
    col_width = max(len(o) for o in options) + 2
    crit_width = max(len(c["name"]) for c in criteria) + 2
    header = "Criterion".ljust(crit_width) + "Wt  " + "  ".join(o.center(col_width) for o in options)
    sep = "-" * len(header)
    rows = [header, sep]
    for crit in criteria:
        name = crit["name"]
        weight = crit.get("weight", 1)
        row = name.ljust(crit_width) + f"{weight:<4}"
        for option in options:
            s = scores[option].get(name, 0)
            ws = round(s * weight, 2)
            row += f"{s}({ws})".center(col_width) + "  "
        rows.append(row)
    rows.append(sep)
    total_row = "TOTAL".ljust(crit_width) + "    "
    for r in sorted(results, key=lambda x: options.index(x["option"])):
        total_row += f"{r['weighted_total']}".center(col_width) + "  "
    rows.append(total_row)
    matrix_text = "\n".join(rows)

    return tool_response(
        results=results, winner=winner,
        matrix=matrix_text,
        criteria_count=len(criteria),
        option_count=len(options),
    )


__all__ = ["build_decision_matrix_tool"]
