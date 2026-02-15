"""Algorithm complexity — Big-O analysis, comparison, ASCII plots."""

import math
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("algorithm-complexity")

COMPLEXITIES = {
    "binary_search": {"best": "O(1)", "average": "O(log n)", "worst": "O(log n)", "space": "O(1)"},
    "linear_search": {"best": "O(1)", "average": "O(n)", "worst": "O(n)", "space": "O(1)"},
    "bubble_sort": {"best": "O(n)", "average": "O(n^2)", "worst": "O(n^2)", "space": "O(1)"},
    "insertion_sort": {"best": "O(n)", "average": "O(n^2)", "worst": "O(n^2)", "space": "O(1)"},
    "merge_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },
    "quick_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n^2)",
        "space": "O(log n)",
    },
    "heap_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(1)",
    },
    "hash_table_lookup": {"best": "O(1)", "average": "O(1)", "worst": "O(n)", "space": "O(n)"},
    "bst_search": {"best": "O(log n)", "average": "O(log n)", "worst": "O(n)", "space": "O(n)"},
    "bfs": {"best": "O(V+E)", "average": "O(V+E)", "worst": "O(V+E)", "space": "O(V)"},
    "dfs": {"best": "O(V+E)", "average": "O(V+E)", "worst": "O(V+E)", "space": "O(V)"},
    "dijkstra": {
        "best": "O(V+E log V)",
        "average": "O(V+E log V)",
        "worst": "O(V+E log V)",
        "space": "O(V)",
    },
}

GROWTH_FNS = {
    "O(1)": lambda n: 1,
    "O(log n)": lambda n: math.log2(max(n, 1)),
    "O(n)": lambda n: n,
    "O(n log n)": lambda n: n * math.log2(max(n, 1)),
    "O(n^2)": lambda n: n * n,
    "O(n^3)": lambda n: n**3,
    "O(2^n)": lambda n: 2 ** min(n, 30),
    "O(n!)": lambda n: math.factorial(min(n, 12)),
}


def _ascii_plot(fns: List[str], max_n: int = 20, height: int = 15) -> str:
    points = {}
    for fn_name in fns:
        fn = GROWTH_FNS.get(fn_name)
        if fn:
            points[fn_name] = [fn(n) for n in range(1, max_n + 1)]
    if not points:
        return "No valid functions to plot"
    all_vals = [v for vals in points.values() for v in vals]
    max_val = max(all_vals) if all_vals else 1
    lines = []
    for row in range(height, -1, -1):
        threshold = max_val * row / height
        line = f"{threshold:>8.0f} |"
        for col in range(max_n):
            chars = []
            for i, fn_name in enumerate(fns):
                if fn_name in points and col < len(points[fn_name]):
                    if abs(points[fn_name][col] - threshold) <= max_val / height / 2 or (
                        row == 0 and points[fn_name][col] <= threshold + max_val / height
                    ):
                        chars.append("*#@$%"[i % 5])
            line += chars[0] if chars else " "
        lines.append(line)
    lines.append("         +" + "-" * max_n)
    lines.append("          " + "".join(str(i % 10) for i in range(1, max_n + 1)) + " -> n")
    legend = "  Legend: " + ", ".join(
        f"{'*#@$%'[i % 5]}={fn}" for i, fn in enumerate(fns) if fn in GROWTH_FNS
    )
    lines.append(legend)
    return "\n".join(lines)


@tool_wrapper(required_params=["operation"])
def complexity_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Big-O analysis: lookup, compare, plot, list."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    try:
        if op == "lookup":
            algo = params.get("algorithm", "").lower().replace(" ", "_")
            if algo not in COMPLEXITIES:
                return tool_error(
                    f"Unknown algorithm: {algo}. Known: {', '.join(COMPLEXITIES.keys())}"
                )
            return tool_response(algorithm=algo, **COMPLEXITIES[algo])
        if op == "compare":
            algos = params.get("algorithms", [])
            results = {}
            for a in algos:
                key = a.lower().replace(" ", "_")
                if key in COMPLEXITIES:
                    results[key] = COMPLEXITIES[key]
            return tool_response(comparison=results)
        if op == "plot":
            fns = params.get("functions", ["O(n)", "O(n log n)", "O(n^2)"])
            max_n = int(params.get("max_n", 20))
            chart = _ascii_plot(fns, max_n)
            return tool_response(plot=chart, functions=fns)
        if op == "list":
            return tool_response(
                algorithms=list(COMPLEXITIES.keys()), growth_functions=list(GROWTH_FNS.keys())
            )
        if op == "classify":
            expr = params.get("expression", "")
            for name in [
                "O(1)",
                "O(log n)",
                "O(n)",
                "O(n log n)",
                "O(n^2)",
                "O(n^3)",
                "O(2^n)",
                "O(n!)",
            ]:
                if name.lower() in expr.lower():
                    category = "constant" if "1" == name[2:-1] else name[2:-1].replace(" ", "-")
                    return tool_response(expression=expr, complexity=name, category=category)
            return tool_response(
                expression=expr,
                complexity="unknown",
                note="Could not classify. Provide standard Big-O notation.",
            )
        return tool_error(f"Unknown op: {op}. Use lookup/compare/plot/list/classify")
    except Exception as e:
        return tool_error(str(e))


__all__ = ["complexity_tool"]
