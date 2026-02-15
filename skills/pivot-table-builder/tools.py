"""Pivot Table Builder Skill — create pivot tables (pure Python)."""
import statistics
from typing import Dict, Any, List, Callable

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("pivot-table-builder")

AGG_FUNCS = {
    "sum": sum,
    "mean": lambda vals: statistics.mean(vals) if vals else 0,
    "count": len,
    "min": min,
    "max": max,
    "median": lambda vals: statistics.median(vals) if vals else 0,
}


@tool_wrapper(required_params=["data", "rows", "values"])
def pivot_table_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a pivot table from data."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    row_field = params["rows"]
    col_field = params.get("columns")
    val_field = params["values"]
    agg_name = params.get("aggfunc", "sum").lower()

    if not isinstance(data, list):
        return tool_error("data must be a list of objects")
    if agg_name not in AGG_FUNCS:
        return tool_error(f"aggfunc must be one of: {list(AGG_FUNCS.keys())}")

    agg_func = AGG_FUNCS[agg_name]

    # Collect values into buckets
    buckets: Dict[str, Dict[str, List[float]]] = {}
    col_keys = set()

    for record in data:
        if not isinstance(record, dict):
            continue
        row_key = str(record.get(row_field, ""))
        val = record.get(val_field)
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        if col_field:
            col_key = str(record.get(col_field, ""))
            col_keys.add(col_key)
        else:
            col_key = val_field

        if row_key not in buckets:
            buckets[row_key] = {}
        if col_key not in buckets[row_key]:
            buckets[row_key][col_key] = []
        buckets[row_key][col_key].append(val)

    # Aggregate
    table = {}
    for row_key, cols in sorted(buckets.items()):
        table[row_key] = {}
        for col_key, vals in sorted(cols.items()):
            result = agg_func(vals)
            table[row_key][col_key] = round(result, 2) if isinstance(result, float) else result

    # Format as ASCII table
    all_cols = sorted(col_keys) if col_field else [val_field]
    col_width = max(max((len(c) for c in all_cols), default=5), 10)
    row_width = max(max((len(r) for r in table.keys()), default=5), 10)

    header = f"{'':<{row_width}} | " + " | ".join(f"{c:>{col_width}}" for c in all_cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for row_key in sorted(table.keys()):
        vals = " | ".join(
            f"{table[row_key].get(c, 0):>{col_width}}" if isinstance(table[row_key].get(c, 0), int)
            else f"{table[row_key].get(c, 0):>{col_width}.2f}"
            for c in all_cols
        )
        lines.append(f"{row_key:<{row_width}} | {vals}")

    formatted = "\n".join(lines)
    return tool_response(table=table, formatted=formatted,
                         rows=len(table), columns=len(all_cols), aggfunc=agg_name)


__all__ = ["pivot_table_tool"]
