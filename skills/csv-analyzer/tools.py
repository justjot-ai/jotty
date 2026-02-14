"""CSV Analyzer Skill — load, filter, aggregate CSV files."""
import csv
import io
import statistics
from pathlib import Path
from typing import Dict, Any, List

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("csv-analyzer")


def _read_csv(file_path: str, delimiter: str = ",", max_rows: int = 10000) -> tuple:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(p, "r", newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    return rows, reader.fieldnames or []


@tool_wrapper(required_params=["file_path"])
def csv_summary_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary statistics for a CSV file."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        rows, columns = _read_csv(params["file_path"], params.get("delimiter", ","))
    except FileNotFoundError as e:
        return tool_error(str(e))

    col_stats = {}
    for col in columns:
        values = [r.get(col, "") for r in rows]
        non_empty = [v for v in values if v.strip()]
        numeric = []
        for v in non_empty:
            try:
                numeric.append(float(v.replace(",", "")))
            except ValueError:
                pass
        info = {"non_null": len(non_empty), "null": len(values) - len(non_empty),
                "unique": len(set(non_empty))}
        if numeric:
            info.update({"mean": round(statistics.mean(numeric), 2),
                         "min": min(numeric), "max": max(numeric),
                         "median": round(statistics.median(numeric), 2)})
            if len(numeric) > 1:
                info["stdev"] = round(statistics.stdev(numeric), 2)
        else:
            top = max(set(non_empty), key=non_empty.count) if non_empty else None
            info["top_value"] = top
        col_stats[col] = info

    return tool_response(rows=len(rows), columns=columns, stats=col_stats)


@tool_wrapper(required_params=["file_path"])
def csv_filter_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter CSV rows by column value."""
    status.set_callback(params.pop("_status_callback", None))
    column = params.get("column")
    value = params.get("value")
    operator = params.get("operator", "equals")
    limit = int(params.get("limit", 100))
    if not column:
        return tool_error("column parameter required")

    try:
        rows, columns = _read_csv(params["file_path"], params.get("delimiter", ","))
    except FileNotFoundError as e:
        return tool_error(str(e))

    if column not in columns:
        return tool_error(f"Column '{column}' not found. Available: {columns}")

    filtered = []
    for row in rows:
        cell = row.get(column, "")
        match = False
        if operator == "equals":
            match = cell == str(value)
        elif operator == "contains":
            match = str(value).lower() in cell.lower()
        elif operator == "gt":
            try:
                match = float(cell) > float(value)
            except ValueError:
                pass
        elif operator == "lt":
            try:
                match = float(cell) < float(value)
            except ValueError:
                pass
        if match:
            filtered.append(row)
            if len(filtered) >= limit:
                break

    return tool_response(results=filtered, count=len(filtered), total_rows=len(rows))


__all__ = ["csv_summary_tool", "csv_filter_tool"]
