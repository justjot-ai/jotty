"""Generate markdown tables from structured data."""
from typing import Dict, Any, List, Optional
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("markdown-table-generator")

_ALIGN = {"left": ":---", "right": "---:", "center": ":---:"}


@tool_wrapper(required_params=["data"])
def generate_markdown_table(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a markdown table from a list of dicts."""
    status.set_callback(params.pop("_status_callback", None))
    data: List[Dict] = params["data"]
    if not data or not isinstance(data, list):
        return tool_error("data must be a non-empty list of dicts")
    align: Dict[str, str] = params.get("align", {})
    headers: Optional[List[str]] = params.get("headers")
    if not headers:
        headers = list(data[0].keys())

    # Calculate column widths
    widths = {h: len(str(h)) for h in headers}
    for row in data:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def pad(val: str, h: str) -> str:
        w = widths[h]
        a = align.get(h, "left")
        if a == "right":
            return val.rjust(w)
        elif a == "center":
            return val.center(w)
        return val.ljust(w)

    header_line = "| " + " | ".join(pad(str(h), h) for h in headers) + " |"
    sep_parts = []
    for h in headers:
        a = align.get(h, "left")
        sep_parts.append(_ALIGN.get(a, "---").ljust(widths[h], "-"))
    sep_line = "| " + " | ".join(sep_parts) + " |"
    rows = []
    for row in data:
        cells = " | ".join(pad(str(row.get(h, "")), h) for h in headers)
        rows.append(f"| {cells} |")
    table = "\n".join([header_line, sep_line] + rows)
    return tool_response(markdown=table, rows=len(data), columns=len(headers))


__all__ = ["generate_markdown_table"]
