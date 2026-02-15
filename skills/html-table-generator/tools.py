"""Generate HTML tables from structured data."""

from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("html-table-generator")


@tool_wrapper(required_params=["data"])
def generate_html_table(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an HTML table from a list of dicts."""
    status.set_callback(params.pop("_status_callback", None))
    data: List[Dict] = params["data"]
    if not data or not isinstance(data, list):
        return tool_error("data must be a non-empty list of dicts")

    sort_by = params.get("sort_by")
    zebra = params.get("zebra", False)
    css_class = params.get("css_class", "")
    caption = params.get("caption", "")

    if sort_by and sort_by in data[0]:
        data = sorted(
            data, key=lambda r: r.get(sort_by, ""), reverse=params.get("descending", False)
        )

    headers = list(data[0].keys())
    cls = f' class="{css_class}"' if css_class else ""
    lines = [f"<table{cls}>"]
    if caption:
        lines.append(f"  <caption>{caption}</caption>")
    lines.append("  <thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
    lines.append("  <tbody>")
    for i, row in enumerate(data):
        style = ' style="background:#f2f2f2"' if zebra and i % 2 == 1 else ""
        empty = ""
        cells = "".join(f"<td>{row.get(h, empty)}</td>" for h in headers)
        lines.append(f"    <tr{style}>{cells}</tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    html = "\n".join(lines)
    return tool_response(html=html, rows=len(data), columns=len(headers))


__all__ = ["generate_html_table"]
