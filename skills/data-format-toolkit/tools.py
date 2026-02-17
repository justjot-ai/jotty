"""
Data Format Toolkit — Unified data format conversion and validation skill.

Consolidates: csv-to-json, json-transformer, toml-parser, yaml-validator,
csv-analyzer, json-diff, markdown-to-html, markdown-table-generator, html-table-generator.
"""

import csv
import io
import json
import re
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("data-format-toolkit")


# =============================================================================
# CSV <-> JSON
# =============================================================================


@tool_wrapper(required_params=["csv_text"])
def csv_to_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CSV text to JSON array of objects."""
    status.set_callback(params.pop("_status_callback", None))
    csv_text = params["csv_text"]
    delimiter = params.get("delimiter", ",")
    has_header = params.get("has_header", True)
    try:
        reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return tool_response(data=[], count=0)
        if has_header:
            headers = rows[0]
            data = [dict(zip(headers, row)) for row in rows[1:]]
        else:
            data = [{"col_" + str(i): v for i, v in enumerate(row)} for row in rows]
        return tool_response(data=data, count=len(data), columns=len(rows[0]) if rows else 0)
    except Exception as e:
        return tool_error(f"CSV parse error: {e}")


@tool_wrapper(required_params=["data"])
def json_to_csv_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON array of objects to CSV text."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    if not isinstance(data, list) or not data:
        return tool_error("data must be a non-empty array of objects")
    try:
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            return tool_response(csv_text=output.getvalue(), rows=len(data), columns=len(headers))
        else:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(data)
            return tool_response(csv_text=output.getvalue(), rows=len(data))
    except Exception as e:
        return tool_error(f"JSON to CSV error: {e}")


# =============================================================================
# YAML / TOML
# =============================================================================


@tool_wrapper(required_params=["yaml_text"])
def yaml_validate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate YAML, returning structured data or error details."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        import yaml
    except ImportError:
        return tool_error("PyYAML not installed. Run: pip install pyyaml")
    try:
        data = yaml.safe_load(params["yaml_text"])
        return tool_response(valid=True, data=data, type=type(data).__name__)
    except yaml.YAMLError as e:
        error_info: Dict[str, Any] = {"message": str(e)}
        if hasattr(e, "problem_mark") and e.problem_mark:
            error_info["line"] = e.problem_mark.line + 1
            error_info["column"] = e.problem_mark.column + 1
        return tool_response(valid=False, error=error_info)


@tool_wrapper(required_params=["toml_text"])
def toml_parse_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse TOML text and return structured data."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return tool_error("TOML parser not available. Use Python 3.11+ or: pip install tomli")
    try:
        data = tomllib.loads(params["toml_text"])
        return tool_response(valid=True, data=data, sections=list(data.keys()))
    except Exception as e:
        return tool_response(valid=False, error=str(e))


# =============================================================================
# JSON DIFF
# =============================================================================


def _json_diff(a: Any, b: Any, path: str = "") -> List[Dict[str, Any]]:
    """Recursively diff two JSON-like objects."""
    diffs: List[Dict[str, Any]] = []
    if type(a) != type(b):
        diffs.append(
            {
                "path": path or "/",
                "type": "type_change",
                "old": str(type(a).__name__),
                "new": str(type(b).__name__),
                "old_value": a,
                "new_value": b,
            }
        )
        return diffs
    if isinstance(a, dict):
        all_keys = set(list(a.keys()) + list(b.keys()))
        for key in sorted(all_keys):
            child_path = f"{path}/{key}"
            if key not in a:
                diffs.append({"path": child_path, "type": "added", "value": b[key]})
            elif key not in b:
                diffs.append({"path": child_path, "type": "removed", "value": a[key]})
            else:
                diffs.extend(_json_diff(a[key], b[key], child_path))
    elif isinstance(a, list):
        for i in range(max(len(a), len(b))):
            child_path = f"{path}[{i}]"
            if i >= len(a):
                diffs.append({"path": child_path, "type": "added", "value": b[i]})
            elif i >= len(b):
                diffs.append({"path": child_path, "type": "removed", "value": a[i]})
            else:
                diffs.extend(_json_diff(a[i], b[i], child_path))
    elif a != b:
        diffs.append({"path": path or "/", "type": "changed", "old": a, "new": b})
    return diffs


@tool_wrapper(required_params=["a", "b"])
def json_diff_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two JSON objects and return detailed differences."""
    status.set_callback(params.pop("_status_callback", None))
    a = params["a"]
    b = params["b"]
    diffs = _json_diff(a, b)
    return tool_response(
        differences=diffs[:100],
        count=len(diffs),
        identical=len(diffs) == 0,
        added=sum(1 for d in diffs if d["type"] == "added"),
        removed=sum(1 for d in diffs if d["type"] == "removed"),
        changed=sum(1 for d in diffs if d["type"] == "changed"),
    )


# =============================================================================
# TABLE GENERATORS
# =============================================================================


@tool_wrapper(required_params=["headers", "rows"])
def markdown_table_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a markdown table from headers and row data."""
    status.set_callback(params.pop("_status_callback", None))
    headers = params["headers"]
    rows = params["rows"]
    alignment = params.get("alignment", ["left"] * len(headers))
    align_map = {"left": ":---", "center": ":---:", "right": "---:"}
    sep = [align_map.get(a, ":---") for a in alignment[: len(headers)]]
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return tool_response(table="\n".join(lines), rows=len(rows), columns=len(headers))


@tool_wrapper(required_params=["headers", "rows"])
def html_table_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an HTML table from headers and row data."""
    status.set_callback(params.pop("_status_callback", None))
    headers = params["headers"]
    rows = params["rows"]
    css_class = params.get("css_class", "")
    cls = f' class="{css_class}"' if css_class else ""
    lines = [f"<table{cls}>", "  <thead>", "    <tr>"]
    for h in headers:
        lines.append(f"      <th>{h}</th>")
    lines += ["    </tr>", "  </thead>", "  <tbody>"]
    for row in rows:
        lines.append("    <tr>")
        for cell in row:
            lines.append(f"      <td>{cell}</td>")
        lines.append("    </tr>")
    lines += ["  </tbody>", "</table>"]
    return tool_response(html="\n".join(lines), rows=len(rows), columns=len(headers))


# =============================================================================
# MARKDOWN TO HTML
# =============================================================================


@tool_wrapper(required_params=["markdown"])
def markdown_to_html_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert markdown text to HTML."""
    status.set_callback(params.pop("_status_callback", None))
    md = params["markdown"]
    try:
        import markdown as md_lib

        extensions = params.get("extensions", ["tables", "fenced_code"])
        html = md_lib.markdown(md, extensions=extensions)
        return tool_response(html=html, length=len(html))
    except ImportError:
        html = md
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        paragraphs = html.split("\n\n")
        html = "\n".join(
            f"<p>{p.strip()}</p>" if not p.strip().startswith("<") else p.strip()
            for p in paragraphs
            if p.strip()
        )
        return tool_response(
            html=html,
            length=len(html),
            note="Basic conversion (install 'markdown' for full support)",
        )


__all__ = [
    "csv_to_json_tool",
    "json_to_csv_tool",
    "yaml_validate_tool",
    "toml_parse_tool",
    "json_diff_tool",
    "markdown_table_tool",
    "html_table_tool",
    "markdown_to_html_tool",
]
