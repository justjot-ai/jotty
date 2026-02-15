"""Convert between CSV and JSON formats."""
import csv
import io
import json
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("csv-to-json")


def _infer_type(val: str) -> Any:
    """Try to convert string to int, float, bool, or keep as str."""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


@tool_wrapper(required_params=["csv_text"])
def csv_to_json(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CSV text to JSON array of objects."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["csv_text"]
    delimiter = params.get("delimiter", ",")
    infer_types = params.get("infer_types", True)
    has_headers = params.get("has_headers", True)

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return tool_error("CSV is empty")
    if has_headers:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        headers = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows
    records: List[Dict] = []
    for row in data_rows:
        rec = {}
        for i, h in enumerate(headers):
            val = row[i] if i < len(row) else ""
            rec[h] = _infer_type(val) if infer_types else val
        records.append(rec)
    return tool_response(records=records, rows=len(records), columns=len(headers),
                         json=json.dumps(records, indent=2))


@tool_wrapper(required_params=["json_data"])
def json_to_csv(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON array of objects to CSV text."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["json_data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON: {e}")
    if not isinstance(data, list) or not data:
        return tool_error("json_data must be a non-empty list of objects")
    delimiter = params.get("delimiter", ",")
    headers = list(data[0].keys())
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter)
    writer.writerow(headers)
    for row in data:
        writer.writerow([row.get(h, "") for h in headers])
    return tool_response(csv=buf.getvalue(), rows=len(data), columns=len(headers))


__all__ = ["csv_to_json", "json_to_csv"]
