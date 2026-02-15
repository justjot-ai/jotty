"""Batch 5a — 10 utility skills with real implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_skills import create_skill

# ── 1. html-table-generator ──────────────────────────────────────────────────
create_skill(
    name="html-table-generator",
    frontmatter_name="html-table-generator",
    description="Generate HTML tables from data with styling, headers, sorting, and zebra stripes",
    category="data/formatting",
    capabilities=["Generate HTML tables from lists of dicts", "Apply zebra striping and custom styles", "Sort by column"],
    triggers=["generate html table", "create html table from data", "format data as html table"],
    eval_tool="generate_html_table",
    eval_input={"data": [{"name": "Alice", "age": 30}], "zebra": True},
    tool_docs="### generate_html_table\nGenerate an HTML table from a list of dicts.\n**Params:** data (list[dict]), sort_by (str), zebra (bool), css_class (str)",
    tools_code='''"""Generate HTML tables from structured data."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

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
        data = sorted(data, key=lambda r: r.get(sort_by, ""), reverse=params.get("descending", False))

    headers = list(data[0].keys())
    cls = f\' class="{css_class}"\' if css_class else ""
    lines = [f"<table{cls}>"]
    if caption:
        lines.append(f"  <caption>{caption}</caption>")
    lines.append("  <thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
    lines.append("  <tbody>")
    for i, row in enumerate(data):
        style = \' style="background:#f2f2f2"\' if zebra and i % 2 == 1 else ""
        cells = "".join(f"<td>{row.get(h, \\'\\')}</td>" for h in headers)
        lines.append(f"    <tr{style}>{cells}</tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    html = "\\n".join(lines)
    return tool_response(html=html, rows=len(data), columns=len(headers))


__all__ = ["generate_html_table"]
''',
)

# ── 2. crontab-scheduler ─────────────────────────────────────────────────────
create_skill(
    name="crontab-scheduler",
    frontmatter_name="crontab-scheduler",
    description="Convert human-readable schedules to cron expressions and vice versa",
    category="devops/scheduling",
    capabilities=["Convert natural language to cron", "Parse cron to human-readable", "Validate cron expressions"],
    triggers=["convert schedule to cron", "parse cron expression", "crontab help"],
    eval_tool="schedule_to_cron",
    eval_input={"schedule": "every day at 3pm"},
    tool_docs="### schedule_to_cron\nConvert human text to cron.\n### cron_to_human\nExplain a cron expression.",
    tools_code='''"""Convert between human-readable schedules and cron expressions."""
import re
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("crontab-scheduler")

_PATTERNS = [
    (r"every\\s+minute", "* * * * *"),
    (r"every\\s+(\\d+)\\s+minutes?", "{0}/{1} * * * *"),
    (r"every\\s+hour", "0 * * * *"),
    (r"every\\s+(\\d+)\\s+hours?", "0 */{1} * * *"),
    (r"every\\s+day\\s+at\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", "_daily"),
    (r"every\\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\s+at\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", "_weekly"),
    (r"every\\s+month\\s+on\\s+day\\s+(\\d{1,2})\\s+at\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", "_monthly"),
]
_DAYS = {"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4, "friday": 5, "saturday": 6, "sunday": 0}
_FIELD_NAMES = ["minute", "hour", "day of month", "month", "day of week"]


def _resolve_hour(h: str, m: str, ampm: str) -> tuple:
    hour = int(h)
    minute = int(m) if m else 0
    if ampm:
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
    return minute, hour


@tool_wrapper(required_params=["schedule"])
def schedule_to_cron(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a human-readable schedule to a cron expression."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["schedule"].strip().lower()

    # every N minutes
    m = re.match(r"every\\s+(\\d+)\\s+minutes?", text)
    if m:
        return tool_response(cron=f"*/{m.group(1)} * * * *", description=text)
    if re.match(r"every\\s+minute", text):
        return tool_response(cron="* * * * *", description=text)

    # every N hours
    m = re.match(r"every\\s+(\\d+)\\s+hours?", text)
    if m:
        return tool_response(cron=f"0 */{m.group(1)} * * *", description=text)
    if re.match(r"every\\s+hour", text):
        return tool_response(cron="0 * * * *", description=text)

    # every day at H:MM am/pm
    m = re.match(r"every\\s+day\\s+at\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", text)
    if m:
        minute, hour = _resolve_hour(m.group(1), m.group(2), m.group(3))
        return tool_response(cron=f"{minute} {hour} * * *", description=text)

    # every <weekday> at H
    m = re.match(r"every\\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\s+at\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", text)
    if m:
        minute, hour = _resolve_hour(m.group(2), m.group(3), m.group(4))
        dow = _DAYS[m.group(1)]
        return tool_response(cron=f"{minute} {hour} * * {dow}", description=text)

    # every month on day N at H
    m = re.match(r"every\\s+month\\s+on\\s+day\\s+(\\d{1,2})\\s+at\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", text)
    if m:
        minute, hour = _resolve_hour(m.group(2), m.group(3), m.group(4))
        return tool_response(cron=f"{minute} {hour} {m.group(1)} * *", description=text)

    return tool_error(f"Could not parse schedule: {text}")


@tool_wrapper(required_params=["cron"])
def cron_to_human(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a cron expression to human-readable text."""
    status.set_callback(params.pop("_status_callback", None))
    parts = params["cron"].strip().split()
    if len(parts) != 5:
        return tool_error("Cron expression must have exactly 5 fields")
    mi, hr, dom, mon, dow = parts
    pieces = []
    if mi == "*" and hr == "*":
        pieces.append("every minute")
    elif mi.startswith("*/"):
        pieces.append(f"every {mi[2:]} minutes")
    elif hr.startswith("*/"):
        pieces.append(f"every {hr[2:]} hours at minute {mi}")
    else:
        h = int(hr) if hr != "*" else None
        m = int(mi) if mi != "*" else 0
        if h is not None:
            ampm = "AM" if h < 12 else "PM"
            display_h = h % 12 or 12
            pieces.append(f"at {display_h}:{m:02d} {ampm}")
    if dow != "*":
        day_names = {v: k for k, v in _DAYS.items()}
        pieces.append(f"on {day_names.get(int(dow), dow)}")
    if dom != "*":
        pieces.append(f"on day {dom} of the month")
    if mon != "*":
        pieces.append(f"in month {mon}")
    return tool_response(human=" ".join(pieces) if pieces else params["cron"], cron=params["cron"])


__all__ = ["schedule_to_cron", "cron_to_human"]
''',
)

# ── 3. yaml-validator ────────────────────────────────────────────────────────
create_skill(
    name="yaml-validator",
    frontmatter_name="yaml-validator",
    description="Validate YAML syntax, check structure, and convert between YAML and JSON",
    category="data/validation",
    capabilities=["Validate YAML syntax", "Convert YAML to JSON", "Convert JSON to YAML"],
    triggers=["validate yaml", "convert yaml to json", "check yaml syntax"],
    eval_tool="validate_yaml",
    eval_input={"yaml_text": "name: test\nvalue: 42"},
    tool_docs="### validate_yaml\nValidate and optionally convert YAML.\n**Params:** yaml_text (str), to_json (bool)",
    deps="PyYAML",
    tools_code='''"""Validate YAML syntax and convert YAML<->JSON."""
import json
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("yaml-validator")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@tool_wrapper(required_params=["yaml_text"])
def validate_yaml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate YAML text and optionally convert to JSON."""
    status.set_callback(params.pop("_status_callback", None))
    if not HAS_YAML:
        return tool_error("PyYAML not installed. Run: pip install pyyaml")
    text = params["yaml_text"]
    to_json = params.get("to_json", False)
    try:
        docs = list(yaml.safe_load_all(text))
        parsed = docs[0] if len(docs) == 1 else docs
    except yaml.YAMLError as e:
        return tool_error(f"Invalid YAML: {e}")
    result = dict(valid=True, document_count=len(docs), type=type(parsed).__name__)
    if to_json:
        result["json"] = json.dumps(parsed, indent=2, default=str)
    return tool_response(data=result)


@tool_wrapper(required_params=["json_text"])
def json_to_yaml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON text to YAML."""
    status.set_callback(params.pop("_status_callback", None))
    if not HAS_YAML:
        return tool_error("PyYAML not installed. Run: pip install pyyaml")
    try:
        data = json.loads(params["json_text"])
    except json.JSONDecodeError as e:
        return tool_error(f"Invalid JSON: {e}")
    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return tool_response(yaml=yaml_text)


__all__ = ["validate_yaml", "json_to_yaml"]
''',
)

# ── 4. toml-parser ───────────────────────────────────────────────────────────
create_skill(
    name="toml-parser",
    frontmatter_name="toml-parser",
    description="Parse and generate TOML files with Python 3.11+ tomllib and fallback support",
    category="data/parsing",
    capabilities=["Parse TOML content", "Generate TOML from dicts", "Read pyproject.toml / Cargo.toml"],
    triggers=["parse toml", "read toml file", "generate toml"],
    eval_tool="parse_toml",
    eval_input={"toml_text": "[project]\nname = \"demo\"\nversion = \"1.0\""},
    tool_docs="### parse_toml\nParse TOML text.\n### generate_toml\nGenerate TOML from a dict.",
    tools_code='''"""Parse and generate TOML content."""
import json
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("toml-parser")

# Python 3.11+ has tomllib; fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _to_toml(data: dict, prefix: str = "") -> str:
    """Simple TOML serializer for basic types."""
    lines = []
    tables = []
    for k, v in data.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            tables.append((full_key, v))
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, list):
            items = ", ".join(json.dumps(i) for i in v)
            lines.append(f"{k} = [{items}]")
        else:
            lines.append(f'{k} = "{v}"')
    result = "\\n".join(lines)
    for tkey, tval in tables:
        result += f"\\n\\n[{tkey}]\\n" + _to_toml(tval)
    return result


@tool_wrapper(required_params=["toml_text"])
def parse_toml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse TOML text and return as dict."""
    status.set_callback(params.pop("_status_callback", None))
    if tomllib is None:
        return tool_error("No TOML parser available. Use Python 3.11+ or pip install tomli")
    try:
        parsed = tomllib.loads(params["toml_text"])
    except Exception as e:
        return tool_error(f"Invalid TOML: {e}")
    return tool_response(data=parsed, sections=list(parsed.keys()))


@tool_wrapper(required_params=["data"])
def generate_toml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate TOML text from a dict."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    if not isinstance(data, dict):
        return tool_error("data must be a dict")
    toml_text = _to_toml(data)
    return tool_response(toml=toml_text)


__all__ = ["parse_toml", "generate_toml"]
''',
)

# ── 5. csv-to-json ───────────────────────────────────────────────────────────
create_skill(
    name="csv-to-json",
    frontmatter_name="csv-to-json",
    description="Convert CSV data to JSON and vice versa with header detection and type inference",
    category="data/conversion",
    capabilities=["Convert CSV to JSON", "Convert JSON to CSV", "Auto-detect delimiters", "Infer types"],
    triggers=["convert csv to json", "csv to json", "json to csv"],
    eval_tool="csv_to_json",
    eval_input={"csv_text": "name,age\nAlice,30\nBob,25"},
    tool_docs="### csv_to_json\nConvert CSV text to JSON.\n### json_to_csv\nConvert JSON array to CSV.",
    tools_code='''"""Convert between CSV and JSON formats."""
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
    return tool_response(data=records, rows=len(records), columns=len(headers),
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
''',
)

# ── 6. markdown-table-generator ──────────────────────────────────────────────
create_skill(
    name="markdown-table-generator",
    frontmatter_name="markdown-table-generator",
    description="Generate formatted markdown tables from structured data",
    category="data/formatting",
    capabilities=["Generate markdown tables from dicts", "Support column alignment", "Auto-size columns"],
    triggers=["create markdown table", "generate md table", "format data as markdown table"],
    eval_tool="generate_markdown_table",
    eval_input={"data": [{"name": "Alice", "score": 95}], "align": {"score": "right"}},
    tool_docs="### generate_markdown_table\nGenerate a markdown table from list of dicts.\n**Params:** data, align (dict), headers (list)",
    tools_code='''"""Generate markdown tables from structured data."""
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
    table = "\\n".join([header_line, sep_line] + rows)
    return tool_response(markdown=table, rows=len(data), columns=len(headers))


__all__ = ["generate_markdown_table"]
''',
)

# ── 7. http-status-lookup ────────────────────────────────────────────────────
create_skill(
    name="http-status-lookup",
    frontmatter_name="http-status-lookup",
    description="Look up HTTP status code meanings, categories, and descriptions",
    category="web/reference",
    capabilities=["Look up any HTTP status code", "List codes by category", "Get detailed descriptions"],
    triggers=["what is http status 404", "http status codes", "look up http code"],
    eval_tool="lookup_http_status",
    eval_input={"code": 404},
    tool_docs="### lookup_http_status\nLook up an HTTP status code.\n**Params:** code (int), category (str)",
    tools_code='''"""Look up HTTP status code meanings and categories."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("http-status-lookup")

_CODES: Dict[int, tuple] = {
    100: ("Continue", "Server received headers, client should proceed"),
    101: ("Switching Protocols", "Server switching to requested protocol"),
    200: ("OK", "Request succeeded"),
    201: ("Created", "Resource created successfully"),
    202: ("Accepted", "Request accepted for processing"),
    204: ("No Content", "Success with no response body"),
    301: ("Moved Permanently", "Resource permanently moved to new URL"),
    302: ("Found", "Resource temporarily at different URL"),
    304: ("Not Modified", "Resource not modified since last request"),
    307: ("Temporary Redirect", "Temporary redirect preserving method"),
    308: ("Permanent Redirect", "Permanent redirect preserving method"),
    400: ("Bad Request", "Server cannot process due to client error"),
    401: ("Unauthorized", "Authentication required"),
    403: ("Forbidden", "Server refuses to authorize request"),
    404: ("Not Found", "Resource not found"),
    405: ("Method Not Allowed", "HTTP method not supported for resource"),
    408: ("Request Timeout", "Server timed out waiting for request"),
    409: ("Conflict", "Request conflicts with current resource state"),
    410: ("Gone", "Resource permanently removed"),
    413: ("Payload Too Large", "Request body exceeds server limit"),
    415: ("Unsupported Media Type", "Media type not supported"),
    418: ("I'm a Teapot", "RFC 2324 - server is a teapot"),
    422: ("Unprocessable Entity", "Request well-formed but has semantic errors"),
    429: ("Too Many Requests", "Rate limit exceeded"),
    500: ("Internal Server Error", "Unexpected server error"),
    501: ("Not Implemented", "Server does not support the functionality"),
    502: ("Bad Gateway", "Invalid response from upstream server"),
    503: ("Service Unavailable", "Server temporarily unavailable"),
    504: ("Gateway Timeout", "Upstream server timed out"),
}
_CATEGORIES = {1: "Informational", 2: "Success", 3: "Redirection", 4: "Client Error", 5: "Server Error"}


@tool_wrapper()
def lookup_http_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Look up an HTTP status code or list codes by category."""
    status.set_callback(params.pop("_status_callback", None))
    code = params.get("code")
    category = params.get("category")

    if code is not None:
        code = int(code)
        info = _CODES.get(code)
        if not info:
            cat = _CATEGORIES.get(code // 100, "Unknown")
            return tool_response(code=code, name="Unknown", category=cat,
                                 description=f"Non-standard {cat.lower()} status code")
        return tool_response(code=code, name=info[0], category=_CATEGORIES.get(code // 100, "Unknown"),
                             description=info[1])

    if category is not None:
        cat_num = int(category[0]) if isinstance(category, str) and category[0].isdigit() else category
        codes = {c: v[0] for c, v in _CODES.items() if c // 100 == int(cat_num)}
        return tool_response(category=_CATEGORIES.get(int(cat_num), "Unknown"), codes=codes)

    return tool_response(categories=_CATEGORIES, total_codes=len(_CODES))


__all__ = ["lookup_http_status"]
''',
)

# ── 8. random-data-generator ─────────────────────────────────────────────────
create_skill(
    name="random-data-generator",
    frontmatter_name="random-data-generator",
    description="Generate random names, emails, addresses, and phone numbers for testing",
    category="testing/data",
    capabilities=["Generate random names", "Generate fake emails", "Generate phone numbers", "Generate addresses"],
    triggers=["generate test data", "random names", "fake data generator"],
    eval_tool="generate_random_data",
    eval_input={"type": "person", "count": 3},
    tool_docs="### generate_random_data\nGenerate random test data.\n**Params:** type (person|email|phone|address), count (int)",
    tools_code='''"""Generate random test data without external dependencies."""
import random
import string
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("random-data-generator")

_FIRST = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
          "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
          "Quinn", "Rose", "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander", "Yara", "Zane"]
_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
         "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas", "Lee", "Harris"]
_DOMAINS = ["example.com", "test.org", "demo.net", "sample.io", "mock.dev"]
_STREETS = ["Main St", "Oak Ave", "Pine Rd", "Elm Dr", "Cedar Ln", "Maple Blvd", "Park Way"]
_CITIES = ["Springfield", "Riverside", "Georgetown", "Fairview", "Madison", "Clinton", "Franklin"]
_STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]


def _person() -> Dict[str, str]:
    first, last = random.choice(_FIRST), random.choice(_LAST)
    email = f"{first.lower()}.{last.lower()}@{random.choice(_DOMAINS)}"
    phone = f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
    return {"first_name": first, "last_name": last, "email": email, "phone": phone}


def _email() -> str:
    user = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
    return f"{user}@{random.choice(_DOMAINS)}"


def _phone() -> str:
    return f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"


def _address() -> Dict[str, str]:
    return {"street": f"{random.randint(1,9999)} {random.choice(_STREETS)}",
            "city": random.choice(_CITIES), "state": random.choice(_STATES),
            "zip": f"{random.randint(10000,99999)}"}


@tool_wrapper(required_params=["type"])
def generate_random_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate random test data of a given type."""
    status.set_callback(params.pop("_status_callback", None))
    dtype = params["type"].lower()
    count = min(int(params.get("count", 5)), 1000)
    seed = params.get("seed")
    if seed is not None:
        random.seed(int(seed))
    generators = {"person": _person, "email": _email, "phone": _phone, "address": _address}
    gen = generators.get(dtype)
    if not gen:
        return tool_error(f"Unknown type: {dtype}. Use: {', '.join(generators)}")
    items: List = [gen() for _ in range(count)]
    return tool_response(data=items, count=count, type=dtype)


__all__ = ["generate_random_data"]
''',
)

# ── 9. checksum-verifier ─────────────────────────────────────────────────────
create_skill(
    name="checksum-verifier",
    frontmatter_name="checksum-verifier",
    description="Calculate and verify MD5, SHA1, and SHA256 checksums for files and strings",
    category="security/verification",
    capabilities=["Calculate MD5/SHA1/SHA256 checksums", "Verify checksums against expected values", "Hash files and strings"],
    triggers=["calculate checksum", "verify sha256", "md5 hash"],
    eval_tool="calculate_checksum",
    eval_input={"text": "hello world", "algorithm": "sha256"},
    tool_docs="### calculate_checksum\nCalculate checksum of text or file.\n### verify_checksum\nVerify a checksum matches.",
    tools_code='''"""Calculate and verify checksums using hashlib."""
import hashlib
from pathlib import Path
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("checksum-verifier")

_ALGOS = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}


def _hash_bytes(data: bytes, algo: str) -> str:
    fn = _ALGOS.get(algo.lower())
    if not fn:
        raise ValueError(f"Unsupported algorithm: {algo}. Use: {', '.join(_ALGOS)}")
    return fn(data).hexdigest()


@tool_wrapper()
def calculate_checksum(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate a checksum for text or a file."""
    status.set_callback(params.pop("_status_callback", None))
    algo = params.get("algorithm", "sha256").lower()
    text = params.get("text")
    file_path = params.get("file_path")
    if not text and not file_path:
        return tool_error("Provide either text or file_path")
    if text:
        digest = _hash_bytes(text.encode("utf-8"), algo)
        return tool_response(checksum=digest, algorithm=algo, input_type="text",
                             length=len(text))
    p = Path(file_path)
    if not p.exists():
        return tool_error(f"File not found: {file_path}")
    h = _ALGOS.get(algo)
    if not h:
        return tool_error(f"Unsupported algorithm: {algo}")
    hasher = h()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return tool_response(checksum=hasher.hexdigest(), algorithm=algo,
                         input_type="file", file=str(p), size=p.stat().st_size)


@tool_wrapper(required_params=["expected"])
def verify_checksum(params: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a checksum matches an expected value."""
    status.set_callback(params.pop("_status_callback", None))
    expected = params["expected"].strip().lower()
    result = calculate_checksum(params)
    if not result.get("success"):
        return result
    actual = result["checksum"]
    match = actual == expected
    return tool_response(match=match, expected=expected, actual=actual,
                         algorithm=result["algorithm"])


__all__ = ["calculate_checksum", "verify_checksum"]
''',
)

# ── 10. port-scanner ─────────────────────────────────────────────────────────
create_skill(
    name="port-scanner",
    frontmatter_name="port-scanner",
    description="Check if specific ports are open on a host using socket connections",
    category="networking/diagnostics",
    capabilities=["Check if ports are open", "Scan port ranges", "Look up common port services"],
    triggers=["scan ports", "check if port is open", "port scanner"],
    eval_tool="scan_ports",
    eval_input={"host": "localhost", "ports": [80, 443]},
    tool_docs="### scan_ports\nCheck if ports are open on a host.\n**Params:** host (str), ports (list[int]), timeout (float)",
    tools_code='''"""Check if ports are open on a host using socket connections."""
import socket
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("port-scanner")

_SERVICES: Dict[int, str] = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 465: "SMTPS",
    587: "SMTP-TLS", 993: "IMAPS", 995: "POP3S", 3306: "MySQL",
    3389: "RDP", 5432: "PostgreSQL", 5672: "RabbitMQ", 6379: "Redis",
    8080: "HTTP-Alt", 8443: "HTTPS-Alt", 9200: "Elasticsearch",
    27017: "MongoDB",
}


def _check_port(host: str, port: int, timeout: float) -> Dict[str, Any]:
    service = _SERVICES.get(port, "unknown")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            is_open = result == 0
    except socket.gaierror:
        return {"port": port, "open": False, "service": service, "error": "DNS resolution failed"}
    except OSError as e:
        return {"port": port, "open": False, "service": service, "error": str(e)}
    return {"port": port, "open": is_open, "service": service}


@tool_wrapper(required_params=["host"])
def scan_ports(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check if specific ports are open on a host."""
    status.set_callback(params.pop("_status_callback", None))
    host = params["host"]
    ports: List[int] = params.get("ports", [80, 443, 22, 8080])
    timeout = float(params.get("timeout", 1.0))
    if len(ports) > 100:
        return tool_error("Maximum 100 ports per scan")
    results = [_check_port(host, int(p), timeout) for p in ports]
    open_ports = [r for r in results if r["open"]]
    return tool_response(host=host, results=results, open_count=len(open_ports),
                         closed_count=len(results) - len(open_ports))


@tool_wrapper()
def list_common_ports(params: Dict[str, Any]) -> Dict[str, Any]:
    """List common ports and their associated services."""
    status.set_callback(params.pop("_status_callback", None))
    return tool_response(ports={str(k): v for k, v in sorted(_SERVICES.items())},
                         count=len(_SERVICES))


__all__ = ["scan_ports", "list_common_ports"]
''',
)

print(f"\nBatch 5a complete — 10 skills generated.")
