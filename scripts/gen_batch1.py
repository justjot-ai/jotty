"""Batch 1: Pure Python stdlib skills (20 skills)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from generate_skills import create_skill

# ── 1. hash-calculator ──────────────────────────────────────────────
create_skill(
    name="hash-calculator",
    frontmatter_name="hashing-data",
    description="Compute and verify MD5, SHA-256, SHA-512, and bcrypt hashes for files and strings. Use when the user wants to hash, checksum, verify integrity.",
    category="development",
    capabilities=["code"],
    triggers=["hash", "checksum", "md5", "sha256", "sha512", "verify hash"],
    eval_tool="hash_tool",
    eval_input={"text": "hello world", "algorithm": "sha256"},
    tool_docs="""### hash_tool
Compute hash of text or file.

**Parameters:**
- `text` (str, optional): Text to hash
- `file_path` (str, optional): File to hash
- `algorithm` (str, optional): md5, sha1, sha256, sha512 (default: sha256)

**Returns:**
- `success` (bool): Whether hashing succeeded
- `hash` (str): Hex digest
- `algorithm` (str): Algorithm used

### verify_hash_tool
Verify a hash matches expected value.

**Parameters:**
- `text` (str, optional): Text to verify
- `file_path` (str, optional): File to verify
- `expected_hash` (str, required): Expected hash value
- `algorithm` (str, optional): Algorithm (default: sha256)

**Returns:**
- `success` (bool): Whether verification succeeded
- `match` (bool): Whether hashes match""",
    tools_code='''"""Hash Calculator Skill — compute and verify hashes."""
import hashlib
from pathlib import Path
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("hash-calculator")

ALGORITHMS = {"md5", "sha1", "sha256", "sha512", "sha384", "sha224"}


@tool_wrapper()
def hash_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute hash of text or file."""
    status.set_callback(params.pop("_status_callback", None))
    text = params.get("text")
    file_path = params.get("file_path")
    algo = params.get("algorithm", "sha256").lower()

    if algo not in ALGORITHMS:
        return tool_error(f"Unsupported algorithm: {algo}. Use one of: {sorted(ALGORITHMS)}")
    if not text and not file_path:
        return tool_error("Provide either text or file_path")

    h = hashlib.new(algo)
    if file_path:
        p = Path(file_path)
        if not p.exists():
            return tool_error(f"File not found: {file_path}")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    else:
        h.update(text.encode("utf-8"))

    return tool_response(hash=h.hexdigest(), algorithm=algo,
                         input_type="file" if file_path else "text")


@tool_wrapper(required_params=["expected_hash"])
def verify_hash_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a hash matches expected value."""
    status.set_callback(params.pop("_status_callback", None))
    result = hash_tool(params)
    if not result.get("success"):
        return result
    match = result["hash"].lower() == params["expected_hash"].strip().lower()
    return tool_response(match=match, computed_hash=result["hash"],
                         expected_hash=params["expected_hash"], algorithm=result["algorithm"])


__all__ = ["hash_tool", "verify_hash_tool"]
''')

# ── 2. password-generator ──────────────────────────────────────────
create_skill(
    name="password-generator",
    frontmatter_name="generating-passwords",
    description="Generate cryptographically secure passwords, passphrases, and PIN codes. Use when the user wants to generate password, passphrase, PIN, secret.",
    category="development",
    capabilities=["code"],
    triggers=["password", "passphrase", "PIN", "generate secret", "random string"],
    eval_tool="generate_password_tool",
    eval_input={"length": 16},
    tool_docs="""### generate_password_tool
Generate a secure random password.

**Parameters:**
- `length` (int, optional): Password length (default: 16, min: 8, max: 128)
- `uppercase` (bool, optional): Include uppercase (default: true)
- `lowercase` (bool, optional): Include lowercase (default: true)
- `digits` (bool, optional): Include digits (default: true)
- `symbols` (bool, optional): Include symbols (default: true)
- `count` (int, optional): Number of passwords (default: 1)

**Returns:**
- `success` (bool)
- `passwords` (list): Generated passwords
- `strength` (str): Estimated strength""",
    tools_code='''"""Password Generator Skill — secure passwords and passphrases."""
import secrets
import string
import math
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("password-generator")

WORD_LIST = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    "acquire", "across", "action", "actor", "actress", "actual", "adapt", "address",
    "adjust", "admit", "adult", "advance", "advice", "afford", "again", "agent",
    "agree", "ahead", "alarm", "album", "alert", "alien", "alley", "allow", "almost",
    "alone", "alpha", "already", "also", "alter", "always", "amateur", "amazing",
    "among", "amount", "anchor", "ancient", "anger", "angle", "animal", "ankle",
    "annual", "answer", "antenna", "apple", "armor", "army", "arrive", "arrow",
    "basket", "battle", "beach", "beauty", "become", "before", "begin", "behind",
    "believe", "below", "bench", "benefit", "beyond", "bicycle", "blanket", "blast",
    "blossom", "board", "bonus", "border", "bottle", "bounce", "brave", "breeze",
    "bridge", "bright", "broken", "bronze", "bubble", "budget", "buffalo", "burden",
    "cabin", "cable", "camera", "canal", "canyon", "carbon", "cargo", "carpet",
    "castle", "catalog", "cattle", "caught", "cause", "ceiling", "cement", "census",
    "certain", "chair", "change", "chapter", "charge", "cherry", "chicken", "choice",
    "circle", "citizen", "civil", "claim", "clap", "clarify", "claw", "click",
    "climb", "clinic", "clock", "close", "cluster", "coach", "coconut", "coffee",
    "collect", "column", "combine", "comfort", "common", "company", "concert",
    "connect", "consider", "control", "convince", "copper", "coral", "core",
    "correct", "cotton", "country", "couple", "course", "cousin", "cover", "craft",
    "cream", "credit", "cricket", "cross", "crowd", "cruel", "cruise", "crystal",
    "custom", "cycle", "damage", "dance", "danger", "daring", "dawn", "debate",
    "decade", "decline", "define", "demand", "depart", "depend", "deposit", "depth",
    "derive", "desert", "design", "detail", "detect", "develop", "device", "devote",
    "diamond", "diary", "diesel", "differ", "digital", "dinner", "dinosaur",
    "direct", "dismiss", "disorder", "display", "distance", "divide", "dolphin",
    "domain", "donkey", "donor", "dragon", "drama", "dream", "drift", "drink",
    "driver", "during", "dutch", "dwarf", "dynamic", "eager", "eagle", "early",
    "earth", "easily", "ecology", "economy", "educate", "effort", "eight", "either",
    "elbow", "elder", "electric", "elegant", "element", "elephant", "elevator",
    "elite", "embark", "embody", "embrace", "emerge", "emotion", "employ", "empower",
    "enable", "endorse", "energy", "enforce", "engage", "engine", "enhance", "enjoy",
    "enrich", "ensure", "enter", "entire", "entry", "envelope", "episode", "equal",
    "equip", "erode", "escape", "essence", "estate", "eternal", "evidence", "evolve",
    "exact", "example", "excess", "exchange", "excite", "exclude", "excuse",
    "execute", "exercise", "exhaust", "exhibit", "exile", "expand", "expect",
    "expire", "explain", "expose", "extend", "extra", "fabric", "faculty", "faint",
    "falcon", "family", "famous", "fancy", "fantasy", "fashion", "father", "fault",
    "favorite", "feature", "federal", "fence", "festival", "fiction", "field",
    "figure", "filter", "final", "finger", "finish", "fitness", "flame", "flash",
    "flavor", "flight", "float", "flock", "floor", "flower", "fluid", "focus",
    "follow", "force", "forest", "forget", "formal", "fortune", "forum", "forward",
    "fossil", "foster", "found", "fragile", "frame", "freedom", "frequent", "fresh",
    "friend", "fringe", "frost", "frozen", "fruit", "fuel", "furnace", "future",
    "gadget", "galaxy", "gallery", "garden", "garlic", "gather", "gauge", "general",
    "gentle", "genius", "genre", "gesture", "giant", "ginger", "giraffe", "glacier",
    "glance", "glimpse", "globe", "gloom", "glory", "glove", "glow", "goddess",
    "golden", "gospel", "govern", "grace", "grain", "grant", "gravity", "grocery",
    "group", "growth", "guard", "guitar", "habit", "hammer", "hamster", "harbor",
    "harvest", "hawk", "hazard", "health", "heart", "heaven", "heavy", "hedgehog",
    "height", "helmet", "hidden", "highway", "history", "hobby", "hockey", "holiday",
    "hollow", "honey", "horizon", "horror", "hospital", "host", "hotel", "hover",
    "humble", "humor", "hundred", "hungry", "hurdle", "hybrid",
]


def _estimate_strength(length: int, charset_size: int) -> str:
    bits = length * math.log2(charset_size) if charset_size > 0 else 0
    if bits >= 128:
        return "very_strong"
    elif bits >= 80:
        return "strong"
    elif bits >= 60:
        return "moderate"
    else:
        return "weak"


@tool_wrapper()
def generate_password_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate secure random passwords."""
    status.set_callback(params.pop("_status_callback", None))
    length = min(max(int(params.get("length", 16)), 4), 128)
    count = min(max(int(params.get("count", 1)), 1), 20)
    inc_upper = params.get("uppercase", True)
    inc_lower = params.get("lowercase", True)
    inc_digits = params.get("digits", True)
    inc_symbols = params.get("symbols", True)

    charset = ""
    if inc_lower:
        charset += string.ascii_lowercase
    if inc_upper:
        charset += string.ascii_uppercase
    if inc_digits:
        charset += string.digits
    if inc_symbols:
        charset += "!@#$%^&*()-_=+[]{}|;:,.<>?"

    if not charset:
        return tool_error("At least one character class must be enabled")

    passwords = ["".join(secrets.choice(charset) for _ in range(length)) for _ in range(count)]
    strength = _estimate_strength(length, len(charset))

    return tool_response(passwords=passwords, strength=strength, length=length,
                         charset_size=len(charset))


@tool_wrapper()
def generate_passphrase_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a passphrase from random words."""
    status.set_callback(params.pop("_status_callback", None))
    word_count = min(max(int(params.get("words", 5)), 3), 12)
    separator = params.get("separator", "-")
    capitalize = params.get("capitalize", True)

    words = [secrets.choice(WORD_LIST) for _ in range(word_count)]
    if capitalize:
        words = [w.capitalize() for w in words]

    passphrase = separator.join(words)
    bits = word_count * math.log2(len(WORD_LIST))

    return tool_response(passphrase=passphrase, word_count=word_count,
                         entropy_bits=round(bits, 1),
                         strength=_estimate_strength(word_count, len(WORD_LIST)))


__all__ = ["generate_password_tool", "generate_passphrase_tool"]
''')

# ── 3. json-transformer ────────────────────────────────────────────
create_skill(
    name="json-transformer",
    frontmatter_name="transforming-json",
    description="Transform, query, flatten, and merge JSON structures. Use when the user wants to transform JSON, flatten, merge, query, jq.",
    category="data-analysis",
    capabilities=["code", "data-fetch"],
    triggers=["json", "transform json", "flatten json", "merge json", "jq", "jsonpath"],
    eval_tool="flatten_json_tool",
    eval_input={"data": {"a": {"b": 1, "c": 2}}},
    tool_docs="""### flatten_json_tool
Flatten nested JSON into dot-notation keys.

**Parameters:**
- `data` (dict, required): JSON object to flatten
- `separator` (str, optional): Key separator (default: ".")

**Returns:**
- `success` (bool)
- `result` (dict): Flattened key-value pairs

### merge_json_tool
Deep merge two or more JSON objects.

**Parameters:**
- `objects` (list, required): List of JSON objects to merge

**Returns:**
- `success` (bool)
- `result` (dict): Merged object

### query_json_tool
Query JSON with dot-notation path.

**Parameters:**
- `data` (dict, required): JSON to query
- `path` (str, required): Dot-notation path (e.g. "users.0.name")

**Returns:**
- `success` (bool)
- `result` (any): Value at path""",
    tools_code='''"""JSON Transformer Skill — flatten, merge, query JSON."""
import json
import copy
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("json-transformer")


def _flatten(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            items.update(_flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}{sep}{i}" if prefix else str(i)
            items.update(_flatten(v, new_key, sep))
    else:
        items[prefix] = obj
    return items


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _query_path(data: Any, path: str) -> Any:
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key not found: {part}")
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise KeyError(f"Invalid index: {part}")
        else:
            raise KeyError(f"Cannot traverse into {type(current).__name__}")
    return current


@tool_wrapper(required_params=["data"])
def flatten_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested JSON into dot-notation keys."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    sep = params.get("separator", ".")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON: {e}")
    result = _flatten(data, sep=sep)
    return tool_response(result=result, keys_count=len(result))


@tool_wrapper(required_params=["objects"])
def merge_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two or more JSON objects."""
    status.set_callback(params.pop("_status_callback", None))
    objects = params["objects"]
    if not isinstance(objects, list) or len(objects) < 2:
        return tool_error("Provide a list of at least 2 objects")
    result = {}
    for obj in objects:
        if isinstance(obj, str):
            obj = json.loads(obj)
        if not isinstance(obj, dict):
            return tool_error("All items must be JSON objects")
        result = _deep_merge(result, obj)
    return tool_response(result=result)


@tool_wrapper(required_params=["data", "path"])
def query_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Query JSON with dot-notation path."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    if isinstance(data, str):
        data = json.loads(data)
    try:
        result = _query_path(data, params["path"])
        return tool_response(result=result, path=params["path"])
    except KeyError as e:
        return tool_error(str(e))


__all__ = ["flatten_json_tool", "merge_json_tool", "query_json_tool"]
''')

# ── 4. csv-analyzer ────────────────────────────────────────────────
create_skill(
    name="csv-analyzer",
    frontmatter_name="analyzing-csv",
    description="Load, filter, aggregate, and summarize CSV files. Use when the user wants to analyze CSV, filter rows, aggregate data, summarize columns.",
    category="data-analysis",
    capabilities=["analyze", "data-fetch"],
    triggers=["csv", "analyze csv", "filter csv", "csv summary", "tabular data"],
    eval_tool="csv_summary_tool",
    eval_input={"file_path": "test.csv"},
    tool_docs="""### csv_summary_tool
Get summary statistics for a CSV file.

**Parameters:**
- `file_path` (str, required): Path to CSV file
- `delimiter` (str, optional): Column delimiter (default: ",")

**Returns:**
- `success` (bool)
- `rows` (int): Number of rows
- `columns` (list): Column names
- `stats` (dict): Per-column statistics""",
    tools_code='''"""CSV Analyzer Skill — load, filter, aggregate CSV files."""
import csv
import io
import statistics
from pathlib import Path
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

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
''')

# ── 5. cron-job-manager ────────────────────────────────────────────
create_skill(
    name="cron-job-manager",
    frontmatter_name="managing-cron-jobs",
    description="Create, validate, and explain cron expressions. Use when the user wants to create cron, schedule, crontab, explain cron.",
    category="development",
    capabilities=["code", "devops"],
    triggers=["cron", "crontab", "schedule", "cron expression", "recurring job"],
    eval_tool="explain_cron_tool",
    eval_input={"expression": "0 9 * * 1-5"},
    tool_docs="""### explain_cron_tool
Explain a cron expression in human-readable format.

**Parameters:**
- `expression` (str, required): Cron expression (5 or 6 fields)

**Returns:**
- `success` (bool)
- `explanation` (str): Human-readable explanation
- `next_runs` (list): Next 5 execution times""",
    tools_code='''"""Cron Job Manager Skill — parse, explain, validate cron expressions."""
from datetime import datetime, timedelta
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("cron-job-manager")

FIELD_NAMES = ["minute", "hour", "day_of_month", "month", "day_of_week"]
FIELD_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DAY_NAMES = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}

COMMON = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0", "@daily": "0 0 * * *", "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}


def _explain_field(value: str, name: str) -> str:
    if value == "*":
        return f"every {name}"
    elif value.startswith("*/"):
        return f"every {value[2:]} {name}s"
    elif "," in value:
        return f"{name} {value}"
    elif "-" in value:
        parts = value.split("-")
        return f"{name} {parts[0]} through {parts[1]}"
    else:
        return f"{name} {value}"


@tool_wrapper(required_params=["expression"])
def explain_cron_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a cron expression in human-readable format."""
    status.set_callback(params.pop("_status_callback", None))
    expr = params["expression"].strip()
    expr = COMMON.get(expr, expr)
    parts = expr.split()

    if len(parts) not in (5, 6):
        return tool_error(f"Expected 5 or 6 fields, got {len(parts)}: {expr}")

    fields = parts[:5]
    explanations = [_explain_field(f, n) for f, n in zip(fields, FIELD_NAMES)]
    explanation = "Runs at " + ", ".join(explanations)

    return tool_response(expression=expr, explanation=explanation,
                         fields={n: f for n, f in zip(FIELD_NAMES, fields)})


@tool_wrapper(required_params=["expression"])
def validate_cron_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a cron expression."""
    status.set_callback(params.pop("_status_callback", None))
    expr = params["expression"].strip()
    expr = COMMON.get(expr, expr)
    parts = expr.split()

    if len(parts) not in (5, 6):
        return tool_response(valid=False, error=f"Expected 5 or 6 fields, got {len(parts)}")

    errors = []
    for i, (field, (lo, hi)) in enumerate(zip(parts[:5], FIELD_RANGES)):
        if field == "*":
            continue
        field_clean = field.replace("*/", "")
        for segment in field_clean.split(","):
            for part in segment.split("-"):
                try:
                    val = int(part)
                    if val < lo or val > hi:
                        errors.append(f"{FIELD_NAMES[i]}: {val} out of range [{lo}-{hi}]")
                except ValueError:
                    if part.lower() not in MONTH_NAMES and part.lower() not in DAY_NAMES:
                        errors.append(f"{FIELD_NAMES[i]}: invalid value '{part}'")

    return tool_response(valid=len(errors) == 0, errors=errors, expression=expr)


@tool_wrapper()
def build_cron_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a cron expression from human-readable description."""
    status.set_callback(params.pop("_status_callback", None))
    minute = params.get("minute", "*")
    hour = params.get("hour", "*")
    dom = params.get("day_of_month", "*")
    month = params.get("month", "*")
    dow = params.get("day_of_week", "*")

    expr = f"{minute} {hour} {dom} {month} {dow}"
    return tool_response(expression=expr)


__all__ = ["explain_cron_tool", "validate_cron_tool", "build_cron_tool"]
''')

# ── 6. base64-encoder ──────────────────────────────────────────────
create_skill(
    name="base64-encoder",
    frontmatter_name="encoding-base64",
    description="Encode and decode Base64, URL-safe Base64, and hex strings. Use when the user wants to encode, decode, base64, hex.",
    category="development",
    capabilities=["code"],
    triggers=["base64", "encode", "decode", "hex encode", "url encode"],
    eval_tool="base64_encode_tool",
    eval_input={"text": "Hello World"},
    tool_docs="""### base64_encode_tool
Encode text or file to Base64.

**Parameters:**
- `text` (str, optional): Text to encode
- `encoding` (str, optional): base64, base64url, hex (default: base64)

**Returns:**
- `success` (bool)
- `encoded` (str): Encoded string""",
    tools_code='''"""Base64 Encoder Skill — encode/decode Base64 and hex."""
import base64
import binascii
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("base64-encoder")


@tool_wrapper()
def base64_encode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Encode text to Base64, URL-safe Base64, or hex."""
    status.set_callback(params.pop("_status_callback", None))
    text = params.get("text", "")
    if not text:
        return tool_error("text parameter required")
    encoding = params.get("encoding", "base64").lower()
    data = text.encode("utf-8")

    if encoding == "base64":
        encoded = base64.b64encode(data).decode("ascii")
    elif encoding in ("base64url", "urlsafe"):
        encoded = base64.urlsafe_b64encode(data).decode("ascii")
    elif encoding == "hex":
        encoded = data.hex()
    else:
        return tool_error(f"Unsupported encoding: {encoding}. Use: base64, base64url, hex")

    return tool_response(encoded=encoded, encoding=encoding, original_length=len(text))


@tool_wrapper()
def base64_decode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Decode Base64, URL-safe Base64, or hex string."""
    status.set_callback(params.pop("_status_callback", None))
    encoded = params.get("encoded", "") or params.get("text", "")
    if not encoded:
        return tool_error("encoded parameter required")
    encoding = params.get("encoding", "base64").lower()

    try:
        if encoding == "base64":
            decoded = base64.b64decode(encoded).decode("utf-8")
        elif encoding in ("base64url", "urlsafe"):
            decoded = base64.urlsafe_b64decode(encoded).decode("utf-8")
        elif encoding == "hex":
            decoded = bytes.fromhex(encoded).decode("utf-8")
        else:
            return tool_error(f"Unsupported encoding: {encoding}")
        return tool_response(decoded=decoded, encoding=encoding)
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        return tool_error(f"Decode failed: {e}")


__all__ = ["base64_encode_tool", "base64_decode_tool"]
''')

# ── 7. jwt-decoder ──────────────────────────────────────────────────
create_skill(
    name="jwt-decoder",
    frontmatter_name="decoding-jwt",
    description="Decode and inspect JWT tokens — view header, payload, claims, and expiry. Use when the user wants to decode JWT, inspect token, check token expiry.",
    category="development",
    capabilities=["code"],
    triggers=["jwt", "decode jwt", "token", "inspect jwt", "json web token"],
    eval_tool="decode_jwt_tool",
    eval_input={"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"},
    tool_docs="""### decode_jwt_tool
Decode a JWT token without verification.

**Parameters:**
- `token` (str, required): JWT token string

**Returns:**
- `success` (bool)
- `header` (dict): Token header (algorithm, type)
- `payload` (dict): Token payload (claims)
- `expired` (bool): Whether token is expired""",
    tools_code='''"""JWT Decoder Skill — decode and inspect JWT tokens."""
import json
import base64
from datetime import datetime, timezone
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("jwt-decoder")


def _decode_segment(segment: str) -> dict:
    padding = 4 - len(segment) % 4
    segment += "=" * padding
    decoded = base64.urlsafe_b64decode(segment)
    return json.loads(decoded)


@tool_wrapper(required_params=["token"])
def decode_jwt_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a JWT token without verification."""
    status.set_callback(params.pop("_status_callback", None))
    token = params["token"].strip()
    parts = token.split(".")

    if len(parts) != 3:
        return tool_error(f"Invalid JWT: expected 3 parts, got {len(parts)}")

    try:
        header = _decode_segment(parts[0])
        payload = _decode_segment(parts[1])
    except (json.JSONDecodeError, Exception) as e:
        return tool_error(f"Failed to decode JWT: {e}")

    expired = None
    expires_at = None
    if "exp" in payload:
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expired = exp_dt < datetime.now(timezone.utc)
        expires_at = exp_dt.isoformat()

    issued_at = None
    if "iat" in payload:
        issued_at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc).isoformat()

    return tool_response(header=header, payload=payload, expired=expired,
                         expires_at=expires_at, issued_at=issued_at,
                         algorithm=header.get("alg", "unknown"))


__all__ = ["decode_jwt_tool"]
''')

# ── 8. regex-tester ─────────────────────────────────────────────────
create_skill(
    name="regex-tester",
    frontmatter_name="testing-regex",
    description="Test, match, and explain regular expressions. Use when the user wants to test regex, match pattern, extract matches, regex.",
    category="development",
    capabilities=["code"],
    triggers=["regex", "regular expression", "pattern match", "test regex", "re.match"],
    eval_tool="regex_match_tool",
    eval_input={"pattern": r"\d+", "text": "abc 123 def 456"},
    tool_docs="""### regex_match_tool
Test a regex pattern against text.

**Parameters:**
- `pattern` (str, required): Regular expression pattern
- `text` (str, required): Text to match against
- `flags` (str, optional): Flags: i=ignorecase, m=multiline, s=dotall

**Returns:**
- `success` (bool)
- `matches` (list): All matches with groups and positions""",
    tools_code='''"""Regex Tester Skill — test, match, and explain regex patterns."""
import re
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("regex-tester")


def _parse_flags(flags_str: str) -> int:
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
    flags = 0
    for c in flags_str.lower():
        if c in flag_map:
            flags |= flag_map[c]
    return flags


@tool_wrapper(required_params=["pattern", "text"])
def regex_match_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Test a regex pattern and return all matches."""
    status.set_callback(params.pop("_status_callback", None))
    pattern = params["pattern"]
    text = params["text"]
    flags = _parse_flags(params.get("flags", ""))

    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")

    matches = []
    for m in compiled.finditer(text):
        match_info = {
            "match": m.group(),
            "start": m.start(),
            "end": m.end(),
            "groups": list(m.groups()),
        }
        if m.groupdict():
            match_info["named_groups"] = m.groupdict()
        matches.append(match_info)

    return tool_response(matches=matches, count=len(matches), pattern=pattern)


@tool_wrapper(required_params=["pattern", "text"])
def regex_replace_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace regex matches in text."""
    status.set_callback(params.pop("_status_callback", None))
    replacement = params.get("replacement", "")
    flags = _parse_flags(params.get("flags", ""))
    count = int(params.get("count", 0))  # 0 = all

    try:
        result, n = re.subn(params["pattern"], replacement, params["text"], count=count, flags=flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")

    return tool_response(result=result, replacements=n)


@tool_wrapper(required_params=["pattern", "text"])
def regex_split_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Split text by regex pattern."""
    status.set_callback(params.pop("_status_callback", None))
    flags = _parse_flags(params.get("flags", ""))
    try:
        parts = re.split(params["pattern"], params["text"], flags=flags)
    except re.error as e:
        return tool_error(f"Invalid regex: {e}")
    return tool_response(parts=parts, count=len(parts))


__all__ = ["regex_match_tool", "regex_replace_tool", "regex_split_tool"]
''')

# ── 9. uuid-generator ──────────────────────────────────────────────
create_skill(
    name="uuid-generator",
    frontmatter_name="generating-uuids",
    description="Generate UUIDs (v1, v4, v5) and ULID identifiers. Use when the user wants to generate UUID, ULID, unique id.",
    category="development",
    capabilities=["code"],
    triggers=["uuid", "guid", "unique id", "ulid", "generate id"],
    eval_tool="generate_uuid_tool",
    eval_input={"version": 4},
    tool_docs="""### generate_uuid_tool
Generate UUID identifiers.

**Parameters:**
- `version` (int, optional): UUID version 1 or 4 (default: 4)
- `count` (int, optional): Number of UUIDs (default: 1, max: 100)
- `uppercase` (bool, optional): Uppercase output (default: false)

**Returns:**
- `success` (bool)
- `uuids` (list): Generated UUIDs""",
    tools_code='''"""UUID Generator Skill — generate UUIDs and ULIDs."""
import uuid
import time
import secrets
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("uuid-generator")

ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _generate_ulid() -> str:
    t = int(time.time() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = ULID_CHARS[t & 0x1F] + time_part
        t >>= 5
    rand_part = "".join(secrets.choice(ULID_CHARS) for _ in range(16))
    return time_part + rand_part


@tool_wrapper()
def generate_uuid_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate UUID identifiers."""
    status.set_callback(params.pop("_status_callback", None))
    version = int(params.get("version", 4))
    count = min(max(int(params.get("count", 1)), 1), 100)
    upper = params.get("uppercase", False)

    uuids = []
    for _ in range(count):
        if version == 1:
            u = str(uuid.uuid1())
        elif version == 4:
            u = str(uuid.uuid4())
        elif version == 5:
            namespace = params.get("namespace", "dns")
            name = params.get("name", "example.com")
            ns = {"dns": uuid.NAMESPACE_DNS, "url": uuid.NAMESPACE_URL,
                  "oid": uuid.NAMESPACE_OID, "x500": uuid.NAMESPACE_X500}.get(
                namespace, uuid.NAMESPACE_DNS)
            u = str(uuid.uuid5(ns, name))
        else:
            return tool_error(f"Unsupported version: {version}. Use 1, 4, or 5")
        uuids.append(u.upper() if upper else u)

    return tool_response(uuids=uuids, version=version, count=len(uuids))


@tool_wrapper()
def generate_ulid_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ULID identifiers (sortable, 128-bit)."""
    status.set_callback(params.pop("_status_callback", None))
    count = min(max(int(params.get("count", 1)), 1), 100)
    ulids = [_generate_ulid() for _ in range(count)]
    return tool_response(ulids=ulids, count=len(ulids))


__all__ = ["generate_uuid_tool", "generate_ulid_tool"]
''')

# ── 10. diff-tool ──────────────────────────────────────────────────
create_skill(
    name="diff-tool",
    frontmatter_name="comparing-text",
    description="Compare two texts or files and show differences. Use when the user wants to diff, compare, text difference.",
    category="development",
    capabilities=["code"],
    triggers=["diff", "compare", "difference", "text diff", "file diff"],
    eval_tool="diff_text_tool",
    eval_input={"text_a": "hello world", "text_b": "hello earth"},
    tool_docs="""### diff_text_tool
Show differences between two texts.

**Parameters:**
- `text_a` (str, required): First text
- `text_b` (str, required): Second text
- `context_lines` (int, optional): Lines of context (default: 3)

**Returns:**
- `success` (bool)
- `diff` (str): Unified diff output
- `changes` (int): Number of changed lines""",
    tools_code='''"""Diff Tool Skill — compare text and files."""
import difflib
from pathlib import Path
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("diff-tool")


@tool_wrapper(required_params=["text_a", "text_b"])
def diff_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Show unified diff between two texts."""
    status.set_callback(params.pop("_status_callback", None))
    a_lines = params["text_a"].splitlines(keepends=True)
    b_lines = params["text_b"].splitlines(keepends=True)
    n = int(params.get("context_lines", 3))

    diff = list(difflib.unified_diff(a_lines, b_lines, fromfile="a", tofile="b", n=n))
    changes = sum(1 for line in diff if line.startswith("+") or line.startswith("-"))

    return tool_response(diff="".join(diff), changes=changes, lines_a=len(a_lines),
                         lines_b=len(b_lines))


@tool_wrapper(required_params=["file_a", "file_b"])
def diff_files_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Show unified diff between two files."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        a = Path(params["file_a"]).read_text(errors="replace").splitlines(keepends=True)
        b = Path(params["file_b"]).read_text(errors="replace").splitlines(keepends=True)
    except FileNotFoundError as e:
        return tool_error(str(e))

    n = int(params.get("context_lines", 3))
    diff = list(difflib.unified_diff(a, b, fromfile=params["file_a"],
                                     tofile=params["file_b"], n=n))
    changes = sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))

    return tool_response(diff="".join(diff), changes=changes)


__all__ = ["diff_text_tool", "diff_files_tool"]
''')

# ── 11. data-anonymizer ────────────────────────────────────────────
create_skill(
    name="data-anonymizer",
    frontmatter_name="anonymizing-data",
    description="Anonymize PII by masking emails, phone numbers, names, IPs, and credit card numbers. Use when the user wants to anonymize, mask PII, redact data.",
    category="data-analysis",
    capabilities=["analyze"],
    triggers=["anonymize", "mask", "redact", "PII", "remove personal data"],
    eval_tool="anonymize_text_tool",
    eval_input={"text": "Contact john@example.com or call 555-123-4567"},
    tool_docs="""### anonymize_text_tool
Anonymize PII in text.

**Parameters:**
- `text` (str, required): Text containing PII
- `mask_emails` (bool, optional): Mask email addresses (default: true)
- `mask_phones` (bool, optional): Mask phone numbers (default: true)
- `mask_ips` (bool, optional): Mask IP addresses (default: true)
- `mask_credit_cards` (bool, optional): Mask credit card numbers (default: true)

**Returns:**
- `success` (bool)
- `anonymized` (str): Text with PII masked
- `detections` (dict): Count of each PII type found""",
    tools_code='''"""Data Anonymizer Skill — mask PII in text."""
import re
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("data-anonymizer")

PATTERNS = {
    "email": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "[EMAIL]"),
    "phone": (r"(?:\\+?1[-.\\s]?)?(?:\\(?\\d{3}\\)?[-.\\s]?)?\\d{3}[-.\\s]?\\d{4}", "[PHONE]"),
    "ip": (r"\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b", "[IP]"),
    "credit_card": (r"\\b(?:\\d{4}[- ]?){3}\\d{4}\\b", "[CREDIT_CARD]"),
    "ssn": (r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "[SSN]"),
}


@tool_wrapper(required_params=["text"])
def anonymize_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize PII in text by replacing with placeholders."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    detections = {}

    mask_map = {
        "email": params.get("mask_emails", True),
        "phone": params.get("mask_phones", True),
        "ip": params.get("mask_ips", True),
        "credit_card": params.get("mask_credit_cards", True),
        "ssn": params.get("mask_ssn", True),
    }

    for pii_type, (pattern, replacement) in PATTERNS.items():
        if not mask_map.get(pii_type, True):
            continue
        matches = re.findall(pattern, text)
        if matches:
            detections[pii_type] = len(matches)
            text = re.sub(pattern, replacement, text)

    return tool_response(anonymized=text, detections=detections,
                         total_redactions=sum(detections.values()))


__all__ = ["anonymize_text_tool"]
''')

# ── 12. markdown-to-html ───────────────────────────────────────────
create_skill(
    name="markdown-to-html",
    frontmatter_name="converting-markdown",
    description="Convert Markdown to styled HTML with syntax highlighting and TOC. Use when the user wants to convert markdown, markdown to html, render markdown.",
    category="document-creation",
    capabilities=["document", "generate"],
    triggers=["markdown", "convert markdown", "markdown to html", "render md"],
    eval_tool="markdown_to_html_tool",
    eval_input={"markdown": "# Hello\n\nThis is **bold**."},
    tool_docs="""### markdown_to_html_tool
Convert Markdown text to HTML.

**Parameters:**
- `markdown` (str, required): Markdown text
- `include_style` (bool, optional): Include CSS styling (default: true)
- `toc` (bool, optional): Generate table of contents (default: false)

**Returns:**
- `success` (bool)
- `html` (str): Rendered HTML""",
    tools_code='''"""Markdown to HTML Skill — convert markdown to styled HTML."""
import re
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("markdown-to-html")

CSS = """<style>
body{font-family:system-ui,sans-serif;line-height:1.6;max-width:800px;margin:0 auto;padding:20px;color:#333}
h1,h2,h3{color:#1a1a1a;border-bottom:1px solid #eee;padding-bottom:0.3em}
code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em}
pre code{display:block;padding:16px;overflow-x:auto}
blockquote{border-left:4px solid #ddd;margin:0;padding:0 16px;color:#666}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f4f4f4}
a{color:#0366d6}
</style>"""


def _md_to_html(md: str) -> str:
    html = md
    # Code blocks
    html = re.sub(r"```(\\w*)\\n(.*?)```", lambda m: f"<pre><code class=\\"{m.group(1)}\\">{m.group(2)}</code></pre>", html, flags=re.DOTALL)
    # Inline code
    html = re.sub(r"`([^`]+)`", r"<code>\\1</code>", html)
    # Headers
    html = re.sub(r"^######\\s+(.+)$", r"<h6>\\1</h6>", html, flags=re.MULTILINE)
    html = re.sub(r"^#####\\s+(.+)$", r"<h5>\\1</h5>", html, flags=re.MULTILINE)
    html = re.sub(r"^####\\s+(.+)$", r"<h4>\\1</h4>", html, flags=re.MULTILINE)
    html = re.sub(r"^###\\s+(.+)$", r"<h3>\\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^##\\s+(.+)$", r"<h2>\\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^#\\s+(.+)$", r"<h1>\\1</h1>", html, flags=re.MULTILINE)
    # Bold and italic
    html = re.sub(r"\\*\\*\\*(.+?)\\*\\*\\*", r"<strong><em>\\1</em></strong>", html)
    html = re.sub(r"\\*\\*(.+?)\\*\\*", r"<strong>\\1</strong>", html)
    html = re.sub(r"\\*(.+?)\\*", r"<em>\\1</em>", html)
    # Links and images
    html = re.sub(r"!\\[([^\\]]*)\\]\\(([^)]+)\\)", r\'<img src="\\2" alt="\\1">\', html)
    html = re.sub(r"\\[([^\\]]*)\\]\\(([^)]+)\\)", r\'<a href="\\2">\\1</a>\', html)
    # Lists
    html = re.sub(r"^[-*]\\s+(.+)$", r"<li>\\1</li>", html, flags=re.MULTILINE)
    # Blockquotes
    html = re.sub(r"^>\\s+(.+)$", r"<blockquote>\\1</blockquote>", html, flags=re.MULTILINE)
    # Horizontal rules
    html = re.sub(r"^---+$", r"<hr>", html, flags=re.MULTILINE)
    # Paragraphs
    html = re.sub(r"\\n\\n+", r"\\n</p><p>\\n", html)
    html = f"<p>{html}</p>"

    return html


@tool_wrapper(required_params=["markdown"])
def markdown_to_html_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Markdown text to HTML."""
    status.set_callback(params.pop("_status_callback", None))
    md = params["markdown"]
    include_style = params.get("include_style", True)

    html = _md_to_html(md)
    if include_style:
        html = CSS + html

    return tool_response(html=html, char_count=len(html))


__all__ = ["markdown_to_html_tool"]
''')

# ── 13. lorem-ipsum-generator ──────────────────────────────────────
create_skill(
    name="lorem-ipsum-generator",
    frontmatter_name="generating-lorem-ipsum",
    description="Generate Lorem Ipsum placeholder text in paragraphs, sentences, or words. Use when the user wants to generate placeholder text, lorem ipsum, dummy text.",
    category="content-creation",
    capabilities=["generate"],
    triggers=["lorem ipsum", "placeholder text", "dummy text", "filler text"],
    eval_tool="lorem_ipsum_tool",
    eval_input={"paragraphs": 2},
    tool_docs="""### lorem_ipsum_tool
Generate Lorem Ipsum placeholder text.

**Parameters:**
- `paragraphs` (int, optional): Number of paragraphs (default: 1)
- `sentences` (int, optional): Number of sentences (overrides paragraphs)
- `words` (int, optional): Number of words (overrides both)

**Returns:**
- `success` (bool)
- `text` (str): Generated Lorem Ipsum text""",
    tools_code='''"""Lorem Ipsum Generator Skill."""
import random
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("lorem-ipsum-generator")

WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
    "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
    "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
    "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
    "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
    "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
    "deserunt", "mollit", "anim", "id", "est", "laborum", "perspiciatis", "unde",
    "omnis", "iste", "natus", "error", "voluptatem", "accusantium", "doloremque",
    "laudantium", "totam", "rem", "aperiam", "eaque", "ipsa", "quae", "ab", "illo",
    "inventore", "veritatis", "quasi", "architecto", "beatae", "vitae", "dicta",
    "explicabo", "nemo", "ipsam", "quia", "voluptas", "aspernatur", "aut", "odit",
    "fugit", "consequuntur", "magni", "dolores", "eos", "ratione",
]


def _sentence(min_words: int = 5, max_words: int = 15) -> str:
    n = random.randint(min_words, max_words)
    s = " ".join(random.choice(WORDS) for _ in range(n))
    return s[0].upper() + s[1:] + "."


def _paragraph(min_sentences: int = 3, max_sentences: int = 7) -> str:
    n = random.randint(min_sentences, max_sentences)
    return " ".join(_sentence() for _ in range(n))


@tool_wrapper()
def lorem_ipsum_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Lorem Ipsum placeholder text."""
    status.set_callback(params.pop("_status_callback", None))
    word_count = params.get("words")
    sent_count = params.get("sentences")
    para_count = params.get("paragraphs", 1)

    if word_count:
        words = [random.choice(WORDS) for _ in range(int(word_count))]
        words[0] = words[0].capitalize()
        text = " ".join(words) + "."
    elif sent_count:
        text = " ".join(_sentence() for _ in range(int(sent_count)))
    else:
        text = "\\n\\n".join(_paragraph() for _ in range(int(para_count)))

    return tool_response(text=text, word_count=len(text.split()))


__all__ = ["lorem_ipsum_tool"]
''')

# ── 14. url-parser ─────────────────────────────────────────────────
create_skill(
    name="url-parser",
    frontmatter_name="parsing-urls",
    description="Parse, build, and manipulate URLs — extract components, add query params. Use when the user wants to parse URL, extract domain, build URL.",
    category="development",
    capabilities=["code"],
    triggers=["url", "parse url", "domain", "query string", "url encode"],
    eval_tool="parse_url_tool",
    eval_input={"url": "https://example.com:8080/path?q=test&page=2#section"},
    tool_docs="""### parse_url_tool
Parse a URL into its components.

**Parameters:**
- `url` (str, required): URL to parse

**Returns:**
- `success` (bool)
- `scheme` (str): Protocol
- `host` (str): Hostname
- `port` (int): Port number
- `path` (str): URL path
- `query` (dict): Query parameters
- `fragment` (str): Fragment identifier""",
    tools_code='''"""URL Parser Skill — parse and manipulate URLs."""
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote, unquote
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("url-parser")


@tool_wrapper(required_params=["url"])
def parse_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a URL into its components."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        parsed = urlparse(params["url"])
        query_params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}
        return tool_response(
            scheme=parsed.scheme, host=parsed.hostname or "",
            port=parsed.port, path=parsed.path,
            query=query_params, fragment=parsed.fragment,
            username=parsed.username, password=parsed.password,
        )
    except Exception as e:
        return tool_error(f"Failed to parse URL: {e}")


@tool_wrapper()
def build_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a URL from components."""
    status.set_callback(params.pop("_status_callback", None))
    scheme = params.get("scheme", "https")
    host = params.get("host", "")
    port = params.get("port")
    path = params.get("path", "/")
    query = params.get("query", {})
    fragment = params.get("fragment", "")

    if not host:
        return tool_error("host parameter required")

    netloc = host
    if port:
        netloc = f"{host}:{port}"
    qs = urlencode(query) if query else ""
    url = urlunparse((scheme, netloc, path, "", qs, fragment))
    return tool_response(url=url)


@tool_wrapper(required_params=["text"])
def url_encode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """URL-encode or decode text."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    mode = params.get("mode", "encode")
    if mode == "decode":
        return tool_response(result=unquote(text))
    return tool_response(result=quote(text, safe=params.get("safe", "")))


__all__ = ["parse_url_tool", "build_url_tool", "url_encode_tool"]
''')

# ── 15. date-calculator ────────────────────────────────────────────
create_skill(
    name="date-calculator",
    frontmatter_name="calculating-dates",
    description="Calculate date differences, add/subtract durations, format dates, find business days. Use when the user wants to calculate date, days between, add days.",
    category="workflow-automation",
    capabilities=["analyze"],
    triggers=["date", "days between", "add days", "date difference", "business days", "timestamp"],
    eval_tool="date_diff_tool",
    eval_input={"date_a": "2024-01-01", "date_b": "2024-12-31"},
    tool_docs="""### date_diff_tool
Calculate difference between two dates.

**Parameters:**
- `date_a` (str, required): First date (YYYY-MM-DD or ISO format)
- `date_b` (str, required): Second date

**Returns:**
- `success` (bool)
- `days` (int): Total days difference
- `weeks` (float): Weeks
- `months` (float): Approximate months
- `years` (float): Approximate years""",
    tools_code='''"""Date Calculator Skill — date arithmetic and formatting."""
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("date-calculator")

FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
           "%Y-%m-%dT%H:%M:%S%z", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
           "%Y%m%d", "%d-%b-%Y"]


def _parse_date(s: str) -> datetime:
    for fmt in FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}. Use YYYY-MM-DD format.")


@tool_wrapper(required_params=["date_a", "date_b"])
def date_diff_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate difference between two dates."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        a = _parse_date(params["date_a"])
        b = _parse_date(params["date_b"])
    except ValueError as e:
        return tool_error(str(e))

    delta = abs(b - a)
    days = delta.days
    return tool_response(days=days, weeks=round(days / 7, 1),
                         months=round(days / 30.44, 1), years=round(days / 365.25, 2))


@tool_wrapper(required_params=["date"])
def date_add_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add or subtract days/weeks/months from a date."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        dt = _parse_date(params["date"])
    except ValueError as e:
        return tool_error(str(e))

    days = int(params.get("days", 0))
    weeks = int(params.get("weeks", 0))
    result = dt + timedelta(days=days, weeks=weeks)

    return tool_response(result=result.strftime("%Y-%m-%d"), original=params["date"],
                         added_days=days + weeks * 7)


@tool_wrapper()
def now_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get current date and time in various formats."""
    status.set_callback(params.pop("_status_callback", None))
    tz_name = params.get("timezone", "UTC")
    now = datetime.now(timezone.utc)
    return tool_response(
        iso=now.isoformat(), date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"), timestamp=int(now.timestamp()),
        day_of_week=now.strftime("%A"), timezone="UTC",
    )


__all__ = ["date_diff_tool", "date_add_tool", "now_tool"]
''')

# ── 16. color-converter ────────────────────────────────────────────
create_skill(
    name="color-converter",
    frontmatter_name="converting-colors",
    description="Convert colors between HEX, RGB, HSL, and named formats. Generate palettes and complementary colors. Use when the user wants to convert color, hex to rgb, color palette.",
    category="content-creation",
    capabilities=["generate"],
    triggers=["color", "hex", "rgb", "hsl", "color palette", "convert color"],
    eval_tool="convert_color_tool",
    eval_input={"color": "#FF5733", "to_format": "rgb"},
    tool_docs="""### convert_color_tool
Convert between color formats.

**Parameters:**
- `color` (str, required): Color value (hex, rgb, hsl, or named)
- `to_format` (str, optional): Target format: hex, rgb, hsl (default: all)

**Returns:**
- `success` (bool)
- `hex` (str): Hex value
- `rgb` (dict): RGB values
- `hsl` (dict): HSL values""",
    tools_code='''"""Color Converter Skill — convert between color formats."""
import re
import colorsys
from typing import Dict, Any, Tuple, Optional

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("color-converter")

NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "white": (255, 255, 255), "black": (0, 0, 0), "yellow": (255, 255, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "orange": (255, 165, 0),
    "purple": (128, 0, 128), "pink": (255, 192, 203), "brown": (165, 42, 42),
    "gray": (128, 128, 128), "grey": (128, 128, 128), "navy": (0, 0, 128),
    "teal": (0, 128, 128), "coral": (255, 127, 80), "salmon": (250, 128, 114),
    "gold": (255, 215, 0), "silver": (192, 192, 192), "lime": (0, 255, 0),
    "indigo": (75, 0, 130), "violet": (238, 130, 238), "maroon": (128, 0, 0),
}


def _parse_color(color: str) -> Tuple[int, int, int]:
    color = color.strip().lower()
    if color in NAMED_COLORS:
        return NAMED_COLORS[color]
    hex_match = re.match(r"^#?([0-9a-f]{6})$", color)
    if hex_match:
        h = hex_match.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rgb_match = re.match(r"rgb\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)", color)
    if rgb_match:
        return int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
    raise ValueError(f"Cannot parse color: {color}")


@tool_wrapper(required_params=["color"])
def convert_color_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between color formats."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        r, g, b = _parse_color(params["color"])
    except ValueError as e:
        return tool_error(str(e))

    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    hex_val = f"#{r:02x}{g:02x}{b:02x}"

    return tool_response(
        hex=hex_val,
        rgb={"r": r, "g": g, "b": b},
        hsl={"h": round(h * 360), "s": round(s * 100), "l": round(l * 100)},
        css_rgb=f"rgb({r}, {g}, {b})",
        css_hsl=f"hsl({round(h * 360)}, {round(s * 100)}%, {round(l * 100)}%)",
    )


@tool_wrapper(required_params=["color"])
def complementary_color_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get complementary and analogous colors."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        r, g, b = _parse_color(params["color"])
    except ValueError as e:
        return tool_error(str(e))

    comp = (255 - r, 255 - g, 255 - b)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    analogous = []
    for offset in [-30, 30]:
        ah = ((h * 360 + offset) % 360) / 360
        ar, ag, ab = colorsys.hls_to_rgb(ah, l, s)
        analogous.append(f"#{int(ar*255):02x}{int(ag*255):02x}{int(ab*255):02x}")

    return tool_response(
        original=f"#{r:02x}{g:02x}{b:02x}",
        complementary=f"#{comp[0]:02x}{comp[1]:02x}{comp[2]:02x}",
        analogous=analogous,
    )


__all__ = ["convert_color_tool", "complementary_color_tool"]
''')

# ── 17. env-config-manager ─────────────────────────────────────────
create_skill(
    name="env-config-manager",
    frontmatter_name="managing-env-configs",
    description="Manage .env files — validate, diff, merge, detect missing variables. Use when the user wants to compare env files, validate env, find missing env vars.",
    category="development",
    capabilities=["code", "devops"],
    triggers=["env", "dotenv", ".env", "environment variables", "env diff"],
    eval_tool="parse_env_tool",
    eval_input={"file_path": ".env"},
    tool_docs="""### parse_env_tool
Parse a .env file and return key-value pairs.

**Parameters:**
- `file_path` (str, required): Path to .env file

**Returns:**
- `success` (bool)
- `variables` (dict): Key-value pairs
- `count` (int): Number of variables""",
    tools_code='''"""Env Config Manager Skill — parse, diff, validate .env files."""
import re
from pathlib import Path
from typing import Dict, Any, Set

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("env-config-manager")


def _parse_env_file(file_path: str) -> Dict[str, str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    env_vars = {}
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(.*)", line)
        if match:
            key = match.group(1)
            val = match.group(2).strip().strip("'").strip('"')
            env_vars[key] = val
    return env_vars


@tool_wrapper(required_params=["file_path"])
def parse_env_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a .env file and return variables."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        variables = _parse_env_file(params["file_path"])
        # Mask sensitive values
        masked = {}
        sensitive = {"key", "secret", "password", "token", "api"}
        for k, v in variables.items():
            if any(s in k.lower() for s in sensitive) and v:
                masked[k] = v[:4] + "****" if len(v) > 4 else "****"
            else:
                masked[k] = v
        return tool_response(variables=masked, count=len(variables))
    except FileNotFoundError as e:
        return tool_error(str(e))


@tool_wrapper(required_params=["file_a", "file_b"])
def diff_env_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two .env files and find differences."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        a = _parse_env_file(params["file_a"])
        b = _parse_env_file(params["file_b"])
    except FileNotFoundError as e:
        return tool_error(str(e))

    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))
    changed = sorted(k for k in set(a.keys()) & set(b.keys()) if a[k] != b[k])

    return tool_response(only_in_a=only_a, only_in_b=only_b, changed=changed,
                         total_a=len(a), total_b=len(b))


__all__ = ["parse_env_tool", "diff_env_tool"]
''')

# ── 18. string-case-converter ──────────────────────────────────────
create_skill(
    name="string-case-converter",
    frontmatter_name="converting-string-case",
    description="Convert strings between camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE. Use when the user wants to convert case, camelCase, snake_case.",
    category="development",
    capabilities=["code"],
    triggers=["camelCase", "snake_case", "kebab-case", "PascalCase", "case convert"],
    eval_tool="convert_case_tool",
    eval_input={"text": "hello_world_example", "to_case": "camelCase"},
    tool_docs="""### convert_case_tool
Convert string between naming conventions.

**Parameters:**
- `text` (str, required): Text to convert
- `to_case` (str, required): Target: camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE, Title Case

**Returns:**
- `success` (bool)
- `result` (str): Converted string""",
    tools_code='''"""String Case Converter Skill."""
import re
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("string-case-converter")


def _split_words(text: str) -> list:
    text = re.sub(r"([a-z])([A-Z])", r"\\1 \\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1 \\2", text)
    return re.split(r"[\\s_\\-]+", text.strip())


@tool_wrapper(required_params=["text", "to_case"])
def convert_case_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string between naming conventions."""
    status.set_callback(params.pop("_status_callback", None))
    words = _split_words(params["text"])
    target = params["to_case"].lower().replace(" ", "").replace("_", "")

    if target == "camelcase":
        result = words[0].lower() + "".join(w.capitalize() for w in words[1:])
    elif target == "pascalcase":
        result = "".join(w.capitalize() for w in words)
    elif target in ("snakecase", "snake"):
        result = "_".join(w.lower() for w in words)
    elif target in ("kebabcase", "kebab"):
        result = "-".join(w.lower() for w in words)
    elif target in ("uppercase", "upper", "screamingsnake"):
        result = "_".join(w.upper() for w in words)
    elif target in ("titlecase", "title"):
        result = " ".join(w.capitalize() for w in words)
    elif target in ("lowercase", "lower"):
        result = " ".join(w.lower() for w in words)
    elif target in ("dotcase", "dot"):
        result = ".".join(w.lower() for w in words)
    else:
        return tool_error(f"Unknown case: {params['to_case']}. Use: camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE, Title Case")

    return tool_response(result=result, from_words=words, to_case=params["to_case"])


__all__ = ["convert_case_tool"]
''')

# ── 19. ip-lookup ──────────────────────────────────────────────────
create_skill(
    name="ip-lookup",
    frontmatter_name="looking-up-ip",
    description="Look up IP address geolocation, ISP, and network info. Use when the user wants to lookup IP, geolocate IP, find IP location, my IP.",
    category="development",
    capabilities=["data-fetch"],
    triggers=["ip", "ip lookup", "geolocate", "my ip", "ip address", "whois"],
    eval_tool="ip_lookup_tool",
    eval_input={"ip": "8.8.8.8"},
    tool_docs="""### ip_lookup_tool
Look up geolocation and network info for an IP address.

**Parameters:**
- `ip` (str, optional): IP address (default: your public IP)

**Returns:**
- `success` (bool)
- `ip` (str): IP address
- `country` (str): Country
- `city` (str): City
- `isp` (str): Internet Service Provider""",
    tools_code='''"""IP Lookup Skill — geolocation and network info."""
import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("ip-lookup")


@tool_wrapper()
def ip_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Look up IP geolocation and network info."""
    status.set_callback(params.pop("_status_callback", None))
    ip = params.get("ip", "")

    try:
        url = f"http://ip-api.com/json/{ip}" if ip else "http://ip-api.com/json/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "fail":
            return tool_error(data.get("message", "Lookup failed"))

        return tool_response(
            ip=data.get("query", ip),
            country=data.get("country", ""),
            country_code=data.get("countryCode", ""),
            region=data.get("regionName", ""),
            city=data.get("city", ""),
            zip=data.get("zip", ""),
            lat=data.get("lat"),
            lon=data.get("lon"),
            timezone=data.get("timezone", ""),
            isp=data.get("isp", ""),
            org=data.get("org", ""),
            as_number=data.get("as", ""),
        )
    except requests.RequestException as e:
        return tool_error(f"Lookup failed: {e}")


__all__ = ["ip_lookup_tool"]
''')

# ── 20. dns-lookup ─────────────────────────────────────────────────
create_skill(
    name="dns-lookup",
    frontmatter_name="looking-up-dns",
    description="Perform DNS lookups — A, AAAA, MX, CNAME, TXT, NS records. Use when the user wants to DNS lookup, resolve domain, check DNS records, MX records.",
    category="development",
    capabilities=["data-fetch", "devops"],
    triggers=["dns", "dns lookup", "resolve domain", "mx records", "nameserver", "dig"],
    eval_tool="dns_lookup_tool",
    eval_input={"domain": "example.com", "record_type": "A"},
    tool_docs="""### dns_lookup_tool
Perform DNS lookup for a domain.

**Parameters:**
- `domain` (str, required): Domain to look up
- `record_type` (str, optional): A, AAAA, MX, CNAME, TXT, NS, SOA (default: A)

**Returns:**
- `success` (bool)
- `domain` (str): Queried domain
- `records` (list): DNS records found""",
    tools_code='''"""DNS Lookup Skill — resolve domain records."""
import socket
import requests
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("dns-lookup")


@tool_wrapper(required_params=["domain"])
def dns_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform DNS lookup using DNS-over-HTTPS (Cloudflare)."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"].strip().rstrip(".")
    record_type = params.get("record_type", "A").upper()
    valid_types = {"A", "AAAA", "MX", "CNAME", "TXT", "NS", "SOA", "SRV", "PTR"}

    if record_type not in valid_types:
        return tool_error(f"Invalid record type. Use one of: {sorted(valid_types)}")

    try:
        resp = requests.get(
            "https://cloudflare-dns.com/dns-query",
            params={"name": domain, "type": record_type},
            headers={"Accept": "application/dns-json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        records = []
        for answer in data.get("Answer", []):
            records.append({
                "name": answer.get("name", ""),
                "type": answer.get("type", 0),
                "ttl": answer.get("TTL", 0),
                "data": answer.get("data", ""),
            })

        return tool_response(domain=domain, record_type=record_type,
                             records=records, count=len(records),
                             status_code=data.get("Status", -1))
    except requests.RequestException as e:
        return tool_error(f"DNS lookup failed: {e}")


@tool_wrapper(required_params=["domain"])
def dns_all_records_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get all common DNS record types for a domain."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"]
    all_records = {}
    for rtype in ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]:
        result = dns_lookup_tool({"domain": domain, "record_type": rtype})
        if result.get("success") and result.get("records"):
            all_records[rtype] = result["records"]
    return tool_response(domain=domain, records=all_records)


__all__ = ["dns_lookup_tool", "dns_all_records_tool"]
''')

print(f"\\nBatch 1 complete: 20 skills created.")
