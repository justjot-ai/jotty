"""Batch 5b: Pure Python utility skills (10 skills)."""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from generate_skills import create_skill

SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _get_regex_builder_code() -> str:
    """Return regex-builder tools.py code, creating it if needed."""
    tools_path = SKILLS_DIR / "regex-builder" / "tools.py"
    if tools_path.exists():
        return tools_path.read_text()
    # First-run: write the file, then return its content
    tools_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        '"""Regex Builder Skill — build, test, and explain regex patterns."""\n'
        'import re\n'
        'from typing import Dict, Any\n'
        '\n'
        'from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper\n'
        'from Jotty.core.utils.skill_status import SkillStatus\n'
        '\n'
        'status = SkillStatus("regex-builder")\n'
        '\n'
        '_PRESETS = {\n'
        '    "email": {"pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "description": "Email address"},\n'
        '    "url": {"pattern": r\'https?://[^\\s<>"]+\', "description": "HTTP/HTTPS URL"},\n'
        '    "phone": {"pattern": r"\\+?1?[-.\\s]?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}", "description": "US phone number"},\n'
        '    "ipv4": {"pattern": r"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b", "description": "IPv4 address"},\n'
        '    "date_iso": {"pattern": r"\\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\\d|3[01])", "description": "ISO date (YYYY-MM-DD)"},\n'
        '    "uuid": {"pattern": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "description": "UUID v4"},\n'
        '    "hex_color": {"pattern": r"#(?:[0-9a-fA-F]{3}){1,2}\\b", "description": "Hex color code"},\n'
        '    "ip": {"pattern": r"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b", "description": "IPv4 address"},\n'
        '    "date": {"pattern": r"\\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\\d|3[01])", "description": "ISO date"},\n'
        '}\n'
        '\n'
        '_EXPLAIN = {\n'
        '    "\\\\d": "digit (0-9)", "\\\\w": "word char (a-z, A-Z, 0-9, _)", "\\\\s": "whitespace",\n'
        '    "\\\\b": "word boundary", ".": "any character", "+": "one or more", "*": "zero or more",\n'
        '    "?": "zero or one (optional)", "^": "start of string", "$": "end of string",\n'
        '    "\\\\D": "non-digit", "\\\\W": "non-word char", "\\\\S": "non-whitespace",\n'
        '}\n'
        '\n'
        '\n'
        '@tool_wrapper(required_params=["action"])\n'
        'def regex_tool(params: Dict[str, Any]) -> Dict[str, Any]:\n'
        '    """Build, test, or explain regular expressions."""\n'
        '    status.set_callback(params.pop("_status_callback", None))\n'
        '    action = params["action"]\n'
        '\n'
        '    if action == "preset":\n'
        '        name = params.get("name", "")\n'
        '        if not name:\n'
        '            return tool_response(available=list(_PRESETS.keys()))\n'
        '        preset = _PRESETS.get(name)\n'
        '        if not preset:\n'
        '            return tool_error(f"Unknown preset: {name}. Available: {list(_PRESETS.keys())}")\n'
        '        return tool_response(name=name, **preset)\n'
        '\n'
        '    if action == "test":\n'
        '        pattern = params.get("pattern", "")\n'
        '        text = params.get("text", "")\n'
        '        if not pattern or not text:\n'
        '            return tool_error("pattern and text required for test")\n'
        '        try:\n'
        '            matches = re.findall(pattern, text)\n'
        '            full = bool(re.fullmatch(pattern, text))\n'
        '            return tool_response(pattern=pattern, matches=matches, count=len(matches), full_match=full)\n'
        '        except re.error as e:\n'
        '            return tool_error(f"Invalid regex: {e}")\n'
        '\n'
        '    if action == "explain":\n'
        '        pattern = params.get("pattern", "")\n'
        '        if not pattern:\n'
        '            return tool_error("pattern required for explain")\n'
        '        parts = []\n'
        '        for token, desc in _EXPLAIN.items():\n'
        '            if token in pattern:\n'
        '                parts.append({"token": token, "meaning": desc})\n'
        '        return tool_response(pattern=pattern, components=parts)\n'
        '\n'
        '    return tool_error(f"Unknown action: {action}. Use: preset, test, explain")\n'
        '\n'
        '\n'
        '__all__ = ["regex_tool"]\n'
    )
    tools_path.write_text(code)
    return code

# ── 1. json-diff ────────────────────────────────────────────────────
create_skill(
    name="json-diff",
    frontmatter_name="comparing-json",
    description="Compare two JSON objects and find additions, deletions, and modifications with JSON path.",
    category="development",
    capabilities=["code"],
    triggers=["json diff", "compare json", "json difference", "diff objects"],
    eval_tool="json_diff_tool",
    eval_input={"a": {"x": 1}, "b": {"x": 2}},
    tool_docs="""### json_diff_tool
Compare two JSON objects.

**Parameters:**
- `a` (dict/str, required): First JSON object or JSON string
- `b` (dict/str, required): Second JSON object or JSON string

**Returns:**
- `added` (list): Paths present in b but not a
- `removed` (list): Paths present in a but not b
- `modified` (list): Paths with different values""",
    tools_code='''"""JSON Diff Skill — compare two JSON objects with path tracking."""
import json
from typing import Dict, Any, List, Tuple

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("json-diff")


def _diff(a: Any, b: Any, path: str = "$") -> Tuple[List, List, List]:
    added, removed, modified = [], [], []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            p = f"{path}.{k}"
            if k not in a:
                added.append({"path": p, "value": b[k]})
            elif k not in b:
                removed.append({"path": p, "value": a[k]})
            else:
                a2, r2, m2 = _diff(a[k], b[k], p)
                added.extend(a2); removed.extend(r2); modified.extend(m2)
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            p = f"{path}[{i}]"
            if i >= len(a):
                added.append({"path": p, "value": b[i]})
            elif i >= len(b):
                removed.append({"path": p, "value": a[i]})
            else:
                a2, r2, m2 = _diff(a[i], b[i], p)
                added.extend(a2); removed.extend(r2); modified.extend(m2)
    elif a != b:
        modified.append({"path": path, "old": a, "new": b})
    return added, removed, modified


def _parse(v: Any) -> Any:
    if isinstance(v, str):
        return json.loads(v)
    return v


@tool_wrapper(required_params=["a", "b"])
def json_diff_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two JSON objects and report differences."""
    status.set_callback(params.pop("_status_callback", None))
    a = _parse(params["a"])
    b = _parse(params["b"])
    added, removed, modified = _diff(a, b)
    return tool_response(
        added=added, removed=removed, modified=modified,
        total_changes=len(added) + len(removed) + len(modified),
    )


__all__ = ["json_diff_tool"]
''',
)

# ── 2. semver-manager ───────────────────────────────────────────────
create_skill(
    name="semver-manager",
    frontmatter_name="managing-semver",
    description="Parse, compare, and bump semantic versions (major.minor.patch). Check version constraints.",
    category="development",
    capabilities=["code"],
    triggers=["semver", "semantic version", "bump version", "version compare"],
    eval_tool="semver_tool",
    eval_input={"action": "bump", "version": "1.2.3", "part": "minor"},
    tool_docs="""### semver_tool
Parse, compare, or bump semantic versions.

**Parameters:**
- `action` (str, required): parse, bump, compare, satisfies
- `version` (str, required): Semantic version string
- `part` (str): major/minor/patch (for bump)
- `other` (str): Second version (for compare)
- `constraint` (str): e.g. >=1.0.0 (for satisfies)""",
    tools_code='''"""Semver Manager Skill — parse, compare, bump semantic versions."""
import re
from typing import Dict, Any, Tuple, Optional

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("semver-manager")

_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([\w.]+))?(?:\+([\w.]+))?$")


def _parse(v: str) -> Tuple[int, int, int, Optional[str], Optional[str]]:
    m = _RE.match(v.strip())
    if not m:
        raise ValueError(f"Invalid semver: {v}")
    return int(m[1]), int(m[2]), int(m[3]), m[4], m[5]


def _fmt(ma: int, mi: int, pa: int, pre: Optional[str] = None, bld: Optional[str] = None) -> str:
    s = f"{ma}.{mi}.{pa}"
    if pre:
        s += f"-{pre}"
    if bld:
        s += f"+{bld}"
    return s


def _cmp(a: str, b: str) -> int:
    a1, a2, a3, _, _ = _parse(a)
    b1, b2, b3, _, _ = _parse(b)
    for x, y in [(a1, b1), (a2, b2), (a3, b3)]:
        if x != y:
            return 1 if x > y else -1
    return 0


@tool_wrapper(required_params=["action", "version"])
def semver_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Manage semantic versions."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]
    ver = params["version"]

    if action == "parse":
        ma, mi, pa, pre, bld = _parse(ver)
        return tool_response(major=ma, minor=mi, patch=pa, prerelease=pre, build=bld)

    if action == "bump":
        ma, mi, pa, _, _ = _parse(ver)
        part = params.get("part", "patch")
        if part == "major":
            ma, mi, pa = ma + 1, 0, 0
        elif part == "minor":
            mi, pa = mi + 1, 0
        else:
            pa += 1
        return tool_response(original=ver, bumped=_fmt(ma, mi, pa), part=part)

    if action == "compare":
        other = params.get("other", "")
        if not other:
            return tool_error("other parameter required for compare")
        c = _cmp(ver, other)
        rel = "equal" if c == 0 else ("greater" if c > 0 else "less")
        return tool_response(version=ver, other=other, result=c, relation=rel)

    if action == "satisfies":
        constraint = params.get("constraint", "")
        if not constraint:
            return tool_error("constraint parameter required")
        m = re.match(r"^([><=!]+)\s*(.+)$", constraint.strip())
        if not m:
            return tool_error(f"Invalid constraint: {constraint}")
        op, cv = m[1], m[2]
        c = _cmp(ver, cv)
        ok = {">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0,
              "==": c == 0, "!=": c != 0, "=": c == 0}.get(op)
        if ok is None:
            return tool_error(f"Unknown operator: {op}")
        return tool_response(version=ver, constraint=constraint, satisfies=ok)

    return tool_error(f"Unknown action: {action}. Use: parse, bump, compare, satisfies")


__all__ = ["semver_tool"]
''',
)

# ── 3. slug-generator ───────────────────────────────────────────────
create_skill(
    name="slug-generator",
    frontmatter_name="generating-slugs",
    description="Generate URL-friendly slugs from titles. Handle unicode, transliteration, custom separators.",
    category="development",
    capabilities=["code"],
    triggers=["slug", "url slug", "slugify", "url-friendly"],
    eval_tool="slugify_tool",
    eval_input={"text": "Hello World!"},
    tool_docs="""### slugify_tool
Generate a URL-friendly slug from text.

**Parameters:**
- `text` (str, required): Text to slugify
- `separator` (str): Separator character (default: -)
- `max_length` (int): Maximum slug length (default: 200)
- `lowercase` (bool): Force lowercase (default: true)""",
    tools_code='''"""Slug Generator Skill — create URL-friendly slugs from text."""
import re
import unicodedata
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("slug-generator")

# Common unicode transliteration replacements
_REPLACEMENTS = {
    "\\u00e4": "ae", "\\u00f6": "oe", "\\u00fc": "ue",
    "\\u00c4": "Ae", "\\u00d6": "Oe", "\\u00dc": "Ue",
    "\\u00df": "ss", "\\u00e9": "e", "\\u00e8": "e",
    "\\u00e0": "a", "\\u00e2": "a", "\\u00f4": "o",
    "\\u00e7": "c", "\\u00f1": "n", "\\u00ee": "i", "\\u00f9": "u",
}


def _transliterate(text: str) -> str:
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@tool_wrapper(required_params=["text"])
def slugify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a URL-friendly slug from text."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    sep = params.get("separator", "-")
    max_len = params.get("max_length", 200)
    lower = params.get("lowercase", True)

    slug = _transliterate(text)
    if lower:
        slug = slug.lower()
    slug = re.sub(r"[^\\w\\s-]", "", slug)
    slug = re.sub(r"[\\s_-]+", sep, slug).strip(sep)
    if max_len and len(slug) > max_len:
        slug = slug[:max_len].rstrip(sep)
    return tool_response(slug=slug, original=text, length=len(slug))


__all__ = ["slugify_tool"]
''',
)

# ── 4. emoji-lookup ─────────────────────────────────────────────────
create_skill(
    name="emoji-lookup",
    frontmatter_name="looking-up-emojis",
    description="Search emojis by name/keyword, get emoji info, convert shortcodes to unicode. Built-in emoji database.",
    category="utilities",
    capabilities=["code"],
    triggers=["emoji", "emoji search", "emoji lookup", "shortcode to emoji"],
    eval_tool="emoji_lookup_tool",
    eval_input={"action": "search", "query": "smile"},
    tool_docs="""### emoji_lookup_tool
Search and convert emojis.

**Parameters:**
- `action` (str, required): search, info, convert
- `query` (str): Search term (for search)
- `emoji` (str): Emoji char (for info)
- `shortcode` (str): Shortcode like :smile: (for convert)""",
    tools_code='''"""Emoji Lookup Skill — search emojis, get info, convert shortcodes."""
import unicodedata
import re
from typing import Dict, Any, List

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("emoji-lookup")

_DB: Dict[str, Dict[str, Any]] = {
    ":smile:": {"emoji": "\\U0001f604", "name": "smiling face with open mouth and smiling eyes", "category": "faces"},
    ":grinning:": {"emoji": "\\U0001f600", "name": "grinning face", "category": "faces"},
    ":heart:": {"emoji": "\\u2764\\ufe0f", "name": "red heart", "category": "symbols"},
    ":thumbsup:": {"emoji": "\\U0001f44d", "name": "thumbs up", "category": "hands"},
    ":thumbsdown:": {"emoji": "\\U0001f44e", "name": "thumbs down", "category": "hands"},
    ":fire:": {"emoji": "\\U0001f525", "name": "fire", "category": "nature"},
    ":star:": {"emoji": "\\u2b50", "name": "star", "category": "symbols"},
    ":sun:": {"emoji": "\\u2600\\ufe0f", "name": "sun", "category": "nature"},
    ":moon:": {"emoji": "\\U0001f319", "name": "crescent moon", "category": "nature"},
    ":rocket:": {"emoji": "\\U0001f680", "name": "rocket", "category": "travel"},
    ":check:": {"emoji": "\\u2705", "name": "check mark", "category": "symbols"},
    ":x:": {"emoji": "\\u274c", "name": "cross mark", "category": "symbols"},
    ":warning:": {"emoji": "\\u26a0\\ufe0f", "name": "warning", "category": "symbols"},
    ":wave:": {"emoji": "\\U0001f44b", "name": "waving hand", "category": "hands"},
    ":clap:": {"emoji": "\\U0001f44f", "name": "clapping hands", "category": "hands"},
    ":cry:": {"emoji": "\\U0001f622", "name": "crying face", "category": "faces"},
    ":laugh:": {"emoji": "\\U0001f602", "name": "face with tears of joy", "category": "faces"},
    ":think:": {"emoji": "\\U0001f914", "name": "thinking face", "category": "faces"},
    ":100:": {"emoji": "\\U0001f4af", "name": "hundred points", "category": "symbols"},
    ":party:": {"emoji": "\\U0001f389", "name": "party popper", "category": "objects"},
    ":globe:": {"emoji": "\\U0001f30d", "name": "globe showing Europe-Africa", "category": "travel"},
    ":coffee:": {"emoji": "\\u2615", "name": "hot beverage", "category": "food"},
    ":bug:": {"emoji": "\\U0001f41b", "name": "bug", "category": "nature"},
    ":lock:": {"emoji": "\\U0001f512", "name": "locked", "category": "objects"},
    ":key:": {"emoji": "\\U0001f511", "name": "key", "category": "objects"},
}


@tool_wrapper(required_params=["action"])
def emoji_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search, get info, or convert emoji shortcodes."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "search":
        query = params.get("query", "").lower()
        if not query:
            return tool_error("query required for search")
        results = []
        for code, info in _DB.items():
            if query in code or query in info["name"]:
                results.append({"shortcode": code, **info})
        return tool_response(results=results, count=len(results))

    if action == "info":
        emoji = params.get("emoji", "")
        if not emoji:
            return tool_error("emoji parameter required")
        name = unicodedata.name(emoji[0], "unknown")
        cp = "+".join(f"U+{ord(c):04X}" for c in emoji)
        return tool_response(emoji=emoji, name=name.lower(), codepoints=cp)

    if action == "convert":
        shortcode = params.get("shortcode", "")
        if not shortcode:
            return tool_error("shortcode parameter required")
        if not shortcode.startswith(":"):
            shortcode = f":{shortcode}:"
        if not shortcode.endswith(":"):
            shortcode = f"{shortcode}:"
        entry = _DB.get(shortcode)
        if not entry:
            return tool_error(f"Unknown shortcode: {shortcode}")
        return tool_response(emoji=entry["emoji"], name=entry["name"], shortcode=shortcode)

    return tool_error(f"Unknown action: {action}. Use: search, info, convert")


__all__ = ["emoji_lookup_tool"]
''',
)

# ── 5. regex-builder ────────────────────────────────────────────────
create_skill(
    name="regex-builder",
    frontmatter_name="building-regex",
    description="Build regex patterns from descriptions. Common patterns: email, URL, phone, IP, date. Explain regex.",
    category="development",
    capabilities=["code"],
    triggers=["regex", "regular expression", "pattern", "regex builder"],
    eval_tool="regex_tool",
    eval_input={"action": "preset", "name": "email"},
    tool_docs="""### regex_tool
Build, test, or explain regex patterns.

**Parameters:**
- `action` (str, required): preset, test, explain
- `name` (str): Preset name (for preset): email, url, phone, ip, date, uuid
- `pattern` (str): Regex pattern (for test/explain)
- `text` (str): Text to test against (for test)""",
    tools_code=_get_regex_builder_code(),
)

# ── 6. binary-converter ─────────────────────────────────────────────
create_skill(
    name="binary-converter",
    frontmatter_name="converting-binary",
    description="Convert between binary, decimal, hex, octal. Bitwise operations (AND, OR, XOR, NOT, shift).",
    category="utilities",
    capabilities=["code"],
    triggers=["binary", "hex", "octal", "bitwise", "binary convert"],
    eval_tool="binary_convert_tool",
    eval_input={"action": "convert", "value": "255", "from_base": "decimal", "to_base": "binary"},
    tool_docs="""### binary_convert_tool
Convert numbers and perform bitwise operations.

**Parameters:**
- `action` (str, required): convert or bitwise
- `value` (str): Number to convert (for convert)
- `from_base` (str): binary/decimal/hex/octal (for convert)
- `to_base` (str): binary/decimal/hex/octal (for convert)
- `op` (str): AND/OR/XOR/NOT/LSHIFT/RSHIFT (for bitwise)
- `a` (int): First operand (for bitwise)
- `b` (int): Second operand (for bitwise)""",
    tools_code='''"""Binary Converter Skill — convert bases and bitwise operations."""
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("binary-converter")

_BASES = {"binary": 2, "decimal": 10, "hex": 16, "octal": 8}
_PREFIX = {"binary": "0b", "hex": "0x", "octal": "0o", "decimal": ""}


def _to_int(value: str, base_name: str) -> int:
    b = _BASES.get(base_name)
    if b is None:
        raise ValueError(f"Unknown base: {base_name}")
    v = value.strip().lower().replace("0b", "").replace("0x", "").replace("0o", "")
    return int(v, b)


def _from_int(n: int, base_name: str) -> str:
    if base_name == "binary":
        return bin(n)
    if base_name == "hex":
        return hex(n)
    if base_name == "octal":
        return oct(n)
    return str(n)


@tool_wrapper(required_params=["action"])
def binary_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between number bases or perform bitwise operations."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "convert":
        value = params.get("value", "")
        fb = params.get("from_base", "decimal")
        tb = params.get("to_base", "binary")
        if not value:
            return tool_error("value required")
        n = _to_int(str(value), fb)
        result = _from_int(n, tb)
        return tool_response(original=value, from_base=fb, to_base=tb,
                             result=result, decimal_value=n)

    if action == "bitwise":
        op = params.get("op", "").upper()
        a = int(params.get("a", 0))
        if op == "NOT":
            return tool_response(op=op, a=a, result=~a, binary=bin(~a & 0xFFFFFFFF))
        b = int(params.get("b", 0))
        ops = {"AND": a & b, "OR": a | b, "XOR": a ^ b,
               "LSHIFT": a << b, "RSHIFT": a >> b}
        if op not in ops:
            return tool_error(f"Unknown op: {op}. Use: AND, OR, XOR, NOT, LSHIFT, RSHIFT")
        r = ops[op]
        return tool_response(op=op, a=a, b=b, result=r, binary=bin(r))

    return tool_error(f"Unknown action: {action}. Use: convert, bitwise")


__all__ = ["binary_convert_tool"]
''',
)

# ── 7. number-base-converter ────────────────────────────────────────
create_skill(
    name="number-base-converter",
    frontmatter_name="converting-number-bases",
    description="Convert numbers between arbitrary bases (2-36). Support custom digit sets.",
    category="utilities",
    capabilities=["code"],
    triggers=["base convert", "number base", "base 2", "base 16", "radix"],
    eval_tool="base_convert_tool",
    eval_input={"number": "ff", "from_base": 16, "to_base": 10},
    tool_docs="""### base_convert_tool
Convert numbers between arbitrary bases 2-36.

**Parameters:**
- `number` (str, required): The number to convert
- `from_base` (int, required): Source base (2-36)
- `to_base` (int, required): Target base (2-36)
- `custom_digits` (str): Custom digit set (optional)""",
    tools_code='''"""Number Base Converter Skill — arbitrary base conversion (2-36)."""
import string
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("number-base-converter")

_DIGITS = string.digits + string.ascii_lowercase  # 0-9a-z = 36 chars


def _to_decimal(number: str, base: int, digits: str = _DIGITS) -> int:
    number = number.strip().lower()
    result = 0
    for ch in number:
        val = digits.index(ch)
        if val >= base:
            raise ValueError(f"Digit '{ch}' invalid for base {base}")
        result = result * base + val
    return result


def _from_decimal(n: int, base: int, digits: str = _DIGITS) -> str:
    if n == 0:
        return digits[0]
    negative = n < 0
    n = abs(n)
    chars = []
    while n > 0:
        chars.append(digits[n % base])
        n //= base
    if negative:
        chars.append("-")
    return "".join(reversed(chars))


@tool_wrapper(required_params=["number", "from_base", "to_base"])
def base_convert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a number between arbitrary bases (2-36)."""
    status.set_callback(params.pop("_status_callback", None))
    number = str(params["number"])
    fb = int(params["from_base"])
    tb = int(params["to_base"])

    if not (2 <= fb <= 36) or not (2 <= tb <= 36):
        return tool_error("Bases must be between 2 and 36")

    digits = params.get("custom_digits", _DIGITS)
    decimal_val = _to_decimal(number, fb, digits)
    result = _from_decimal(decimal_val, tb, digits)
    return tool_response(
        original=number, from_base=fb, to_base=tb,
        result=result, decimal_value=decimal_val,
    )


__all__ = ["base_convert_tool"]
''',
)

# ── 8. roman-numeral-converter ──────────────────────────────────────
create_skill(
    name="roman-numeral-converter",
    frontmatter_name="converting-roman-numerals",
    description="Convert between Roman numerals and integers. Validate Roman numeral strings.",
    category="utilities",
    capabilities=["code"],
    triggers=["roman numeral", "roman to integer", "integer to roman", "roman convert"],
    eval_tool="roman_tool",
    eval_input={"action": "to_roman", "number": 2024},
    tool_docs="""### roman_tool
Convert between Roman numerals and integers.

**Parameters:**
- `action` (str, required): to_roman, to_integer, validate
- `number` (int): Integer to convert (for to_roman)
- `roman` (str): Roman numeral string (for to_integer/validate)""",
    tools_code='''"""Roman Numeral Converter Skill — convert between Roman numerals and integers."""
import re
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("roman-numeral-converter")

_TO_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]
_FROM_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_VALID_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _int_to_roman(n: int) -> str:
    parts = []
    for value, numeral in _TO_ROMAN:
        while n >= value:
            parts.append(numeral)
            n -= value
    return "".join(parts)


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _FROM_ROMAN.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


@tool_wrapper(required_params=["action"])
def roman_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between Roman numerals and integers."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "to_roman":
        n = params.get("number")
        if n is None:
            return tool_error("number required")
        n = int(n)
        if n < 1 or n > 3999:
            return tool_error("Number must be 1-3999")
        return tool_response(number=n, roman=_int_to_roman(n))

    if action == "to_integer":
        roman = params.get("roman", "")
        if not roman:
            return tool_error("roman required")
        val = _roman_to_int(roman)
        return tool_response(roman=roman.upper(), number=val)

    if action == "validate":
        roman = params.get("roman", "")
        if not roman:
            return tool_error("roman required")
        valid = bool(_VALID_RE.match(roman.upper().strip()))
        return tool_response(roman=roman.upper(), valid=valid)

    return tool_error(f"Unknown action: {action}. Use: to_roman, to_integer, validate")


__all__ = ["roman_tool"]
''',
)

# ── 9. morse-code-translator ────────────────────────────────────────
create_skill(
    name="morse-code-translator",
    frontmatter_name="translating-morse-code",
    description="Convert text to/from Morse code. Support letters, numbers, common punctuation.",
    category="utilities",
    capabilities=["code"],
    triggers=["morse code", "morse translate", "text to morse", "morse to text"],
    eval_tool="morse_tool",
    eval_input={"action": "encode", "text": "HELLO"},
    tool_docs="""### morse_tool
Convert text to/from Morse code.

**Parameters:**
- `action` (str, required): encode or decode
- `text` (str): Text to encode (for encode)
- `morse` (str): Morse code to decode (for decode), dots and dashes separated by spaces""",
    tools_code='''"""Morse Code Translator Skill — encode/decode Morse code."""
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("morse-code-translator")

_ENCODE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.",
    "!": "-.-.--", "/": "-..-.", "(": "-.--.", ")": "-.--.-",
    "&": ".-...", ":": "---...", ";": "-.-.-.", "=": "-...-",
    "+": ".-.-.", "-": "-....-", "_": "..--.-", '"': ".-..-.",
    "$": "...-..-", "@": ".--.-.",
}
_DECODE = {v: k for k, v in _ENCODE.items()}


@tool_wrapper(required_params=["action"])
def morse_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Encode text to Morse code or decode Morse to text."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "encode":
        text = params.get("text", "")
        if not text:
            return tool_error("text required")
        words = text.upper().split()
        coded_words = []
        for word in words:
            letters = []
            for ch in word:
                code = _ENCODE.get(ch)
                if code:
                    letters.append(code)
            coded_words.append(" ".join(letters))
        morse = " / ".join(coded_words)
        return tool_response(text=text, morse=morse)

    if action == "decode":
        morse = params.get("morse", "")
        if not morse:
            return tool_error("morse required")
        words = morse.strip().split(" / ")
        decoded = []
        for word in words:
            chars = []
            for code in word.strip().split():
                ch = _DECODE.get(code, "?")
                chars.append(ch)
            decoded.append("".join(chars))
        text = " ".join(decoded)
        return tool_response(morse=morse, text=text)

    return tool_error(f"Unknown action: {action}. Use: encode, decode")


__all__ = ["morse_tool"]
''',
)

# ── 10. nato-phonetic-alphabet ──────────────────────────────────────
create_skill(
    name="nato-phonetic-alphabet",
    frontmatter_name="spelling-with-nato",
    description="Convert text to NATO phonetic alphabet spelling. ABC -> Alfa Bravo Charlie.",
    category="utilities",
    capabilities=["code"],
    triggers=["nato phonetic", "phonetic alphabet", "nato spelling", "spell out"],
    eval_tool="nato_tool",
    eval_input={"action": "encode", "text": "ABC"},
    tool_docs="""### nato_tool
Convert text to/from NATO phonetic alphabet.

**Parameters:**
- `action` (str, required): encode or decode
- `text` (str): Text to spell out (for encode)
- `words` (str): NATO words to decode (for decode)""",
    tools_code='''"""NATO Phonetic Alphabet Skill — convert text to NATO spelling."""
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("nato-phonetic-alphabet")

_NATO = {
    "A": "Alfa", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo",
    "F": "Foxtrot", "G": "Golf", "H": "Hotel", "I": "India", "J": "Juliett",
    "K": "Kilo", "L": "Lima", "M": "Mike", "N": "November", "O": "Oscar",
    "P": "Papa", "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango",
    "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray", "Y": "Yankee",
    "Z": "Zulu",
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Niner",
}
_REVERSE = {v.upper(): k for k, v in _NATO.items()}


@tool_wrapper(required_params=["action"])
def nato_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert text to NATO phonetic alphabet or decode back."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "encode":
        text = params.get("text", "")
        if not text:
            return tool_error("text required")
        result = []
        for ch in text.upper():
            if ch == " ":
                result.append("[space]")
            elif ch in _NATO:
                result.append(_NATO[ch])
            else:
                result.append(ch)
        return tool_response(
            text=text, nato=" ".join(result),
            words=result, count=len(result),
        )

    if action == "decode":
        words_str = params.get("words", "")
        if not words_str:
            return tool_error("words required")
        words = words_str.split()
        chars = []
        for w in words:
            if w == "[space]":
                chars.append(" ")
            else:
                chars.append(_REVERSE.get(w.upper(), w))
        text = "".join(chars)
        return tool_response(words=words_str, text=text)

    return tool_error(f"Unknown action: {action}. Use: encode, decode")


__all__ = ["nato_tool"]
''',
)

print(f"\nBatch 5b complete: 10 skills generated.")
