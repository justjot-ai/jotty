"""
Batch 4a Skill Generator — 10 developer-tooling skills.

Usage:
    python scripts/gen_batch4a.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_skills import create_skill

# ---------------------------------------------------------------------------
# 1. sql-query-builder
# ---------------------------------------------------------------------------
SQL_QUERY_BUILDER = '''\
"""Build SQL queries programmatically (SELECT, INSERT, UPDATE, DELETE)."""
from typing import Dict, Any, List, Optional
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("sql-query-builder")


def _quote(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    return "'" + str(val).replace("'", "''") + "'"


def _build_where(conditions: List[Dict[str, Any]]) -> str:
    if not conditions:
        return ""
    parts = []
    for c in conditions:
        col = c["column"]
        op = c.get("op", "=")
        val = _quote(c["value"])
        parts.append(f"{col} {op} {val}")
    joiner = " AND "
    return " WHERE " + joiner.join(parts)


@tool_wrapper(required_params=["operation", "table"])
def build_sql_query(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a SQL query string from structured parameters.

    Params:
        operation: SELECT | INSERT | UPDATE | DELETE
        table: table name
        columns: list of column names (SELECT/INSERT)
        values: list of values or list-of-lists (INSERT)
        updates: dict of column->value (UPDATE)
        conditions: list of {column, op, value} (WHERE)
        joins: list of {table, on, type} (SELECT)
        order_by: list of {column, direction} (SELECT)
        limit: int (SELECT)
        group_by: list of columns (SELECT)
    """
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].upper()
    table = params["table"]
    columns = params.get("columns", ["*"])
    conditions = params.get("conditions", [])
    where = _build_where(conditions)

    if op == "SELECT":
        cols = ", ".join(columns)
        sql = f"SELECT {cols} FROM {table}"
        for j in params.get("joins", []):
            jtype = j.get("type", "INNER").upper()
            sql += f" {jtype} JOIN {j['table']} ON {j['on']}"
        sql += where
        if params.get("group_by"):
            sql += " GROUP BY " + ", ".join(params["group_by"])
        for ob in params.get("order_by", []):
            direction = ob.get("direction", "ASC").upper()
            sql += f" ORDER BY {ob['column']} {direction}"
        if params.get("limit"):
            sql += f" LIMIT {int(params['limit'])}"
    elif op == "INSERT":
        cols = ", ".join(columns)
        rows = params.get("values", [])
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        row_strs = [", ".join(_quote(v) for v in row) for row in rows]
        vals = ", ".join(f"({r})" for r in row_strs)
        sql = f"INSERT INTO {table} ({cols}) VALUES {vals}"
    elif op == "UPDATE":
        updates = params.get("updates", {})
        if not updates:
            return tool_error("updates dict required for UPDATE")
        sets = ", ".join(f"{k} = {_quote(v)}" for k, v in updates.items())
        sql = f"UPDATE {table} SET {sets}{where}"
    elif op == "DELETE":
        sql = f"DELETE FROM {table}{where}"
    else:
        return tool_error(f"Unsupported operation: {op}")

    return tool_response(query=sql, operation=op, table=table)


__all__ = ["build_sql_query"]
'''

# ---------------------------------------------------------------------------
# 2. readme-generator
# ---------------------------------------------------------------------------
README_GENERATOR = '''\
"""Generate README.md files from project info."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("readme-generator")


@tool_wrapper(required_params=["name", "description"])
def generate_readme(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a README.md from project metadata.

    Params:
        name: project name
        description: short description
        features: list of feature strings
        installation: install instructions string
        usage: usage example string
        license: license name (e.g. MIT)
        badges: list of {label, url} badge dicts
        contributing: contributing guidelines
        author: author name
    """
    status.set_callback(params.pop("_status_callback", None))
    name = params["name"]
    desc = params["description"]
    sections = []

    # Title + description
    sections.append(f"# {name}\\n\\n{desc}")

    # Badges
    for b in params.get("badges", []):
        sections.append(f"![{b.get('label', '')}]({b.get('url', '')})")

    # Features
    features = params.get("features", [])
    if features:
        items = "\\n".join(f"- {f}" for f in features)
        sections.append(f"## Features\\n\\n{items}")

    # Installation
    install = params.get("installation", "")
    if install:
        sections.append(f"## Installation\\n\\n```bash\\n{install}\\n```")

    # Usage
    usage = params.get("usage", "")
    if usage:
        sections.append(f"## Usage\\n\\n```\\n{usage}\\n```")

    # Contributing
    contrib = params.get("contributing", "")
    if contrib:
        sections.append(f"## Contributing\\n\\n{contrib}")

    # Author
    author = params.get("author", "")
    if author:
        sections.append(f"## Author\\n\\n{author}")

    # License
    lic = params.get("license", "")
    if lic:
        sections.append(f"## License\\n\\n{lic}")

    readme = "\\n\\n".join(sections) + "\\n"
    return tool_response(readme=readme, sections=len(sections))


__all__ = ["generate_readme"]
'''

# ---------------------------------------------------------------------------
# 3. gitignore-generator
# ---------------------------------------------------------------------------
GITIGNORE_GENERATOR = '''\
"""Generate .gitignore files for languages/frameworks."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("gitignore-generator")

TEMPLATES: Dict[str, List[str]] = {
    "python": [
        "__pycache__/", "*.py[cod]", "*$py.class", "*.so", "dist/", "build/",
        "*.egg-info/", ".eggs/", "*.egg", ".venv/", "venv/", "env/",
        ".env", ".pytest_cache/", ".mypy_cache/", "*.pyo", "htmlcov/",
        ".coverage", ".tox/",
    ],
    "node": [
        "node_modules/", "npm-debug.log*", "yarn-debug.log*", "yarn-error.log*",
        ".npm", ".yarn-integrity", "dist/", "build/", ".env", ".env.local",
        ".env.*.local", "coverage/", ".next/", ".nuxt/",
    ],
    "java": [
        "*.class", "*.jar", "*.war", "*.ear", "target/", ".gradle/",
        "build/", ".settings/", ".project", ".classpath", "*.iml",
        ".idea/", "out/",
    ],
    "go": [
        "*.exe", "*.exe~", "*.dll", "*.so", "*.dylib", "*.test",
        "*.out", "vendor/", "go.sum", ".env",
    ],
    "rust": [
        "target/", "**/*.rs.bk", "Cargo.lock",
    ],
    "c": [
        "*.o", "*.obj", "*.so", "*.dylib", "*.dll", "*.a", "*.lib",
        "*.exe", "*.out", "build/", "cmake-build-*/",
    ],
    "general": [
        ".DS_Store", "Thumbs.db", "*.swp", "*.swo", "*~", ".idea/",
        ".vscode/", "*.log",
    ],
}


@tool_wrapper(required_params=["languages"])
def generate_gitignore(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a .gitignore file for given languages/frameworks.

    Params:
        languages: list of language keys (python, node, java, go, rust, c, general)
        extras: list of additional patterns to include
    """
    status.set_callback(params.pop("_status_callback", None))
    langs = params["languages"]
    if isinstance(langs, str):
        langs = [l.strip() for l in langs.split(",")]
    extras = params.get("extras", [])

    lines: List[str] = ["# Auto-generated .gitignore", ""]
    for lang in langs:
        key = lang.lower().strip()
        patterns = TEMPLATES.get(key, [])
        if not patterns:
            lines.append(f"# Unknown language: {key}")
            continue
        lines.append(f"# --- {key.title()} ---")
        lines.extend(patterns)
        lines.append("")

    if extras:
        lines.append("# --- Custom ---")
        lines.extend(extras)
        lines.append("")

    content = "\\n".join(lines)
    return tool_response(gitignore=content, languages=langs)


__all__ = ["generate_gitignore"]
'''

# ---------------------------------------------------------------------------
# 4. license-generator
# ---------------------------------------------------------------------------
LICENSE_GENERATOR = '''\
"""Generate open-source license text (MIT, Apache-2.0, GPL-3.0, BSD-2, ISC)."""
from typing import Dict, Any
from datetime import datetime
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("license-generator")

_MIT = """MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

_APACHE2 = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright {year} {author}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

_BSD2 = """BSD 2-Clause License

Copyright (c) {year}, {author}
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

_ISC = """ISC License

Copyright (c) {year}, {author}

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE."""

_GPL3_SHORT = """GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (C) {year} {author}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>."""

LICENSES = {
    "mit": _MIT,
    "apache-2.0": _APACHE2,
    "apache2": _APACHE2,
    "gpl-3.0": _GPL3_SHORT,
    "gpl3": _GPL3_SHORT,
    "bsd-2": _BSD2,
    "bsd2": _BSD2,
    "isc": _ISC,
}


@tool_wrapper(required_params=["license_type", "author"])
def generate_license(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate license text for a given license type.

    Params:
        license_type: mit | apache-2.0 | gpl-3.0 | bsd-2 | isc
        author: copyright holder name
        year: copyright year (defaults to current year)
    """
    status.set_callback(params.pop("_status_callback", None))
    lt = params["license_type"].lower().strip()
    author = params["author"]
    year = str(params.get("year", datetime.now().year))

    template = LICENSES.get(lt)
    if not template:
        supported = ", ".join(sorted(set(LICENSES.keys())))
        return tool_error(f"Unknown license: {lt}. Supported: {supported}")

    text = template.replace("{year}", year).replace("{author}", author)
    return tool_response(license_text=text, license_type=lt, year=year)


__all__ = ["generate_license"]
'''

# ---------------------------------------------------------------------------
# 5. api-docs-generator
# ---------------------------------------------------------------------------
API_DOCS_GENERATOR = '''\
"""Generate OpenAPI/Swagger docs from endpoint definitions."""
import json
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("api-docs-generator")


@tool_wrapper(required_params=["title", "endpoints"])
def generate_api_docs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate OpenAPI 3.0 spec from endpoint definitions.

    Params:
        title: API title
        version: API version (default "1.0.0")
        description: API description
        base_url: server base URL
        endpoints: list of endpoint dicts:
            - path: URL path (e.g. /users/{id})
            - method: HTTP method
            - summary: endpoint summary
            - description: detailed description
            - parameters: list of {name, in, type, required, description}
            - request_body: {type, properties} for POST/PUT
            - responses: dict of {status_code: {description, schema}}
            - tags: list of tag strings
    """
    status.set_callback(params.pop("_status_callback", None))
    title = params["title"]
    version = params.get("version", "1.0.0")
    desc = params.get("description", "")
    base_url = params.get("base_url", "http://localhost:8000")
    endpoints = params["endpoints"]

    spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version, "description": desc},
        "servers": [{"url": base_url}],
        "paths": {},
    }

    for ep in endpoints:
        path = ep.get("path", "/")
        method = ep.get("method", "get").lower()
        operation: Dict[str, Any] = {
            "summary": ep.get("summary", ""),
            "description": ep.get("description", ""),
        }
        if ep.get("tags"):
            operation["tags"] = ep["tags"]

        # Parameters
        if ep.get("parameters"):
            operation["parameters"] = []
            for p in ep["parameters"]:
                operation["parameters"].append({
                    "name": p["name"],
                    "in": p.get("in", "query"),
                    "required": p.get("required", False),
                    "description": p.get("description", ""),
                    "schema": {"type": p.get("type", "string")},
                })

        # Request body
        if ep.get("request_body") and method in ("post", "put", "patch"):
            rb = ep["request_body"]
            props = {}
            for k, v in rb.get("properties", {}).items():
                props[k] = {"type": v} if isinstance(v, str) else v
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "properties": props}
                    }
                },
            }

        # Responses
        responses = ep.get("responses", {"200": {"description": "Success"}})
        operation["responses"] = {}
        for code, info in responses.items():
            resp: Dict[str, Any] = {"description": info.get("description", "")}
            if info.get("schema"):
                resp["content"] = {
                    "application/json": {"schema": info["schema"]}
                }
            operation["responses"][str(code)] = resp

        spec["paths"].setdefault(path, {})[method] = operation

    spec_json = json.dumps(spec, indent=2)
    return tool_response(openapi_spec=spec, spec_json=spec_json, endpoint_count=len(endpoints))


__all__ = ["generate_api_docs"]
'''

# ---------------------------------------------------------------------------
# 6. db-seed-generator
# ---------------------------------------------------------------------------
DB_SEED_GENERATOR = '''\
"""Generate SQL INSERT statements or JSON seed data with realistic fake data."""
import random
import string
import json
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("db-seed-generator")

_FIRST = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Ivy","Jack",
           "Karen","Leo","Mia","Noah","Olivia","Pat","Quinn","Rosa","Sam","Tina"]
_LAST = ["Smith","Jones","Brown","Davis","Wilson","Moore","Taylor","Anderson",
          "Thomas","Jackson","White","Harris","Martin","Garcia","Clark","Lewis"]
_DOMAINS = ["example.com","test.org","demo.net","sample.io","mail.com"]
_CITIES = ["New York","London","Tokyo","Berlin","Sydney","Toronto","Paris","Mumbai"]
_COMPANIES = ["Acme Corp","Globex","Initech","Umbrella","Wayne Enterprises",
              "Stark Industries","Cyberdyne","Soylent Corp"]


def _fake(col_type: str) -> Any:
    t = col_type.lower()
    if t in ("string", "varchar", "text", "name"):
        return random.choice(_FIRST) + " " + random.choice(_LAST)
    if t in ("email",):
        name = random.choice(_FIRST).lower()
        return f"{name}{random.randint(1,999)}@{random.choice(_DOMAINS)}"
    if t in ("int", "integer", "number"):
        return random.randint(1, 10000)
    if t in ("float", "decimal", "double"):
        return round(random.uniform(0.01, 9999.99), 2)
    if t in ("bool", "boolean"):
        return random.choice([True, False])
    if t in ("date",):
        return f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    if t in ("city",):
        return random.choice(_CITIES)
    if t in ("company",):
        return random.choice(_COMPANIES)
    if t in ("phone",):
        return f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    if t in ("uuid", "id"):
        return "".join(random.choices(string.hexdigits[:16], k=32))
    return f"sample_{random.randint(1,999)}"


def _quote_sql(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    return "'" + str(val).replace("'", "''") + "'"


@tool_wrapper(required_params=["table", "columns"])
def generate_seed_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate seed data (SQL INSERTs or JSON) from a schema definition.

    Params:
        table: table name
        columns: list of {name, type} dicts
        count: number of rows (default 10, max 1000)
        format: 'sql' or 'json' (default 'sql')
        seed: random seed for reproducibility
    """
    status.set_callback(params.pop("_status_callback", None))
    table = params["table"]
    columns = params["columns"]
    count = min(int(params.get("count", 10)), 1000)
    fmt = params.get("format", "sql").lower()
    if params.get("seed") is not None:
        random.seed(int(params["seed"]))

    col_names = [c["name"] for c in columns]
    col_types = [c.get("type", "string") for c in columns]

    rows = []
    for _ in range(count):
        row = {name: _fake(ctype) for name, ctype in zip(col_names, col_types)}
        rows.append(row)

    if fmt == "json":
        output = json.dumps(rows, indent=2)
    else:
        col_list = ", ".join(col_names)
        stmts = []
        for row in rows:
            vals = ", ".join(_quote_sql(row[c]) for c in col_names)
            stmts.append(f"INSERT INTO {table} ({col_list}) VALUES ({vals});")
        output = "\\n".join(stmts)

    return tool_response(seed_data=output, format=fmt, row_count=count, table=table)


__all__ = ["generate_seed_data"]
'''

# ---------------------------------------------------------------------------
# 7. code-complexity-analyzer
# ---------------------------------------------------------------------------
CODE_COMPLEXITY_ANALYZER = '''\
"""Calculate cyclomatic complexity, LOC, function count from Python source."""
import ast
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("code-complexity-analyzer")


def _cyclomatic(node: ast.AST) -> int:
    """Count decision points in an AST node."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, (ast.Assert, ast.With)):
            complexity += 1
        elif isinstance(child, ast.comprehension):
            complexity += 1 + len(child.ifs)
    return complexity


@tool_wrapper(required_params=["source"])
def analyze_complexity(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Python source code complexity.

    Params:
        source: Python source code string
        threshold: complexity warning threshold (default 10)
    """
    status.set_callback(params.pop("_status_callback", None))
    source = params["source"]
    threshold = int(params.get("threshold", 10))

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return tool_error(f"Syntax error: {e}")

    lines = source.splitlines()
    total_loc = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    code_lines = total_loc - blank_lines - comment_lines

    functions: List[Dict[str, Any]] = []
    classes: List[str] = []
    imports = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _cyclomatic(node)
            end = getattr(node, "end_lineno", node.lineno)
            func_lines = end - node.lineno + 1
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "complexity": cc,
                "lines": func_lines,
                "high_complexity": cc > threshold,
            })
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1

    overall_cc = _cyclomatic(tree)
    high = [f for f in functions if f["high_complexity"]]

    return tool_response(
        total_loc=total_loc,
        code_lines=code_lines,
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        function_count=len(functions),
        class_count=len(classes),
        import_count=imports,
        overall_complexity=overall_cc,
        functions=functions,
        high_complexity_functions=high,
        threshold=threshold,
    )


__all__ = ["analyze_complexity"]
'''

# ---------------------------------------------------------------------------
# 8. dependency-checker
# ---------------------------------------------------------------------------
DEPENDENCY_CHECKER = '''\
"""Parse requirements.txt/package.json and check for issues."""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("dependency-checker")

# Known packages with security advisories or deprecations
_KNOWN_ISSUES: Dict[str, str] = {
    "requests": "Ensure >= 2.31.0 (CVE-2023-32681)",
    "urllib3": "Ensure >= 2.0.7 (CVE-2023-45803)",
    "cryptography": "Ensure >= 41.0.6 (multiple CVEs)",
    "pillow": "Ensure >= 10.2.0 (CVE-2023-50447)",
    "django": "Ensure >= 4.2.8 (security updates)",
    "flask": "Ensure >= 3.0.0 for latest security patches",
    "jinja2": "Ensure >= 3.1.3 (CVE-2024-22195)",
    "setuptools": "Ensure >= 70.0.0 (CVE-2024-6345)",
    "certifi": "Ensure >= 2024.2.2 for updated CA bundle",
    "axios": "Ensure >= 1.6.0 (CVE-2023-45857)",
    "lodash": "Ensure >= 4.17.21 (prototype pollution)",
    "express": "Ensure >= 4.19.2 (CVE-2024-29041)",
    "jsonwebtoken": "Ensure >= 9.0.0 (CVE-2022-23529)",
}


def _parse_requirements(content: str) -> List[Dict[str, Any]]:
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([a-zA-Z0-9_.-]+)\\s*([><=!~]+)?\\s*([\\d.*]+)?", line)
        if match:
            name = match.group(1).lower()
            op = match.group(2) or ""
            ver = match.group(3) or ""
            issue = _KNOWN_ISSUES.get(name)
            deps.append({"name": name, "operator": op, "version": ver,
                         "pinned": op == "==", "advisory": issue})
    return deps


def _parse_package_json(content: str) -> List[Dict[str, Any]]:
    deps = []
    try:
        pkg = json.loads(content)
    except json.JSONDecodeError:
        return deps
    for section in ("dependencies", "devDependencies"):
        for name, ver in pkg.get(section, {}).items():
            clean = re.sub(r"[^\\d.]", "", ver)
            issue = _KNOWN_ISSUES.get(name)
            deps.append({"name": name, "version": ver, "clean_version": clean,
                         "pinned": not ver.startswith("^") and not ver.startswith("~"),
                         "dev": section == "devDependencies", "advisory": issue})
    return deps


@tool_wrapper(required_params=["content"])
def check_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and check dependencies for issues.

    Params:
        content: file content (requirements.txt or package.json)
        file_type: 'requirements' or 'package_json' (auto-detected if omitted)
    """
    status.set_callback(params.pop("_status_callback", None))
    content = params["content"]
    ftype = params.get("file_type", "")

    if not ftype:
        ftype = "package_json" if content.strip().startswith("{") else "requirements"

    if ftype == "package_json":
        deps = _parse_package_json(content)
    else:
        deps = _parse_requirements(content)

    advisories = [d for d in deps if d.get("advisory")]
    unpinned = [d for d in deps if not d.get("pinned")]

    return tool_response(
        dependencies=deps,
        total=len(deps),
        advisories=advisories,
        advisory_count=len(advisories),
        unpinned=unpinned,
        unpinned_count=len(unpinned),
        file_type=ftype,
    )


__all__ = ["check_dependencies"]
'''

# ---------------------------------------------------------------------------
# 9. sitemap-generator
# ---------------------------------------------------------------------------
SITEMAP_GENERATOR = '''\
"""Generate XML sitemaps from URL lists."""
from typing import Dict, Any, List
from datetime import datetime
from xml.sax.saxutils import escape
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("sitemap-generator")


@tool_wrapper(required_params=["urls"])
def generate_sitemap(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an XML sitemap from a list of URLs.

    Params:
        urls: list of URL strings or list of dicts:
            - url: the URL (required)
            - lastmod: last modification date (YYYY-MM-DD)
            - changefreq: always|hourly|daily|weekly|monthly|yearly|never
            - priority: 0.0 to 1.0
        default_changefreq: default changefreq for simple URL strings
        default_priority: default priority for simple URL strings (default 0.5)
    """
    status.set_callback(params.pop("_status_callback", None))
    urls = params["urls"]
    default_cf = params.get("default_changefreq", "weekly")
    default_pri = float(params.get("default_priority", 0.5))

    valid_freqs = {"always","hourly","daily","weekly","monthly","yearly","never"}

    lines = [
        \'\'\'<?xml version="1.0" encoding="UTF-8"?>\'\'\',
        \'\'\'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\'\'\',
    ]

    count = 0
    for entry in urls:
        if isinstance(entry, str):
            entry = {"url": entry}
        url = entry.get("url", "")
        if not url:
            continue
        lastmod = entry.get("lastmod", datetime.now().strftime("%Y-%m-%d"))
        cf = entry.get("changefreq", default_cf)
        if cf not in valid_freqs:
            cf = default_cf
        pri = float(entry.get("priority", default_pri))
        pri = max(0.0, min(1.0, pri))

        lines.append("  <url>")
        lines.append(f"    <loc>{escape(url)}</loc>")
        lines.append(f"    <lastmod>{escape(lastmod)}</lastmod>")
        lines.append(f"    <changefreq>{cf}</changefreq>")
        lines.append(f"    <priority>{pri:.1f}</priority>")
        lines.append("  </url>")
        count += 1

    lines.append("</urlset>")
    xml = "\\n".join(lines)
    return tool_response(sitemap_xml=xml, url_count=count)


__all__ = ["generate_sitemap"]
'''

# ---------------------------------------------------------------------------
# 10. robots-txt-generator
# ---------------------------------------------------------------------------
ROBOTS_TXT_GENERATOR = '''\
"""Generate robots.txt files with user-agent rules."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("robots-txt-generator")


@tool_wrapper(required_params=["rules"])
def generate_robots_txt(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a robots.txt file.

    Params:
        rules: list of rule dicts:
            - user_agent: bot name or "*" (default "*")
            - allow: list of allowed paths
            - disallow: list of disallowed paths
            - crawl_delay: delay in seconds (optional)
        sitemaps: list of sitemap URLs
        host: preferred host (optional)
    """
    status.set_callback(params.pop("_status_callback", None))
    rules = params["rules"]
    sitemaps = params.get("sitemaps", [])
    host = params.get("host", "")

    lines: List[str] = ["# robots.txt", "# Auto-generated", ""]

    for rule in rules:
        ua = rule.get("user_agent", "*")
        lines.append(f"User-agent: {ua}")

        for path in rule.get("allow", []):
            lines.append(f"Allow: {path}")
        for path in rule.get("disallow", []):
            lines.append(f"Disallow: {path}")

        delay = rule.get("crawl_delay")
        if delay is not None:
            lines.append(f"Crawl-delay: {delay}")
        lines.append("")

    for sm in sitemaps:
        lines.append(f"Sitemap: {sm}")

    if host:
        lines.append(f"Host: {host}")

    if sitemaps or host:
        lines.append("")

    content = "\\n".join(lines)
    return tool_response(robots_txt=content, rule_count=len(rules))


__all__ = ["generate_robots_txt"]
'''


# ===========================================================================
# Create all 10 skills
# ===========================================================================
def main():
    print("Generating Batch 4a skills (10 developer-tooling skills)...")

    create_skill(
        name="sql-query-builder",
        frontmatter_name="sql-query-builder",
        description="Build SQL queries programmatically (SELECT, INSERT, UPDATE, DELETE with WHERE, JOIN, ORDER BY)",
        category="development",
        capabilities=[
            "Build SELECT queries with JOIN, WHERE, ORDER BY, GROUP BY, LIMIT",
            "Build INSERT queries with multiple rows",
            "Build UPDATE queries with SET and WHERE",
            "Build DELETE queries with WHERE conditions",
        ],
        triggers=["build SQL query", "generate SQL", "create SELECT statement", "SQL builder"],
        tools_code=SQL_QUERY_BUILDER,
        tool_docs="### build_sql_query\nBuild SQL queries from structured parameters.",
        eval_tool="build_sql_query",
        eval_input={"operation": "SELECT", "table": "users", "columns": ["id", "name"]},
    )

    create_skill(
        name="readme-generator",
        frontmatter_name="readme-generator",
        description="Generate README.md files from project info (name, description, features, installation, usage, license)",
        category="documentation",
        capabilities=[
            "Generate README with features, install, usage sections",
            "Support badges, contributing, author, license sections",
        ],
        triggers=["generate README", "create README.md", "project readme"],
        tools_code=README_GENERATOR,
        tool_docs="### generate_readme\nGenerate README.md from project metadata.",
        eval_tool="generate_readme",
        eval_input={"name": "MyProject", "description": "A sample project"},
    )

    create_skill(
        name="gitignore-generator",
        frontmatter_name="gitignore-generator",
        description="Generate .gitignore files for languages/frameworks (Python, Node, Java, Go, Rust, etc)",
        category="development",
        capabilities=[
            "Generate .gitignore for Python, Node, Java, Go, Rust, C",
            "Combine multiple language templates",
            "Add custom extra patterns",
        ],
        triggers=["generate gitignore", "create .gitignore", "gitignore for Python"],
        tools_code=GITIGNORE_GENERATOR,
        tool_docs="### generate_gitignore\nGenerate .gitignore for given languages.",
        eval_tool="generate_gitignore",
        eval_input={"languages": ["python", "node"]},
    )

    create_skill(
        name="license-generator",
        frontmatter_name="license-generator",
        description="Generate open-source license text (MIT, Apache-2.0, GPL-3.0, BSD-2, ISC) with year/author substitution",
        category="documentation",
        capabilities=[
            "Generate MIT, Apache-2.0, GPL-3.0, BSD-2, ISC license text",
            "Substitute year and author into templates",
        ],
        triggers=["generate license", "create MIT license", "license file generator"],
        tools_code=LICENSE_GENERATOR,
        tool_docs="### generate_license\nGenerate license text for a given type.",
        eval_tool="generate_license",
        eval_input={"license_type": "mit", "author": "Test Author"},
    )

    create_skill(
        name="api-docs-generator",
        frontmatter_name="api-docs-generator",
        description="Generate OpenAPI/Swagger docs from endpoint definitions (path, method, params, responses)",
        category="documentation",
        capabilities=[
            "Generate OpenAPI 3.0 JSON spec from endpoint definitions",
            "Support parameters, request bodies, responses, tags",
        ],
        triggers=["generate API docs", "create OpenAPI spec", "Swagger generator"],
        tools_code=API_DOCS_GENERATOR,
        tool_docs="### generate_api_docs\nGenerate OpenAPI spec from endpoint definitions.",
        eval_tool="generate_api_docs",
        eval_input={
            "title": "My API",
            "endpoints": [{"path": "/users", "method": "get", "summary": "List users"}],
        },
    )

    create_skill(
        name="db-seed-generator",
        frontmatter_name="db-seed-generator",
        description="Generate SQL INSERT statements or JSON seed data from schema definitions with realistic fake data",
        category="development",
        capabilities=[
            "Generate SQL INSERT statements with fake data",
            "Generate JSON seed data",
            "Support multiple column types (name, email, int, date, etc.)",
        ],
        triggers=["generate seed data", "create test data", "database seed generator"],
        tools_code=DB_SEED_GENERATOR,
        tool_docs="### generate_seed_data\nGenerate seed data from schema.",
        eval_tool="generate_seed_data",
        eval_input={
            "table": "users",
            "columns": [{"name": "id", "type": "integer"}, {"name": "name", "type": "name"}],
        },
    )

    create_skill(
        name="code-complexity-analyzer",
        frontmatter_name="code-complexity-analyzer",
        description="Calculate cyclomatic complexity, LOC, function count from Python source code using ast module",
        category="development",
        capabilities=[
            "Calculate cyclomatic complexity per function",
            "Count lines of code, blank lines, comment lines",
            "Identify high-complexity functions",
        ],
        triggers=["analyze code complexity", "cyclomatic complexity", "code quality metrics"],
        tools_code=CODE_COMPLEXITY_ANALYZER,
        tool_docs="### analyze_complexity\nAnalyze Python source complexity.",
        eval_tool="analyze_complexity",
        eval_input={"source": "def hello():\n    print('hi')\n"},
    )

    create_skill(
        name="dependency-checker",
        frontmatter_name="dependency-checker",
        description="Parse requirements.txt/package.json and check for outdated versions or known issues",
        category="development",
        capabilities=[
            "Parse requirements.txt and package.json",
            "Check against known security advisories",
            "Identify unpinned dependencies",
        ],
        triggers=["check dependencies", "dependency audit", "requirements checker"],
        tools_code=DEPENDENCY_CHECKER,
        tool_docs="### check_dependencies\nParse and check dependencies for issues.",
        eval_tool="check_dependencies",
        eval_input={"content": "requests==2.28.0\nflask>=2.0"},
    )

    create_skill(
        name="sitemap-generator",
        frontmatter_name="sitemap-generator",
        description="Generate XML sitemaps from URL lists with lastmod, changefreq, priority",
        category="web",
        capabilities=[
            "Generate XML sitemaps conforming to sitemaps.org schema",
            "Support lastmod, changefreq, priority per URL",
            "Accept simple URL strings or detailed URL dicts",
        ],
        triggers=["generate sitemap", "create XML sitemap", "sitemap.xml generator"],
        tools_code=SITEMAP_GENERATOR,
        tool_docs="### generate_sitemap\nGenerate XML sitemap from URL list.",
        eval_tool="generate_sitemap",
        eval_input={"urls": ["https://example.com", "https://example.com/about"]},
    )

    create_skill(
        name="robots-txt-generator",
        frontmatter_name="robots-txt-generator",
        description="Generate robots.txt files with user-agent rules, Allow/Disallow, sitemap reference",
        category="web",
        capabilities=[
            "Generate robots.txt with multiple user-agent rules",
            "Support Allow, Disallow, Crawl-delay directives",
            "Add Sitemap and Host directives",
        ],
        triggers=["generate robots.txt", "create robots.txt", "robots txt generator"],
        tools_code=ROBOTS_TXT_GENERATOR,
        tool_docs="### generate_robots_txt\nGenerate robots.txt from rules.",
        eval_tool="generate_robots_txt",
        eval_input={"rules": [{"user_agent": "*", "disallow": ["/admin"]}]},
    )

    print("\nDone! 10 skills created.")


if __name__ == "__main__":
    main()
