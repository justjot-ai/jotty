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
        output = "\n".join(stmts)

    return tool_response(seed_data=output, format=fmt, row_count=count, table=table)


__all__ = ["generate_seed_data"]
