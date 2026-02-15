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
