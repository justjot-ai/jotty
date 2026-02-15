"""Expense Tracker Skill - track expenses with JSON storage."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("expense-tracker")

CATEGORIES = {
    "food",
    "transport",
    "utilities",
    "entertainment",
    "shopping",
    "health",
    "education",
    "housing",
    "insurance",
    "savings",
    "other",
}
DEFAULT_FILE = "expenses.json"


def _load(path: str) -> List[dict]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save(path: str, data: List[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


@tool_wrapper(required_params=["amount", "category"])
def add_expense_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Add an expense entry."""
    status.set_callback(params.pop("_status_callback", None))
    amount = round(float(params["amount"]), 2)
    category = params["category"].lower()
    description = params.get("description", "")
    date_str = params.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    storage = params.get("storage_file", DEFAULT_FILE)

    if amount <= 0:
        return tool_error("Amount must be positive")
    if category not in CATEGORIES:
        return tool_error(f"Invalid category. Use: {sorted(CATEGORIES)}")

    expenses = _load(storage)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "amount": amount,
        "category": category,
        "description": description,
        "date": date_str,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    expenses.append(entry)
    _save(storage, expenses)

    running_total = round(sum(e["amount"] for e in expenses), 2)

    return tool_response(expense=entry, running_total=running_total, total_entries=len(expenses))


@tool_wrapper()
def expense_summary_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get expense summary with category breakdown."""
    status.set_callback(params.pop("_status_callback", None))
    storage = params.get("storage_file", DEFAULT_FILE)
    month_filter = params.get("month", "")
    cat_filter = params.get("category", "").lower()

    expenses = _load(storage)

    if month_filter:
        expenses = [e for e in expenses if e.get("date", "").startswith(month_filter)]
    if cat_filter:
        expenses = [e for e in expenses if e.get("category") == cat_filter]

    total = round(sum(e["amount"] for e in expenses), 2)
    by_category = {}
    for e in expenses:
        cat = e.get("category", "other")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "count": 0}
        by_category[cat]["total"] = round(by_category[cat]["total"] + e["amount"], 2)
        by_category[cat]["count"] += 1

    # Sort by total descending
    by_category = dict(sorted(by_category.items(), key=lambda x: x[1]["total"], reverse=True))

    return tool_response(
        total=total,
        count=len(expenses),
        by_category=by_category,
        filters={"month": month_filter, "category": cat_filter},
    )


__all__ = ["add_expense_tool", "expense_summary_tool"]
