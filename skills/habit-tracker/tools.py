"""Habit Tracker Skill - track habits with streaks."""
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("habit-tracker")
DEFAULT_FILE = "habits.json"


def _load(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _calc_streak(dates: List[str]) -> int:
    if not dates:
        return 0
    sorted_dates = sorted(set(dates), reverse=True)
    streak = 1
    for i in range(len(sorted_dates) - 1):
        curr = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
        prev = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")
        if (curr - prev).days == 1:
            streak += 1
        else:
            break
    return streak


def _longest_streak(dates: List[str]) -> int:
    if not dates:
        return 0
    sorted_dates = sorted(set(dates))
    longest = 1
    current = 1
    for i in range(1, len(sorted_dates)):
        curr = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
        prev = datetime.strptime(sorted_dates[i - 1], "%Y-%m-%d")
        if (curr - prev).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


@tool_wrapper(required_params=["habit"])
def log_habit_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Log a habit completion."""
    status.set_callback(params.pop("_status_callback", None))
    habit = params["habit"].lower().strip()
    completed = params.get("completed", True)
    date = params.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    notes = params.get("notes", "")
    storage = params.get("storage_file", DEFAULT_FILE)

    data = _load(storage)
    if habit not in data:
        data[habit] = {"entries": [], "created": date}

    # Remove existing entry for same date
    data[habit]["entries"] = [e for e in data[habit]["entries"] if e.get("date") != date]
    data[habit]["entries"].append({
        "date": date,
        "completed": completed,
        "notes": notes,
    })

    _save(storage, data)

    completed_dates = [e["date"] for e in data[habit]["entries"] if e["completed"]]
    streak = _calc_streak(completed_dates)

    return tool_response(
        habit=habit, date=date, completed=completed,
        current_streak=streak,
        total_completions=len(completed_dates),
    )


@tool_wrapper()
def habit_stats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get habit statistics and streaks."""
    status.set_callback(params.pop("_status_callback", None))
    storage = params.get("storage_file", DEFAULT_FILE)
    habit_filter = params.get("habit", "").lower().strip()

    data = _load(storage)
    if habit_filter and habit_filter in data:
        data = {habit_filter: data[habit_filter]}
    elif habit_filter and habit_filter not in data:
        return tool_error(f"Habit '{habit_filter}' not found. Available: {list(data.keys())}")

    habits_stats = {}
    for habit, info in data.items():
        entries = info.get("entries", [])
        completed_dates = [e["date"] for e in entries if e.get("completed")]
        total_entries = len(entries)
        total_completed = len(completed_dates)

        habits_stats[habit] = {
            "total_entries": total_entries,
            "total_completed": total_completed,
            "completion_rate": round(total_completed / max(total_entries, 1) * 100, 1),
            "current_streak": _calc_streak(completed_dates),
            "longest_streak": _longest_streak(completed_dates),
            "created": info.get("created", "unknown"),
        }

    return tool_response(habits=habits_stats, habit_count=len(habits_stats))


__all__ = ["log_habit_tool", "habit_stats_tool"]
