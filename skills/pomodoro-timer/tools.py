"""Pomodoro Timer Skill - track focus sessions and breaks."""
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("pomodoro-timer")
DEFAULT_FILE = "pomodoro_sessions.json"


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


@tool_wrapper(required_params=["task"])
def start_pomodoro_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Start a Pomodoro session."""
    status.set_callback(params.pop("_status_callback", None))
    task = params["task"]
    work_min = int(params.get("work_minutes", 25))
    break_min = int(params.get("break_minutes", 5))
    long_break = int(params.get("long_break_minutes", 15))
    storage = params.get("storage_file", DEFAULT_FILE)

    now = datetime.now(timezone.utc)
    sessions = _load(storage)

    # Count today's sessions for this task
    today = now.strftime("%Y-%m-%d")
    today_count = sum(1 for s in sessions if s.get("date") == today)
    session_number = today_count + 1
    is_long_break = session_number % 4 == 0
    actual_break = long_break if is_long_break else break_min

    session = {
        "id": uuid.uuid4().hex[:8],
        "task": task,
        "date": today,
        "started_at": now.isoformat(),
        "work_minutes": work_min,
        "break_minutes": actual_break,
        "work_ends_at": (now + timedelta(minutes=work_min)).isoformat(),
        "break_ends_at": (now + timedelta(minutes=work_min + actual_break)).isoformat(),
        "session_number": session_number,
        "is_long_break": is_long_break,
        "status": "active",
    }

    sessions.append(session)
    _save(storage, sessions)

    return tool_response(
        session=session,
        message=f"Pomodoro #{session_number} started! Focus for {work_min} min, then {'long' if is_long_break else 'short'} break ({actual_break} min).",
    )


@tool_wrapper()
def pomodoro_stats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get Pomodoro productivity statistics."""
    status.set_callback(params.pop("_status_callback", None))
    storage = params.get("storage_file", DEFAULT_FILE)
    sessions = _load(storage)

    total_focus = sum(s.get("work_minutes", 25) for s in sessions)
    total_break = sum(s.get("break_minutes", 5) for s in sessions)

    by_task = {}
    for s in sessions:
        task = s.get("task", "Unknown")
        if task not in by_task:
            by_task[task] = {"sessions": 0, "focus_minutes": 0}
        by_task[task]["sessions"] += 1
        by_task[task]["focus_minutes"] += s.get("work_minutes", 25)

    by_date = {}
    for s in sessions:
        date = s.get("date", "unknown")
        by_date[date] = by_date.get(date, 0) + 1

    return tool_response(
        total_sessions=len(sessions),
        total_focus_minutes=total_focus,
        total_break_minutes=total_break,
        by_task=by_task,
        by_date=by_date,
        avg_sessions_per_day=round(len(sessions) / max(len(by_date), 1), 1),
    )


__all__ = ["start_pomodoro_tool", "pomodoro_stats_tool"]
