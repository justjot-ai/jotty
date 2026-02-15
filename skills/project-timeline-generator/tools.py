"""Project Timeline Generator Skill - text-based Gantt charts."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("project-timeline-generator")


def _resolve_dependencies(tasks: List[dict]) -> List[dict]:
    """Resolve task dependencies and adjust start days."""
    task_map = {t["name"]: t for t in tasks}
    resolved = []

    for task in tasks:
        t = dict(task)
        dep = t.get("depends_on")
        if dep and dep in task_map:
            dep_task = task_map[dep]
            dep_end = dep_task.get("start_day", 1) + dep_task.get("duration", 1)
            t["start_day"] = max(t.get("start_day", dep_end), dep_end)
        resolved.append(t)

    return resolved


@tool_wrapper(required_params=["tasks"])
def generate_timeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a text-based Gantt chart."""
    status.set_callback(params.pop("_status_callback", None))
    tasks = params["tasks"]
    title = params.get("title", "Project Timeline")
    scale = params.get("scale", "day").lower()

    if not tasks:
        return tool_error("Need at least one task")

    tasks = _resolve_dependencies(tasks)

    # Calculate project bounds
    max_end = 0
    for t in tasks:
        start = t.get("start_day", 1)
        duration = t.get("duration", 1)
        end = start + duration - 1
        t["_end"] = end
        max_end = max(max_end, end)

    total_days = max_end
    name_width = max(len(t.get("name", "Task")) for t in tasks) + 2

    # Scale factor for chart width
    if scale == "week":
        chart_width = min((total_days // 7) + 1, 52)
        scale_factor = 7
    else:
        chart_width = min(total_days, 80)
        scale_factor = max(1, total_days // chart_width) if chart_width > 0 else 1

    # Build header
    lines = [f"  {title}", "  " + "=" * (name_width + chart_width + 10)]

    # Day/week markers
    marker_line = " " * (name_width + 2)
    for i in range(0, chart_width, 5):
        day_num = i * scale_factor + 1
        label = str(day_num)
        marker_line += label.ljust(5)
    lines.append(marker_line)
    lines.append(" " * (name_width + 2) + "|" * chart_width)

    # Task bars
    for t in tasks:
        name = t.get("name", "Task")
        start = t.get("start_day", 1)
        duration = t.get("duration", 1)

        bar_start = (start - 1) // scale_factor
        bar_len = max(1, duration // scale_factor)

        bar_start = min(bar_start, chart_width - 1)
        bar_len = min(bar_len, chart_width - bar_start)

        bar = " " * bar_start + "\u2588" * bar_len
        padding = chart_width - len(bar)
        if padding > 0:
            bar += " " * padding

        day_range = f"(d{start}-d{start + duration - 1})"
        lines.append(f"  {name.ljust(name_width)}{bar}  {day_range}")

    lines.append(" " * (name_width + 2) + "-" * chart_width)
    lines.append(f"  Total duration: {total_days} days")

    gantt_text = "\n".join(lines)

    # Find critical path (longest path)
    critical = sorted(tasks, key=lambda t: t.get("_end", 0), reverse=True)
    critical_path = [t["name"] for t in critical if t.get("_end") == max_end]

    task_summary = []
    for t in tasks:
        task_summary.append({
            "name": t.get("name"),
            "start_day": t.get("start_day"),
            "duration": t.get("duration"),
            "end_day": t.get("_end"),
            "depends_on": t.get("depends_on"),
        })

    return tool_response(
        gantt_chart=gantt_text,
        total_duration=total_days,
        task_count=len(tasks),
        tasks=task_summary,
        critical_path=critical_path,
        title=title,
    )


__all__ = ["generate_timeline_tool"]
