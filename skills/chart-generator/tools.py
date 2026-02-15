"""Chart Generator Skill — ASCII bar and line charts (pure Python)."""

from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("chart-generator")

BLOCK_CHARS = " ▏▎▍▌▋▊▉█"


@tool_wrapper(required_params=["data"])
def bar_chart_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ASCII horizontal bar chart."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    width = int(params.get("width", 50))
    sort_desc = params.get("sort", True)

    if not isinstance(data, dict) or not data:
        return tool_error("data must be a non-empty dict of label: value pairs")

    items = list(data.items())
    if sort_desc:
        items.sort(key=lambda x: float(x[1]), reverse=True)

    max_val = max(float(v) for _, v in items)
    max_label = max(len(str(k)) for k, _ in items)

    lines = []
    for label, value in items:
        val = float(value)
        if max_val > 0:
            filled = val / max_val * width
            full_blocks = int(filled)
            fraction = filled - full_blocks
            frac_idx = int(fraction * 8)
            bar = "█" * full_blocks
            if frac_idx > 0 and full_blocks < width:
                bar += BLOCK_CHARS[frac_idx]
        else:
            bar = ""
        lines.append(f"{str(label):>{max_label}} │ {bar} {val:g}")

    chart = "\n".join(lines)
    return tool_response(chart=chart, items=len(items))


@tool_wrapper(required_params=["data"])
def line_chart_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ASCII line chart from sequential data."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    height = int(params.get("height", 15))
    width = int(params.get("width", 60))

    if isinstance(data, dict):
        labels = list(data.keys())
        values = [float(v) for v in data.values()]
    elif isinstance(data, list):
        values = [float(v) for v in data]
        labels = [str(i) for i in range(len(values))]
    else:
        return tool_error("data must be a list of numbers or dict of label: value")

    if not values:
        return tool_error("No data points provided")

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    # Build grid
    grid = [[" "] * len(values) for _ in range(height)]
    for i, v in enumerate(values):
        row = height - 1 - int((v - min_val) / val_range * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][i] = "●"
        # Draw line to next point
        if i < len(values) - 1:
            next_row = height - 1 - int((values[i + 1] - min_val) / val_range * (height - 1))
            next_row = max(0, min(height - 1, next_row))
            if next_row != row:
                step = 1 if next_row > row else -1
                for r in range(row + step, next_row, step):
                    if grid[r][i] == " ":
                        grid[r][i] = "│"

    # Render
    lines = []
    for r in range(height):
        y_val = max_val - r * val_range / (height - 1) if height > 1 else max_val
        row_str = "".join(grid[r])
        lines.append(f"{y_val:>8.1f} ┤ {row_str}")
    lines.append(" " * 10 + "└" + "─" * len(values))

    chart = "\n".join(lines)
    return tool_response(chart=chart, points=len(values), min_value=min_val, max_value=max_val)


__all__ = ["bar_chart_tool", "line_chart_tool"]
