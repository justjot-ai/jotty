"""
CLI Status Callback - Real-time agent activity display
======================================================

Extracted from app.py to keep it focused on orchestration.
Provides beautiful real-time status output during task execution.
"""

import re
import sys
from typing import Any, Callable

# Agent activity icons
AGENT_ICONS = {
    "analyze": "\U0001f9e0",
    "analyzing": "\U0001f9e0",
    "search": "\U0001f50d",
    "searching": "\U0001f50d",
    "generate": "\U0001f4dd",
    "generating": "\U0001f4dd",
    "write": "\u270d\ufe0f",
    "writing": "\u270d\ufe0f",
    "read": "\U0001f4d6",
    "reading": "\U0001f4d6",
    "save": "\U0001f4be",
    "saving": "\U0001f4be",
    "send": "\U0001f4e4",
    "sending": "\U0001f4e4",
    "decision": "\U0001f3af",
    "deciding": "\U0001f3af",
    "output": "\U0001f4e6",
    "created": "\u2705",
}


def get_agent_icon(stage: str) -> str:
    """Get icon for agent activity."""
    stage_lower = stage.lower()
    for key, icon in AGENT_ICONS.items():
        if key in stage_lower:
            return icon
    return "\u2192"


def create_status_callback(renderer: Any) -> Callable[[str, str], None]:
    """
    Create a status callback for real-time CLI output.

    Args:
        renderer: RichRenderer instance

    Returns:
        Callback function(stage, detail)
    """

    def status_callback(stage: str, detail: str = "") -> Any:
        stage_lower = stage.lower()
        icon = get_agent_icon(stage)

        # Search results
        if "search" in stage_lower and "result" in stage_lower:
            count = 0
            if detail:
                match = re.search(r"(\d+)\s*results?", detail)
                if match:
                    count = int(match.group(1))
            renderer.search_query(detail or stage, count if count else None)

        # Step progress
        elif stage_lower.startswith("step "):
            match = re.match(r"step\s+(\d+)/(\d+)", stage_lower)
            if match:
                step_num = int(match.group(1))
                total = int(match.group(2))
                status = (
                    "done" if "\u2713" in detail or "succeeded" in detail.lower() else "running"
                )
                if "\u2717" in detail or "failed" in detail.lower():
                    status = "failed"
                renderer.step_progress(step_num, total, detail, status)
            else:
                renderer.print(
                    f"  [{renderer.theme.muted}]{icon}[/{renderer.theme.muted}] {stage}: {detail}"
                    if detail
                    else f"  {icon} {stage}"
                )

        # File operations
        elif any(x in stage_lower for x in ["writing", "creating file", "saving"]):
            renderer.print(f"  [bold green]{icon}[/bold green] [dim]{stage}[/dim]")
            if detail:
                renderer.print(f"     [cyan]{detail}[/cyan]")

        elif any(x in stage_lower for x in ["reading", "loading"]):
            renderer.print(f"  [bold blue]{icon}[/bold blue] [dim]{stage}[/dim]")
            if detail:
                renderer.print(f"     [cyan]{detail}[/cyan]")

        # Agent activity
        elif any(x in stage_lower for x in ["analyz", "decision", "generat", "search"]):
            renderer.print(f"  [bold yellow]{icon}[/bold yellow] [white]{stage}[/white]")
            if detail:
                display_detail = detail[:80] + "..." if len(detail) > 80 else detail
                renderer.print(f"     [dim]{display_detail}[/dim]")

        # Tool success
        elif "\u2713" in stage and detail:
            renderer.tool_output(
                stage.replace("\u2713", "").strip(), detail if "/" in detail else None
            )

        # Default
        else:
            if detail:
                renderer.print(f"  [cyan]{icon}[/cyan] {stage}: {detail}")
            else:
                renderer.print(f"  [cyan]{icon}[/cyan] {stage}")

        sys.stdout.flush()

    return status_callback
