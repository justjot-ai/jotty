"""Diff Tool Skill — compare text and files."""

import difflib
from pathlib import Path
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("diff-tool")


@tool_wrapper(required_params=["text_a", "text_b"])
def diff_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Show unified diff between two texts."""
    status.set_callback(params.pop("_status_callback", None))
    a_lines = params["text_a"].splitlines(keepends=True)
    b_lines = params["text_b"].splitlines(keepends=True)
    n = int(params.get("context_lines", 3))

    diff = list(difflib.unified_diff(a_lines, b_lines, fromfile="a", tofile="b", n=n))
    changes = sum(1 for line in diff if line.startswith("+") or line.startswith("-"))

    return tool_response(
        diff="".join(diff), changes=changes, lines_a=len(a_lines), lines_b=len(b_lines)
    )


@tool_wrapper(required_params=["file_a", "file_b"])
def diff_files_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Show unified diff between two files."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        a = Path(params["file_a"]).read_text(errors="replace").splitlines(keepends=True)
        b = Path(params["file_b"]).read_text(errors="replace").splitlines(keepends=True)
    except FileNotFoundError as e:
        return tool_error(str(e))

    n = int(params.get("context_lines", 3))
    diff = list(difflib.unified_diff(a, b, fromfile=params["file_a"], tofile=params["file_b"], n=n))
    changes = sum(
        1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )

    return tool_response(diff="".join(diff), changes=changes)


__all__ = ["diff_text_tool", "diff_files_tool"]
