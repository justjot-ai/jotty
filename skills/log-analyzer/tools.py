"""Log Analyzer Skill — parse and summarize log files (pure Python)."""

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("log-analyzer")

LOG_PATTERNS = [
    # Standard: 2024-01-01 10:00:00 ERROR message
    re.compile(
        r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^ ]*)\s+(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL|TRACE)\s+(.+)$",
        re.IGNORECASE,
    ),
    # Bracketed: [2024-01-01 10:00:00] [ERROR] message
    re.compile(
        r"^\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\]]*)\]\s*\[(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL)\]\s+(.+)$",
        re.IGNORECASE,
    ),
    # Syslog-ish: Jan  1 10:00:00 hostname service: message
    re.compile(r"^(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+\S+\s+(\S+):\s+(.+)$"),
    # Nginx/Apache access log
    re.compile(r'^(\S+)\s+\S+\s+\S+\s+\[([^\]]+)\]\s+"(\S+)\s+(\S+)\s+\S+"\s+(\d{3})\s+(\d+)'),
]

ERROR_LEVELS = {"error", "fatal", "critical"}
WARN_LEVELS = {"warn", "warning"}


@tool_wrapper()
def analyze_log_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and summarize log content."""
    status.set_callback(params.pop("_status_callback", None))
    content = params.get("content", "")
    file_path = params.get("file_path")
    level_filter = params.get("level_filter", "").upper()

    if file_path:
        p = Path(file_path)
        if not p.exists():
            return tool_error(f"File not found: {file_path}")
        content = p.read_text(errors="replace")

    if not content:
        return tool_error("Provide either content or file_path")

    lines = content.splitlines()
    level_counts = Counter()
    errors = []
    warnings = []
    timestamps = []
    parsed_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        matched = False
        for pattern in LOG_PATTERNS[:2]:
            m = pattern.match(line)
            if m:
                ts, level, msg = m.group(1), m.group(2).upper(), m.group(3)
                timestamps.append(ts)
                level_norm = level.replace("WARNING", "WARN")
                level_counts[level_norm] += 1
                if level.lower() in ERROR_LEVELS:
                    errors.append({"timestamp": ts, "message": msg[:200]})
                elif level.lower() in WARN_LEVELS:
                    warnings.append({"timestamp": ts, "message": msg[:200]})
                parsed_count += 1
                matched = True
                break
        if not matched:
            level_counts["UNPARSED"] += 1

    # Apply filter
    filtered_errors = errors
    if level_filter == "ERROR":
        pass  # already filtered
    elif level_filter == "WARN":
        filtered_errors = warnings

    # Find common error patterns
    error_messages = [e["message"] for e in errors]
    common_errors = Counter(error_messages).most_common(10)

    return tool_response(
        total_lines=len(lines),
        parsed_lines=parsed_count,
        level_counts=dict(level_counts),
        errors=errors[:50],
        warnings=warnings[:50],
        common_errors=[{"message": msg, "count": cnt} for msg, cnt in common_errors],
        time_range={
            "start": timestamps[0] if timestamps else None,
            "end": timestamps[-1] if timestamps else None,
        },
    )


@tool_wrapper(required_params=["content", "pattern"])
def search_log_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search log content for a pattern."""
    status.set_callback(params.pop("_status_callback", None))
    content = params["content"]
    pattern = params["pattern"]
    limit = int(params.get("limit", 50))

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return tool_error(f"Invalid pattern: {e}")

    matches = []
    for i, line in enumerate(content.splitlines()):
        if regex.search(line):
            matches.append({"line_number": i + 1, "content": line.strip()[:300]})
            if len(matches) >= limit:
                break

    return tool_response(matches=matches, count=len(matches), pattern=pattern)


__all__ = ["analyze_log_tool", "search_log_tool"]
