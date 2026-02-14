"""Env Config Manager Skill — parse, diff, validate .env files."""
import re
from pathlib import Path
from typing import Dict, Any, Set

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("env-config-manager")


def _parse_env_file(file_path: str) -> Dict[str, str]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    env_vars = {}
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if match:
            key = match.group(1)
            val = match.group(2).strip().strip("'").strip('"')
            env_vars[key] = val
    return env_vars


@tool_wrapper(required_params=["file_path"])
def parse_env_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a .env file and return variables."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        variables = _parse_env_file(params["file_path"])
        # Mask sensitive values
        masked = {}
        sensitive = {"key", "secret", "password", "token", "api"}
        for k, v in variables.items():
            if any(s in k.lower() for s in sensitive) and v:
                masked[k] = v[:4] + "****" if len(v) > 4 else "****"
            else:
                masked[k] = v
        return tool_response(variables=masked, count=len(variables))
    except FileNotFoundError as e:
        return tool_error(str(e))


@tool_wrapper(required_params=["file_a", "file_b"])
def diff_env_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two .env files and find differences."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        a = _parse_env_file(params["file_a"])
        b = _parse_env_file(params["file_b"])
    except FileNotFoundError as e:
        return tool_error(str(e))

    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))
    changed = sorted(k for k in set(a.keys()) & set(b.keys()) if a[k] != b[k])

    return tool_response(only_in_a=only_a, only_in_b=only_b, changed=changed,
                         total_a=len(a), total_b=len(b))


__all__ = ["parse_env_tool", "diff_env_tool"]
