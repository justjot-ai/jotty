"""OpenClaw Skill Adapter — parse OpenClaw SKILL.md format into Jotty skills.

OpenClaw skills are pure SKILL.md files (markdown with YAML frontmatter or
table-based metadata). They teach an LLM how to use CLI tools — no Python code.

This adapter detects OpenClaw-format SKILL.md files and converts them into
Jotty SkillDefinitions backed by shell-exec (subprocess).

Supported formats:
  - YAML frontmatter (--- delimited) with optional metadata.openclaw
  - Markdown table (| name | ... |) header format

References:
  - https://github.com/openclaw — 5,494 community skills
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_openclaw_skill(skill_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse OpenClaw SKILL.md format into Jotty SkillDefinition kwargs.

    Returns None if not an OpenClaw skill (has tools.py, or lacks
    OpenClaw-style frontmatter).

    Args:
        skill_dir: Directory containing SKILL.md (and no tools.py).

    Returns:
        Dict with: name, description, category, tags, instructions,
        requires_bins, or None if not OpenClaw format.
    """
    skill_md = skill_dir / "SKILL.md"
    tools_py = skill_dir / "tools.py"

    # Must have SKILL.md and NOT have tools.py
    if tools_py.exists() or not skill_md.exists():
        return None

    try:
        content = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Try YAML frontmatter first, then table format
    result = _parse_yaml_frontmatter(content)
    if result is None:
        result = _parse_table_format(content)

    if result is None:
        return None

    # Extract the markdown body (instructions for the LLM)
    body = _extract_body(content)
    result["instructions"] = body
    result.setdefault("name", skill_dir.name)
    result.setdefault("description", f"OpenClaw skill: {result['name']}")
    result.setdefault("category", "openclaw")
    result.setdefault("tags", ["openclaw", "community"])
    result.setdefault("requires_bins", [])

    return result


def _parse_yaml_frontmatter(content: str) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter (--- delimited)."""
    if not content.startswith("---"):
        return None

    end_idx = content.find("---", 3)
    if end_idx < 0:
        return None

    frontmatter = content[3:end_idx].strip()
    result: Dict[str, Any] = {}

    for line in frontmatter.split("\n"):
        line = line.strip()
        if line.startswith("name:"):
            result["name"] = line[5:].strip().strip("\"'")
        elif line.startswith("description:"):
            result["description"] = line[12:].strip().strip("\"'")
        elif line.startswith("category:"):
            result["category"] = line[9:].strip().strip("\"'")
        elif line.startswith("metadata:"):
            # Single-line JSON metadata (OpenClaw style)
            meta_str = line[9:].strip()
            try:
                meta = json.loads(meta_str)
                if isinstance(meta, dict):
                    if "openclaw" in meta:
                        requires = meta["openclaw"].get("requires", {})
                        result["requires_bins"] = requires.get("bins", [])
                    result["_meta"] = meta
            except (json.JSONDecodeError, TypeError):
                pass
        elif line.startswith("tags:"):
            tag_str = line[5:].strip()
            # Handle inline list: [tag1, tag2] or comma-separated
            tag_str = tag_str.strip("[]")
            result["tags"] = [t.strip().strip("\"'") for t in tag_str.split(",") if t.strip()]

    # Must have at least a name or description to be valid
    if not result.get("name") and not result.get("description"):
        return None

    return result


def _parse_table_format(content: str) -> Optional[Dict[str, Any]]:
    """Parse table-based metadata (| name | value |)."""
    lines = content.split("\n")
    result: Dict[str, Any] = {}
    found_table = False

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if found_table:
                break  # Table ended
            continue

        # Skip separator rows (|---|---|)
        if re.match(r"^\|[\s\-|]+\|$", stripped):
            found_table = True
            continue

        cells = [c.strip() for c in stripped.split("|")]
        cells = [c for c in cells if c]  # Remove empty cells from leading/trailing |

        if len(cells) >= 2:
            found_table = True
            key = cells[0].lower().strip()
            value = cells[1].strip()

            if key == "name":
                result["name"] = value
            elif key == "description":
                result["description"] = value
            elif key == "category":
                result["category"] = value
            elif key == "metadata":
                try:
                    meta = json.loads(value)
                    if isinstance(meta, dict) and "openclaw" in meta:
                        requires = meta["openclaw"].get("requires", {})
                        result["requires_bins"] = requires.get("bins", [])
                except (json.JSONDecodeError, TypeError):
                    pass

    if not found_table or (not result.get("name") and not result.get("description")):
        return None

    return result


def _extract_body(content: str) -> str:
    """Extract the markdown body after frontmatter/table header."""
    # Skip YAML frontmatter
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx >= 0:
            return content[end_idx + 3 :].strip()

    # Skip leading table
    lines = content.split("\n")
    body_start = 0
    in_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            in_table = True
        elif in_table and not stripped.startswith("|"):
            body_start = i
            break

    return "\n".join(lines[body_start:]).strip()


def make_shell_exec_tools(
    skill_name: str, instructions: str, requires_bins: List[str]
) -> Dict[str, Any]:
    """Create a shell-exec tool dict for an OpenClaw skill.

    The tool runs shell commands as instructed by the LLM,
    with the SKILL.md instructions provided as context.
    """
    import asyncio

    async def shell_exec_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command (OpenClaw skill backend).

        Args:
            params: Dict with 'command' (str) — the shell command to run.

        Returns:
            Dict with 'success', 'stdout', 'stderr', 'returncode'.
        """
        command = params.get("command", "")
        if not command:
            return {"success": False, "error": "No command provided"}

        timeout = params.get("timeout", 30)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace")[:4000],
                "stderr": stderr.decode(errors="replace")[:2000],
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    shell_exec_tool.__name__ = f"{skill_name.replace('-', '_')}_shell_tool"
    shell_exec_tool.__doc__ = (
        f"OpenClaw skill: {skill_name}\n\n"
        f"Instructions:\n{instructions[:500]}\n\n"
        f"Requires: {', '.join(requires_bins) or 'none'}"
    )
    # Mark required params for Jotty's tool system
    shell_exec_tool._required_params = ["command"]  # type: ignore[attr-defined]

    return {shell_exec_tool.__name__: shell_exec_tool}
