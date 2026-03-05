"""Community Skill Installer — fetch and install skills from external sources.

Supports:
  - GitHub: "github:user/repo/path/to/skill" or full URL
  - Local: "/path/to/skill-dir" (copies)

Installs to: skills/community/<skill-name>/

Usage:
    from Jotty.core.capabilities.registry.skill_installer import install_skill
    result = await install_skill("github:user/repo/skills/my-skill")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default install target
_JOTTY_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # -> Jotty/
_COMMUNITY_DIR = _JOTTY_ROOT / "skills" / "community"


async def install_skill(
    source: str,
    target_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Install a skill from GitHub URL or local path.

    Args:
        source: Skill source. Formats:
            - "github:user/repo/skills/skill-name"
            - "https://github.com/user/repo/tree/main/skills/skill-name"
            - "/path/to/local/skill-dir" (local copy)
        target_dir: Override install directory (default: skills/community/).

    Returns:
        Dict with: success, skill_name, installed_path, source.
    """
    target = target_dir or _COMMUNITY_DIR
    target.mkdir(parents=True, exist_ok=True)

    if source.startswith("github:") or "github.com" in source:
        return await _install_from_github(source, target)
    elif Path(source).is_dir():
        return _install_from_local(Path(source), target)
    else:
        return {"success": False, "error": f"Unsupported source format: {source}"}


async def _install_from_github(source: str, target: Path) -> Dict[str, Any]:
    """Download a skill directory from GitHub."""
    import asyncio

    # Parse source into owner/repo/path
    if source.startswith("github:"):
        # github:user/repo/skills/skill-name
        parts = source[7:].split("/", 2)
        if len(parts) < 3:
            return {
                "success": False,
                "error": f"Invalid github source: {source}. " "Use github:user/repo/path/to/skill",
            }
        owner, repo, path = parts[0], parts[1], parts[2]
    elif "github.com" in source:
        # https://github.com/user/repo/tree/branch/path/to/skill
        import re

        match = re.search(r"github\.com/([^/]+)/([^/]+)/tree/[^/]+/(.*)", source)
        if not match:
            return {"success": False, "error": f"Could not parse GitHub URL: {source}"}
        owner, repo, path = match.group(1), match.group(2), match.group(3)
    else:
        return {"success": False, "error": f"Not a GitHub source: {source}"}

    skill_name = path.rstrip("/").split("/")[-1]
    install_path = target / skill_name

    if install_path.exists():
        return {
            "success": False,
            "error": f"Skill already installed: {install_path}",
            "skill_name": skill_name,
        }

    # Use git sparse-checkout to download just the skill directory
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            f"https://github.com/{owner}/{repo}.git",
            str(target / f".tmp-{skill_name}"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            return {"success": False, "error": f"git clone failed: {stderr.decode()[:200]}"}

        tmp_dir = target / f".tmp-{skill_name}"

        # Sparse checkout the specific path
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(tmp_dir),
            "sparse-checkout",
            "set",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

        # Move skill directory to final location
        src = tmp_dir / path
        if src.is_dir():
            shutil.copytree(src, install_path)
        else:
            # Fallback: copy entire checkout
            shutil.copytree(tmp_dir, install_path, ignore=shutil.ignore_patterns(".git"))

        # Cleanup temp
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(f"Installed skill '{skill_name}' from {source} → {install_path}")
        return {
            "success": True,
            "skill_name": skill_name,
            "installed_path": str(install_path),
            "source": source,
        }

    except Exception as e:
        # Cleanup on failure
        tmp_dir = target / f".tmp-{skill_name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"success": False, "error": str(e)}


def _install_from_local(source: Path, target: Path) -> Dict[str, Any]:
    """Copy a local skill directory."""
    skill_name = source.name
    install_path = target / skill_name

    if install_path.exists():
        return {
            "success": False,
            "error": f"Skill already installed: {install_path}",
            "skill_name": skill_name,
        }

    try:
        shutil.copytree(source, install_path)
        logger.info(f"Installed skill '{skill_name}' from {source} → {install_path}")
        return {
            "success": True,
            "skill_name": skill_name,
            "installed_path": str(install_path),
            "source": str(source),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_installed() -> list[Dict[str, str]]:
    """List all installed community skills."""
    if not _COMMUNITY_DIR.exists():
        return []

    skills = []
    for d in sorted(_COMMUNITY_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            skill_md = d / "SKILL.md"
            description = ""
            if skill_md.exists():
                for line in skill_md.read_text().split("\n"):
                    if line.strip() and not line.startswith("#") and not line.startswith("---"):
                        description = line.strip()
                        break
            skills.append({"name": d.name, "path": str(d), "description": description})
    return skills


def uninstall_skill(skill_name: str) -> Dict[str, Any]:
    """Remove an installed community skill."""
    install_path = _COMMUNITY_DIR / skill_name
    if not install_path.exists():
        return {"success": False, "error": f"Skill not found: {skill_name}"}

    shutil.rmtree(install_path)
    logger.info(f"Uninstalled skill: {skill_name}")
    return {"success": True, "skill_name": skill_name}
