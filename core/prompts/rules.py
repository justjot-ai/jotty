"""
Rule Files — project-specific agent instructions (Cline .clinerules pattern).

Loads rules from the workspace so agents adapt to project conventions.
Supports multiple formats (Cline compatibility):
  .jottyrules    — Jotty native
  .clinerules    — Cline rules (compatible)
  .cursorrules   — Cursor rules (compatible)
  CLAUDE.md      — Claude Code rules (compatible)

Rules are plain text/markdown injected into agent system prompts.
Loaded once per task, cached per workspace path.

KISS: just reads files and returns their content. No parsing, no schema.

Usage:
    rules = load_project_rules("/path/to/project")
    # Returns: "Always use TypeScript strict mode.\nPrefer functional components.\n..."
"""

import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Supported rule file names, in priority order
RULE_FILES = [
    ".jottyrules",
    ".clinerules",
    ".cursorrules",
    "CLAUDE.md",
]

# Max size to prevent loading huge files (64KB)
_MAX_RULES_SIZE = 65536


@lru_cache(maxsize=16)
def load_project_rules(workspace_dir: str) -> str:
    """
    Load project-specific rules from the workspace directory.

    Checks for rule files in priority order. If multiple exist,
    concatenates them (first = highest priority).

    Args:
        workspace_dir: Path to the project/workspace root

    Returns:
        Combined rules text, or "" if no rule files found
    """
    root = Path(workspace_dir)
    if not root.is_dir():
        return ""

    parts = []
    for filename in RULE_FILES:
        filepath = root / filename
        if filepath.is_file():
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                if len(content) > _MAX_RULES_SIZE:
                    content = content[:_MAX_RULES_SIZE] + "\n[...truncated]"
                    logger.warning(f"Rule file {filepath} truncated to {_MAX_RULES_SIZE} bytes")
                if content.strip():
                    parts.append(f"# Rules from {filename}\n{content.strip()}")
                    logger.info(f"Loaded rules from {filepath} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")

    return "\n\n".join(parts)


def clear_rules_cache():
    """Clear the rules cache (call when workspace changes)."""
    load_project_rules.cache_clear()
