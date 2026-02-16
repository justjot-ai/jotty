#!/usr/bin/env python3
"""
Update imports to use new core/execution/ structure.

Migrates:
- from Jotty.core.modes.agent → from Jotty.core.execution.base
- from Jotty.core.execution.swarms → from Jotty.core.execution.swarms
- from Jotty.core.modes.workflow → from Jotty.core.execution.workflows
"""

import re
import sys
from pathlib import Path
from typing import Tuple

# Import mappings
IMPORT_REPLACEMENTS = [
    # Agent imports
    (
        r"from Jotty\.core\.modes\.agent\.base import",
        "from Jotty.core.execution.base import",
    ),
    (
        r"from Jotty\.core\.modes\.agent\.agents import",
        "from Jotty.core.execution.agents import",
    ),
    (
        r"from Jotty\.core\.modes\.agent import",
        "from Jotty.core.execution.base import",
    ),
    # Swarm imports (now canonical location is core.intelligence.swarms)
    (
        r"from Jotty\.core\.execution\.swarms\.templates import",
        "from Jotty.core.execution.swarms.templates import",
    ),
    (
        r"from Jotty\.core\.execution\.swarms\.base import",
        "from Jotty.core.execution.swarms.base import",
    ),
    (
        r"from Jotty\.core\.execution\.swarms import",
        "from Jotty.core.execution.swarms import",
    ),
    # Workflow imports
    (
        r"from Jotty\.core\.modes\.workflow import",
        "from Jotty.core.execution.workflows import",
    ),
    # Intelligence reasoning experts (now in execution/agents)
    (
        r"from Jotty\.core\.intelligence\.reasoning\.experts import",
        "from Jotty.core.execution.agents import",
    ),
]


def update_file(file_path: Path) -> Tuple[bool, int]:
    """
    Update imports in a single file.

    Returns:
        (changed, num_replacements)
    """
    try:
        content = file_path.read_text()
        original = content
        replacements = 0

        for pattern, replacement in IMPORT_REPLACEMENTS:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                replacements += count

        if content != original:
            file_path.write_text(content)
            return True, replacements

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False, 0


def main():
    """Update all Python files in core/ and tests/."""
    project_root = Path(__file__).parent.parent

    # Directories to process
    dirs_to_process = [
        project_root / "core",
        project_root / "tests",
    ]

    total_files = 0
    total_changed = 0
    total_replacements = 0

    for directory in dirs_to_process:
        if not directory.exists():
            print(f"Skipping {directory} (doesn't exist)")
            continue

        print(f"\nProcessing {directory}...")

        for py_file in directory.rglob("*.py"):
            # Skip pycache
            if "__pycache__" in str(py_file):
                continue

            total_files += 1
            changed, replacements = update_file(py_file)

            if changed:
                total_changed += 1
                total_replacements += replacements
                print(f"  ✓ {py_file.relative_to(project_root)} ({replacements} replacements)")

    print(f"\n{'='*80}")
    print("Import Update Summary")
    print(f"{'='*80}")
    print(f"Files processed: {total_files}")
    print(f"Files changed:   {total_changed}")
    print(f"Total replacements: {total_replacements}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
