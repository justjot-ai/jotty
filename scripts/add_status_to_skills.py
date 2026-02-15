#!/usr/bin/env python3
"""
Add status callback support to all Jotty skills.

This script:
1. Adds SkillStatus import to each tools.py
2. Adds status initialization
3. Adds status.set_callback() to each tool function
4. Adds basic status.emit() calls based on patterns

Run from Jotty root:
    python scripts/add_status_to_skills.py
"""

import os
import re
from pathlib import Path


def get_skill_name(file_path: Path) -> str:
    """Extract skill name from path."""
    return file_path.parent.name


def already_has_status(content: str) -> bool:
    """Check if file already has status support."""
    return "SkillStatus" in content or "_status_callback" in content or "emit_status" in content


def add_status_import(content: str) -> str:
    """Add SkillStatus import after other imports."""
    import_line = "from Jotty.core.infrastructure.utils.skill_status import SkillStatus\n"

    # Find last import line
    lines = content.split("\n")
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    # Insert after last import
    lines.insert(last_import_idx + 1, "")
    lines.insert(last_import_idx + 2, import_line.strip())

    return "\n".join(lines)


def add_status_init(content: str, skill_name: str) -> str:
    """Add status initialization after imports."""
    init_code = f'\n# Status emitter for progress updates\nstatus = SkillStatus("{skill_name}")\n'

    # Find where to insert (after imports, before first function)
    lines = content.split("\n")
    insert_idx = 0

    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from ") or line.strip().startswith("#"):
            insert_idx = i + 1
        elif line.startswith("def ") or line.startswith("async def ") or line.startswith("class "):
            break

    # Insert init code
    lines.insert(insert_idx, init_code)

    return "\n".join(lines)


def add_callback_to_tool(content: str) -> str:
    """Add status.set_callback() to each tool function."""
    # Pattern to find tool functions
    tool_pattern = r'((?:async )?def \w+_tool\(params[^)]*\)[^:]*:)\n(\s*)("""[^"]*""")?'

    def add_callback(match):
        func_def = match.group(1)
        indent = match.group(2)
        docstring = match.group(3) or ""

        callback_line = f"{indent}status.set_callback(params.pop('_status_callback', None))\n"

        if docstring:
            return f"{func_def}\n{indent}{docstring}\n{callback_line}"
        else:
            return f"{func_def}\n{callback_line}"

    return re.sub(tool_pattern, add_callback, content)


def add_basic_status_calls(content: str, skill_name: str) -> str:
    """Add basic status.emit() calls based on common patterns."""
    # Add status at start of tool after callback setup
    # This is tricky to do automatically, so we'll just add a simple pattern

    # Find patterns like requests.get, search, fetch, etc. and add status before them
    patterns = [
        (r"(\s+)(requests\.(get|post)\([^)]+\))", r'\1status.fetching("web data")\n\1\2'),
        (r"(\s+)(search\([^)]+\))", r'\1status.searching("results")\n\1\2'),
        (r"(\s+)(\.write_pdf\([^)]+\))", r'\1status.creating("PDF")\n\1\2'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, count=1)

    return content


def process_skill(tools_path: Path) -> bool:
    """Process a single skill's tools.py file."""
    try:
        content = tools_path.read_text()
        skill_name = get_skill_name(tools_path)

        # Skip if already has status support
        if already_has_status(content):
            print(f"  ⏭️  {skill_name}: Already has status support")
            return False

        # Skip very short files (likely stubs)
        if len(content) < 200:
            print(f"  ⏭️  {skill_name}: Stub file, skipping")
            return False

        # Add status support
        new_content = content
        new_content = add_status_import(new_content)
        new_content = add_status_init(new_content, skill_name)
        new_content = add_callback_to_tool(new_content)

        # Write back
        tools_path.write_text(new_content)
        print(f"  ✅ {skill_name}: Added status support")
        return True

    except Exception as e:
        print(f"  ❌ {skill_name}: Error - {e}")
        return False


def main():
    """Main entry point."""
    skills_dir = Path(__file__).parent.parent / "skills"

    if not skills_dir.exists():
        print(f"Skills directory not found: {skills_dir}")
        return

    print(f"Processing skills in: {skills_dir}")
    print()

    updated = 0
    skipped = 0
    errors = 0

    for tools_path in sorted(skills_dir.glob("*/tools.py")):
        result = process_skill(tools_path)
        if result:
            updated += 1
        elif result is False:
            skipped += 1
        else:
            errors += 1

    print()
    print(f"Summary: {updated} updated, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
