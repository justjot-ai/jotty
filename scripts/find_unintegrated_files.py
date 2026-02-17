#!/usr/bin/env python3
"""
Find unintegrated files in core/ folder.

Identifies Python files that are never imported by any other file,
indicating they may be orphaned, dead code, or not integrated.
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def find_all_python_files(root_dir):
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__ and .git
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".mypy_cache"]]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def extract_module_path(file_path, base_dir):
    """Convert file path to module import path."""
    # Remove base directory
    rel_path = os.path.relpath(file_path, base_dir)

    # Remove .py extension
    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]

    # Convert to module path
    module_path = rel_path.replace(os.sep, ".")

    # Remove __init__ from end
    if module_path.endswith(".__init__"):
        module_path = module_path[:-9]

    return module_path


def find_imports_in_file(file_path):
    """Extract all import statements from a file."""
    imports = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Match: from X import Y
        from_imports = re.findall(r"from\s+([\w.]+)\s+import", content)
        imports.update(from_imports)

        # Match: import X
        direct_imports = re.findall(r"(?:^|\n)import\s+([\w.]+)", content)
        imports.update(direct_imports)

        # Match: import X as Y
        as_imports = re.findall(r"import\s+([\w.]+)\s+as", content)
        imports.update(as_imports)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return imports


def analyze_integration(base_dir):
    """Analyze which files are integrated (imported)."""
    python_files = find_all_python_files(base_dir)

    # Build module path for each file
    file_to_module = {}
    for file_path in python_files:
        module_path = extract_module_path(file_path, base_dir)
        file_to_module[file_path] = module_path

    # Track all imports across the codebase
    all_imports = set()

    print(f"Analyzing {len(python_files)} Python files...")

    for file_path in python_files:
        imports = find_imports_in_file(file_path)
        all_imports.update(imports)

    # Check which core files are never imported
    unintegrated = []

    for file_path, module_path in file_to_module.items():
        # Skip __init__.py files (they're not directly imported)
        if file_path.endswith("__init__.py"):
            continue

        # Check if module is imported anywhere
        is_imported = False

        # Check exact match
        if module_path in all_imports:
            is_imported = True

        # Check if imported as part of package
        # e.g., "core.intelligence.reasoning.agents.mermaid_agent" imported via "from core.intelligence.reasoning.agents import mermaid_agent"
        for imp in all_imports:
            if module_path.startswith(imp) or imp.startswith(module_path):
                is_imported = True
                break

        if not is_imported:
            # Calculate relative path from project root
            rel_path = os.path.relpath(file_path, base_dir)
            unintegrated.append((rel_path, module_path))

    return unintegrated


def main():
    """Main analysis."""
    project_root = Path(__file__).parent.parent
    core_dir = project_root / "core"

    print("=" * 80)
    print("UNINTEGRATED FILES ANALYSIS - core/ folder")
    print("=" * 80)
    print()

    unintegrated = analyze_integration(str(project_root))

    # Filter to only core/ files
    core_unintegrated = [
        (path, module) for path, module in unintegrated if path.startswith("core/")
    ]

    if not core_unintegrated:
        print("✅ All files in core/ are integrated!")
        return

    # Group by directory
    by_dir = defaultdict(list)
    for path, module in core_unintegrated:
        dir_name = os.path.dirname(path)
        by_dir[dir_name].append((path, module))

    print(f"Found {len(core_unintegrated)} unintegrated files in core/\n")

    for dir_name in sorted(by_dir.keys()):
        files = by_dir[dir_name]
        print(f"\n{dir_name}/ ({len(files)} files)")
        print("-" * 80)

        for path, module in sorted(files):
            file_name = os.path.basename(path)

            # Calculate file size
            full_path = project_root / path
            try:
                size = os.path.getsize(full_path)
                lines = sum(1 for _ in open(full_path))
                print(f"  • {file_name:<40} ({lines:>4} lines, {size:>6} bytes)")
            except:
                print(f"  • {file_name}")

    print("\n" + "=" * 80)
    print(f"TOTAL: {len(core_unintegrated)} unintegrated files")
    print("=" * 80)

    # Calculate total lines
    total_lines = 0
    for path, module in core_unintegrated:
        full_path = project_root / path
        try:
            lines = sum(1 for _ in open(full_path))
            total_lines += lines
        except:
            pass

    print(f"\nTotal lines of unintegrated code: {total_lines:,}")


if __name__ == "__main__":
    main()
