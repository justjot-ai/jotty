#!/usr/bin/env python3
"""
Bulk Type Hint Addition Tool
=============================

Automatically adds missing type hints to Python files.
Focuses on return type hints (-> None) for methods without returns.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def add_return_hint_to_line(line: str, return_type: str = "None") -> str:
    """Add return hint to a function definition line."""
    line = line.rstrip()

    # Already has return hint
    if '->' in line:
        return line

    # Find the colon
    if ':' not in line:
        return line

    # Insert -> None before the colon
    parts = line.rsplit(':', 1)
    return f"{parts[0]} -> {return_type}:{parts[1]}"


def process_file(filepath: Path, dry_run: bool = True) -> Tuple[int, List[str]]:
    """
    Add missing return hints to a file.

    Returns:
        (num_changes, list_of_changes)
    """
    try:
        with open(filepath) as f:
            content = f.read()
            lines = content.split('\n')

        tree = ast.parse(content, filename=str(filepath))
    except Exception as e:
        return 0, [f"ERROR parsing {filepath}: {e}"]

    changes = []
    lines_to_modify = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip if already has return hint
            if node.returns is not None:
                continue

            # Skip private/dunder methods (focus on public APIs first)
            if node.name.startswith('_'):
                continue

            # Check if function has explicit return statements with values
            has_return_value = False
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Return) and subnode.value is not None:
                    has_return_value = True
                    break

            # If no return values, add -> None
            if not has_return_value:
                lineno = node.lineno - 1  # ast uses 1-based indexing
                if lineno < len(lines):
                    lines_to_modify[lineno] = "None"
                    changes.append(f"  Line {node.lineno}: {node.name} -> None")

    if not lines_to_modify:
        return 0, []

    # Apply modifications
    if not dry_run:
        for lineno, return_type in lines_to_modify.items():
            lines[lineno] = add_return_hint_to_line(lines[lineno], return_type)

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    return len(lines_to_modify), changes


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add missing type hints')
    parser.add_argument('paths', nargs='+', help='Files or directories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--apply', action='store_true', help='Apply changes')

    args = parser.parse_args()

    if not args.apply:
        args.dry_run = True

    total_changes = 0
    files_modified = 0

    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file() and path.suffix == '.py':
            files = [path]
        elif path.is_dir():
            files = list(path.rglob('*.py'))
        else:
            print(f"Skipping {path_str}")
            continue

        for filepath in files:
            if '__pycache__' in str(filepath):
                continue

            num_changes, change_list = process_file(filepath, dry_run=args.dry_run)

            if num_changes > 0:
                files_modified += 1
                total_changes += num_changes

                rel_path = filepath.relative_to(Path.cwd()) if filepath.is_relative_to(Path.cwd()) else filepath
                print(f"\n{'[DRY RUN] ' if args.dry_run else ''}📝 {rel_path}")
                print(f"   {num_changes} return hints added:")
                for change in change_list[:10]:  # Show first 10
                    print(change)
                if len(change_list) > 10:
                    print(f"   ... and {len(change_list) - 10} more")

    print(f"\n{'='*70}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}Summary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Total hints added: {total_changes}")

    if args.dry_run:
        print(f"\n💡 Run with --apply to make changes")
        return 1
    else:
        print(f"\n✅ Changes applied successfully!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
