#!/usr/bin/env python3
"""
Add Type Hints Automatically
=============================

Adds return type hints to functions missing them.
Uses heuristics to determine appropriate types.

Usage:
    python scripts/add_type_hints.py path/to/file.py    # Add hints to one file
    python scripts/add_type_hints.py --swarms            # Add hints to all swarms
    python scripts/add_type_hints.py --check             # Check only (no changes)
"""

import ast
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple


def infer_return_type(func_node: ast.FunctionDef, source_code: str) -> Optional[str]:
    """Infer return type from function body."""

    # Get function source
    func_lines = source_code.split('\n')[func_node.lineno-1:func_node.end_lineno]
    func_source = '\n'.join(func_lines)

    # Check for explicit return statements
    returns = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            if node.value is None:
                returns.append('None')
            elif isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    returns.append('str')
                elif isinstance(node.value.value, int):
                    returns.append('int')
                elif isinstance(node.value.value, float):
                    returns.append('float')
                elif isinstance(node.value.value, bool):
                    returns.append('bool')
                elif node.value.value is None:
                    returns.append('None')
            elif isinstance(node.value, ast.Dict):
                returns.append('Dict[str, Any]')
            elif isinstance(node.value, ast.List):
                returns.append('List[Any]')
            elif isinstance(node.value, ast.Tuple):
                returns.append('Tuple')

    # No return statements → likely None
    if not returns:
        # Check if it's a generator
        if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(func_node)):
            return None  # Generators are complex, skip
        return 'None'

    # All returns are None
    if all(r == 'None' for r in returns):
        return 'None'

    # Mixed returns with None → Optional[...]
    if 'None' in returns and len(set(returns)) > 1:
        non_none = [r for r in returns if r != 'None']
        if len(set(non_none)) == 1:
            return f'Optional[{non_none[0]}]'
        return None  # Complex, skip

    # Single consistent type
    if len(set(returns)) == 1:
        return returns[0]

    # Mixed types → too complex, skip
    return None


def add_type_hint_to_function(source_code: str, func_node: ast.FunctionDef, return_type: str) -> str:
    """Add return type hint to function signature."""

    lines = source_code.split('\n')

    # Find the closing parenthesis of the function signature
    start_line = func_node.lineno - 1

    # Simple case: single-line signature
    sig_line = lines[start_line]

    # Check if already has type hint
    if '->' in sig_line:
        return source_code

    # Find closing paren
    if ')' in sig_line and ':' in sig_line:
        # Single line signature
        sig_line = sig_line.replace('):', f') -> {return_type}:')
        lines[start_line] = sig_line
        return '\n'.join(lines)

    # Multi-line signature (complex, skip for now)
    return source_code


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """Process a single file and add type hints."""

    try:
        source_code = file_path.read_text()
        tree = ast.parse(source_code, str(file_path))
    except Exception as e:
        print(f"⚠️  Could not parse {file_path}: {e}")
        return 0

    changes = 0
    modified_code = source_code

    # Find functions needing hints
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip if already has return type
            if node.returns:
                continue

            # Skip special methods
            if node.name in ['__init__', '__str__', '__repr__', '__eq__', '__post_init__']:
                continue

            # Infer return type
            return_type = infer_return_type(node, source_code)

            if return_type:
                # Try to add hint
                new_code = add_type_hint_to_function(modified_code, node, return_type)
                if new_code != modified_code:
                    modified_code = new_code
                    changes += 1
                    if not dry_run:
                        print(f"  ✅ Line {node.lineno}: {node.name}() -> {return_type}")

    if changes > 0 and not dry_run:
        file_path.write_text(modified_code)
        print(f"✅ {file_path.name}: Added {changes} type hints")
    elif changes > 0:
        print(f"Would add {changes} type hints to {file_path.name}")

    return changes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add type hints to Python files")
    parser.add_argument('path', nargs='?', help='File or directory to process')
    parser.add_argument('--swarms', action='store_true', help='Process all swarm files')
    parser.add_argument('--check', action='store_true', help='Check only (no changes)')

    args = parser.parse_args()

    root = Path(__file__).parent.parent
    files = []

    if args.swarms:
        files = list((root / 'core' / 'swarms').rglob('*.py'))
    elif args.path:
        p = Path(args.path)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = list(p.rglob('*.py'))
    else:
        print("Usage: add_type_hints.py <path> or --swarms")
        return

    print(f"🔍 Processing {len(files)} files...")

    total_changes = 0
    for file_path in sorted(files):
        changes = process_file(file_path, dry_run=args.check)
        total_changes += changes

    print(f"\n📊 Summary: {total_changes} type hints {'would be ' if args.check else ''}added")

    if args.check and total_changes > 0:
        print(f"Run without --check to apply changes")


if __name__ == "__main__":
    main()
