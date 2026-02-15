#!/usr/bin/env python3
"""
Advanced Type Hint Addition Tool
=================================

Infers and adds type hints using AST analysis and heuristics.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class TypeInferrer(ast.NodeVisitor):
    """Infer return types from function bodies."""

    def __init__(self):
        self.return_types: Set[str] = set()
        self.has_explicit_none = False

    def visit_Return(self, node):
        """Analyze return statements."""
        if node.value is None:
            self.has_explicit_none = True
        elif isinstance(node.value, ast.Constant):
            if node.value.value is None:
                self.has_explicit_none = True
            elif isinstance(node.value.value, bool):
                self.return_types.add("bool")
            elif isinstance(node.value.value, int):
                self.return_types.add("int")
            elif isinstance(node.value.value, str):
                self.return_types.add("str")
            elif isinstance(node.value.value, float):
                self.return_types.add("float")
        elif isinstance(node.value, ast.Dict):
            self.return_types.add("Dict")
        elif isinstance(node.value, ast.List):
            self.return_types.add("List")
        elif isinstance(node.value, ast.Tuple):
            self.return_types.add("Tuple")
        elif isinstance(node.value, ast.Set):
            self.return_types.add("Set")
        elif isinstance(node.value, ast.Name):
            if node.value.id in ("True", "False"):
                self.return_types.add("bool")
            elif node.value.id == "None":
                self.has_explicit_none = True
        self.generic_visit(node)


def infer_return_type(func_node: ast.FunctionDef) -> Optional[str]:
    """Infer return type from function body."""
    inferrer = TypeInferrer()
    inferrer.visit(func_node)

    # If only returns None
    if inferrer.has_explicit_none and not inferrer.return_types:
        return "None"

    # If single consistent type
    if len(inferrer.return_types) == 1:
        return list(inferrer.return_types)[0]

    # If multiple types, use Any
    if inferrer.return_types:
        return "Any"

    return None


def infer_param_type(param_name: str, default_value=None) -> Optional[str]:
    """Infer parameter type from name and default."""
    # Common patterns
    if param_name in ("kwargs", "kw", "options"):
        return "Dict[str, Any]"
    if param_name == "args":
        return "Tuple[Any, ...]"

    # From default value
    if default_value is not None:
        if isinstance(default_value, ast.Constant):
            val = default_value.value
            if isinstance(val, bool):
                return "bool"
            elif isinstance(val, int):
                return "int"
            elif isinstance(val, str):
                return "str"
            elif isinstance(val, float):
                return "float"
        elif isinstance(default_value, ast.List):
            return "List"
        elif isinstance(default_value, ast.Dict):
            return "Dict"

    # From name patterns
    if param_name.endswith("_id") or param_name.endswith("_name"):
        return "str"
    if (
        param_name.startswith("is_")
        or param_name.startswith("has_")
        or param_name.startswith("should_")
    ):
        return "bool"
    if (
        param_name.endswith("_count")
        or param_name == "count"
        or param_name == "limit"
        or param_name == "max"
    ):
        return "int"
    if param_name.endswith("_dict") or param_name == "config" or param_name == "params":
        return "Dict"
    if param_name.endswith("_list") or param_name in ("items", "values", "keys"):
        return "List"

    return None


def add_hints_to_function(lines: List[str], func_node: ast.FunctionDef) -> bool:
    """Add type hints to a function. Returns True if modified."""
    modified = False
    lineno = func_node.lineno - 1

    if lineno >= len(lines):
        return False

    func_line = lines[lineno]

    # Skip if already has return hint
    if func_node.returns is None:
        inferred_return = infer_return_type(func_node)
        if inferred_return:
            # Add return hint
            if ":" in func_line and "->" not in func_line:
                func_line = func_line.rstrip()
                parts = func_line.rsplit(":", 1)

                # Handle type imports
                needs_import = inferred_return in (
                    "Dict",
                    "List",
                    "Tuple",
                    "Set",
                    "Any",
                    "Optional",
                )

                func_line = f"{parts[0]} -> {inferred_return}:{parts[1]}"
                lines[lineno] = func_line
                modified = True

    return modified


def process_file_advanced(filepath: Path, dry_run: bool = True) -> Tuple[int, List[str]]:
    """Add advanced type hints to a file."""
    try:
        with open(filepath) as f:
            content = f.read()
            lines = content.split("\n")

        tree = ast.parse(content, filename=str(filepath))
    except Exception as e:
        return 0, [f"ERROR: {e}"]

    changes = []
    modified_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private methods for now
            if node.name.startswith("__"):
                continue

            # Only add hints to functions missing return hints
            if node.returns is None:
                if add_hints_to_function(lines, node):
                    modified_count += 1
                    inferred = infer_return_type(node)
                    changes.append(f"  Line {node.lineno}: {node.name} -> {inferred}")

    if modified_count > 0 and not dry_run:
        # Check if we need to add imports
        has_typing_import = "from typing import" in content or "import typing" in content
        needs_typing = any(
            t in "\n".join(changes) for t in ["Dict", "List", "Tuple", "Any", "Optional", "Set"]
        )

        if needs_typing and not has_typing_import:
            # Add typing import after other imports
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    continue
                else:
                    lines.insert(i, "from typing import Dict, List, Tuple, Any, Optional, Set")
                    break

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    return modified_count, changes


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    if not args.apply:
        args.dry_run = True

    total_changes = 0
    files_modified = 0

    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            continue

        for filepath in files:
            if "__pycache__" in str(filepath):
                continue

            num_changes, change_list = process_file_advanced(filepath, dry_run=args.dry_run)

            if num_changes > 0:
                files_modified += 1
                total_changes += num_changes

                rel_path = (
                    filepath.relative_to(Path.cwd())
                    if filepath.is_relative_to(Path.cwd())
                    else filepath
                )
                print(f"\n{'[DRY RUN] ' if args.dry_run else ''}📝 {rel_path}")
                print(f"   {num_changes} type hints inferred:")
                for change in change_list[:5]:
                    print(change)
                if len(change_list) > 5:
                    print(f"   ... and {len(change_list) - 5} more")

    print(f"\n{'='*70}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}Summary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Total hints added: {total_changes}")

    if args.dry_run:
        print(f"\n💡 Run with --apply to make changes")
        return 1
    else:
        print(f"\n✅ Changes applied!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
