#!/usr/bin/env python3
"""
Add type annotations to reach 100% coverage.
- Adds : Any to untyped parameters (except self/cls).
- Adds -> None for __init__, __post_init__, __setattr__, __exit__, __enter__ (context managers).
- Adds -> bool for __bool__, __contains__, __eq__-like; -> str for __str__, __repr__; -> int for __len__.
- Otherwise -> Any.
- Ensures 'from typing import Any' (and Optional if we add Optional) in each modified file.
Runs in dry_run by default; pass --write to apply changes.
"""

import ast
import re
import sys
from pathlib import Path

# Return type by name (method/function name -> return type string)
RETURN_NONE = (
    "__init__", "__post_init__", "__setattr__", "__set_name__",
)
RETURN_BOOL = (
    "__bool__", "__contains__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__exit__", "__aexit__",  # return bool (suppress exception or not)
)
RETURN_ANY_CONTEXT = ("__enter__", "__aenter__")  # return Any (context value; Self in 3.11+)
RETURN_STR = ("__str__", "__repr__", "__format__")
RETURN_INT = ("__len__", "__index__", "__hash__")


def return_type_for(name: str) -> str:
    if name in RETURN_NONE:
        return "None"
    if name in RETURN_BOOL:
        return "bool"
    if name in RETURN_STR:
        return "str"
    if name in RETURN_INT:
        return "int"
    if name in RETURN_ANY_CONTEXT:
        return "Any"
    return "Any"


def needs_param_annotation(arg: ast.arg) -> bool:
    if arg.annotation is not None:
        return False
    return True


def format_arg(arg: ast.arg, default: ast.expr | None, unparse_default: bool = True) -> str:
    ann = arg.annotation
    if ann is not None:
        type_str = ast.unparse(ann)
    else:
        type_str = "Any"
    part = f"{arg.arg}: {type_str}"
    if default is not None:
        part += " = " + (ast.unparse(default) if unparse_default else "...")
    return part


def signature_needs_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for arg in node.args.args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            return True
    for arg in (node.args.posonlyargs or []) + (node.args.kwonlyargs or []):
        if arg.annotation is None:
            return True
    if node.args.vararg and node.args.vararg.annotation is None:
        return True
    if node.args.kwarg and node.args.kwarg.annotation is None:
        return True
    if node.returns is None:
        return True
    return False


def get_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr | None]:
    """Return list of default for each arg in args.args; None means no default."""
    args = node.args.args
    defaults = list(node.args.defaults)
    # defaults align to last N args
    n_without = len(args) - len(defaults)
    return [None] * n_without + defaults


def build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    name = node.name
    defaults = get_defaults(node)
    parts = []
    for i, arg in enumerate(node.args.args):
        default = defaults[i] if i < len(defaults) else None
        if arg.arg in ("self", "cls"):
            if default is not None:
                try:
                    parts.append(f"{arg.arg}={ast.unparse(default)}")
                except Exception:
                    parts.append(f"{arg.arg}=...")
            else:
                parts.append(arg.arg)
        else:
            parts.append(format_arg(arg, default))
    if node.args.posonlyargs:
        for i, arg in enumerate(node.args.posonlyargs):
            parts.append(format_arg(arg, None))
        if not parts:
            parts.append("/")
        elif parts and parts[-1] != "/":
            parts.append("/")
    if node.args.vararg:
        v = node.args.vararg
        ann = ast.unparse(v.annotation) if v.annotation else "Any"
        parts.append(f"*{v.arg}: {ann}")
    elif node.args.kwonlyargs:
        parts.append("*")
    for arg in node.args.kwonlyargs or []:
        default = None
        if node.args.kw_defaults:
            idx = node.args.kwonlyargs.index(arg)
            default = node.args.kw_defaults[idx]
        parts.append(format_arg(arg, default))
    if node.args.kwarg:
        k = node.args.kwarg
        ann = ast.unparse(k.annotation) if k.annotation else "Any"
        parts.append(f"**{k.arg}: {ann}")
    args_str = ", ".join(parts)
    ret = node.returns
    if ret is not None:
        ret_str = ast.unparse(ret)
    else:
        ret_str = return_type_for(name)
    return f"{prefix}{name}({args_str}) -> {ret_str}:"


def find_signature_end_line(lines: list[str], start_lineno: int) -> int:
    """Return 0-based index of the line that ends the signature (contains ): or ) ->)."""
    for i in range(start_lineno, len(lines)):
        line = lines[i]
        if ") ->" in line or "):" in line:
            return i
    return start_lineno


def ensure_typing_any(lines: list[str]) -> tuple[list[str], bool]:
    """Ensure 'from typing import ...' includes Any. Returns (new_lines, added_any)."""
    added = False
    for i, line in enumerate(lines):
        if line.startswith("from typing import") or line.startswith("from typing import "):
            if "Any" not in line:
                # append Any to the import
                line = line.rstrip()
                if line.endswith(")"):
                    line = line[:-1] + ", Any)"
                else:
                    line = line.rstrip(",") + ", Any"
                lines[i] = line + "\n"
                added = True
            break
        if line.startswith("from __future__ import"):
            continue
        if i > 30:
            break
    if not added:
        # Check if we need to add a new import
        has_typing = any("from typing import" in l for l in lines[:40])
        if not has_typing:
            # Find first non-comment, non-docstring, non-empty line after future/encoding
            insert_at = 0
            for j, line in enumerate(lines[:25]):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if stripped.startswith("from __future__"):
                    insert_at = j + 1
                    continue
                if stripped and not stripped.startswith("from ") and not stripped.startswith("import "):
                    break
                if stripped.startswith("from ") or stripped.startswith("import "):
                    insert_at = j + 1
            lines.insert(insert_at, "from typing import Any\n")
            added = True
    return lines, added


def process_file(path: Path, dry_run: bool) -> tuple[int, bool]:
    """Add annotations to untyped functions. Returns (count_modified, file_modified)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  skip read {path}: {e}", file=sys.stderr)
        return 0, False
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"  skip parse {path}: {e}", file=sys.stderr)
        return 0, False

    edits: list[tuple[int, int, str]] = []  # (start_line_0based, end_line_0based_excl, new_text)
    need_any = False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not signature_needs_annotation(node):
            continue
        start_lineno_1 = node.lineno
        start_lineno_0 = start_lineno_1 - 1
        end_sig_0 = find_signature_end_line(lines, start_lineno_0)
        # Build new signature with annotations
        new_sig = build_signature(node)
        indent = ""
        for c in lines[start_lineno_0]:
            if c in " \t":
                indent += c
            else:
                break
        new_sig_line = indent + new_sig + "\n"
        old_slice = "".join(lines[start_lineno_0 : end_sig_0 + 1])
        # Only replace when signature ends with ":" (no same-line body), to avoid dropping " pass" etc.
        if old_slice.strip().endswith(":"):
            edits.append((start_lineno_0, end_sig_0 + 1, new_sig_line))
            need_any = True

    if not edits:
        return 0, False

    # Apply edits in reverse order so line numbers don't shift
    edits.sort(key=lambda x: -x[0])
    new_lines = list(lines)
    for start, end, new_text in edits:
        new_lines[start:end] = [new_text]

    new_lines, _ = ensure_typing_any(new_lines)
    if need_any:
        new_lines, _ = ensure_typing_any(new_lines)

    if not dry_run:
        path.write_text("".join(new_lines), encoding="utf-8")
    return len(edits), True


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Apply changes (default: dry run)")
    ap.add_argument("dirs", nargs="*", default=["core", "cli"], help="Directories to process")
    args = ap.parse_args()
    dry_run = not args.write
    root = Path(__file__).resolve().parent.parent
    total = 0
    files_done = 0
    for d in args.dirs:
        dirpath = root / d
        if not dirpath.is_dir():
            print(f"Skip (not a dir): {dirpath}")
            continue
        for path in sorted(dirpath.rglob("*.py")):
            n, modified = process_file(path, dry_run)
            if modified:
                total += n
                files_done += 1
                print(f"  {'[dry-run] ' if dry_run else ''}{path.relative_to(root)} ({n} defs)")
    print(f"\nTotal: {total} signatures in {files_done} files. {'(dry run; use --write to apply)' if dry_run else 'Done.'}")


if __name__ == "__main__":
    main()
