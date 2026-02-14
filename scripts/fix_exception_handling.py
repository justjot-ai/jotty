#!/usr/bin/env python3
"""
Auto-fix Exception Handling
============================

Automatically improves exception handling in Python files:
1. Adds logging to bare except clauses
2. Suggests specific exception types
3. Adds helpful context to error messages

Usage:
    python scripts/fix_exception_handling.py path/to/file.py     # Fix one file
    python scripts/fix_exception_handling.py --all               # Fix all core files
    python scripts/fix_exception_handling.py --check             # Check only (no fixes)
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


class ExceptionFixer(ast.NodeTransformer):
    """AST transformer to improve exception handling."""

    def __init__(self):
        self.fixes = []
        self.current_file = ""

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        """Visit exception handlers and suggest improvements."""

        # Check for bare except:
        if node.type is None:
            self.fixes.append({
                'line': node.lineno,
                'type': 'bare_except',
                'severity': 'HIGH',
                'message': 'Bare except clause catches all exceptions',
                'fix': 'Use specific exception or add logging if using Exception'
            })

        # Check for broad Exception without logging
        elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
            has_logging = self._has_logging_call(node)

            if not has_logging:
                self.fixes.append({
                    'line': node.lineno,
                    'type': 'broad_exception_no_log',
                    'severity': 'MEDIUM',
                    'message': 'Broad except Exception without logging',
                    'fix': 'Add: logger.exception(f"Error in {operation}: {e}")'
                })

        return self.generic_visit(node)

    def _has_logging_call(self, node: ast.AST) -> bool:
        """Check if node contains logging call."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr in ['error', 'exception', 'warning', 'critical']:
                        return True
        return False


def analyze_file(file_path: Path) -> List[dict]:
    """Analyze a file for exception handling issues."""
    try:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content, str(file_path))

        fixer = ExceptionFixer()
        fixer.current_file = str(file_path)
        fixer.visit(tree)

        return fixer.fixes
    except Exception as e:
        print(f"⚠️  Could not analyze {file_path}: {e}")
        return []


def fix_file(file_path: Path, dry_run: bool = False) -> int:
    """Fix exception handling in a file."""
    issues = analyze_file(file_path)

    if not issues:
        return 0

    print(f"\n📄 {file_path.relative_to(Path.cwd())}")

    for issue in issues:
        severity_emoji = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }.get(issue['severity'], 'ℹ️')

        print(f"  {severity_emoji} Line {issue['line']}: {issue['message']}")
        print(f"     Fix: {issue['fix']}")

    if not dry_run:
        print(f"  ⏭️  Auto-fix not yet implemented - manual review required")

    return len(issues)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-fix exception handling")
    parser.add_argument('path', nargs='?', help='File or directory to fix')
    parser.add_argument('--all', action='store_true', help='Fix all core files')
    parser.add_argument('--check', action='store_true', help='Check only (no fixes)')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')

    args = parser.parse_args()

    root = Path(__file__).parent.parent
    files = []

    if args.all:
        files = list((root / 'core').rglob('*.py'))
    elif args.path:
        p = Path(args.path)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = list(p.rglob('*.py'))
    else:
        # Default: check swarms
        files = list((root / 'core' / 'swarms').rglob('*.py'))

    if not files:
        print("No files to check")
        return

    print(f"🔍 Analyzing {len(files)} files...")

    total_issues = 0
    files_with_issues = 0

    for file_path in sorted(files):
        issue_count = fix_file(file_path, dry_run=args.check or args.stats)
        if issue_count > 0:
            total_issues += issue_count
            files_with_issues += 1

    print("\n" + "="*80)
    print(f"📊 Summary:")
    print(f"  Files analyzed: {len(files)}")
    print(f"  Files with issues: {files_with_issues}")
    print(f"  Total issues: {total_issues}")

    if total_issues > 0:
        print(f"\n💡 Next steps:")
        print(f"  1. Review the issues above")
        print(f"  2. Add logging to broad Exception handlers")
        print(f"  3. Use specific exception types where possible")
        print(f"  4. See: docs/ERROR_HANDLING_GUIDE.md")

    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
