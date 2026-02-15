#!/usr/bin/env python3
"""
Auto-fix common mypy type errors.

This script:
1. Runs mypy to detect type errors
2. Parses the output to categorize fixable errors
3. Automatically fixes common patterns:
   - List[T] = None → Optional[List[T]] = None
   - x = {} → x: dict[str, Any] = {}
   - Missing Optional[] for None assignments
4. Re-runs to verify fixes

Usage:
    python scripts/autofix_mypy_errors.py [--dry-run] [--category CATEGORY]

Options:
    --dry-run           Show what would be fixed without making changes
    --category CATEGORY Only fix specific category (none-assignment, missing-annotation, etc.)

Exit code: 0 if all fixable errors fixed; 1 if manual intervention needed
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple


@dataclass
class MypyError:
    """Represents a single mypy error."""
    file: str
    line: int
    column: int
    error_type: str
    message: str
    code: str  # e.g., "assignment", "var-annotated"

    def __repr__(self):
        return f"MypyError({self.file}:{self.line} [{self.code}] {self.message[:50]}...)"


class MypyErrorParser:
    """Parse mypy output to extract errors."""

    # Regex to match mypy error lines like:
    # core/modes/workflow/smart_swarm_registry.py:54:33: error: Incompatible types in assignment (expression has type "None", variable has type "list[str]")  [assignment]
    ERROR_PATTERN = re.compile(
        r'^(?P<file>[\w/._-]+\.py):(?P<line>\d+):(?P<column>\d+):\s+'
        r'error:\s+(?P<message>.+?)\s+\[(?P<code>[\w-]+)\]',
        re.MULTILINE
    )

    @classmethod
    def parse(cls, output: str) -> List[MypyError]:
        """Parse mypy output and return list of errors."""
        errors = []
        for match in cls.ERROR_PATTERN.finditer(output):
            error = MypyError(
                file=match.group('file'),
                line=int(match.group('line')),
                column=int(match.group('column')),
                error_type='error',
                message=match.group('message'),
                code=match.group('code')
            )
            errors.append(error)
        return errors


class MypyErrorFixer:
    """Auto-fix common mypy errors."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.fixes_applied = 0
        self.files_modified: Set[str] = set()

    def fix_none_assignment(self, error: MypyError) -> bool:
        """
        Fix: List[str] = None → Optional[List[str]] = None
        Error code: assignment
        Message: "Incompatible types in assignment (expression has type \"None\", variable has type..."
        """
        if error.code != 'assignment':
            return False
        if 'expression has type "None"' not in error.message:
            return False

        # Handle Jotty/ prefix in file path
        file_path = Path(error.file)
        if not file_path.exists() and error.file.startswith('Jotty/'):
            file_path = Path(error.file.replace('Jotty/', '', 1))
        if not file_path.exists():
            return False

        lines = file_path.read_text().splitlines()
        if error.line > len(lines):
            return False

        line = lines[error.line - 1]  # Line numbers are 1-indexed

        # Pattern: variable: Type = None
        # Should be: variable: Optional[Type] = None
        pattern = r'(\s*)(\w+):\s*([^=]+?)\s*=\s*None'
        match = re.search(pattern, line)
        if not match:
            return False

        indent, var_name, type_hint = match.groups()
        type_hint = type_hint.strip()

        # Skip if already Optional
        if 'Optional' in type_hint or 'None' in type_hint or '|' in type_hint:
            return False

        # Create fixed line
        fixed_line = f"{indent}{var_name}: Optional[{type_hint}] = None"

        if self.dry_run:
            print(f"  Would fix {error.file}:{error.line}")
            print(f"    - {line}")
            print(f"    + {fixed_line}")
            return True

        # Apply fix
        lines[error.line - 1] = fixed_line

        # Check if Optional is imported
        needs_optional_import = True
        for i, l in enumerate(lines):
            if 'from typing import' in l and 'Optional' in l:
                needs_optional_import = False
                break
            if l.startswith('from ') or l.startswith('import '):
                continue
            else:
                break  # Past imports

        # Add Optional import if needed
        if needs_optional_import:
            # Find typing import or add new one
            typing_import_line = None
            for i, l in enumerate(lines):
                if 'from typing import' in l:
                    typing_import_line = i
                    break

            if typing_import_line is not None:
                # Add Optional to existing import
                current_import = lines[typing_import_line]
                if current_import.strip().endswith(')'):
                    # Multi-line import
                    lines[typing_import_line] = current_import.replace(')', ', Optional)')
                else:
                    # Single line import
                    lines[typing_import_line] = current_import.rstrip() + ', Optional'
            else:
                # Add new import after other imports
                insert_pos = 0
                for i, l in enumerate(lines):
                    if l.startswith('from ') or l.startswith('import '):
                        insert_pos = i + 1
                    elif l.strip() == '':
                        continue
                    else:
                        break
                lines.insert(insert_pos, 'from typing import Optional')

        # Write back
        file_path.write_text('\n'.join(lines) + '\n')
        self.fixes_applied += 1
        self.files_modified.add(error.file)
        print(f"  ✅ Fixed {error.file}:{error.line} (None assignment)")
        return True

    def fix_missing_annotation(self, error: MypyError) -> bool:
        """
        Fix: x = {} → x: dict[str, Any] = {}
        Error code: var-annotated
        Message: "Need type annotation for..."
        """
        if error.code != 'var-annotated':
            return False

        # Handle Jotty/ prefix in file path
        file_path = Path(error.file)
        if not file_path.exists() and error.file.startswith('Jotty/'):
            file_path = Path(error.file.replace('Jotty/', '', 1))
        if not file_path.exists():
            return False

        lines = file_path.read_text().splitlines()
        if error.line > len(lines):
            return False

        line = lines[error.line - 1]

        # Pattern: variable = {}
        # Hint is in the message like: 'hint: "by_category: dict[<type>, <type>] = ..."'
        hint_match = re.search(r'hint: "([^"]+)"', error.message)
        if not hint_match:
            return False

        hint = hint_match.group(1)

        # Extract variable name and suggested type
        var_match = re.search(r'(\w+):\s*(\w+)\[', hint)
        if not var_match:
            return False

        var_name = var_match.group(1)
        type_name = var_match.group(2)  # e.g., "dict"

        # Find the assignment in the line
        assignment_pattern = rf'(\s*){var_name}\s*=\s*(.+)'
        match = re.search(assignment_pattern, line)
        if not match:
            return False

        indent, value = match.groups()

        # Determine appropriate type based on value
        if value.strip() == '{}':
            type_hint = "dict[str, Any]"  # Default dict type
        elif value.strip() == '[]':
            type_hint = "list[Any]"  # Default list type
        else:
            # Try to infer from hint
            type_hint = type_name

        fixed_line = f"{indent}{var_name}: {type_hint} = {value}"

        if self.dry_run:
            print(f"  Would fix {error.file}:{error.line}")
            print(f"    - {line}")
            print(f"    + {fixed_line}")
            return True

        # Apply fix
        lines[error.line - 1] = fixed_line

        # Check if Any is imported
        needs_any_import = 'Any' in fixed_line
        if needs_any_import:
            has_any = False
            for l in lines:
                if 'from typing import' in l and 'Any' in l:
                    has_any = True
                    break

            if not has_any:
                # Add Any to imports
                for i, l in enumerate(lines):
                    if 'from typing import' in l:
                        lines[i] = l.rstrip() + ', Any' if not l.strip().endswith(')') else l.replace(')', ', Any)')
                        break
                else:
                    # Add new import
                    lines.insert(0, 'from typing import Any')

        # Write back
        file_path.write_text('\n'.join(lines) + '\n')
        self.fixes_applied += 1
        self.files_modified.add(error.file)
        print(f"  ✅ Fixed {error.file}:{error.line} (missing annotation)")
        return True

    def fix_errors(self, errors: List[MypyError], category: Optional[str] = None) -> int:
        """
        Fix all fixable errors.

        Args:
            errors: List of mypy errors
            category: Optional category filter

        Returns:
            Number of errors fixed
        """
        self.fixes_applied = 0
        self.files_modified.clear()

        for error in errors:
            # Filter by category if specified
            if category and error.code != category:
                continue

            # Try different fix strategies
            if self.fix_none_assignment(error):
                continue
            if self.fix_missing_annotation(error):
                continue

        return self.fixes_applied


def run_mypy() -> Tuple[int, str]:
    """Run mypy and return (exit_code, output)."""
    try:
        # Run from parent directory so Jotty.* imports work
        parent_dir = Path.cwd().parent if Path.cwd().name == 'Jotty' else Path.cwd()

        result = subprocess.run(
            [sys.executable, '-m', 'mypy', 'Jotty/core', 'Jotty/apps', 'Jotty/sdk',
             '--config-file', 'Jotty/mypy.ini'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=parent_dir
        )
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        return 1, f"Error running mypy: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fix common mypy type errors"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be fixed without making changes")
    parser.add_argument('--category', type=str,
                       help="Only fix specific error category (e.g., 'assignment', 'var-annotated')")
    args = parser.parse_args()

    print("🔍 Running mypy to detect type errors...")
    exit_code, output = run_mypy()

    if exit_code == 0:
        print("✅ No mypy errors detected!")
        return 0

    print("\n📋 Parsing mypy errors...")
    errors = MypyErrorParser.parse(output)

    if not errors:
        print("⚠️  Mypy failed but no errors were parsed.")
        print("    Check mypy output for syntax errors or configuration issues.")
        return 1

    # Categorize errors
    errors_by_code: Dict[str, List[MypyError]] = {}
    for error in errors:
        if error.code not in errors_by_code:
            errors_by_code[error.code] = []
        errors_by_code[error.code].append(error)

    print(f"\n📊 Found {len(errors)} error(s) across {len(errors_by_code)} categories:")
    for code, errs in sorted(errors_by_code.items()):
        print(f"  • {code}: {len(errs)} error(s)")

    # Auto-fixable categories
    fixable_categories = {'assignment', 'var-annotated'}
    fixable_errors = [e for e in errors if e.code in fixable_categories]

    if args.category:
        fixable_errors = [e for e in fixable_errors if e.code == args.category]

    if not fixable_errors:
        print("\n⚠️  No auto-fixable errors found.")
        print("    Auto-fix supports: assignment, var-annotated")
        print("    Other errors require manual intervention.")
        return 1

    print(f"\n🔧 Found {len(fixable_errors)} auto-fixable error(s)")

    # Apply fixes
    fixer = MypyErrorFixer(dry_run=args.dry_run)
    fixed_count = fixer.fix_errors(fixable_errors, args.category)

    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would fix {fixed_count} error(s) in {len(fixer.files_modified)} file(s)")
        print("Run without --dry-run to apply changes")
        return 0

    if fixed_count > 0:
        print(f"\n✅ Fixed {fixed_count} error(s) in {len(fixer.files_modified)} file(s)")

        # Re-run to verify
        print("\n🔍 Re-running mypy to verify fixes...")
        exit_code, output = run_mypy()

        remaining_errors = MypyErrorParser.parse(output)
        if exit_code == 0:
            print("✅ All errors fixed! Mypy now passes.")
            return 0
        else:
            remaining_fixable = [e for e in remaining_errors if e.code in fixable_categories]
            print(f"⚠️  {len(remaining_errors)} error(s) remain ({len(remaining_fixable)} auto-fixable)")
            if remaining_fixable:
                print("    Run script again to fix remaining auto-fixable errors")
            return 1
    else:
        print("\n⚠️  Could not auto-fix any errors")
        print("    Manual intervention required")
        return 1


if __name__ == '__main__':
    sys.exit(main())
