#!/usr/bin/env python3
"""
Auto-discover and auto-fix import-linter violations.

This script:
1. Runs import-linter to detect violations
2. Parses the output to extract forbidden imports
3. Automatically adds them to .importlinter as exceptions (with TODO comments)
4. Re-runs to verify the fix

Usage:
    python scripts/autofix_imports.py [--dry-run] [--no-todo]

Options:
    --dry-run    Show what would be fixed without making changes
    --no-todo    Don't add TODO comments to auto-fixed imports

Exit code: 0 if all violations fixed; 1 if manual intervention needed
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ImportViolation:
    """Represents a single import violation detected by import-linter."""

    def __init__(self, source: str, target: str, location: str, contract: str):
        self.source = source  # e.g., "Jotty.apps.cli.commands.telegram_bot"
        self.target = target  # e.g., "Jotty.apps.telegram_bot.bot"
        self.location = location  # e.g., "l.59"
        self.contract = contract  # e.g., "CLI must not import from other apps"

    def __repr__(self):
        return f"ImportViolation({self.source} -> {self.target} @ {self.location})"

    def to_ignore_line(self, add_todo: bool = True) -> str:
        """Convert to an ignore_imports line for .importlinter config."""
        comment = "# Auto-fixed: TODO - review if this should be allowed"
        if not add_todo:
            comment = "# Auto-discovered import"
        return f"    {self.source} -> {self.target}  {comment}"


class ImportLinterParser:
    """Parse import-linter output to extract violations."""

    # Regex to match violation lines like:
    # -   Jotty.apps.cli.commands.telegram_bot -> Jotty.apps.telegram_bot.bot (l.59)
    VIOLATION_PATTERN = re.compile(
        r"-\s+(?P<source>[\w.]+)\s+->\s+(?P<target>[\w.]+)\s+\((?P<location>l\.\d+)\)"
    )

    # Regex to match contract headers like:
    # CLI must not import from other apps
    # -----------------------------------
    CONTRACT_PATTERN = re.compile(r"^(?P<name>.+?)\s*$\n-+$", re.MULTILINE)

    @classmethod
    def parse(cls, output: str) -> Dict[str, List[ImportViolation]]:
        """
        Parse import-linter output and return violations grouped by contract.

        Returns:
            Dict mapping contract name to list of violations
        """
        violations_by_contract = {}

        # Split output into sections
        lines = output.split("\n")
        current_contract = None

        for i, line in enumerate(lines):
            # Check if this is a contract header
            if i + 1 < len(lines) and lines[i + 1].startswith("---"):
                current_contract = line.strip()
                violations_by_contract[current_contract] = []
                continue

            # Check if this is a violation line
            if current_contract:
                match = cls.VIOLATION_PATTERN.match(line)
                if match:
                    violation = ImportViolation(
                        source=match.group("source"),
                        target=match.group("target"),
                        location=match.group("location"),
                        contract=current_contract,
                    )
                    violations_by_contract[current_contract].append(violation)

        return violations_by_contract


class ImportLinterFixer:
    """Auto-fix import-linter violations by updating .importlinter config."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config_content = config_path.read_text()

    def add_ignore(
        self, contract_name: str, violations: List[ImportViolation], add_todo: bool = True
    ) -> bool:
        """
        Add violations to ignore_imports for a specific contract.

        Returns True if config was modified, False otherwise.
        """
        if not violations:
            return False

        # Find the contract section
        contract_section_pattern = re.compile(
            rf"\[importlinter:contract:[^\]]+\]\s*\n" rf"name\s*=\s*{re.escape(contract_name)}",
            re.MULTILINE,
        )

        match = contract_section_pattern.search(self.config_content)
        if not match:
            print(f"⚠️  Could not find contract '{contract_name}' in config")
            return False

        # Find where to insert ignore_imports
        contract_start = match.start()

        # Look for existing ignore_imports section
        next_section = self.config_content.find("[importlinter:", contract_start + 1)
        if next_section == -1:
            next_section = len(self.config_content)

        contract_section = self.config_content[contract_start:next_section]

        # Check if ignore_imports already exists
        if "ignore_imports" in contract_section:
            # Find the end of ignore_imports section
            ignore_start = self.config_content.find("ignore_imports", contract_start)
            ignore_end = ignore_start

            # Find the next section or end of contract
            for line in self.config_content[ignore_start:next_section].split("\n"):
                if (
                    line
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and line != "ignore_imports"
                ):
                    break
                ignore_end = self.config_content.find(line, ignore_end) + len(line)

            # Insert before the next section
            insert_pos = ignore_end
            prefix = "\n"
        else:
            # Add ignore_imports section before next section
            # Find the last line of the contract before the next section
            insert_pos = next_section
            if insert_pos > contract_start:
                # Find last non-empty line
                prev_newline = self.config_content.rfind("\n", contract_start, insert_pos - 1)
                insert_pos = prev_newline + 1

            prefix = "ignore_imports =\n"

        # Generate ignore lines
        ignore_lines = [violation.to_ignore_line(add_todo) for violation in violations]
        insert_text = prefix + "\n".join(ignore_lines) + "\n"

        # Insert into config
        self.config_content = (
            self.config_content[:insert_pos] + insert_text + self.config_content[insert_pos:]
        )

        return True

    def save(self) -> None:
        """Save modified config back to file."""
        self.config_path.write_text(self.config_content)


def run_import_linter(verbose: bool = False) -> Tuple[int, str]:
    """
    Run import-linter and return (exit_code, output).
    """
    try:
        result = subprocess.run(
            ["lint-imports"] + (["--verbose"] if verbose else []),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "Error: lint-imports timed out after 30 seconds"
    except FileNotFoundError:
        return 1, "Error: lint-imports not found. Run: pip install import-linter"


def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover and auto-fix import-linter violations"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fixed without making changes"
    )
    parser.add_argument(
        "--no-todo", action="store_true", help="Don't add TODO comments to auto-fixed imports"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output from import-linter")
    args = parser.parse_args()

    # Find .importlinter config
    config_path = Path(".importlinter")
    if not config_path.exists():
        print("❌ .importlinter not found in current directory")
        print("   Run from repository root: cd /path/to/Jotty")
        return 1

    print("🔍 Running import-linter to detect violations...")
    exit_code, output = run_import_linter(args.verbose)

    if exit_code == 0:
        print("✅ No violations detected! Import-linter passed.")
        return 0

    print("\n📋 Parsing violations...")
    violations_by_contract = ImportLinterParser.parse(output)

    if not violations_by_contract:
        print("⚠️  import-linter failed but no violations were parsed.")
        print(
            "    This might be a different type of error (e.g., 'Modules have shared descendants')"
        )
        print("\n" + "=" * 80)
        print("IMPORT-LINTER OUTPUT:")
        print("=" * 80)
        print(output)
        print("=" * 80)
        print("\n💡 Manual intervention required - check output above")
        return 1

    # Display found violations
    total_violations = sum(len(v) for v in violations_by_contract.values())
    print(
        f"\n📊 Found {total_violations} violation(s) across {len(violations_by_contract)} contract(s):\n"
    )

    for contract, violations in violations_by_contract.items():
        print(f"  📋 {contract}: {len(violations)} violation(s)")
        for v in violations:
            print(f"     • {v.source} -> {v.target} ({v.location})")

    if args.dry_run:
        print("\n🔍 DRY RUN - Would add these to .importlinter:")
        for contract, violations in violations_by_contract.items():
            print(f"\n  Contract: {contract}")
            for v in violations:
                print(f"  {v.to_ignore_line(not args.no_todo)}")
        print("\nRun without --dry-run to apply changes")
        return 0

    # Apply fixes
    print(f"\n🔧 Auto-fixing by adding to ignore_imports...")
    fixer = ImportLinterFixer(config_path)

    modified = False
    for contract, violations in violations_by_contract.items():
        if fixer.add_ignore(contract, violations, add_todo=not args.no_todo):
            print(f"  ✅ Added {len(violations)} ignore(s) to '{contract}'")
            modified = True
        else:
            print(f"  ⚠️  Could not auto-fix '{contract}'")

    if modified:
        fixer.save()
        print(f"\n💾 Saved changes to {config_path}")

        # Re-run to verify
        print("\n🔍 Re-running import-linter to verify fix...")
        exit_code, output = run_import_linter(args.verbose)

        if exit_code == 0:
            print("✅ All violations fixed! Import-linter now passes.")
            print("\n📝 NOTE: Review auto-fixed imports in .importlinter")
            print("         Some may need architectural refactoring instead of ignoring")
            return 0
        else:
            print("⚠️  Some violations remain after auto-fix:")
            print(output)
            return 1
    else:
        print("\n⚠️  No changes made - manual intervention required")
        return 1


if __name__ == "__main__":
    sys.exit(main())
