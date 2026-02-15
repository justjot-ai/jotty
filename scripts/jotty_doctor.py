#!/usr/bin/env python3
"""
Jotty Doctor - Automated Health Check Tool
==========================================

Scans the codebase for common issues and suggests fixes.

Usage:
    python scripts/jotty_doctor.py              # Full scan
    python scripts/jotty_doctor.py --fix        # Auto-fix issues
    python scripts/jotty_doctor.py --imports    # Check imports only
"""

import ast
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class Severity(Enum):
    CRITICAL = "🔴 CRITICAL"
    HIGH = "🟠 HIGH"
    MEDIUM = "🟡 MEDIUM"
    LOW = "🟢 LOW"
    INFO = "ℹ️ INFO"


@dataclass
class Issue:
    severity: Severity
    category: str
    file: Path
    line: int
    message: str
    fix: str = ""


class JottyDoctor:
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.issues: List[Issue] = []

    def check_imports(self) -> None:
        """Check for import inconsistencies."""
        print("🔍 Checking imports...")

        # Check for wildcard imports (except intentional re-exports)
        for py_file in self.root_dir.rglob("core/**/*.py"):
            try:
                content = py_file.read_text()

                # Check for wildcard imports
                if "import *" in content and "skill_sdk" not in str(py_file):
                    for i, line in enumerate(content.split("\n"), 1):
                        if "import *" in line and "# noqa" not in line:
                            self.issues.append(
                                Issue(
                                    severity=Severity.MEDIUM,
                                    category="Import",
                                    file=py_file,
                                    line=i,
                                    message="Wildcard import makes dependencies unclear",
                                    fix="Use explicit imports: from module import specific_name",
                                )
                            )
            except (OSError, UnicodeDecodeError):
                # Skip files that can't be read or decoded
                pass

    def check_secrets(self) -> None:
        """Check for hardcoded secrets."""
        print("🔍 Checking for hardcoded secrets...")

        patterns = [
            (r'["\']([0-9]{10,}:[A-Za-z0-9_-]{35})["\']', "Telegram Bot Token"),
            (r"sk-[A-Za-z0-9]{48}", "OpenAI API Key"),
            (r"xoxb-[0-9]{11,12}-[0-9]{11,12}-[A-Za-z0-9]{24}", "Slack Bot Token"),
            (r'(password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded Password"),
        ]

        for py_file in self.root_dir.rglob("core/**/*.py"):
            try:
                content = py_file.read_text()

                for pattern, secret_type in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip if using environment variables
                        start = max(0, match.start() - 100)
                        end = min(len(content), match.end() + 100)
                        context = content[start:end]

                        if "getenv" not in context and "environ" not in context:
                            line_num = content[: match.start()].count("\n") + 1
                            self.issues.append(
                                Issue(
                                    severity=Severity.CRITICAL,
                                    category="Security",
                                    file=py_file,
                                    line=line_num,
                                    message=f"Hardcoded {secret_type} detected",
                                    fix=f"Use: os.getenv('{secret_type.upper().replace(' ', '_')}')",
                                )
                            )
            except (OSError, UnicodeDecodeError):
                # Skip files that can't be read or decoded
                pass

    def check_type_hints(self, sample_size: int = 50) -> None:
        """Check type hint coverage (sample to avoid slowness)."""
        print(f"🔍 Checking type hints (sampling {sample_size} files)...")

        files = list(self.root_dir.rglob("core/**/*.py"))
        sample = files[:sample_size]

        no_hints = 0
        total_funcs = 0

        for py_file in sample:
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read(), str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_funcs += 1
                        if not node.returns and node.name not in [
                            "__init__",
                            "__str__",
                            "__repr__",
                            "__eq__",
                        ]:
                            no_hints += 1
            except (OSError, UnicodeDecodeError, SyntaxError):
                # Skip files with parse errors
                pass

        if total_funcs > 0:
            coverage = ((total_funcs - no_hints) / total_funcs) * 100
            if coverage < 50:
                severity = Severity.HIGH
            elif coverage < 80:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            self.issues.append(
                Issue(
                    severity=severity,
                    category="Type Hints",
                    file=self.root_dir / "core",
                    line=0,
                    message=f"Type hint coverage: {coverage:.1f}% ({no_hints}/{total_funcs} functions missing)",
                    fix="Run: mypy --strict and add type hints to all functions",
                )
            )

    def check_exception_handling(self, sample_size: int = 50) -> None:
        """Check for broad exception handling."""
        print(f"🔍 Checking exception handling (sampling {sample_size} files)...")

        files = list(self.root_dir.rglob("core/**/*.py"))
        sample = files[:sample_size]

        for py_file in sample:
            try:
                with open(py_file) as f:
                    content = f.read()
                    tree = ast.parse(content, str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler):
                        if node.type is None:
                            line_num = node.lineno
                            self.issues.append(
                                Issue(
                                    severity=Severity.MEDIUM,
                                    category="Exception",
                                    file=py_file,
                                    line=line_num,
                                    message="Bare except clause catches all exceptions",
                                    fix="Use specific exception: except ValueError: or except (KeyError, TypeError):",
                                )
                            )
                        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                            # Check if there's a good reason (re-raise, log, etc.)
                            has_logging = any(
                                isinstance(child, ast.Call)
                                and isinstance(child.func, ast.Attribute)
                                and child.func.attr in ["error", "exception", "warning"]
                                for child in ast.walk(node)
                            )

                            if not has_logging:
                                self.issues.append(
                                    Issue(
                                        severity=Severity.LOW,
                                        category="Exception",
                                        file=py_file,
                                        line=node.lineno,
                                        message="Broad 'except Exception' without logging",
                                        fix="Add logging or use specific exception type",
                                    )
                                )
            except (OSError, UnicodeDecodeError, SyntaxError):
                # Skip files with parse errors
                pass

    def check_todos(self) -> None:
        """Check TODO/FIXME markers."""
        print("🔍 Checking TODO/FIXME markers...")

        patterns = [
            (r"#\s*TODO", "TODO"),
            (r"#\s*FIXME", "FIXME"),
            (r"#\s*XXX", "XXX"),
            (r"#\s*HACK", "HACK"),
        ]

        todo_count = 0

        for py_file in self.root_dir.rglob("core/**/*.py"):
            try:
                content = py_file.read_text()

                for pattern, marker_type in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        todo_count += 1
            except (OSError, UnicodeDecodeError):
                # Skip files that can't be read
                pass

        if todo_count > 50:
            severity = Severity.MEDIUM
        elif todo_count > 20:
            severity = Severity.LOW
        else:
            severity = Severity.INFO

        if todo_count > 0:
            self.issues.append(
                Issue(
                    severity=severity,
                    category="Maintenance",
                    file=self.root_dir / "core",
                    line=0,
                    message=f"{todo_count} TODO/FIXME/XXX/HACK markers found",
                    fix="Triage: convert to GitHub issues or fix immediately",
                )
            )

    def run_all_checks(self) -> None:
        """Run all health checks."""
        print("🏥 Jotty Doctor - Running Health Checks...\n")

        self.check_imports()
        self.check_secrets()
        self.check_type_hints(sample_size=50)
        self.check_exception_handling(sample_size=50)
        self.check_todos()

    def report(self) -> None:
        """Print health check report."""
        print("\n" + "=" * 80)
        print("📋 HEALTH CHECK REPORT")
        print("=" * 80 + "\n")

        if not self.issues:
            print("✅ No issues found! Jotty is healthy.")
            return

        # Group by severity
        by_severity = {}
        for issue in self.issues:
            severity = issue.severity
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)

        # Print summary
        print(f"Total Issues: {len(self.issues)}\n")
        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]:
            if severity in by_severity:
                count = len(by_severity[severity])
                print(f"{severity.value}: {count}")

        print("\n" + "=" * 80 + "\n")

        # Print details
        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]:
            if severity not in by_severity:
                continue

            print(f"\n{severity.value} ISSUES:")
            print("-" * 80)

            for issue in by_severity[severity]:
                print(f"\n[{issue.category}] {issue.file}:{issue.line}")
                print(f"  Issue: {issue.message}")
                if issue.fix:
                    print(f"  Fix: {issue.fix}")

        print("\n" + "=" * 80)

        # Exit code
        if any(i.severity in [Severity.CRITICAL, Severity.HIGH] for i in self.issues):
            print("\n⚠️  Critical or High severity issues found. Please address before production.")
            sys.exit(1)
        else:
            print("\n✅ No critical issues. Review medium/low priority items.")
            sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Jotty Doctor - Automated Health Check")
    parser.add_argument("--imports", action="store_true", help="Check imports only")
    parser.add_argument("--secrets", action="store_true", help="Check secrets only")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues (not implemented yet)")

    args = parser.parse_args()

    doctor = JottyDoctor()

    if args.imports:
        doctor.check_imports()
    elif args.secrets:
        doctor.check_secrets()
    else:
        doctor.run_all_checks()

    doctor.report()


if __name__ == "__main__":
    main()
