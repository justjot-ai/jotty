#!/usr/bin/env python3
"""
Run import-linter, mypy, and optionally python-code-analyzer in one go.
Use this to detect the reason for failures: each tool's output explains its own issues.

Usage:
    pip install -e ".[dev]"   # ensures import-linter, mypy, python-code-analyzer
    python scripts/lint_all.py [--no-analyzer] [--no-mypy] [--no-imports]

Exit code: 0 if all enabled checks pass; 1 if any fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], label: str) -> tuple[int, str]:
    """Run command; return (exit_code, combined stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).resolve().parent.parent,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return result.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 1, f"{label}: timed out after 300s"
    except FileNotFoundError:
        return 1, f"{label}: command not found (install with: pip install -e \".[dev]\")"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run import-linter, mypy, and optional code analyzer")
    parser.add_argument("--no-analyzer", action="store_true", help="Skip python-code-analyzer (self-test)")
    parser.add_argument("--no-mypy", action="store_true", help="Skip mypy")
    parser.add_argument("--no-imports", action="store_true", help="Skip import-linter")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    failures: list[str] = []

    # 1. Import linter (detects architecture / forbidden import violations)
    if not args.no_imports:
        code, out = run(["lint-imports"], "import-linter")
        print("=== import-linter (lint_imports) ===")
        print(out or "(no output)")
        print()
        if code != 0:
            failures.append("import-linter")

    # 2. Mypy (type checking)
    if not args.no_mypy:
        code, out = run(
            [
                sys.executable, "-m", "mypy",
                "core", "apps", "sdk",
                "--no-error-summary",
                "--explicit-package-bases",
            ],
            "mypy",
        )
        print("=== mypy ===")
        print(out or "(no output)")
        print()
        if code != 0:
            failures.append("mypy")

    # 3. Python code analyzer (runtime trace; self-test only to avoid zipimport crash)
    if not args.no_analyzer:
        script = repo / "scripts" / "analyze_jotty_runtime.py"
        code, out = run(
            [sys.executable, str(script), "--scenario", "self-test"],
            "python-code-analyzer",
        )
        print("=== python-code-analyzer (self-test) ===")
        # Only show last few lines unless failed
        lines = out.splitlines()
        if code != 0:
            print(out)
            failures.append("python-code-analyzer")
        else:
            print("\n".join(lines[-5:]) if len(lines) > 5 else out)
        print()

    if failures:
        print("Failed:", ", ".join(failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
