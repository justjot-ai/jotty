#!/usr/bin/env python3
"""
Auto-discover and auto-fix mypy type stub issues.

This script:
1. Runs mypy to detect missing stub packages
2. Parses the output to extract stub package names
3. Automatically installs missing stubs
4. Re-runs to verify the fix

Usage:
    python scripts/autofix_mypy.py [--dry-run]

Options:
    --dry-run    Show what would be installed without making changes

Exit code: 0 if all stubs installed; 1 if manual intervention needed
"""

import argparse
import re
import subprocess
import sys
from typing import List, Set, Tuple


class MypyStubParser:
    """Parse mypy output to extract missing stub package names."""

    # Regex to match stub hints like:
    # note: Hint: "python3 -m pip install types-PyYAML"
    STUB_HINT_PATTERN = re.compile(
        r'note: Hint: "python3 -m pip install (?P<package>types-[\w-]+)"'
    )

    @classmethod
    def parse(cls, output: str) -> Set[str]:
        """
        Parse mypy output and return set of missing stub packages.

        Returns:
            Set of package names like {'types-PyYAML', 'types-Markdown'}
        """
        missing_stubs = set()

        for line in output.split('\n'):
            match = cls.STUB_HINT_PATTERN.search(line)
            if match:
                missing_stubs.add(match.group('package'))

        return missing_stubs


def run_mypy() -> Tuple[int, str]:
    """
    Run mypy and return (exit_code, output).
    Configuration is read from mypy.ini
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'mypy'],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "Error: mypy timed out after 60 seconds"
    except Exception as e:
        return 1, f"Error running mypy: {e}"


def install_stubs(packages: List[str], dry_run: bool = False) -> bool:
    """
    Install stub packages using pip.

    Returns True if installation succeeded, False otherwise.
    """
    if not packages:
        return True

    if dry_run:
        print(f"\n🔍 DRY RUN - Would install: {' '.join(packages)}")
        return True

    print(f"\n📦 Installing {len(packages)} stub package(s)...")
    for package in packages:
        print(f"   • {package}")

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install'] + packages,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("✅ Successfully installed all stub packages")
            return True
        else:
            print(f"❌ Installation failed:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Installation timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover and auto-fix mypy type stub issues"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be installed without making changes")
    args = parser.parse_args()

    print("🔍 Running mypy to detect missing stubs...")
    exit_code, output = run_mypy()

    if exit_code == 0:
        print("✅ No mypy errors detected!")
        return 0

    print("\n📋 Parsing mypy output for missing stubs...")
    missing_stubs = MypyStubParser.parse(output)

    if not missing_stubs:
        print("⚠️  Mypy failed but no missing stubs were detected.")
        print("    There might be actual type errors requiring manual fixes.")
        print("\n" + "="*80)
        print("MYPY OUTPUT (first 50 lines):")
        print("="*80)
        print('\n'.join(output.split('\n')[:50]))
        print("="*80)
        print("\n💡 Manual intervention required - check output above")
        return 1

    # Display found stubs
    print(f"\n📊 Found {len(missing_stubs)} missing stub package(s):")
    for stub in sorted(missing_stubs):
        print(f"  • {stub}")

    if args.dry_run:
        print("\nRun without --dry-run to install these packages")
        return 0

    # Install stubs
    if install_stubs(sorted(missing_stubs), dry_run=args.dry_run):
        # Re-run to verify
        print("\n🔍 Re-running mypy to verify fix...")
        exit_code, output = run_mypy()

        if exit_code == 0:
            print("✅ All stub issues fixed! Mypy now passes.")
            return 0
        else:
            # Check if there are still missing stubs
            remaining_stubs = MypyStubParser.parse(output)
            if remaining_stubs:
                print(f"⚠️  Still missing {len(remaining_stubs)} stub(s):")
                for stub in sorted(remaining_stubs):
                    print(f"  • {stub}")
                print("\nRun script again to install remaining stubs")
                return 1
            else:
                print("⚠️  Stubs installed but mypy still has errors.")
                print("    These are likely actual type errors requiring manual fixes.")
                print("\n" + "="*80)
                print("REMAINING MYPY ERRORS (first 30 lines):")
                print("="*80)
                print('\n'.join(output.split('\n')[:30]))
                print("="*80)
                return 1
    else:
        print("\n❌ Failed to install stub packages")
        return 1


if __name__ == '__main__':
    sys.exit(main())
