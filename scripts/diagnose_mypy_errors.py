#!/usr/bin/env python3
"""
Diagnose mypy errors to find root causes and patterns.

This script:
1. Runs mypy and collects all errors
2. Categorizes errors by type and pattern
3. Analyzes common patterns (import paths, missing modules, etc.)
4. Suggests fixes for the most common issues

Usage:
    python scripts/diagnose_mypy_errors.py [--category CATEGORY] [--top N]
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Tuple


@dataclass
class MypyError:
    """Represents a single mypy error."""
    file: str
    line: int
    column: int
    message: str
    code: str

    @property
    def module_path(self) -> str:
        """Extract module path from message if it's an import/attr error."""
        # Match patterns like: Cannot find ... "Jotty.core.infrastructure.foundation.data_structures"
        match = re.search(r'"([Jotty\.\w.]+)"', self.message)
        if match:
            return match.group(1)

        # Match patterns like: Module "Jotty.core.xyz" has no attribute "foo"
        match = re.search(r'Module "([^"]+)" has no attribute', self.message)
        if match:
            return match.group(1)

        return ""

    @property
    def missing_attribute(self) -> str:
        """Extract missing attribute name from attr-defined errors."""
        # Match: Module "X" has no attribute "Y"
        match = re.search(r'has no attribute "([^"]+)"', self.message)
        if match:
            return match.group(1)
        return ""


class MypyDiagnostics:
    """Analyze mypy errors to find root causes."""

    def __init__(self):
        self.errors: List[MypyError] = []
        self.errors_by_code: Dict[str, List[MypyError]] = defaultdict(list)

    def run_mypy(self) -> Tuple[int, str]:
        """Run mypy and return (exit_code, output)."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'mypy'],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout + result.stderr
        except Exception as e:
            return 1, f"Error running mypy: {e}"

    def parse_errors(self, output: str) -> List[MypyError]:
        """Parse mypy output and return list of errors."""
        errors = []

        # Pattern: file.py:line:column: error: message [code]
        pattern = re.compile(
            r'^(?P<file>[\w/._-]+\.py):(?P<line>\d+):(?P<column>\d+):\s+'
            r'error:\s+(?P<message>.+?)\s+\[(?P<code>[\w-]+)\]',
            re.MULTILINE
        )

        for match in pattern.finditer(output):
            error = MypyError(
                file=match.group('file'),
                line=int(match.group('line')),
                column=int(match.group('column')),
                message=match.group('message'),
                code=match.group('code')
            )
            errors.append(error)
            self.errors_by_code[error.code].append(error)

        return errors

    def analyze_attr_defined_errors(self) -> Dict[str, any]:
        """Deep analysis of attr-defined errors to find patterns."""
        attr_errors = self.errors_by_code.get('attr-defined', [])

        if not attr_errors:
            return {}

        # Pattern 1: Missing modules
        missing_modules = Counter()
        # Pattern 2: Missing attributes per module
        missing_attrs = defaultdict(Counter)
        # Pattern 3: Files with most errors
        files_with_errors = Counter()
        # Pattern 4: Import patterns
        import_patterns = Counter()

        for error in attr_errors:
            files_with_errors[error.file] += 1

            module = error.module_path
            attr = error.missing_attribute

            if module:
                missing_modules[module] += 1
                if attr:
                    missing_attrs[module][attr] += 1

                # Analyze import pattern
                if module.startswith('Jotty.'):
                    import_patterns['Jotty.* imports'] += 1
                elif '.' in module:
                    import_patterns['Relative imports'] += 1
                else:
                    import_patterns['Simple imports'] += 1

        return {
            'total': len(attr_errors),
            'missing_modules': missing_modules.most_common(20),
            'missing_attrs': dict(missing_attrs),
            'top_files': files_with_errors.most_common(10),
            'import_patterns': import_patterns.most_common(),
        }

    def analyze_import_errors(self) -> Dict[str, any]:
        """Analyze import-not-found and import errors."""
        import_errors = (
            self.errors_by_code.get('import-not-found', []) +
            self.errors_by_code.get('import', [])
        )

        if not import_errors:
            return {}

        # Extract missing module names
        missing_modules = Counter()
        jotty_imports = []
        external_imports = []

        for error in import_errors:
            module = error.module_path
            if module:
                missing_modules[module] += 1
                if module.startswith('Jotty.'):
                    jotty_imports.append(module)
                else:
                    external_imports.append(module)

        return {
            'total': len(import_errors),
            'missing_modules': missing_modules.most_common(20),
            'jotty_imports': list(set(jotty_imports)),
            'external_imports': list(set(external_imports)),
        }

    def check_module_exists(self, module_path: str) -> bool:
        """Check if a module actually exists in the codebase."""
        # Convert Jotty.core.xyz to Jotty/core/xyz.py
        parts = module_path.split('.')

        # Try as .py file
        file_path = Path('/'.join(parts) + '.py')
        if file_path.exists():
            return True

        # Try as package (__init__.py)
        package_path = Path('/'.join(parts) + '/__init__.py')
        if package_path.exists():
            return True

        return False

    def suggest_fixes(self, analysis: Dict[str, any]) -> List[str]:
        """Suggest fixes based on analysis."""
        suggestions = []

        # Analyze attr-defined errors
        if 'attr-defined' in self.errors_by_code:
            attr_analysis = analysis.get('attr_defined', {})
            missing_mods = attr_analysis.get('missing_modules', [])

            if missing_mods:
                top_module, count = missing_mods[0]

                # Check if module exists
                if self.check_module_exists(top_module):
                    suggestions.append(
                        f"🔍 Top missing module '{top_module}' ({count} errors) EXISTS in codebase!\n"
                        f"   → This suggests mypy can't find it due to PYTHONPATH or mypy.ini configuration\n"
                        f"   → Fix: Update mypy.ini with correct mypy_path or namespace_packages settings"
                    )
                else:
                    suggestions.append(
                        f"❌ Top missing module '{top_module}' ({count} errors) DOES NOT EXIST\n"
                        f"   → This might be from refactoring or old imports\n"
                        f"   → Fix: Update imports or create the module"
                    )

        # Analyze import errors
        if 'import-not-found' in self.errors_by_code or 'import' in self.errors_by_code:
            import_analysis = analysis.get('import', {})
            jotty_imports = import_analysis.get('jotty_imports', [])

            if jotty_imports:
                suggestions.append(
                    f"🔍 Found {len(jotty_imports)} missing Jotty.* imports\n"
                    f"   → Examples: {', '.join(jotty_imports[:3])}\n"
                    f"   → Fix: Mypy can't resolve absolute Jotty.* imports\n"
                    f"   → Update mypy.ini: set mypy_path = . or use namespace_packages = True"
                )

        # Check for systematic issues
        total_errors = len(self.errors)
        attr_errors = len(self.errors_by_code.get('attr-defined', []))

        if attr_errors > total_errors * 0.8:
            suggestions.append(
                f"⚠️  {attr_errors}/{total_errors} ({attr_errors*100//total_errors}%) errors are attr-defined!\n"
                f"   → This is likely a SYSTEMATIC configuration issue, not individual bugs\n"
                f"   → Root cause: Mypy can't find modules/attributes that exist\n"
                f"   → Fix: Update mypy.ini configuration (paths, namespace packages)"
            )

        return suggestions

    def print_report(self, top_n: int = 20, category: str = None):
        """Print diagnostic report."""
        print("="*80)
        print("MYPY ERROR DIAGNOSTICS")
        print("="*80)

        print(f"\n📊 SUMMARY")
        print(f"   Total errors: {len(self.errors)}")
        print(f"   Categories: {len(self.errors_by_code)}")
        print()

        # Category breakdown
        print("📋 ERRORS BY CATEGORY:")
        for code, errors in sorted(self.errors_by_code.items(), key=lambda x: -len(x[1])):
            if category and code != category:
                continue
            pct = len(errors) * 100 // len(self.errors)
            print(f"   • {code:20s}: {len(errors):4d} ({pct:2d}%)")
        print()

        # Detailed analysis for attr-defined
        if not category or category == 'attr-defined':
            print("="*80)
            print("🔍 DETAILED ANALYSIS: attr-defined errors")
            print("="*80)

            analysis = self.analyze_attr_defined_errors()
            if analysis:
                print(f"\n📊 Total attr-defined errors: {analysis['total']}")

                print(f"\n🔝 TOP {top_n} MISSING MODULES:")
                for module, count in analysis['missing_modules'][:top_n]:
                    exists = "✅ EXISTS" if self.check_module_exists(module) else "❌ MISSING"
                    print(f"   • {module:60s}: {count:3d} errors  {exists}")

                print(f"\n📁 TOP 10 FILES WITH MOST ERRORS:")
                for file, count in analysis['top_files']:
                    print(f"   • {file:60s}: {count:3d} errors")

                print(f"\n📦 IMPORT PATTERNS:")
                for pattern, count in analysis['import_patterns']:
                    pct = count * 100 // analysis['total']
                    print(f"   • {pattern:30s}: {count:3d} ({pct:2d}%)")

        # Detailed analysis for import errors
        if not category or category in ['import-not-found', 'import']:
            print("\n" + "="*80)
            print("🔍 DETAILED ANALYSIS: import errors")
            print("="*80)

            analysis = self.analyze_import_errors()
            if analysis:
                print(f"\n📊 Total import errors: {analysis['total']}")

                print(f"\n🔝 TOP {top_n} MISSING MODULES:")
                for module, count in analysis['missing_modules'][:top_n]:
                    exists = "✅ EXISTS" if self.check_module_exists(module) else "❌ MISSING"
                    print(f"   • {module:60s}: {count:3d} errors  {exists}")

                if analysis['jotty_imports']:
                    print(f"\n📦 JOTTY.* IMPORTS ({len(analysis['jotty_imports'])}):")
                    for module in analysis['jotty_imports'][:10]:
                        print(f"   • {module}")

                if analysis['external_imports']:
                    print(f"\n📦 EXTERNAL IMPORTS ({len(analysis['external_imports'])}):")
                    for module in analysis['external_imports'][:10]:
                        print(f"   • {module}")

        # Suggestions
        print("\n" + "="*80)
        print("💡 SUGGESTED FIXES")
        print("="*80)

        all_analysis = {
            'attr_defined': self.analyze_attr_defined_errors(),
            'import': self.analyze_import_errors(),
        }
        suggestions = self.suggest_fixes(all_analysis)

        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"\n{i}. {suggestion}")
        else:
            print("\n   No automated suggestions available.")
            print("   Manual review of errors needed.")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Diagnose mypy errors")
    parser.add_argument('--category', help="Focus on specific category")
    parser.add_argument('--top', type=int, default=20, help="Show top N results")
    args = parser.parse_args()

    print("🔍 Running mypy to collect errors...")

    diagnostics = MypyDiagnostics()
    exit_code, output = diagnostics.run_mypy()

    if exit_code == 0:
        print("✅ No mypy errors found!")
        return 0

    print("📋 Parsing mypy output...")
    diagnostics.errors = diagnostics.parse_errors(output)

    if not diagnostics.errors:
        print("⚠️  Mypy failed but no errors were parsed.")
        print(output[:500])
        return 1

    print(f"✅ Parsed {len(diagnostics.errors)} errors\n")

    diagnostics.print_report(top_n=args.top, category=args.category)

    return 0


if __name__ == '__main__':
    sys.exit(main())
