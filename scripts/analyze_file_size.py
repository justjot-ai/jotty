#!/usr/bin/env python3
"""
File Size Analyzer - Auto-detect files that need splitting

Analyzes Python files to find those with >1,000 lines of actual code
(excluding documentation), and suggests how to split them.

Usage:
    # Analyze all files
    python scripts/analyze_file_size.py

    # Analyze specific file
    python scripts/analyze_file_size.py path/to/file.py

    # Pre-commit hook mode (non-blocking)
    python scripts/analyze_file_size.py --hook

    # CI mode (fails if files >1,000 code lines found)
    python scripts/analyze_file_size.py --ci

Integration:
    Add to .pre-commit-config.yaml:

    - repo: local
      hooks:
        - id: file-size-check
          name: Check for oversized files
          entry: python scripts/analyze_file_size.py --hook
          language: system
          pass_filenames: false
          always_run: true
          verbose: true
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional


class FileSizeAnalyzer:
    """Analyzes Python files for size and complexity."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.oversized_files = []

    def count_code_vs_docs(self, file_path: Path) -> Optional[Dict[str, int]]:
        """Count actual code lines vs documentation."""
        try:
            content = file_path.read_text()
            lines = content.split("\n")

            in_docstring = False
            code_lines = 0
            doc_lines = 0
            blank_lines = 0
            comment_lines = 0

            for line in lines:
                stripped = line.strip()

                # Check for docstring markers
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    doc_lines += 1
                elif in_docstring:
                    doc_lines += 1
                elif not stripped:
                    blank_lines += 1
                elif stripped.startswith("#"):
                    comment_lines += 1
                else:
                    code_lines += 1

            return {
                "total": len(lines),
                "code": code_lines,
                "docs": doc_lines,
                "comments": comment_lines,
                "blank": blank_lines,
            }
        except Exception:
            return None

    def analyze_structure(self, file_path: Path) -> Dict[str, any]:
        """Analyze file structure to suggest splits."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            classes = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            start = item.lineno
                            end = item.end_lineno if hasattr(item, "end_lineno") else start + 10
                            methods.append(
                                {
                                    "name": item.name,
                                    "lines": end - start,
                                }
                            )

                    classes[node.name] = {
                        "methods": methods,
                        "total_methods": len(methods),
                        "total_lines": sum(m["lines"] for m in methods),
                    }

            return self._suggest_split(file_path, classes)
        except Exception:
            return {"suggestion": "Manual review needed"}

    def _suggest_split(self, file_path: Path, classes: Dict) -> Dict[str, any]:
        """Suggest how to split a file based on its structure."""
        suggestions = []

        # If multiple classes, split by class
        if len(classes) > 1:
            for class_name, info in classes.items():
                suggestions.append(
                    f"Extract {class_name} to {file_path.stem}_{class_name.lower()}.py "
                    f"({info['total_methods']} methods, ~{info['total_lines']} lines)"
                )
            return {"suggestion": "Split by class", "details": suggestions}

        # If single large class, analyze methods
        if len(classes) == 1:
            class_name, info = list(classes.items())[0]
            methods = info["methods"]

            # Group methods by prefix/responsibility
            groups = {}
            for method in methods:
                # Extract prefix (first word before underscore or camelCase)
                name = method["name"]
                if name.startswith("_"):
                    prefix = "private"
                elif "_" in name:
                    prefix = name.split("_")[0]
                else:
                    # CamelCase - take first lowercase part
                    prefix = "".join(c for c in name if c.islower())[:5]

                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(method)

            # Suggest splitting if we have clear groups
            if len(groups) > 2:
                for prefix, methods_list in sorted(
                    groups.items(), key=lambda x: len(x[1]), reverse=True
                ):
                    if len(methods_list) >= 3:
                        total = sum(m["lines"] for m in methods_list)
                        suggestions.append(
                            f"Extract '{prefix}' methods to {file_path.stem}_{prefix}.py "
                            f"({len(methods_list)} methods, ~{total} lines)"
                        )
                return {"suggestion": "Split by responsibility", "details": suggestions}

        return {
            "suggestion": "Manual review needed",
            "details": [
                "File is complex but doesn't have obvious split points.",
                "Consider extracting helper functions or refactoring for cohesion.",
            ],
        }

    def analyze_file(self, file_path: Path) -> Optional[Dict]:
        """Analyze a single file."""
        stats = self.count_code_vs_docs(file_path)
        if not stats:
            return None

        # Only flag if >1,000 lines of code
        if stats["code"] < 1000:
            return None

        rel_path = str(file_path.relative_to(self.base_path))
        structure = self.analyze_structure(file_path)

        return {
            "file": rel_path,
            "stats": stats,
            "structure": structure,
        }

    def analyze_all(self, exclude_tests: bool = True) -> List[Dict]:
        """Analyze all Python files in the codebase."""
        results = []

        for py_file in self.base_path.rglob("*.py"):
            if exclude_tests and "test" in str(py_file):
                continue

            result = self.analyze_file(py_file)
            if result:
                results.append(result)

        return sorted(results, key=lambda x: x["stats"]["code"], reverse=True)

    def report(self, results: List[Dict], mode: str = "full") -> int:
        """Print analysis report. Returns number of oversized files."""
        if not results:
            print("✓ No files exceed 1,000 lines of code (excluding documentation)")
            return 0

        print("=" * 100)
        print("⚠️  FILES WITH >1,000 LINES OF CODE (excluding documentation)")
        print("=" * 100)

        for i, result in enumerate(results, 1):
            file = result["file"]
            stats = result["stats"]
            structure = result["structure"]

            doc_pct = stats["docs"] / stats["total"] * 100

            print(f"\n{i}. {file}")
            print(
                f"   Total: {stats['total']:,} lines | Code: {stats['code']:,} lines | Docs: {stats['docs']:,} lines ({doc_pct:.0f}%)"
            )

            if mode == "full":
                print(f"\n   📋 Suggestion: {structure['suggestion']}")
                if "details" in structure and structure["details"]:
                    for detail in structure["details"][:3]:  # Limit to top 3
                        print(f"      • {detail}")

        print("\n" + "=" * 100)
        print(f"Total: {len(results)} file(s) need attention")
        print("=" * 100)

        return len(results)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Python file sizes")
    parser.add_argument("path", nargs="?", help="Specific file to analyze")
    parser.add_argument("--hook", action="store_true", help="Pre-commit hook mode (non-blocking)")
    parser.add_argument(
        "--ci", action="store_true", help="CI mode (fails if oversized files found)"
    )
    parser.add_argument("--brief", action="store_true", help="Brief output (no split suggestions)")

    args = parser.parse_args()

    # Determine base path
    script_dir = Path(__file__).parent
    base_path = script_dir.parent  # Jotty root

    analyzer = FileSizeAnalyzer(base_path)

    # Analyze specific file or all files
    if args.path:
        file_path = Path(args.path)
        if not file_path.is_absolute():
            file_path = base_path / file_path
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

        result = analyzer.analyze_file(file_path)
        if result:
            results = [result]
        else:
            print(f"✓ {file_path.name} is under 1,000 lines of code")
            return 0
    else:
        results = analyzer.analyze_all()

    # Report based on mode
    mode = "brief" if args.brief else "full"
    count = analyzer.report(results, mode=mode)

    # Exit code
    if args.ci and count > 0:
        print(f"\n❌ CI FAILURE: {count} file(s) exceed 1,000 lines of code")
        return 1
    elif args.hook and count > 0:
        print(f"\n⚠️  Warning: {count} file(s) exceed 1,000 lines of code")
        print("    Consider splitting these files for better maintainability.")
        return 0  # Non-blocking

    return 0


if __name__ == "__main__":
    sys.exit(main())
