#!/usr/bin/env python3
"""
Advanced Dead Code Detection
=============================

Distinguishes between:
1. Truly dead code (never used, not part of any pattern)
2. Lazy-loaded modules (via __getattr__, _LAZY_IMPORTS, etc.)
3. Plugin-based architectures (skill registry, entry points)
4. Public API exports (meant for external use)

This script filters out false positives from basic dead code analysis.
"""

import ast
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [
            d for d in dirs if d not in ["__pycache__", ".git", ".mypy_cache", ".pytest_cache"]
        ]
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_lazy_loading_patterns(file_path: str) -> Dict[str, List[str]]:
    """
    Detect lazy loading patterns in __init__.py files.

    Returns:
        Dict with pattern type -> list of lazy-loaded names
    """
    patterns = {
        "lazy_imports": [],  # _LAZY_IMPORTS dict
        "getattr_dynamic": [],  # __getattr__ function
        "all_exports": [],  # __all__ list
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            # Detect _LAZY_IMPORTS dictionary
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and "LAZY" in target.id.upper():
                        if isinstance(node.value, ast.Dict):
                            for key in node.value.keys:
                                if isinstance(key, ast.Constant):
                                    patterns["lazy_imports"].append(key.value)

            # Detect __getattr__ function
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                patterns["getattr_dynamic"].append("__getattr__")

            # Detect __all__ exports
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    patterns["all_exports"].append(elt.value)

    except Exception as e:
        pass

    return patterns


def find_lazy_loaded_modules(project_root: str) -> Set[str]:
    """Find all modules that are lazy-loaded via patterns."""
    lazy_modules = set()

    # Find all __init__.py files
    init_files = []
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".mypy_cache"]]
        if "__init__.py" in files:
            init_files.append(os.path.join(root, "__init__.py"))

    for init_file in init_files:
        patterns = extract_lazy_loading_patterns(init_file)

        if patterns["lazy_imports"] or patterns["getattr_dynamic"]:
            # This package uses lazy loading
            package_dir = os.path.dirname(init_file)
            rel_package = os.path.relpath(package_dir, project_root)

            # All Python files in this directory are potentially lazy-loaded
            for py_file in find_python_files(package_dir):
                if not py_file.endswith("__init__.py"):
                    rel_path = os.path.relpath(py_file, project_root)
                    lazy_modules.add(rel_path)

    return lazy_modules


def find_skill_registry_patterns(project_root: str) -> Set[str]:
    """Find files that are part of skill registry (plugin architecture)."""
    skill_files = set()

    # All files in skills/ directory are plugins
    skills_dir = os.path.join(project_root, "skills")
    if os.path.exists(skills_dir):
        for py_file in find_python_files(skills_dir):
            rel_path = os.path.relpath(py_file, project_root)
            skill_files.add(rel_path)

    return skill_files


def find_public_api_exports(project_root: str) -> Set[str]:
    """Find files that are public API exports (meant for external use)."""
    api_exports = set()

    # Check top-level __init__.py and core/__init__.py
    top_level_inits = [
        os.path.join(project_root, "__init__.py"),
        os.path.join(project_root, "core", "__init__.py"),
        os.path.join(project_root, "sdk", "__init__.py"),
    ]

    for init_file in top_level_inits:
        if not os.path.exists(init_file):
            continue

        patterns = extract_lazy_loading_patterns(init_file)

        # Files exported in __all__ at top level are public API
        for export in patterns["all_exports"]:
            # Try to find corresponding file
            parent_dir = os.path.dirname(init_file)
            potential_file = os.path.join(parent_dir, f"{export}.py")
            if os.path.exists(potential_file):
                rel_path = os.path.relpath(potential_file, project_root)
                api_exports.add(rel_path)

    return api_exports


def find_test_fixtures_and_helpers(project_root: str) -> Set[str]:
    """Find test fixtures and helpers (used only in tests)."""
    test_helpers = set()

    # conftest.py files are pytest fixtures
    for py_file in find_python_files(project_root):
        if "conftest.py" in py_file or "test_helpers" in py_file or "fixtures" in py_file:
            rel_path = os.path.relpath(py_file, project_root)
            test_helpers.add(rel_path)

    return test_helpers


def get_file_stats(project_root: Path, file_path: str) -> Tuple[int, int]:
    """Get file statistics."""
    full_path = os.path.join(project_root, file_path)
    try:
        lines = sum(1 for _ in open(full_path))
        size = os.path.getsize(full_path)
        return lines, size
    except:
        return 0, 0


def analyze_truly_dead_code(project_root: str) -> Dict[str, any]:
    """
    Identify truly dead code by filtering out:
    - Lazy-loaded modules
    - Plugin architectures
    - Public API exports
    - Test fixtures
    """

    print("🔍 Analyzing code patterns...")

    # Find architectural patterns
    lazy_loaded = find_lazy_loaded_modules(project_root)
    skill_plugins = find_skill_registry_patterns(project_root)
    api_exports = find_public_api_exports(project_root)
    test_helpers = find_test_fixtures_and_helpers(project_root)

    print(f"   ✅ {len(lazy_loaded)} lazy-loaded modules")
    print(f"   ✅ {len(skill_plugins)} skill plugins")
    print(f"   ✅ {len(api_exports)} public API exports")
    print(f"   ✅ {len(test_helpers)} test helpers")

    # Run original dead code analysis
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from find_dead_code import analyze_usage

    dead_results = analyze_usage(project_root)

    # Filter results
    truly_dead = {"only_in_init": [], "only_in_tests": [], "rarely_used": [], "unused": []}

    # Filter "only_in_init" - exclude lazy-loaded, plugins, API exports
    for item in dead_results["only_in_init"]:
        file_path = item["file"]

        # Skip if it's part of a known pattern
        if (
            file_path in lazy_loaded
            or file_path in skill_plugins
            or file_path in api_exports
            or file_path in test_helpers
        ):
            continue

        # Also skip if it's in a directory with lazy loading
        parent_dir = os.path.dirname(file_path)
        init_file = os.path.join(project_root, parent_dir, "__init__.py")
        if os.path.exists(init_file):
            patterns = extract_lazy_loading_patterns(init_file)
            if patterns["getattr_dynamic"]:
                continue  # Skip - parent uses lazy loading

        truly_dead["only_in_init"].append(item)

    # Filter "only_in_tests" - exclude test helpers
    for item in dead_results["only_in_tests"]:
        file_path = item["file"]

        if file_path in test_helpers:
            continue

        # Test-only code might be legitimate (test utilities)
        # Only flag if it's in core/ and claims to be production code
        if file_path.startswith("core/") and "test" not in file_path.lower():
            truly_dead["only_in_tests"].append(item)

    # Filter "rarely_used" - exclude plugins and lazy-loaded
    for item in dead_results["rarely_used"]:
        file_path = item["file"]

        if file_path in lazy_loaded or file_path in skill_plugins or file_path in api_exports:
            continue

        truly_dead["rarely_used"].append(item)

    # "unused" is usually truly dead
    truly_dead["unused"] = dead_results["unused"]

    return truly_dead, {
        "lazy_loaded": lazy_loaded,
        "skill_plugins": skill_plugins,
        "api_exports": api_exports,
        "test_helpers": test_helpers,
    }


def main():
    """Main analysis."""
    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("TRULY DEAD CODE ANALYSIS")
    print("=" * 80)
    print()
    print("This analysis filters out false positives:")
    print("  • Lazy-loaded modules (via __getattr__)")
    print("  • Plugin architectures (skill registry)")
    print("  • Public API exports (meant for external use)")
    print("  • Test fixtures and helpers")
    print()

    truly_dead, patterns = analyze_truly_dead_code(str(project_root))

    # Report findings
    print("\n" + "=" * 80)
    print("RESULTS: Files that are ACTUALLY dead (not architectural patterns)")
    print("=" * 80)
    print()

    total_truly_dead = 0
    total_lines = 0

    # Category 1: Only in __init__.py (but not lazy-loaded or public API)
    if truly_dead["only_in_init"]:
        print(f"1. ONLY IN __init__.py (RE-EXPORTED BUT NOT USED)")
        print(f"   ({len(truly_dead['only_in_init'])} files)")
        print("-" * 80)

        cat_lines = 0
        for item in sorted(truly_dead["only_in_init"], key=lambda x: x["file"]):
            lines, _ = get_file_stats(project_root, item["file"])
            cat_lines += lines
            print(f"  • {item['file']:<60} ({lines:>4} lines)")

        print(f"\nSubtotal: {len(truly_dead['only_in_init'])} files, {cat_lines:,} lines\n")
        total_truly_dead += len(truly_dead["only_in_init"])
        total_lines += cat_lines
    else:
        print("1. ONLY IN __init__.py: ✅ None (all are valid patterns)\n")

    # Category 2: Only in tests (production code in core/)
    if truly_dead["only_in_tests"]:
        print(f"2. ONLY IN TESTS (PRODUCTION CODE NOT USED)")
        print(f"   ({len(truly_dead['only_in_tests'])} files)")
        print("-" * 80)

        cat_lines = 0
        for item in sorted(truly_dead["only_in_tests"], key=lambda x: x["file"]):
            lines, _ = get_file_stats(project_root, item["file"])
            cat_lines += lines
            print(f"  • {item['file']:<60} ({lines:>4} lines)")

        print(f"\nSubtotal: {len(truly_dead['only_in_tests'])} files, {cat_lines:,} lines\n")
        total_truly_dead += len(truly_dead["only_in_tests"])
        total_lines += cat_lines
    else:
        print("2. ONLY IN TESTS: ✅ None (all are valid test utilities)\n")

    # Category 3: Rarely used
    if truly_dead["rarely_used"]:
        print(f"3. RARELY USED (1-2 IMPORTS)")
        print(f"   ({len(truly_dead['rarely_used'])} files)")
        print("-" * 80)

        cat_lines = 0
        for item in sorted(truly_dead["rarely_used"], key=lambda x: x.get("real_count", 0)):
            lines, _ = get_file_stats(project_root, item["file"])
            cat_lines += lines
            real_usages = [
                u["file"] for u in item["usages"] if not u["is_init"] and not u["is_test"]
            ]
            print(f"  • {item['file']:<60} ({lines:>4} lines)")
            print(f"    Used in {item['real_count']} files: {', '.join(real_usages[:2])}")

        print(f"\nSubtotal: {len(truly_dead['rarely_used'])} files, {cat_lines:,} lines\n")
        total_truly_dead += len(truly_dead["rarely_used"])
        total_lines += cat_lines
    else:
        print("3. RARELY USED: ✅ None\n")

    # Category 4: Completely unused
    if truly_dead["unused"]:
        print(f"4. COMPLETELY UNUSED")
        print(f"   ({len(truly_dead['unused'])} files)")
        print("-" * 80)

        cat_lines = 0
        for file_path in sorted(truly_dead["unused"]):
            lines, _ = get_file_stats(project_root, file_path)
            cat_lines += lines
            print(f"  • {file_path:<60} ({lines:>4} lines)")

        print(f"\nSubtotal: {len(truly_dead['unused'])} files, {cat_lines:,} lines\n")
        total_truly_dead += len(truly_dead["unused"])
        total_lines += cat_lines
    else:
        print("4. COMPLETELY UNUSED: ✅ None\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Truly dead code:        {total_truly_dead:>3} files ({total_lines:,} lines)")
    print(
        f"Lazy-loaded modules:    {len(patterns['lazy_loaded']):>3} files (architectural pattern)"
    )
    print(
        f"Skill plugins:          {len(patterns['skill_plugins']):>3} files (plugin architecture)"
    )
    print(f"Public API exports:     {len(patterns['api_exports']):>3} files (external interface)")
    print(f"Test helpers:           {len(patterns['test_helpers']):>3} files (test infrastructure)")
    print()
    print("✅ All architectural patterns preserved")
    print("⚠️  Consider reviewing truly dead code for removal")


if __name__ == "__main__":
    main()
