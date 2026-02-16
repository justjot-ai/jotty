#!/usr/bin/env python3
"""
Find potentially dead/orphaned code in core/ folder.

Identifies files that:
1. Are only imported in __init__.py (re-exported but not used)
2. Are only imported in tests
3. Have very few imports (likely unused)
4. Are standalone utilities not integrated into main flow
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def find_all_python_files(root_dir):
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".mypy_cache"]]
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_imports_with_context(file_path):
    """Extract imports and check what's actually used."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        imports = []

        # from X import Y
        from_imports = re.findall(r"from\s+([\w.]+)\s+import\s+([\w,\s]+)", content)
        for module, items in from_imports:
            imports.append(
                {
                    "type": "from",
                    "module": module,
                    "items": [i.strip() for i in items.split(",")],
                    "file": file_path,
                }
            )

        # import X
        direct_imports = re.findall(r"(?:^|\n)import\s+([\w.]+)", content)
        for module in direct_imports:
            imports.append({"type": "import", "module": module, "items": [], "file": file_path})

        return imports
    except Exception as e:
        return []


def analyze_usage(project_root):
    """Analyze actual usage of core modules."""
    core_dir = os.path.join(project_root, "core")

    # Find all core files
    core_files = [f for f in find_all_python_files(core_dir) if not f.endswith("__init__.py")]

    # Find all files that import from core
    all_files = find_all_python_files(project_root)

    # Track where each core module is imported
    module_usage = defaultdict(list)

    print(f"Analyzing {len(all_files)} files...")

    for file_path in all_files:
        imports = extract_imports_with_context(file_path)

        for imp in imports:
            module = imp["module"]

            # Check if it's a core module
            if module.startswith("Jotty.core") or module.startswith("core."):
                # Normalize module path
                if module.startswith("Jotty."):
                    module = module[6:]  # Remove 'Jotty.'

                # Track where it's imported
                rel_file = os.path.relpath(file_path, project_root)
                is_init = file_path.endswith("__init__.py")
                is_test = "test" in rel_file

                module_usage[module].append(
                    {
                        "file": rel_file,
                        "is_init": is_init,
                        "is_test": is_test,
                        "items": imp.get("items", []),
                    }
                )

    # Analyze each core file
    results = {"only_in_init": [], "only_in_tests": [], "rarely_used": [], "unused": []}

    for core_file in core_files:
        # Convert to module path
        rel_path = os.path.relpath(core_file, project_root)
        module_path = rel_path[:-3].replace(os.sep, ".")  # Remove .py

        # Find usages
        usages = []
        for key, usage_list in module_usage.items():
            if module_path.endswith(key) or key in module_path:
                usages.extend(usage_list)

        if not usages:
            results["unused"].append(rel_path)
            continue

        # Categorize usage
        init_count = sum(1 for u in usages if u["is_init"])
        test_count = sum(1 for u in usages if u["is_test"])
        real_count = sum(1 for u in usages if not u["is_init"] and not u["is_test"])

        if real_count == 0:
            if init_count > 0 and test_count == 0:
                results["only_in_init"].append(
                    {"file": rel_path, "init_count": init_count, "usages": usages}
                )
            elif test_count > 0 and init_count == 0:
                results["only_in_tests"].append(
                    {"file": rel_path, "test_count": test_count, "usages": usages}
                )
            elif test_count > 0 and init_count > 0:
                results["only_in_init"].append(
                    {
                        "file": rel_path,
                        "init_count": init_count,
                        "test_count": test_count,
                        "usages": usages,
                    }
                )
        elif real_count <= 2:
            results["rarely_used"].append(
                {
                    "file": rel_path,
                    "real_count": real_count,
                    "init_count": init_count,
                    "test_count": test_count,
                    "usages": usages,
                }
            )

    return results


def get_file_stats(project_root, file_path):
    """Get file statistics."""
    full_path = os.path.join(project_root, file_path)
    try:
        lines = sum(1 for _ in open(full_path))
        size = os.path.getsize(full_path)
        return lines, size
    except:
        return 0, 0


def main():
    """Main analysis."""
    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("DEAD CODE ANALYSIS - core/ folder")
    print("=" * 80)
    print()

    results = analyze_usage(str(project_root))

    # Report findings
    print("\n" + "=" * 80)
    print("1. FILES ONLY IMPORTED IN __init__.py (Re-exported but not used)")
    print("=" * 80)

    if results["only_in_init"]:
        total_lines = 0
        for item in sorted(results["only_in_init"], key=lambda x: x["file"]):
            lines, size = get_file_stats(project_root, item["file"])
            total_lines += lines
            print(f"  • {item['file']:<60} ({lines:>4} lines)")
            init_files = [u["file"] for u in item["usages"] if u["is_init"]]
            print(f"    Only in: {', '.join(set(init_files))}")
        print(f"\nTotal: {len(results['only_in_init'])} files, {total_lines:,} lines")
    else:
        print("  ✅ None found")

    print("\n" + "=" * 80)
    print("2. FILES ONLY IMPORTED IN TESTS (Not used in production code)")
    print("=" * 80)

    if results["only_in_tests"]:
        total_lines = 0
        for item in sorted(results["only_in_tests"], key=lambda x: x["file"]):
            lines, size = get_file_stats(project_root, item["file"])
            total_lines += lines
            print(f"  • {item['file']:<60} ({lines:>4} lines)")
        print(f"\nTotal: {len(results['only_in_tests'])} files, {total_lines:,} lines")
    else:
        print("  ✅ None found")

    print("\n" + "=" * 80)
    print("3. RARELY USED FILES (1-2 non-test imports)")
    print("=" * 80)

    if results["rarely_used"]:
        total_lines = 0
        for item in sorted(results["rarely_used"], key=lambda x: x.get("real_count", 0)):
            lines, size = get_file_stats(project_root, item["file"])
            total_lines += lines
            real_usages = [
                u["file"] for u in item["usages"] if not u["is_init"] and not u["is_test"]
            ]
            print(f"  • {item['file']:<60} ({lines:>4} lines)")
            print(f"    Used in {item['real_count']} files: {', '.join(set(real_usages[:3]))}")
        print(f"\nTotal: {len(results['rarely_used'])} files, {total_lines:,} lines")
    else:
        print("  ✅ None found")

    print("\n" + "=" * 80)
    print("4. COMPLETELY UNUSED FILES")
    print("=" * 80)

    if results["unused"]:
        total_lines = 0
        for file_path in sorted(results["unused"]):
            lines, size = get_file_stats(project_root, file_path)
            total_lines += lines
            print(f"  • {file_path:<60} ({lines:>4} lines)")
        print(f"\nTotal: {len(results['unused'])} files, {total_lines:,} lines")
    else:
        print("  ✅ None found")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_files = (
        len(results["only_in_init"])
        + len(results["only_in_tests"])
        + len(results["rarely_used"])
        + len(results["unused"])
    )
    print(f"Only in __init__:  {len(results['only_in_init']):>3} files")
    print(f"Only in tests:     {len(results['only_in_tests']):>3} files")
    print(f"Rarely used:       {len(results['rarely_used']):>3} files")
    print(f"Completely unused: {len(results['unused']):>3} files")
    print(f"{'─' * 30}")
    print(f"TOTAL:             {total_files:>3} files")


if __name__ == "__main__":
    main()
