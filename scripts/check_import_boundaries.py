#!/usr/bin/env python3
"""
Import Boundary Linter for Jotty Core
======================================

Enforces cross-module dependency rules within core/ to prevent coupling creep.

How it works:
    1. Scans all .py files in core/ using AST parsing
    2. Extracts cross-module imports (from Jotty.core.X or from ..X)
    3. Checks against ALLOWED_DEPS rules
    4. Reports violations and exits non-zero if any found

Runs in CI to prevent new unwanted dependencies from being added.

Usage:
    python scripts/check_import_boundaries.py           # Check all
    python scripts/check_import_boundaries.py --verbose  # Show all imports
    python scripts/check_import_boundaries.py --strict   # Fail on deferred violations too

Exit codes:
    0 = No violations
    1 = Violations found
"""

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# ALLOWED DEPENDENCY RULES
# =============================================================================
# Each module maps to the set of core/ modules it MAY import from.
# "*" means the module is a hub and can import anything (orchestration).
#
# These rules capture the INTENDED architecture. The linter enforces them.
# To add a new dependency: add it here, get it code-reviewed, then import.
#
# Tier 1 — Leaf modules (only foundation + infrastructure)
# Tier 2 — Mid-coupling (import Tier 1)
# Tier 3 — Hub (imports everything)

# Infrastructure modules that have zero or near-zero deps
INFRASTRUCTURE = {
    "monitoring",
    "observability",
    "persistence",
    "prompts",
    "presets",
    "services",
    "tools",
    "interfaces",
    "swarm_prompts",
    "validation_prompts",
    "optimization",
}

ALLOWED_DEPS: Dict[str, Set[str]] = {
    # --- Tier 0: Skill SDK (depends only on foundation) ---
    "skill_sdk": {"foundation", "utils"},
    # --- Tier 1: Leaves (depend only on foundation + infrastructure) ---
    "memory": {"foundation", "observability"},
    "context": {"foundation", "utils"},
    "utils": {"foundation", "context", "data", "learning"},
    "learning": {"foundation", "context", "memory", "integration"},
    "metadata": {"foundation"},
    "semantic": {"foundation"},
    "ui": {"foundation"},
    "llm": {"foundation", "monitoring"},
    "use_cases": {"foundation", "ui"},
    "evaluation": {"foundation", "execution"},
    # --- Tier 2: Mid-coupling (import Tier 1 modules) ---
    "agents": {
        "foundation",
        "memory",
        "learning",
        "context",
        "utils",
        "swarms",
        "registry",
        "execution",
        "integration",
        "persistence",
        "prompts",
        "orchestration",
        "ui",
    },
    "swarms": {"foundation", "agents", "integration", "orchestration", "registry", "skills"},
    "registry": {"foundation", "agents", "metadata", "utils"},
    "skills": {"foundation", "registry", "integration", "orchestration"},
    "execution": {
        "foundation",
        "agents",
        "swarms",
        "orchestration",
        "registry",
        "monitoring",
        "observability",
    },
    "integration": {"foundation", "context", "orchestration", "utils"},
    "api": {"foundation", "agents", "memory", "orchestration", "registry", "use_cases"},
    "autonomous": {"agents", "registry"},
    "experts": {"foundation", "memory", "orchestration"},
    "job_queue": {"foundation", "orchestration"},
    "lotus": {"foundation", "orchestration"},
    "data": {"agents"},
    # --- Tier 3: Hub (explicit deps — no more wildcard) ---
    "orchestration": {
        "foundation",
        "agents",
        "autonomous",
        "context",
        "data",
        "integration",
        "interfaces",
        "learning",
        "llm",
        "lotus",
        "memory",
        "metadata",
        "monitoring",
        "observability",
        "persistence",
        "prompts",
        "registry",
        "skills",
        "ui",
        "utils",
    },
    # --- Foundation: should be a leaf, but has a few legacy deps ---
    # These are tracked so we can shrink them over time.
    "foundation": {"evaluation", "integration", "monitoring", "utils"},
    # --- Infrastructure: no cross-deps (leaf by definition) ---
    "monitoring": set(),
    "observability": set(),
    "persistence": set(),
    "prompts": set(),
    "presets": set(),
    "services": set(),
    "tools": set(),
    "interfaces": set(),
    "swarm_prompts": set(),
    "validation_prompts": set(),
    "optimization": set(),
}


# =============================================================================
# INTERNAL SUB-MODULE BOUNDARIES (within orchestration/)
# =============================================================================
# Orchestration is 60+ files. These rules enforce logical grouping so that
# sub-modules don't reach across internal boundaries unexpectedly.
#
# Key: sub-module name (directory or logical group)
# Value: set of files belonging to that sub-module
#
# The INTERNAL_ALLOWED_DEPS maps sub-module -> set of sub-modules it may import.

ORCHESTRATION_SUB_MODULES: Dict[str, Set[str]] = {
    "llm_providers": {
        "llm_providers/adapter.py",
        "llm_providers/anthropic.py",
        "llm_providers/base.py",
        "llm_providers/factory.py",
        "llm_providers/google.py",
        "llm_providers/__init__.py",
        "llm_providers/openai.py",
        "llm_providers/types.py",
    },
    "intelligence": {
        "swarm_intelligence.py",
        "paradigm_executor.py",
        "ensemble_manager.py",
        "swarm_ensemble.py",
    },
    "public_api": {
        "facade.py",
        "swarm.py",
    },
    "routing": {
        "swarm_router.py",
        "model_tier_router.py",
        "morph_scoring.py",
        "_morph_mixin.py",
    },
    "monitoring": {
        "metrics_collector.py",
        "benchmarking.py",
    },
    "learning": {
        "swarm_learner.py",
        "learning_pipeline.py",
        "learning_delegate.py",
        "_learning_delegation_mixin.py",
        "mas_learning.py",
        "credit_assignment.py",
        "adaptive_learning.py",
        "stigmergy.py",
        "swarm_workflow_learner.py",
        "curriculum_generator.py",
        "policy_explorer.py",
    },
}

# Which sub-modules can import from which
INTERNAL_ALLOWED_DEPS: Dict[str, Set[str]] = {
    "llm_providers": set(),  # Leaf: no orchestration-internal deps
    "monitoring": set(),  # Leaf: no orchestration-internal deps
    "routing": {"intelligence", "llm_providers"},
    "intelligence": {"llm_providers"},
    "learning": {"intelligence", "routing"},
    "public_api": {"intelligence", "routing", "llm_providers", "learning", "monitoring"},
}


def _get_orchestration_sub_module(filepath: str) -> Optional[str]:
    """Get the orchestration sub-module for a file, if any."""
    # Extract path relative to orchestration/
    parts = filepath.split("orchestration/")
    if len(parts) < 2:
        return None
    rel = parts[1]
    for sub_mod, files in ORCHESTRATION_SUB_MODULES.items():
        if rel in files:
            return sub_mod
    return None


def check_internal_boundaries(core_root: str) -> List[str]:
    """Check orchestration internal sub-module boundaries.

    Returns list of violation descriptions (empty = clean).
    """
    orch_root = os.path.join(core_root, "orchestration")
    if not os.path.isdir(orch_root):
        return []

    violations = []

    # Build file -> sub-module map
    file_to_sub = {}
    for dirpath, dirnames, filenames in os.walk(orch_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if fname.endswith(".py"):
                filepath = os.path.join(dirpath, fname)
                sub = _get_orchestration_sub_module(filepath)
                if sub:
                    file_to_sub[filepath] = sub

    # Check each file's imports against sub-module rules
    for filepath, from_sub in file_to_sub.items():
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue

            # Only check intra-orchestration imports
            import_path = node.module
            if not ("orchestration" in import_path or import_path.startswith(".")):
                continue

            # Resolve what sub-module the import target belongs to
            target_sub = None

            # Build the target filename from the import path
            if import_path.startswith("."):
                # Relative import within orchestration
                rel_module = import_path.lstrip(".")
                if not rel_module:
                    continue
                # Convert dots to path: .llm_providers.base -> llm_providers/base.py
                target_file = rel_module.replace(".", "/") + ".py"
            else:
                # Absolute import: Jotty.core.orchestration.X -> X.py
                if "orchestration." not in import_path:
                    continue
                after_orch = import_path.split("orchestration.")[-1]
                target_file = after_orch.replace(".", "/") + ".py"

            # Match against known sub-module files (exact match)
            for sub, files in ORCHESTRATION_SUB_MODULES.items():
                if target_file in files:
                    target_sub = sub
                    break

            # Only report if target is definitively in a different sub-module
            if target_sub and target_sub != from_sub:
                # Verify the match is exact (avoid false positives from partial matches)
                allowed = INTERNAL_ALLOWED_DEPS.get(from_sub, set())
                if target_sub not in allowed:
                    rel_path = os.path.relpath(filepath, os.path.dirname(core_root))
                    violations.append(
                        f"  {rel_path}:{node.lineno} " f"({from_sub} -> {target_sub})"
                    )

    return violations


# =============================================================================
# AST-BASED IMPORT SCANNER
# =============================================================================


class ImportInfo:
    """A single cross-module import."""

    __slots__ = ("source_file", "line", "from_module", "to_module", "is_deferred")

    def __init__(
        self, source_file: str, line: int, from_module: str, to_module: str, is_deferred: bool
    ):
        self.source_file = source_file
        self.line = line
        self.from_module = from_module
        self.to_module = to_module
        self.is_deferred = is_deferred

    def __repr__(self):
        defer = " (deferred)" if self.is_deferred else ""
        return f"{self.source_file}:{self.line} {self.from_module} -> {self.to_module}{defer}"


def _resolve_module(import_path: str) -> Optional[str]:
    """Extract the core/ module name from an import path.

    Returns None if the import doesn't reference a core/ module.

    Examples:
        'Jotty.core.memory.cortex' -> 'memory'
        'Jotty.core.foundation.types' -> 'foundation'
    """
    # Handle absolute imports: from Jotty.core.X.Y.Z
    if import_path.startswith("Jotty.core."):
        parts = import_path.split(".")
        if len(parts) >= 3:
            return parts[2]  # The module name after Jotty.core.
    return None


def _is_inside_function_or_typecheck(node: ast.AST, tree: ast.Module) -> bool:
    """Check if a node is inside a function, method, or TYPE_CHECKING block."""
    # Walk the tree to find the parent chain
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is node:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return True
                if isinstance(parent, ast.If):
                    # Check for TYPE_CHECKING guard
                    test = parent.test
                    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                        return True
                    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                        return True
    return False


def _get_module_from_file(filepath: str, core_root: str) -> Optional[str]:
    """Get the module name for a file within core/."""
    rel = os.path.relpath(filepath, core_root)
    parts = rel.split(os.sep)
    if parts and parts[0] != ".":
        return parts[0]
    return None


def scan_file(filepath: str, core_root: str) -> List[ImportInfo]:
    """Scan a single Python file for cross-module imports."""
    results = []
    from_module = _get_module_from_file(filepath, core_root)
    if not from_module:
        return results

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return results

    # Build parent map for deferred-import detection
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore

    for node in ast.walk(tree):
        import_module = None

        if isinstance(node, ast.ImportFrom) and node.module:
            import_module = node.module

        if import_module is None:
            continue

        to_module = _resolve_module(import_module)
        if to_module is None:
            continue
        if to_module == from_module:
            continue  # Self-import, skip

        # Check if deferred (inside function/method or TYPE_CHECKING)
        is_deferred = False
        parent = getattr(node, "_parent", None)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_deferred = True
                break
            if isinstance(parent, ast.If):
                test = parent.test
                if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                    is_deferred = True
                    break
                if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                    is_deferred = True
                    break
            parent = getattr(parent, "_parent", None)

        rel_path = os.path.relpath(filepath, os.path.dirname(core_root))
        results.append(
            ImportInfo(
                source_file=rel_path,
                line=node.lineno,
                from_module=from_module,
                to_module=to_module,
                is_deferred=is_deferred,
            )
        )

    return results


def scan_core(core_root: str) -> List[ImportInfo]:
    """Scan all Python files in core/ for cross-module imports."""
    all_imports = []
    for dirpath, dirnames, filenames in os.walk(core_root):
        # Skip __pycache__
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if fname.endswith(".py"):
                filepath = os.path.join(dirpath, fname)
                all_imports.extend(scan_file(filepath, core_root))
    return all_imports


# =============================================================================
# VIOLATION CHECKER
# =============================================================================


def check_violations(
    imports: List[ImportInfo], strict: bool = False
) -> Tuple[List[ImportInfo], List[ImportInfo]]:
    """Check imports against ALLOWED_DEPS rules.

    Returns:
        (top_level_violations, deferred_violations)
    """
    top_level = []
    deferred = []

    for imp in imports:
        allowed = ALLOWED_DEPS.get(imp.from_module)
        if allowed is None:
            # Module not in rules — skip (it's not a tracked module)
            continue
        if "*" in allowed:
            continue  # Hub module, anything goes

        if imp.to_module not in allowed:
            if imp.is_deferred:
                deferred.append(imp)
            else:
                top_level.append(imp)

    return top_level, deferred


# =============================================================================
# REPORTING
# =============================================================================


def build_summary(imports: List[ImportInfo]) -> Dict[str, Set[str]]:
    """Build module -> set of depended modules."""
    summary: Dict[str, Set[str]] = defaultdict(set)
    for imp in imports:
        summary[imp.from_module].add(imp.to_module)
    return dict(summary)


def print_matrix(imports: List[ImportInfo]):
    """Print the dependency matrix."""
    summary = build_summary(imports)
    print("\n=== Dependency Matrix ===")
    for mod in sorted(summary):
        deps = sorted(summary[mod])
        print(f"  {mod:20s} → {', '.join(deps)}")
    print()


def print_violations(violations: List[ImportInfo], label: str):
    """Print violations grouped by (from_module → to_module)."""
    if not violations:
        return

    grouped: Dict[Tuple[str, str], List[ImportInfo]] = defaultdict(list)
    for v in violations:
        grouped[(v.from_module, v.to_module)].append(v)

    print(f"\n{'='*60}")
    print(f"  {label}: {len(violations)} violation(s)")
    print(f"{'='*60}")

    for (from_mod, to_mod), items in sorted(grouped.items()):
        allowed = ALLOWED_DEPS.get(from_mod, set())
        print(f"\n  {from_mod} → {to_mod}  (not in allowed: {sorted(allowed)})")
        for item in sorted(items, key=lambda x: (x.source_file, x.line)):
            print(f"    {item.source_file}:{item.line}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    verbose = "--verbose" in sys.argv
    strict = "--strict" in sys.argv

    # Find core/ directory
    script_dir = Path(__file__).resolve().parent
    core_root = script_dir.parent / "core"

    if not core_root.is_dir():
        print(f"ERROR: core/ not found at {core_root}")
        sys.exit(1)

    print(f"Scanning {core_root} ...")
    imports = scan_core(str(core_root))
    print(
        f"Found {len(imports)} cross-module imports ({sum(1 for i in imports if i.is_deferred)} deferred)"
    )

    if verbose:
        print_matrix(imports)

    top_level_violations, deferred_violations = check_violations(imports, strict)

    if top_level_violations:
        print_violations(top_level_violations, "TOP-LEVEL VIOLATIONS (must fix)")

    if deferred_violations and (verbose or strict):
        print_violations(
            deferred_violations, "DEFERRED VIOLATIONS (inside functions/TYPE_CHECKING)"
        )

    # Internal sub-module boundaries (orchestration)
    internal_violations = check_internal_boundaries(str(core_root))
    if internal_violations and verbose:
        print(f"\n{'='*60}")
        print(f"  ORCHESTRATION INTERNAL BOUNDARY VIOLATIONS: {len(internal_violations)}")
        print(f"{'='*60}")
        for v in internal_violations:
            print(v)

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Top-level violations: {len(top_level_violations)}")
    print(f"  Deferred violations:  {len(deferred_violations)}")
    print(f"  Internal violations:  {len(internal_violations)} (orchestration sub-modules)")

    if top_level_violations:
        print(f"\nFAILED: {len(top_level_violations)} top-level boundary violation(s)")
        print("Fix: Add the dependency to ALLOWED_DEPS in scripts/check_import_boundaries.py")
        print("     (or remove the import if it shouldn't exist)")
        sys.exit(1)
    elif strict and deferred_violations:
        print(f"\nFAILED (strict): {len(deferred_violations)} deferred boundary violation(s)")
        sys.exit(1)
    else:
        print("\nPASSED: All imports within allowed boundaries")
        sys.exit(0)


def check_programmatic(strict: bool = False) -> Tuple[List[ImportInfo], List[ImportInfo]]:
    """Run the linter programmatically and return violations.

    Useful for testing:
        top, deferred = check_programmatic()
        assert len(top) == 0, f"Found {len(top)} violations"
    """
    script_dir = Path(__file__).resolve().parent
    core_root = script_dir.parent / "core"
    imports = scan_core(str(core_root))
    return check_violations(imports, strict)


if __name__ == "__main__":
    main()
