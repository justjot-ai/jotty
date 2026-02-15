#!/usr/bin/env python3
"""Fix broken imports in mixin files."""

import re
from pathlib import Path

files_to_fix = [
    "core/intelligence/orchestration/_session_mixin.py",
    "core/intelligence/orchestration/protocols/coordination.py",
    "core/intelligence/orchestration/protocols/routing.py",
    "core/intelligence/orchestration/protocols/resilience.py",
    "core/intelligence/orchestration/protocols/lifecycle.py",
    "core/intelligence/orchestration/_morph_mixin.py",
    "core/intelligence/orchestration/_consensus_mixin.py",
    "core/intelligence/orchestration/_ensemble_mixin.py",
    "core/intelligence/orchestration/_feature_engineering_mixin.py",
    "core/intelligence/orchestration/_feature_selection_mixin.py",
    "core/intelligence/orchestration/_learning_delegation_mixin.py",
    "core/intelligence/orchestration/_mas_zero_mixin.py",
    "core/intelligence/orchestration/_model_pipeline_mixin.py",
    "core/intelligence/orchestration/templates/_deployment_mixin.py",
    "core/intelligence/orchestration/templates/_fairness_mixin.py",
    "core/intelligence/orchestration/templates/_error_analysis_mixin.py",
    "core/intelligence/orchestration/templates/_drift_mixin.py",
    "core/intelligence/orchestration/templates/_interpretability_mixin.py",
    "core/intelligence/orchestration/templates/_mlflow_mixin.py",
    "core/intelligence/orchestration/templates/_report_mixin.py",
    "core/intelligence/orchestration/templates/_rendering_mixin.py",
    "core/intelligence/orchestration/templates/_telegram_mixin.py",
    "core/intelligence/orchestration/templates/_world_class_report_mixin.py",
    "core/intelligence/swarms/_learning_mixin.py",
    "core/intelligence/memory/_consolidation_mixin.py",
]

for file_path in files_to_fix:
    p = Path(file_path)
    if not p.exists():
        print(f"⏭️  {file_path} - doesn't exist")
        continue

    content = p.read_text()
    lines = content.split("\n")

    # Find and fix broken first line
    fixed_lines = []
    i = 0

    # Handle broken future import
    if i < len(lines) and "from __future__" in lines[i]:
        # Clean up the line - extract everything after 'from __future__'
        line = lines[i]
        if "from __future__ from pathlib" in line or "from __future__ from typing" in line:
            # This line is broken, replace with correct future import
            fixed_lines.append("from __future__ import annotations")
            i += 1
        elif line.strip() == "from __future__ import annotations":
            fixed_lines.append(line)
            i += 1
        else:
            fixed_lines.append("from __future__ import annotations")
            i += 1
    else:
        # Add future import if missing
        fixed_lines.append("from __future__ import annotations")

    # Skip broken lines (like "import annotations", "import Path" alone)
    while i < len(lines) and (
        lines[i].strip() in ["import annotations", "import Path", ""]
        or lines[i].strip().startswith("#")
    ):
        if lines[i].strip() == "":
            fixed_lines.append("")
        i += 1

    # Add the rest
    fixed_lines.extend(lines[i:])

    # Now ensure we have the necessary imports
    new_content = "\n".join(fixed_lines)

    # Add pathlib if needed
    if "from pathlib import Path" not in new_content:
        # Find where to insert (after __future__, before other imports)
        lines = new_content.split("\n")
        for idx, line in enumerate(lines):
            if (
                line.startswith('"""')
                or line.startswith("import ")
                or line.startswith("from ")
                and "__future__" not in line
            ):
                lines.insert(idx, "from pathlib import Path")
                lines.insert(idx + 1, "")
                break
        new_content = "\n".join(lines)

    # Write back
    p.write_text(new_content)
    print(f"✅ Fixed {file_path}")

print("\n✅ All imports fixed!")
