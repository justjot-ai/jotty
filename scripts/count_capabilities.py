#!/usr/bin/env python3
"""Count skills and swarms from the repo tree.

Usage:
  python scripts/count_capabilities.py
  python scripts/count_capabilities.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_skills(root: Path) -> int:
    skills_dir = root / "skills"
    if not skills_dir.exists():
        return 0
    count = 0
    for entry in skills_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("_"):
            continue
        count += 1
    return count


def count_swarms(root: Path) -> int:
    swarms_dir = root / "core" / "intelligence" / "orchestration" / "swarms"
    if not swarms_dir.exists():
        return 0
    exclude = {"_base", "base", "templates", "__pycache__"}
    count = 0
    for entry in swarms_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("_"):
            continue
        if name in exclude:
            continue
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    skills = count_skills(root)
    swarms = count_swarms(root)

    if args.json:
        print(json.dumps({"skills": skills, "swarms": swarms}))
    else:
        print(f"skills: {skills}")
        print(f"swarms: {swarms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
