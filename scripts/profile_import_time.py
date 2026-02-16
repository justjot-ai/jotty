#!/usr/bin/env python3
"""Profile import times to find slow initialization."""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def time_import(module_name: str) -> float:
    """Time how long an import takes."""
    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        return elapsed
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return -1


print("🔍 Profiling Import Times")
print("=" * 80)

imports = [
    "Jotty",
    "Jotty.core",
    "Jotty.core.execution",
    "Jotty.core.execution.swarms",
    "Jotty.core.execution.swarms.olympiad_learning_swarm",
    "Jotty.core.intelligence",
    "Jotty.core.execution.swarms",
    "Jotty.core.infrastructure",
    "dspy",
]

results = []
for module in imports:
    t = time_import(module)
    if t >= 0:
        results.append((module, t))
        if t > 0.5:
            print(f"⚠️  SLOW: {module}: {t:.2f}s")
        elif t > 0.1:
            print(f"⏱️  {module}: {t:.2f}s")
        else:
            print(f"✅ {module}: {t:.2f}s")

print("\n" + "=" * 80)
print("📊 Summary (slowest first)")
print("=" * 80)
results.sort(key=lambda x: x[1], reverse=True)
for module, t in results[:10]:
    print(f"{t:6.2f}s - {module}")
