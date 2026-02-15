#!/usr/bin/env python3
"""
Jotty Performance Benchmarks
=============================

Measures performance of key operations.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --verbose
"""

import sys
import time
from pathlib import Path


def benchmark_import_time():
    """Measure import time."""
    start = time.time()
    import Jotty

    end = time.time()
    return (end - start) * 1000  # Convert to ms


def benchmark_discovery():
    """Measure discovery time."""
    from Jotty import capabilities

    start = time.time()
    caps = capabilities()
    end = time.time()
    return (end - start) * 1000


def benchmark_swarm_init():
    """Measure swarm initialization."""
    from Jotty.core.intelligence.swarms.olympiad_learning_swarm import OlympiadLearningSwarm

    start = time.time()
    swarm = OlympiadLearningSwarm()
    end = time.time()
    return (end - start) * 1000


def main():
    verbose = "--verbose" in sys.argv

    print("🏃 Running Jotty Performance Benchmarks...\n")

    benchmarks = [
        ("Import Time", benchmark_import_time, 1000),  # Target: <1000ms
        ("Discovery Time", benchmark_discovery, 100),  # Target: <100ms
        ("Swarm Init", benchmark_swarm_init, 500),  # Target: <500ms
    ]

    results = []
    for name, func, target in benchmarks:
        try:
            duration = func()
            status = "✅" if duration < target else "⚠️"
            results.append((name, duration, target, status))

            if verbose:
                print(f"{status} {name}: {duration:.2f}ms (target: <{target}ms)")
        except Exception as e:
            print(f"❌ {name}: Failed ({e})")
            results.append((name, 0, target, "❌"))

    # Summary
    print("\n📊 Performance Summary:")
    print("-" * 60)
    print(f"{'Benchmark':<20} {'Time (ms)':<15} {'Target':<15} {'Status'}")
    print("-" * 60)

    for name, duration, target, status in results:
        print(f"{name:<20} {duration:>10.2f}ms {target:>10}ms     {status}")

    passed = sum(1 for _, duration, target, _ in results if duration > 0 and duration < target)
    total = len(results)

    print("-" * 60)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All benchmarks passed!")
        return 0
    else:
        print("\n⚠️  Some benchmarks above target")
        return 1


if __name__ == "__main__":
    sys.exit(main())
