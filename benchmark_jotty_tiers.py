"""
Benchmark Jotty Tiers
=====================

Measures import time and memory usage for each tier of Jotty:
- Tier 0 (Minimal): jotty_minimal.py
- Current (Full): core.orchestration.conductor (MultiAgentsOrchestrator)

Goal:
- Tier 0: Import < 0.5s, Memory < 50MB
- Current: Baseline for comparison

Usage:
    python benchmark_jotty_tiers.py
"""

import time
import sys
import subprocess
import psutil

def benchmark_tier(tier_name: str, import_statement: str):
    """Benchmark import time and memory for a tier"""
    print(f"\n{'='*80}")
    print(f"TIER: {tier_name}")
    print('='*80)

    # Measure import time
    code = f"""
import time
import tracemalloc
import sys

start_time = time.time()
start_mem = tracemalloc.start()

{import_statement}

end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"IMPORT_TIME={{end_time - start_time:.3f}}")
print(f"MEMORY_CURRENT={{current / 1024 / 1024:.2f}}")
print(f"MEMORY_PEAK={{peak / 1024 / 1024:.2f}}")
"""

    try:
        # Run in subprocess to get clean measurements
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"❌ Failed to benchmark {tier_name}")
            print(f"Error: {result.stderr}")
            return None

        # Parse output
        output = result.stdout
        import_time = None
        memory_current = None
        memory_peak = None

        for line in output.split("\n"):
            if line.startswith("IMPORT_TIME="):
                import_time = float(line.split("=")[1])
            elif line.startswith("MEMORY_CURRENT="):
                memory_current = float(line.split("=")[1])
            elif line.startswith("MEMORY_PEAK="):
                memory_peak = float(line.split("=")[1])

        if import_time is None:
            print(f"❌ Could not parse benchmark results")
            return None

        # Print results
        print(f"  Import Time: {import_time:.3f}s")
        print(f"  Memory (Current): {memory_current:.2f} MB")
        print(f"  Memory (Peak): {memory_peak:.2f} MB")

        # Check against targets
        if tier_name == "Tier 0 (Minimal)":
            import_ok = "✅" if import_time < 0.5 else "❌"
            memory_ok = "✅" if memory_peak < 50 else "❌"
            print(f"\n  Targets:")
            print(f"    {import_ok} Import < 0.5s: {import_time:.3f}s")
            print(f"    {memory_ok} Memory < 50MB: {memory_peak:.2f}MB")

        return {
            "tier": tier_name,
            "import_time": import_time,
            "memory_current": memory_current,
            "memory_peak": memory_peak
        }

    except subprocess.TimeoutExpired:
        print(f"❌ Benchmark timed out after 30s")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def print_comparison(results):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print("COMPARISON")
    print('='*80)

    if not results or len(results) < 2:
        print("Insufficient results for comparison")
        return

    # Find baseline (current full Jotty)
    baseline = next((r for r in results if "Current" in r["tier"]), None)
    minimal = next((r for r in results if "Minimal" in r["tier"]), None)

    if not baseline or not minimal:
        print("Missing baseline or minimal tier results")
        return

    # Calculate improvements
    import_speedup = baseline["import_time"] / minimal["import_time"]
    memory_reduction = (baseline["memory_peak"] - minimal["memory_peak"]) / baseline["memory_peak"] * 100

    print(f"\n  Tier 0 (Minimal) vs Current (Full):")
    print(f"    Import Time: {minimal['import_time']:.3f}s vs {baseline['import_time']:.3f}s")
    print(f"    Speedup: {import_speedup:.1f}x faster")
    print(f"\n    Memory Peak: {minimal['memory_peak']:.2f}MB vs {baseline['memory_peak']:.2f}MB")
    print(f"    Reduction: {memory_reduction:.1f}% less memory")

    # Success criteria
    print(f"\n  Success Criteria (Tier 0):")
    print(f"    ✅ Import < 0.5s: {minimal['import_time']:.3f}s" if minimal['import_time'] < 0.5 else f"    ❌ Import < 0.5s: {minimal['import_time']:.3f}s")
    print(f"    ✅ Memory < 50MB: {minimal['memory_peak']:.2f}MB" if minimal['memory_peak'] < 50 else f"    ❌ Memory < 50MB: {minimal['memory_peak']:.2f}MB")
    print(f"    ✅ Speedup > 2x: {import_speedup:.1f}x" if import_speedup > 2 else f"    ❌ Speedup > 2x: {import_speedup:.1f}x")

def main():
    """Run benchmarks"""
    print("#"*80)
    print("# JOTTY TIER BENCHMARKS")
    print("#"*80)
    print("\nMeasuring import time and memory usage for each tier...\n")

    results = []

    # Benchmark Tier 0 (Minimal)
    result = benchmark_tier(
        "Tier 0 (Minimal)",
        "from jotty_minimal import Orchestrator"
    )
    if result:
        results.append(result)

    # Benchmark Current (Full Jotty)
    result = benchmark_tier(
        "Current (Full Jotty)",
        "from core.orchestration.conductor import MultiAgentsOrchestrator"
    )
    if result:
        results.append(result)

    # Print comparison
    print_comparison(results)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print('='*80)

if __name__ == "__main__":
    main()
