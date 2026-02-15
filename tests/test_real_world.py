#!/usr/bin/env python3
"""
JOTTY V2 — REAL-WORLD USE CASE TEST
====================================

Tests 3 genuinely practical tasks a real user would ask:
  1. Comparison report (analysis + file output)
  2. Python CLI tool (code generation + save)
  3. Technical explanation (knowledge Q&A)

Measures: latency, output quality, and practical usefulness.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

# ── Load API keys ──
for env_file in [
    Path(__file__).parents[2] / ".env.anthropic",
    Path(__file__).parents[1] / ".env.anthropic",
    Path(__file__).parents[1] / ".env",
]:
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if v and k not in os.environ:
                        os.environ[k] = v

import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("real_world_test")
logger.setLevel(logging.INFO)

B = "\033[1m"
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
D = "\033[2m"
E = "\033[0m"


def extract_output(result) -> str:
    """Extract text output from any result format."""
    if isinstance(result, dict):
        return str(result.get("final_output") or result.get("output") or result.get("result", ""))
    return str(getattr(result, "output", result))


# ══════════════════════════════════════════════════════════════
# USE CASE 1: Comparison Report
# ══════════════════════════════════════════════════════════════
async def test_comparison_report():
    from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator

    print(f'\n{B}{"═"*60}{E}')
    print(f"{B}  USE CASE 1: AI Coding Assistant Comparison Report{E}")
    print(f"{D}  Task: Compare Cursor vs Copilot vs Windsurf with pricing,{E}")
    print(f"{D}  features, pros/cons, and recommendation. Save as .md{E}")
    print(f'{B}{"═"*60}{E}')

    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Compare Cursor, GitHub Copilot, and Windsurf AI coding assistants. "
        "Include: pricing tiers, key features, language support, pros/cons for each, "
        "and a final recommendation for a 5-person startup team. "
        "Save as comparison.md",
        ensemble=False,
        status_callback=lambda stage, detail="": trail.append(
            (time.time() - t0, stage, detail[:90])
        ),
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    checks = {
        "Cursor": "cursor" in output.lower(),
        "Copilot": "copilot" in output.lower(),
        "Windsurf": "windsurf" in output.lower(),
        "Pricing": any(
            w in output.lower() for w in ["price", "pricing", "$", "free", "/month", "plan"]
        ),
        "Recommendation": any(
            w in output.lower() for w in ["recommend", "best", "choose", "suggest", "winner"]
        ),
    }
    quality = sum(checks.values()) / len(checks) * 100

    status = f"{G}PASS{E}" if quality >= 60 else f"{R}FAIL{E}"
    print(f"  {status} | {elapsed:.0f}s | {len(output):,} chars | quality={quality:.0f}%")
    for k, v in checks.items():
        print(f'    {G+"✓"+E if v else R+"✗"+E} {k}')
    for t, stage, detail in trail[-5:]:
        print(f"{D}  [{t:.1f}s] {stage}: {detail}{E}")

    # Print first 500 chars of output
    print(f"\n{D}  Output preview:{E}")
    preview = output[:600].replace("\n", "\n  ")
    print(f"  {preview}...")

    return ("Comparison Report", elapsed, quality, len(output))


# ══════════════════════════════════════════════════════════════
# USE CASE 2: Python CLI Tool
# ══════════════════════════════════════════════════════════════
async def test_cli_tool():
    from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator

    print(f'\n{B}{"═"*60}{E}')
    print(f"{B}  USE CASE 2: Python CSV Analyzer CLI Tool{E}")
    print(f"{D}  Task: Build a complete CLI tool with argparse, pandas,{E}")
    print(f"{D}  statistics, error handling. Save as csv_analyzer.py{E}")
    print(f'{B}{"═"*60}{E}')

    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Write a Python CLI tool (csv_analyzer.py) that: "
        "1) Takes a CSV file path as a command-line argument using argparse "
        "2) Reads the CSV with pandas "
        "3) Prints summary statistics (count, mean, median, std, min, max) for all numeric columns "
        "4) Generates a text report and saves it as report.txt "
        "Include error handling and a --help flag. Save as csv_analyzer.py",
        ensemble=False,
        status_callback=lambda stage, detail="": trail.append(
            (time.time() - t0, stage, detail[:90])
        ),
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    checks = {
        "argparse": "argparse" in output or "ArgumentParser" in output,
        "pandas": "pandas" in output or "read_csv" in output,
        "statistics": any(w in output.lower() for w in ["mean", "median", "std", "describe"]),
        "error_handling": "except" in output or "try" in output,
        "save_file": any(w in output.lower() for w in ["open(", "write(", "report"]),
    }
    quality = sum(checks.values()) / len(checks) * 100

    status = f"{G}PASS{E}" if quality >= 60 else f"{R}FAIL{E}"
    print(f"  {status} | {elapsed:.0f}s | {len(output):,} chars | quality={quality:.0f}%")
    for k, v in checks.items():
        print(f'    {G+"✓"+E if v else R+"✗"+E} {k}')
    for t, stage, detail in trail[-5:]:
        print(f"{D}  [{t:.1f}s] {stage}: {detail}{E}")

    print(f"\n{D}  Output preview:{E}")
    preview = output[:600].replace("\n", "\n  ")
    print(f"  {preview}...")

    return ("CLI Tool Code", elapsed, quality, len(output))


# ══════════════════════════════════════════════════════════════
# USE CASE 3: Technical Explanation (Knowledge Q&A)
# ══════════════════════════════════════════════════════════════
async def test_cap_theorem():
    from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator

    print(f'\n{B}{"═"*60}{E}')
    print(f"{B}  USE CASE 3: CAP Theorem Technical Explanation{E}")
    print(f"{D}  Task: Explain CAP theorem with real DB examples,{E}")
    print(f"{D}  CP vs AP decision matrix, formatted with headers{E}")
    print(f'{B}{"═"*60}{E}')

    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Explain the CAP theorem in distributed systems. "
        "For each pair (CP, AP, CA), give a real-world database example "
        "(e.g., MongoDB, Cassandra, PostgreSQL). "
        "Include a decision matrix showing when to choose CP vs AP based on "
        "use case requirements. Format clearly with headers and bullet points.",
        ensemble=False,
        status_callback=lambda stage, detail="": trail.append(
            (time.time() - t0, stage, detail[:90])
        ),
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    checks = {
        "CAP_explained": any(
            w in output.lower() for w in ["consistency", "availability", "partition"]
        ),
        "DB_examples": sum(
            1
            for db in [
                "mongodb",
                "cassandra",
                "postgres",
                "dynamo",
                "redis",
                "hbase",
                "cockroach",
                "spanner",
                "zookeeper",
            ]
            if db in output.lower()
        )
        >= 2,
        "CP_AP": "cp" in output.lower() and "ap" in output.lower(),
        "Decision_matrix": any(
            w in output.lower() for w in ["matrix", "decision", "when to", "choose", "trade-off"]
        ),
        "Formatted": any(w in output for w in ["##", "**", "- ", "| ", "###"]),
    }
    quality = sum(checks.values()) / len(checks) * 100

    status = f"{G}PASS{E}" if quality >= 60 else f"{R}FAIL{E}"
    print(f"  {status} | {elapsed:.0f}s | {len(output):,} chars | quality={quality:.0f}%")
    for k, v in checks.items():
        print(f'    {G+"✓"+E if v else R+"✗"+E} {k}')
    for t, stage, detail in trail[-5:]:
        print(f"{D}  [{t:.1f}s] {stage}: {detail}{E}")

    print(f"\n{D}  Output preview:{E}")
    preview = output[:600].replace("\n", "\n  ")
    print(f"  {preview}...")

    return ("CAP Theorem Explain", elapsed, quality, len(output))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print(f'{B}{"═"*60}{E}')
    print(f"{B}  JOTTY V2 — REAL-WORLD USE CASE TEST{E}")
    print(f"{D}  Tiered: Gemini Flash (routing) + Sonnet (output){E}")
    print(f'{D}  Time: {time.strftime("%Y-%m-%d %H:%M:%S")}{E}')
    print(f'{B}{"═"*60}{E}')

    results = []
    results.append(await test_comparison_report())
    results.append(await test_cli_tool())
    results.append(await test_cap_theorem())

    # Summary
    print(f'\n{B}{"═"*60}{E}')
    print(f"{B}  FINAL SUMMARY{E}")
    print(f'{B}{"═"*60}{E}')
    print(f'  {"Use Case":<22s} {"Time":>6s} {"Chars":>8s} {"Quality":>8s}')
    print(f'  {"─"*48}')
    total_time = 0
    for name, t, q, c in results:
        total_time += t
        qmark = G + "✓" + E if q >= 60 else R + "✗" + E
        print(f"  {name:<22s} {t:5.0f}s {c:7,} {q:6.0f}%  {qmark}")
    print(f'  {"─"*48}')
    avg_q = sum(r[2] for r in results) / len(results)
    passed = sum(1 for r in results if r[2] >= 60)
    print(f'  {"TOTAL":<22s} {total_time:5.0f}s          {avg_q:6.0f}%')
    print(f"\n  {B}Result: {passed}/{len(results)} passed | Avg quality: {avg_q:.0f}%{E}")


if __name__ == "__main__":
    asyncio.run(main())
