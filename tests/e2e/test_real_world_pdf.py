#!/usr/bin/env python3
"""
JOTTY V2 — REAL-WORLD PDF REPORT TEST
======================================

Tests a genuinely complex multi-step task:
  Research a topic → synthesize findings → generate PDF report

This exercises the FULL Orchestrator pipeline:
  IntentParser → SkillDiscovery → TaskPlanner → SkillExecution → PDF output

Unlike simple Q&A (which hits the fast path), this requires:
  - Multi-skill orchestration (research + document-converter)
  - File I/O (write markdown, convert to PDF)
  - Content synthesis (raw search → structured report)

Measures: latency, output quality, PDF existence, and file size.
"""

import asyncio
import glob
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
logger = logging.getLogger("real_world_pdf_test")
logger.setLevel(logging.INFO)

B = "\033[1m"
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
D = "\033[2m"
E = "\033[0m"

OUTPUT_DIR = Path(__file__).parents[1].parent / "outputs"


def extract_output(result) -> str:
    """Extract text output from any result format."""
    if isinstance(result, dict):
        return str(result.get("final_output") or result.get("output") or result.get("result", ""))
    return str(getattr(result, "output", result))


def find_pdf_files(before_set: set) -> list:
    """Find new PDF files created since before_set snapshot."""
    current = set()
    # Check common output locations
    for search_dir in [OUTPUT_DIR, Path.home() / "jotty" / "reports", Path("/tmp")]:
        if search_dir.exists():
            current.update(search_dir.glob("**/*.pdf"))
    return sorted(current - before_set, key=lambda p: p.stat().st_mtime, reverse=True)


def snapshot_pdfs() -> set:
    """Snapshot existing PDF files in output directories."""
    existing = set()
    for search_dir in [OUTPUT_DIR, Path.home() / "jotty" / "reports", Path("/tmp")]:
        if search_dir.exists():
            existing.update(search_dir.glob("**/*.pdf"))
    return existing


# ══════════════════════════════════════════════════════════════
# USE CASE: Research Topic → PDF Report
# ══════════════════════════════════════════════════════════════
async def test_research_to_pdf():
    from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator

    print(f'\n{B}{"═"*60}{E}')
    print(f"{B}  USE CASE: Research → PDF Report{E}")
    print(f"{D}  Task: Research multi-agent AI systems in 2025,{E}")
    print(f"{D}  synthesize findings into structured report, save as PDF.{E}")
    print(f"{D}  Exercises: IntentParser → SkillDiscovery → Planner →{E}")
    print(f"{D}             research-to-pdf skill → document-converter{E}")
    print(f'{B}{"═"*60}{E}')

    # Snapshot PDFs before test
    pdfs_before = snapshot_pdfs()

    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Research the current state of multi-agent AI systems in 2025. "
        "Cover key frameworks (CrewAI, AutoGen, LangGraph, Jotty), "
        "recent breakthroughs, and industry adoption trends. "
        "Generate a comprehensive PDF report and save it. "
        "Include an executive summary, detailed analysis, and recommendations.",
        ensemble=False,
        status_callback=lambda stage, detail="": (
            trail.append((time.time() - t0, stage, detail[:90])),
            print(f"{D}  [{time.time() - t0:5.1f}s] {stage}: {detail[:80]}{E}"),
        ),
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    # ── Quality checks ──
    # 1. Content quality (does the output mention the right topics?)
    content_checks = {
        "multi_agent": any(w in output.lower() for w in ["multi-agent", "multi agent", "mas"]),
        "frameworks": sum(
            1 for fw in ["crewai", "autogen", "langgraph", "jotty", "swarm"] if fw in output.lower()
        )
        >= 2,
        "analysis": any(
            w in output.lower() for w in ["analysis", "findings", "research", "report"]
        ),
        "recommendations": any(
            w in output.lower() for w in ["recommend", "conclusion", "outlook", "future"]
        ),
        "structured": any(w in output for w in ["##", "**", "- ", "###", "Summary", "Executive"]),
    }

    # 2. PDF output check
    pdfs_after = find_pdf_files(pdfs_before)
    pdf_found = len(pdfs_after) > 0

    # Also check if the output mentions a PDF path
    pdf_path_in_output = any(w in output.lower() for w in [".pdf", "pdf_path", "generated pdf"])

    # 3. File size check (if PDF found)
    pdf_size = 0
    pdf_path = None
    if pdfs_after:
        pdf_path = pdfs_after[0]
        pdf_size = pdf_path.stat().st_size

    output_checks = {
        "pdf_created": pdf_found,
        "pdf_referenced": pdf_path_in_output,
        "pdf_size_ok": pdf_size > 1024,  # At least 1KB
    }

    all_checks = {**content_checks, **output_checks}
    quality = sum(all_checks.values()) / len(all_checks) * 100

    # ── Display results ──
    status = f"{G}PASS{E}" if quality >= 50 else f"{R}FAIL{E}"
    print(f"\n{B}Result: {status}{E}")
    print(f"  Time: {elapsed:.1f}s | Output: {len(output):,} chars | Quality: {quality:.0f}%")

    print(f"\n  {B}Content checks:{E}")
    for k, v in content_checks.items():
        print(f'    {G + "✓" + E if v else R + "✗" + E} {k}')

    print(f"\n  {B}PDF output checks:{E}")
    for k, v in output_checks.items():
        print(f'    {G + "✓" + E if v else R + "✗" + E} {k}')

    if pdf_path:
        print(f"\n  {G}PDF: {pdf_path}{E}")
        print(f"  {D}Size: {pdf_size:,} bytes ({pdf_size / 1024:.1f} KB){E}")
    else:
        print(f"\n  {Y}No PDF file found in output directories{E}")
        print(f"  {D}Searched: {OUTPUT_DIR}, ~/jotty/reports, /tmp{E}")

    # Show execution trail (last 10 events)
    print(f"\n  {B}Execution trail:{E}")
    for t, stage, detail in trail[-10:]:
        print(f"  {D}  [{t:5.1f}s] {stage}: {detail}{E}")

    # Output preview
    print(f"\n{D}Output preview (first 600 chars):{E}")
    preview = output[:600].replace("\n", "\n  ")
    print(f"  {preview}...")

    return ("Research → PDF", elapsed, quality, len(output), pdf_path, pdf_size)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print(f'{B}{"═"*60}{E}')
    print(f"{B}  JOTTY V2 — REAL-WORLD PDF REPORT TEST{E}")
    print(f"{D}  Tests full pipeline: research → synthesize → PDF{E}")
    print(f'{D}  Time: {time.strftime("%Y-%m-%d %H:%M:%S")}{E}')
    print(f'{B}{"═"*60}{E}')

    name, elapsed, quality, chars, pdf_path, pdf_size = await test_research_to_pdf()

    # Final summary
    print(f'\n{B}{"═"*60}{E}')
    print(f"{B}  FINAL SUMMARY{E}")
    print(f'{B}{"═"*60}{E}')
    print(f"  Task:    {name}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"  Output:  {chars:,} chars")
    print(f"  Quality: {quality:.0f}%")
    if pdf_path:
        print(f"  PDF:     {pdf_path} ({pdf_size / 1024:.1f} KB)")
    else:
        print(f"  PDF:     {R}not generated{E}")

    if quality >= 50:
        print(f"\n  {G}PASSED — Orchestrator end-to-end real-world test{E}")
    else:
        print(f"\n  {R}FAILED — quality {quality:.0f}% below 50% threshold{E}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
