#!/usr/bin/env python3
"""
Orchestrator V2 — Complex Use Case Tests
=========================================

Exercises Orchestrator on multi-step, comparison, creation, and analysis tasks:
  1. Multi-step: research → summarize → save to file
  2. Comparison: compare 3+ items with table and recommendation
  3. Creation: generate artifact with dependencies (e.g. code + instructions)
  4. Analysis: structured list with criteria and ranking
  5. Mixed: search + summarize + suggest follow-up

Run:
  pytest tests/test_swarm_manager_complex.py -v -s
  python tests/test_swarm_manager_complex.py
"""

import asyncio
import os
import time
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

# Load API keys from project root or tests dir
for env_file in [
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parents[1] / ".env.anthropic",
    Path(__file__).resolve().parents[0] / ".env",
]:
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if v and k not in os.environ:
                        os.environ[k] = v

import logging
import pytest

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("swarm_complex")
logger.setLevel(logging.INFO)

B = "\033[1m"
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
D = "\033[2m"
E = "\033[0m"


def extract_output(result) -> str:
    """Extract text output from EpisodeResult or ExecutionResult."""
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(
            result.get("final_output")
            or result.get("output")
            or result.get("result")
            or result.get("text", "")
        )
    out = getattr(result, "output", result)
    if hasattr(out, "outputs") and isinstance(getattr(out, "outputs"), dict):
        parts = []
        for k, v in (out.outputs or {}).items():
            if isinstance(v, dict):
                if v.get("text"):
                    parts.append(str(v["text"]))
                elif v.get("path") or v.get("output_path"):
                    parts.append(f"[{k}: {v.get('path') or v.get('output_path')}]")
                elif "code" in v:
                    parts.append(str(v.get("code", "")))
            elif isinstance(v, str):
                parts.append(v)
        if parts:
            return "\n".join(parts)
        # Fallback: stringify key artifacts so assertions can see path/code hints
        desc = " ".join(
            f"{k}={v.get('path', v.get('output_path', v))}"
            for k, v in (out.outputs or {}).items()
            if isinstance(v, dict)
        )
        if desc:
            return desc
    return str(out)


def _status_trail():
    trail = []
    t0 = time.time()

    def cb(stage: str, detail: str = ""):
        trail.append((time.time() - t0, stage, (detail or "")[:90]))

    return trail, t0, cb


# ══════════════════════════════════════════════════════════════
# 1. Multi-step: Research → Summarize → Save
# ══════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_multistep_research_summarize_save():
    """Multi-step: research a topic, summarize, and save to file."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator

    goal = (
        "Research 'Python 3.12 new features' (use web search if needed), "
        "write a short summary in 3–5 bullet points, and save it to a file named python312_summary.md. "
        "Do not skip any step."
    )
    trail, t0, status_cb = _status_trail()
    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)

    result = await sm.run(
        goal,
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=status_cb,
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    assert result is not None, "Result should not be None"
    assert getattr(result, "success", True) or len(output) > 100, "Should succeed or return substantial output"
    assert "python" in output.lower() or "3.12" in output or "feature" in output.lower(), "Output should mention Python 3.12 or features"

    logger.info(f"Multi-step research: {elapsed:.0f}s, {len(output)} chars, success={getattr(result, 'success', None)}")
    for t, stage, detail in trail[-6:]:
        logger.info(f"  [{t:.1f}s] {stage}: {detail}")
    return result


# ══════════════════════════════════════════════════════════════
# 2. Comparison with structured output
# ══════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_comparison_three_items_table():
    """Compare 3 items with table and recommendation."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator

    goal = (
        "Compare these three Python web frameworks: FastAPI, Flask, and Django. "
        "Include: main use case, learning curve, performance, and ecosystem. "
        "Output a markdown table and one final recommendation for a new startup building a REST API."
    )
    trail, t0, status_cb = _status_trail()
    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)

    result = await sm.run(
        goal,
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=status_cb,
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    assert result is not None
    checks = {
        "FastAPI": "fastapi" in output.lower(),
        "Flask": "flask" in output.lower(),
        "Django": "django" in output.lower(),
        "table_or_structured": "|" in output or "table" in output.lower() or "---" in output,
        "recommendation": any(w in output.lower() for w in ["recommend", "best", "choose", "suggest"]),
    }
    quality = sum(checks.values()) / len(checks) * 100
    assert quality >= 40, f"Comparison quality too low: {checks}"

    logger.info(f"Comparison: {elapsed:.0f}s, quality={quality:.0f}%, {checks}")
    return result


# ══════════════════════════════════════════════════════════════
# 3. Creation with dependency (code + instructions)
# ══════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_creation_code_and_docs():
    """Create a small script and brief usage instructions."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator

    goal = (
        "Create a minimal Python script that reads a JSON file path from the command line, "
        "loads the JSON, and prints the keys of the top-level object. "
        "Add a short comment at the top with usage: python script.py <file.json>. "
        "Output the full script as the main result."
    )
    trail, t0, status_cb = _status_trail()
    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)

    result = await sm.run(
        goal,
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=status_cb,
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    assert result is not None
    # Swarm may return ExecutionResult with outputs (e.g. script path) or inline code
    has_json = "json" in output.lower() or "load" in output.lower()
    has_code = "def " in output or "import " in output or "argparse" in output or "sys" in output
    has_artifact = "script.py" in output or "path=" in output or ".py" in output
    assert has_json or has_code or has_artifact, "Should contain JSON handling, code, or script artifact path"

    logger.info(f"Creation code+docs: {elapsed:.0f}s, {len(output)} chars")
    return result


# ══════════════════════════════════════════════════════════════
# 4. Analysis: structured list with criteria
# ══════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_analysis_ranked_list():
    """Produce a ranked list with brief explanations."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator

    goal = (
        "List the top 5 risks of using AI in healthcare, in order of severity. "
        "For each risk give a one-sentence explanation and one mitigation. "
        "Use numbered list and clear headings."
    )
    trail, t0, status_cb = _status_trail()
    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)

    result = await sm.run(
        goal,
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=status_cb,
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    assert result is not None
    assert "risk" in output.lower() or "health" in output.lower() or "ai" in output.lower(), "Should mention AI/healthcare/risk"
    assert any(c in output for c in ["1.", "2.", "3.", "1)", "#"]) or "first" in output.lower(), "Should be a list"

    logger.info(f"Analysis ranked list: {elapsed:.0f}s, {len(output)} chars")
    return result


# ══════════════════════════════════════════════════════════════
# 5. Mixed: search + summarize + follow-up
# ══════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_mixed_search_summarize_followup():
    """Search (or reason), summarize, and suggest a follow-up."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator

    goal = (
        "What are the main benefits of TypeScript over JavaScript for large codebases? "
        "Summarize in 3 bullet points and at the end suggest one follow-up question a developer might ask."
    )
    trail, t0, status_cb = _status_trail()
    sm = Orchestrator(enable_lotus=False, enable_zero_config=False)

    result = await sm.run(
        goal,
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=status_cb,
    )
    elapsed = time.time() - t0
    output = extract_output(result)

    assert result is not None
    assert "typescript" in output.lower() or "javascript" in output.lower(), "Should mention TS/JS"
    assert "bullet" in output.lower() or "•" in output or "- " in output or "1." in output or "benefit" in output.lower(), "Should have list or benefits"

    logger.info(f"Mixed search+summarize: {elapsed:.0f}s, {len(output)} chars")
    return result


# ══════════════════════════════════════════════════════════════
# Run as script
# ══════════════════════════════════════════════════════════════
async def _run_all():
    cases = [
        ("Multi-step research → summarize → save", test_multistep_research_summarize_save),
        ("Comparison (3 items + table + recommendation)", test_comparison_three_items_table),
        ("Creation (code + instructions)", test_creation_code_and_docs),
        ("Analysis (ranked list)", test_analysis_ranked_list),
        ("Mixed (search + summarize + follow-up)", test_mixed_search_summarize_followup),
    ]
    print(f"\n{B}{'═'*60}{E}")
    print(f"{B}  Orchestrator V2 — Complex Use Cases{E}")
    print(f"{D}  {len(cases)} scenarios (enable_lotus=False, skip_autonomous_setup=True where used){E}")
    print(f"{B}{'═'*60}{E}\n")

    results = []
    for name, test_fn in cases:
        print(f"\n{B}▶ {name}{E}")
        try:
            result = await test_fn()
            results.append((name, True, result))
            print(f"  {G}PASS{E}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  {R}FAIL{E}: {e}")
            logger.exception(e)

    print(f"\n{B}{'═'*60}{E}")
    print(f"{B}  Summary{E}")
    print(f"{B}{'═'*60}{E}")
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, _ in results:
        print(f"  {G+'✓'+E if ok else R+'✗'+E} {name}")
    print(f"\n  {passed}/{len(results)} passed\n")


if __name__ == "__main__":
    asyncio.run(_run_all())
