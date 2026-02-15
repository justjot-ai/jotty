#!/usr/bin/env python3
"""
LLM Provider Benchmark
======================

Compares speed/quality across:
  1. DirectAnthropicLM (Haiku)  — cheapest, fastest Anthropic
  2. DirectAnthropicLM (Sonnet) — current default, best quality
  3. dspy.LM via OpenRouter     — free/cheap models (GLM, Llama, etc.)
  4. PersistentClaudeCLI        — subprocess fallback (baseline)

Usage:
  python tests/test_llm_benchmark.py
  OPENROUTER_API_KEY=sk-or-... python tests/test_llm_benchmark.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

# ── Load API keys ──
for env_file in [Path(__file__).parents[2] / '.env.anthropic',
                 Path(__file__).parents[1] / '.env.anthropic',
                 Path(__file__).parents[1] / '.env']:
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if v and k not in os.environ:
                        os.environ[k] = v

import logging
logging.basicConfig(level=logging.WARNING)

B = '\033[1m'; G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; D = '\033[2m'; E = '\033[0m'

# ── Test prompts ──
PROMPTS = {
    "simple": "What is the capital of France? Answer in one sentence.",
    "reasoning": "Explain 3 key differences between REST and GraphQL APIs. Be concise.",
    "code": "Write a Python function that finds the longest palindromic substring. Include type hints.",
}


def measure_call(lm, prompt: str, label: str) -> dict:
    """Call LM, measure time and output quality."""
    t0 = time.time()
    try:
        result = lm(prompt=prompt)
        elapsed = time.time() - t0
        text = result[0] if isinstance(result, list) else str(result)
        return {
            "label": label,
            "success": True,
            "time": elapsed,
            "output_len": len(text),
            "output_preview": text[:150].replace('\n', ' '),
            "usage": lm.history[-1].get("usage", {}) if hasattr(lm, 'history') and lm.history else {},
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "label": label,
            "success": False,
            "time": elapsed,
            "error": str(e)[:100],
        }


def benchmark_provider(name: str, lm, prompts: dict):
    """Run all prompts against a provider."""
    print(f"\n{B}{'─'*60}{E}")
    print(f"{B}  Provider: {name}{E}")
    print(f"{B}  Model: {getattr(lm, 'model_id', getattr(lm, 'model', '?'))}{E}")
    print(f"{B}{'─'*60}{E}")

    results = []
    for pname, prompt in prompts.items():
        r = measure_call(lm, prompt, f"{name}/{pname}")
        results.append(r)
        if r["success"]:
            usage_str = ""
            if r.get("usage"):
                u = r["usage"]
                usage_str = f" | tokens: {u.get('input_tokens','?')}→{u.get('output_tokens','?')}"
            print(f"  {G}✓{E} {pname:12s}  {r['time']:5.2f}s  {r['output_len']:5d} chars{usage_str}")
            print(f"    {D}{r['output_preview'][:100]}...{E}")
        else:
            print(f"  {R}✗{E} {pname:12s}  {r['time']:5.2f}s  ERROR: {r.get('error','')}")

    avg_time = sum(r["time"] for r in results if r["success"]) / max(1, sum(1 for r in results if r["success"]))
    total_chars = sum(r.get("output_len", 0) for r in results if r["success"])
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    print(f"\n  {Y}Summary:{E} avg={avg_time:.2f}s | total_chars={total_chars} | success={success_rate:.0%}")
    return results


def main():
    print(f"\n{B}{'═'*60}{E}")
    print(f"{B}   JOTTY LLM PROVIDER BENCHMARK{E}")
    print(f"{B}{'═'*60}{E}")

    all_results = {}

    # ── 1. DirectAnthropicLM (Haiku) ──
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from Jotty.core.infrastructure.foundation.direct_anthropic_lm import DirectAnthropicLM
            lm = DirectAnthropicLM(model="haiku", max_tokens=1024)
            all_results["anthropic-haiku"] = benchmark_provider(
                "DirectAnthropicLM (Haiku)", lm, PROMPTS
            )
        except Exception as e:
            print(f"\n  {R}Skip Haiku: {e}{E}")

        # ── 2. DirectAnthropicLM (Sonnet) ──
        try:
            lm = DirectAnthropicLM(model="sonnet", max_tokens=1024)
            all_results["anthropic-sonnet"] = benchmark_provider(
                "DirectAnthropicLM (Sonnet)", lm, PROMPTS
            )
        except Exception as e:
            print(f"\n  {R}Skip Sonnet: {e}{E}")
    else:
        print(f"\n  {Y}⚠ ANTHROPIC_API_KEY not set — skipping DirectAnthropicLM{E}")

    # ── 3. OpenRouter (free models) ──
    if os.environ.get("OPENROUTER_API_KEY"):
        import dspy

        # Test multiple free/cheap models
        openrouter_models = [
            ("meta-llama/llama-3.3-70b-instruct:free", "Llama-3.3-70B (free)"),
            ("google/gemma-2-9b-it:free", "Gemma-2-9B (free)"),
            ("qwen/qwen-2.5-72b-instruct:free", "Qwen-2.5-72B (free)"),
        ]

        for model_id, label in openrouter_models:
            try:
                lm = dspy.LM(
                    f"openrouter/{model_id}",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    max_tokens=1024,
                )
                all_results[f"openrouter-{label}"] = benchmark_provider(
                    f"OpenRouter: {label}", lm, PROMPTS
                )
            except Exception as e:
                print(f"\n  {R}Skip {label}: {e}{E}")
    else:
        print(f"\n  {Y}⚠ OPENROUTER_API_KEY not set — skipping OpenRouter models{E}")
        print(f"  {D}Get free key at: https://openrouter.ai/keys{E}")
        print(f"  {D}Then run: OPENROUTER_API_KEY=sk-or-... python tests/test_llm_benchmark.py{E}")

    # ── 4. PersistentClaudeCLI (subprocess baseline) ──
    import shutil
    if shutil.which("claude"):
        try:
            from Jotty.core.infrastructure.foundation.persistent_claude_lm import PersistentClaudeCLI
            lm = PersistentClaudeCLI(model="haiku")
            # Only test simple prompt — CLI is slow
            print(f"\n{B}{'─'*60}{E}")
            print(f"{B}  Provider: PersistentClaudeCLI (Haiku) — baseline{E}")
            print(f"{B}{'─'*60}{E}")
            r = measure_call(lm, PROMPTS["simple"], "cli/simple")
            if r["success"]:
                print(f"  {G}✓{E} simple  {r['time']:5.2f}s  {r['output_len']:5d} chars")
            else:
                print(f"  {R}✗{E} simple  {r['time']:5.2f}s  ERROR: {r.get('error','')}")
            all_results["cli-haiku"] = [r]
        except Exception as e:
            print(f"\n  {R}Skip CLI: {e}{E}")
    else:
        print(f"\n  {Y}⚠ claude CLI not found — skipping CLI baseline{E}")

    # ── Final comparison ──
    print(f"\n\n{B}{'═'*60}{E}")
    print(f"{B}   COMPARISON SUMMARY{E}")
    print(f"{B}{'═'*60}{E}")
    print(f"\n  {'Provider':<35s} {'Avg Time':>10s} {'Success':>10s}")
    print(f"  {'─'*55}")

    for name, results in all_results.items():
        successes = [r for r in results if r.get("success")]
        if successes:
            avg = sum(r["time"] for r in successes) / len(successes)
            rate = len(successes) / len(results)
            marker = G + "★" + E if avg < 2.0 and rate == 1.0 else ""
            print(f"  {name:<35s} {avg:>8.2f}s {rate:>9.0%}  {marker}")
        else:
            print(f"  {name:<35s} {'FAILED':>10s}")

    print(f"\n  {D}★ = recommended (fast + reliable){E}")
    print()


if __name__ == "__main__":
    main()
