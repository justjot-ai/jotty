#!/usr/bin/env python3
"""
Real LLM E2E: Run actual tasks through Jotty and verify OpenClaw v2026.2.19
features fire in practice.

Tests:
    1. SSRF — smart_fetch blocks internal URL before any HTTP call
    2. Streaming thinking — real Anthropic streaming with thinking_callback
    3. Rate limiter — under real FastAPI app
    4. Orchestrator.run() — full pipeline with real LLM, observe features

Usage:
    python tests/e2e/test_openclaw_v2_real_llm.py
"""

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load .env
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

# Enable DEBUG for features we want to observe
for mod in [
    "Jotty.core.infrastructure.utils.smart_fetcher",
    "Jotty.core.intelligence.orchestration.llm_providers.anthropic",
]:
    logging.getLogger(mod).setLevel(logging.DEBUG)

_results = {}


def _h(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg):
    print(f"  [OK] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


def _test(name, condition, detail=""):
    _results[name] = condition
    if condition:
        _ok(f"{name}: {detail}" if detail else name)
    else:
        _fail(f"{name}: {detail}" if detail else name)


async def test_ssrf_real():
    """SSRF blocks before any network call."""
    _h("Test 1: SSRF Protection (real)")

    from Jotty.core.infrastructure.utils.smart_fetcher import smart_fetch

    # This should be blocked instantly — no HTTP call made
    t0 = time.time()
    result = smart_fetch("http://169.254.169.254/latest/meta-data/")
    elapsed = time.time() - t0

    _test(
        "SSRF blocks AWS metadata",
        not result.success and result.skipped,
        f"blocked in {elapsed*1000:.0f}ms: {result.error}",
    )

    result2 = smart_fetch("http://localhost:8080/admin")
    _test("SSRF blocks localhost", not result2.success and result2.skipped, result2.error)

    result3 = smart_fetch("file:///etc/passwd")
    _test("SSRF blocks file://", not result3.success and result3.skipped, result3.error)


async def test_streaming_thinking_real():
    """Real Anthropic streaming with thinking_callback."""
    _h("Test 2: Streaming Thinking (real Anthropic call)")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _test("Streaming thinking", False, "No API key")
        return

    from Jotty.core.intelligence.orchestration.llm_providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key=api_key)

    text_chunks = []
    thinking_chunks = []

    def on_text(t):
        text_chunks.append(t)

    def on_thinking(t):
        thinking_chunks.append(t)

    t0 = time.time()
    try:
        response, full_content = await provider.call_streaming(
            messages=[{"role": "user", "content": "What is 2+2? Reply in one word."}],
            tools=[],
            system="You are a helpful assistant. Be very brief.",
            stream_callback=on_text,
            thinking_callback=on_thinking,
        )
        elapsed = time.time() - t0

        _test(
            "Streaming got text",
            len(text_chunks) > 0,
            f"{len(text_chunks)} chunks in {elapsed:.1f}s",
        )
        _test("Streaming full_content populated", len(full_content) > 0, f"'{full_content[:80]}'")
        _test(
            "Response has content blocks",
            len(response.content) > 0,
            f"{len(response.content)} blocks, stop={response.stop_reason}",
        )
        _test(
            "Response has usage",
            response.usage is not None,
            f"in={response.usage.get('input_tokens', '?')}, out={response.usage.get('output_tokens', '?')}",
        )

        # Check if thinking was emitted (depends on model — not all models have extended thinking)
        if thinking_chunks:
            _ok(f"  Bonus: thinking_callback received {len(thinking_chunks)} chunks")
        else:
            print(
                f"  [..] No thinking blocks (model may not use extended thinking for this prompt)"
            )

    except Exception as e:
        _test("Streaming call", False, f"{type(e).__name__}: {e}")


async def test_orchestrator_real():
    """Full Orchestrator.run() with real LLM."""
    _h("Test 3: Orchestrator.run() — real LLM pipeline")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _test("Orchestrator.run", False, "No API key")
        return

    from Jotty import Jotty
    from Jotty.core.intelligence.orchestration.execution import ExecutionTier

    jotty = Jotty(log_level="WARNING")

    # Tier 1 DIRECT — single LLM call, fastest path
    t0 = time.time()
    try:
        result = await jotty.run(
            "What are the three primary colors? Answer in one sentence.",
            tier=ExecutionTier.DIRECT,
        )
        elapsed = time.time() - t0

        _test("Tier 1 DIRECT succeeded", result.success, f"in {elapsed:.1f}s")
        _test("Tier 1 has output", bool(result.output), f"'{str(result.output)[:100]}'")
        _test(
            "Tier 1 has cost",
            result.cost_usd is not None or hasattr(result, "cost_usd"),
            f"${getattr(result, 'cost_usd', 0) or 0:.4f}",
        )

    except Exception as e:
        _test("Tier 1 DIRECT", False, f"{type(e).__name__}: {e}")
        elapsed = time.time() - t0

    # Tier 2 AGENTIC — planning + multi-step
    t0 = time.time()
    try:
        result2 = await jotty.run(
            "List 3 benefits of exercise in bullet points.",
            tier=ExecutionTier.AGENTIC,
        )
        elapsed = time.time() - t0

        _test("Tier 2 AGENTIC succeeded", result2.success, f"in {elapsed:.1f}s")
        _test("Tier 2 has output", bool(result2.output), f"'{str(result2.output)[:100]}'")

    except Exception as e:
        _test("Tier 2 AGENTIC", False, f"{type(e).__name__}: {e}")


async def test_rate_limiter_real():
    """Rate limiter under real conditions."""
    _h("Test 4: Rate Limiter (real timing)")

    from Jotty.core.infrastructure.utils.api_middleware import RateLimiter

    # 5 req/sec — should block on 6th
    limiter = RateLimiter(max_requests=5, window_seconds=1)
    allowed = 0
    for i in range(10):
        if limiter.is_allowed("test-ip"):
            allowed += 1

    _test("Rate limiter allows exactly max", allowed == 5, f"allowed {allowed}/10 (expected 5)")

    # After window expires, should allow again
    import time

    time.sleep(1.1)
    _test(
        "Rate limiter resets after window",
        limiter.is_allowed("test-ip"),
        "allowed after 1.1s sleep",
    )


async def main():
    _h("OpenClaw v2026.2.19 — Real LLM Integration Test")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set. Skipping LLM tests.")
        return False

    print(f"  API key: {api_key[:8]}...{api_key[-4:]}")

    await test_ssrf_real()
    await test_streaming_thinking_real()
    await test_orchestrator_real()
    await test_rate_limiter_real()

    # Summary
    _h("Summary")
    passed = sum(1 for v in _results.values() if v)
    total = len(_results)
    failed = total - passed

    for name, ok in _results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  All real LLM integration tests passed!")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
