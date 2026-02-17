"""
Super-complex real-world E2E test for Orchestrator.

Exercises the FULL stack:
- Orchestrator.run() and .chat() with real Anthropic Claude
- learn=True (records to LearningService + SwarmLearningPipeline)
- learn=False (skips learning — verifies no side effects)
- Multi-turn conversation with context carry
- Streaming with token-level events
- Complex multi-step reasoning requiring structured output
- LearningService verification (episodes recorded, queryable)

Requires: ANTHROPIC_API_KEY in .env or .env.anthropic
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(test_name: str, content: str, metadata: dict = None) -> None:
    """Save test output to results/ folder for review."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{test_name}.md"
    path = RESULTS_DIR / filename

    header = f"# {test_name}\n\n"
    header += f"**Timestamp:** {datetime.now().isoformat()}\n"
    if metadata:
        for k, v in metadata.items():
            header += f"**{k}:** {v}\n"
    header += "\n---\n\n"

    path.write_text(header + content, encoding="utf-8")
    print(f"  [Saved] {path}")


# Load env
try:
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
    anthropic_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env.anthropic")
    if os.path.exists(anthropic_env):
        load_dotenv(anthropic_env, override=True)
except ImportError:
    pass

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
pytestmark = [
    pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"),
    pytest.mark.e2e,
    pytest.mark.timeout(120),
]


def _get_orchestrator():
    """Create an Orchestrator wired for real Anthropic execution."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = object.__new__(Orchestrator)
    orch.config = type(
        "C",
        (),
        {
            "domain": "general",
            "base_path": None,
            "learning_wait_timeout_seconds": 0,
        },
    )()
    orch.agents = []
    orch.mode = "single"
    orch.runners = {}
    orch._runners_built = False
    orch._efficiency_stats = {}
    orch._intelligence_metrics = {}
    orch._engine = None
    orch._learning_ready = asyncio.Event()
    orch._learning_ready.set()
    return orch


# ============================================================================
# Test 1: Complex analytical reasoning via chat()
# ============================================================================


class TestComplexChat:
    """Multi-turn analytical reasoning with real LLM."""

    @pytest.mark.asyncio
    async def test_complex_analytical_reasoning(self):
        """
        Task: Multi-step financial analysis requiring:
        1. Understanding compound interest formula
        2. Applying it to a real scenario with multiple variables
        3. Comparing two strategies
        4. Providing a recommendation with reasoning

        This tests: deep reasoning, structured output, mathematical accuracy.
        """
        orch = _get_orchestrator()

        result = await orch.chat(
            message=(
                "I have $50,000 to invest for 20 years. Compare these two strategies:\n"
                "Strategy A: Put everything in an index fund returning 9.5% annually, compounded monthly.\n"
                "Strategy B: Split 70/30 — 70% in the index fund (9.5%), "
                "30% in bonds returning 4.2% annually, compounded quarterly. "
                "Rebalance annually.\n\n"
                "For each strategy, calculate:\n"
                "1. Final value after 20 years\n"
                "2. Total return percentage\n"
                "3. Risk-adjusted return (assume index fund Sharpe ratio 0.5, bonds 0.3)\n"
                "4. Maximum drawdown estimate (index: -50%, bonds: -10%)\n\n"
                "Then recommend which strategy and explain WHY considering "
                "a 45-year-old planning for retirement at 65.\n\n"
                "Do NOT use any tools — calculate everything from your knowledge."
            ),
            provider="anthropic",
            learn=True,
        )

        content = getattr(result, "content", str(result))
        assert content, "Expected non-empty response"
        assert len(content) > 500, f"Expected detailed analysis, got {len(content)} chars"

        content_lower = content.lower()
        assert any(
            w in content_lower for w in ["strategy a", "strategy b", "index fund"]
        ), "Expected comparison of strategies"
        assert any(
            w in content_lower for w in ["recommend", "suggest", "advise"]
        ), "Expected a recommendation"
        assert any(
            w in content_lower for w in ["compound", "return", "interest", "growth"]
        ), "Expected financial terminology"

        print(f"\n[PASS] Complex analytical chat: {len(content)} chars")
        print(f"  First 200 chars: {content[:200]}...")
        _save_result("complex_financial_analysis", content, {"chars": len(content)})

    @pytest.mark.asyncio
    async def test_multi_turn_with_context_carry(self):
        """
        Multi-turn conversation where each turn builds on previous context.
        Tests: history handling, context retention, progressive reasoning.
        """
        orch = _get_orchestrator()

        # Turn 1: Set up a scenario (concise to avoid tool-calling timeouts)
        r1 = await orch.chat(
            message=(
                "You are a distributed systems architect. Propose a high-level "
                "architecture for a real-time IoT event processing system handling "
                "500K events/sec across 3 continents. Name specific technologies. "
                "Keep it to 3-4 paragraphs. Do NOT use any tools."
            ),
            provider="anthropic",
            learn=True,
        )

        c1 = getattr(r1, "content", str(r1))
        assert len(c1) > 200, "Expected detailed architecture proposal"

        # Turn 2: Challenge with edge cases
        history = [
            {"role": "user", "content": "Design a 500K events/sec IoT system"},
            {"role": "assistant", "content": c1[:1500]},
        ]

        r2 = await orch.chat(
            message=(
                "Now consider: what happens if an entire AWS region goes down "
                "during peak traffic? And how do you handle a rogue sensor "
                "sending 100K malformed events/second? 2-3 paragraphs, no tools."
            ),
            history=history,
            provider="anthropic",
            learn=True,
        )

        c2 = getattr(r2, "content", str(r2))
        assert len(c2) > 200, "Expected detailed failure analysis"
        assert any(
            w in c2.lower()
            for w in [
                "failover",
                "replica",
                "redundan",
                "backup",
                "region",
                "multi-region",
                "rate limit",
                "filter",
                "circuit",
            ]
        ), "Expected failure handling discussion"

        print(f"\n[PASS] 2-turn architecture discussion:")
        print(f"  Turn 1: {len(c1)} chars (architecture)")
        print(f"  Turn 2: {len(c2)} chars (failure handling)")
        _save_result(
            "multi_turn_architecture",
            (
                "## Turn 1: Architecture Proposal\n\n"
                + c1
                + "\n\n---\n\n## Turn 2: Failure Handling\n\n"
                + c2
            ),
            {"turn1_chars": len(c1), "turn2_chars": len(c2)},
        )


# ============================================================================
# Test 2: Streaming with complex output
# ============================================================================


class TestComplexStreaming:
    """Real-time streaming with complex structured output."""

    @pytest.mark.asyncio
    async def test_streaming_complex_analysis(self):
        """
        Stream a complex analysis and verify:
        - Events arrive incrementally
        - Total content is substantial
        - Streaming completes without errors

        StreamEvent has .type ('text', 'tool_start', 'tool_end', 'complete', 'error')
        and .data (the content string for type='text').
        """
        orch = _get_orchestrator()

        stream = await orch.chat(
            message=(
                "Compare PostgreSQL vs MongoDB vs Cassandra for a social media "
                "platform with 100M users. For each, evaluate: write throughput, "
                "read latency, and operational complexity. Be concise but specific "
                "with numbers. Do NOT use any tools — answer from your knowledge."
            ),
            stream=True,
            provider="anthropic",
            learn=True,
        )

        events = []
        content_chunks = []
        start = time.time()
        first_event_time = None

        async for event in stream:
            events.append(event)
            if first_event_time is None:
                first_event_time = time.time()

            # StreamEvent has .type and .data
            if hasattr(event, "type") and hasattr(event, "data"):
                if event.type == "text" and event.data:
                    content_chunks.append(str(event.data))
            elif isinstance(event, dict):
                if event.get("type") == "text" and event.get("data"):
                    content_chunks.append(str(event["data"]))
                elif event.get("content"):
                    content_chunks.append(event["content"])

        elapsed = time.time() - start
        ttfb = (first_event_time - start) if first_event_time else elapsed
        full_content = "".join(content_chunks)

        assert len(events) >= 2, f"Expected streaming events, got {len(events)}"
        assert (
            len(full_content) > 200
        ), f"Expected substantial content, got {len(full_content)} chars"
        # Verify we got both text and complete events
        event_types = [getattr(e, "type", None) for e in events]
        assert "text" in event_types, f"Expected 'text' event, got types: {event_types}"
        assert "complete" in event_types, f"Expected 'complete' event, got types: {event_types}"

        print(f"\n[PASS] Streaming complex analysis:")
        print(f"  Events: {len(events)} (text: {len(content_chunks)})")
        print(f"  Content: {len(full_content)} chars")
        print(f"  TTFB: {ttfb:.2f}s")
        print(f"  Total: {elapsed:.2f}s")
        print(f"  First 200 chars: {full_content[:200]}...")
        _save_result(
            "streaming_db_comparison",
            full_content,
            {
                "events": len(events),
                "ttfb_s": f"{ttfb:.2f}",
                "total_s": f"{elapsed:.2f}",
            },
        )


# ============================================================================
# Test 3: learn=True vs learn=False verification
# ============================================================================


class TestLearnFlag:
    """Verify learn flag controls LearningService recording."""

    @pytest.mark.asyncio
    async def test_learn_true_records_episode(self):
        """learn=True should record to LearningService (verified via SQLite count)."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()
        baseline = ls._store.get_episode_count()

        orch = _get_orchestrator()
        result = await orch.chat(
            message="What is the capital of France? One word answer.",
            provider="anthropic",
            learn=True,
        )

        content = getattr(result, "content", str(result))
        assert "paris" in content.lower(), f"Expected 'Paris', got: {content[:100]}"

        new_count = ls._store.get_episode_count()
        assert (
            new_count > baseline
        ), f"Expected episode count to increase: {baseline} -> {new_count}"
        print(f"\n[PASS] learn=True recorded episode: {baseline} -> {new_count}")

    @pytest.mark.asyncio
    async def test_learn_false_skips_recording(self):
        """learn=False should NOT record to LearningService."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()
        baseline = ls._store.get_episode_count()

        orch = _get_orchestrator()
        result = await orch.chat(
            message="What is 2 + 2? One number answer.",
            provider="anthropic",
            learn=False,
        )

        content = getattr(result, "content", str(result))
        assert "4" in content, f"Expected '4', got: {content[:100]}"

        new_count = ls._store.get_episode_count()
        assert new_count == baseline, f"learn=False should not record: {baseline} -> {new_count}"
        print(f"\n[PASS] learn=False skipped recording: count stayed at {baseline}")


# ============================================================================
# Test 4: Super complex multi-domain reasoning
# ============================================================================


class TestSuperComplexReasoning:
    """The hardest test: requires cross-domain synthesis."""

    @pytest.mark.asyncio
    async def test_cross_domain_synthesis(self):
        """
        A genuinely hard problem requiring synthesis across:
        - Computer science (algorithms, complexity)
        - Economics (game theory, mechanism design)
        - Biology (evolutionary strategies)
        - Mathematics (optimization, probability)

        This tests whether the LLM can connect ideas across domains
        and produce original synthesis — not just regurgitate.
        """
        orch = _get_orchestrator()

        result = await orch.chat(
            message=(
                "I need a novel solution to this problem:\n\n"
                "A decentralized marketplace has 10,000 sellers and 1M buyers. "
                "Sellers can lie about product quality. Buyers can dispute transactions "
                "fraudulently. Traditional escrow is too slow (48h) and too expensive (5% fee).\n\n"
                "Design a mechanism that:\n"
                "1. Settles disputes in <1 minute with >95% accuracy\n"
                "2. Costs <0.1% per transaction\n"
                "3. Is resistant to collusion between buyers and sellers\n"
                "4. Incentivizes honest behavior without requiring trust\n"
                "5. Scales to 1M transactions/day\n\n"
                "Draw inspiration from:\n"
                "- Biological immune systems (self/non-self recognition)\n"
                "- Game theory (mechanism design, VCG auctions)\n"
                "- Distributed consensus (Byzantine fault tolerance)\n"
                "- Machine learning (anomaly detection, reputation systems)\n\n"
                "Provide:\n"
                "A. The core mechanism (how it works, step by step)\n"
                "B. Mathematical proof sketch of incentive compatibility\n"
                "C. Computational complexity analysis\n"
                "D. Attack vectors and mitigations\n"
                "E. Why this is better than existing solutions (Kleros, Aragon Court, eBay)\n\n"
                "Do NOT use any tools — answer entirely from your knowledge."
            ),
            provider="anthropic",
            learn=True,
        )

        content = getattr(result, "content", str(result))
        assert len(content) > 1000, f"Expected deep analysis (>1000 chars), got {len(content)}"

        content_lower = content.lower()
        domain_hits = sum(
            [
                any(
                    w in content_lower
                    for w in ["incentive", "mechanism", "nash", "equilibrium", "game"]
                ),
                any(w in content_lower for w in ["byzantine", "consensus", "fault", "distributed"]),
                any(w in content_lower for w in ["complexity", "algorithm", "o(n)", "scalab"]),
                any(w in content_lower for w in ["reputation", "anomaly", "detection", "learning"]),
            ]
        )
        assert domain_hits >= 3, f"Expected cross-domain synthesis (hit {domain_hits}/4 domains)"

        sections = sum(
            [
                any(marker in content for marker in ["A.", "A)", "**A", "### A", "Step"]),
                any(
                    marker in content_lower
                    for marker in ["proof", "mathematical", "incentive compat"]
                ),
                any(marker in content_lower for marker in ["complexity", "computational", "o("]),
                any(
                    marker in content_lower
                    for marker in ["attack", "vector", "mitigation", "collusion"]
                ),
            ]
        )
        assert sections >= 3, f"Expected structured response with sections (hit {sections}/4)"

        print(
            f"\n[PASS] Cross-domain synthesis: {len(content)} chars, {domain_hits}/4 domains, {sections}/4 sections"
        )
        print(f"  First 300 chars: {content[:300]}...")
        _save_result(
            "cross_domain_synthesis",
            content,
            {
                "chars": len(content),
                "domains_hit": f"{domain_hits}/4",
                "sections_hit": f"{sections}/4",
            },
        )

    @pytest.mark.asyncio
    async def test_adversarial_code_review(self):
        """
        Present intentionally buggy code with subtle issues and ask for review.
        Tests: code understanding, bug detection, security awareness.
        """
        orch = _get_orchestrator()

        buggy_code = '''
def transfer_funds(from_account, to_account, amount, db):
    """Transfer funds between accounts."""
    # Check balance
    balance = db.query(f"SELECT balance FROM accounts WHERE id = {from_account}")

    if balance >= amount:
        # Deduct from sender
        db.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE id = {from_account}")

        # Add to receiver
        db.execute(f"UPDATE accounts SET balance = balance + {amount} WHERE id = {to_account}")

        # Log transaction
        db.execute(f"INSERT INTO transactions (from_id, to_id, amount) VALUES ({from_account}, {to_account}, {amount})")

        return {"success": True, "new_balance": balance - amount}

    return {"success": False, "error": "Insufficient funds"}


class UserAuth:
    def login(self, username, password):
        user = self.db.query(f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'")
        if user:
            token = hashlib.md5(f"{username}{time.time()}".encode()).hexdigest()
            self.sessions[token] = {"user": username, "role": user["role"]}
            return {"token": token}
        return None

    def is_admin(self, token):
        session = self.sessions.get(token, {})
        return session.get("role") == "admin"

    def delete_user(self, token, user_id):
        if self.is_admin(token):
            self.db.execute(f"DELETE FROM users WHERE id = {user_id}")
            return True
        return False
'''

        result = await orch.chat(
            message=(
                f"Review this Python code for ALL bugs, security vulnerabilities, "
                f"and design flaws. For each issue, explain:\n"
                f"1. What the vulnerability is\n"
                f"2. How an attacker could exploit it\n"
                f"3. The fix\n\n"
                f"Be exhaustive — there are at least 8 distinct issues.\n\n"
                f"```python\n{buggy_code}\n```"
            ),
            provider="anthropic",
            learn=True,
        )

        content = getattr(result, "content", str(result))
        content_lower = content.lower()

        # Should catch these critical bugs:
        found_issues = {
            "sql_injection": any(
                w in content_lower for w in ["sql injection", "parameterized", "prepared statement"]
            ),
            "race_condition": any(
                w in content_lower
                for w in ["race condition", "transaction", "atomic", "concurrent"]
            ),
            "weak_hash": any(w in content_lower for w in ["md5", "weak hash", "bcrypt", "argon"]),
            "no_transaction": any(
                w in content_lower for w in ["transaction", "rollback", "atomic", "commit"]
            ),
            "plaintext_password": any(
                w in content_lower for w in ["plaintext", "plain text", "hash", "salt"]
            ),
            "float_money": any(
                w in content_lower for w in ["float", "decimal", "precision", "rounding"]
            ),
        }

        found_count = sum(found_issues.values())
        assert (
            found_count >= 4
        ), f"Expected at least 4 issues found, got {found_count}: {found_issues}"

        print(f"\n[PASS] Adversarial code review: found {found_count}/6 critical issues")
        for issue, found in found_issues.items():
            print(f"  {'[x]' if found else '[ ]'} {issue}")
        _save_result(
            "adversarial_code_review",
            content,
            {
                "issues_found": f"{found_count}/6",
                **{k: "found" if v else "missed" for k, v in found_issues.items()},
            },
        )


# ============================================================================
# Test 5: End-to-end learning verification
# ============================================================================


class TestEndToEndLearning:
    """Verify the full learning pipeline fires and records data."""

    @pytest.mark.asyncio
    async def test_learning_service_records_and_queries(self):
        """
        Run two tasks, then verify LearningService can query
        guidance based on recorded data.
        """
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()
        orch = _get_orchestrator()

        # Task 1: Simple success
        r1 = await orch.chat(
            message="What programming language is Python named after? One sentence.",
            provider="anthropic",
            learn=True,
        )
        c1 = getattr(r1, "content", str(r1))
        assert "monty" in c1.lower() or "python" in c1.lower(), f"Unexpected: {c1[:100]}"

        # Task 2: Another success
        r2 = await orch.chat(
            message="What year was the first iPhone released? Just the year.",
            provider="anthropic",
            learn=True,
        )
        c2 = getattr(r2, "content", str(r2))
        assert "2007" in c2, f"Expected 2007, got: {c2[:100]}"

        # Query learning for guidance
        guidance = ls.query(
            domain="conversational",
            task_type="chat",
            context={"message": "test query"},
        )
        assert isinstance(guidance, dict), f"Expected dict guidance, got {type(guidance)}"

        # Build context string
        ctx_str = ls.build_context_string(domain="conversational", task_type="chat")
        assert isinstance(ctx_str, str), f"Expected string context, got {type(ctx_str)}"

        print(f"\n[PASS] E2E learning verification:")
        print(f"  Task 1: '{c1[:60]}...'")
        print(f"  Task 2: '{c2[:60]}...'")
        print(f"  Guidance keys: {list(guidance.keys()) if guidance else 'empty'}")
        print(f"  Context string: {len(ctx_str)} chars")
        _save_result(
            "learning_verification",
            (
                f"## Task 1\n\n{c1}\n\n## Task 2\n\n{c2}\n\n"
                f"## LearningService Guidance\n\n```json\n"
                f"{json.dumps(guidance, indent=2, default=str)}\n```\n\n"
                f"## Context String\n\n{ctx_str}"
            ),
            {"guidance_keys": str(list(guidance.keys()))},
        )


# ============================================================================
# Runner
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120", "-s"])
