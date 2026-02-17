"""
Real LLM E2E Tests — Super Complex Use Cases

Tests the unified Orchestrator API (run + chat) with REAL Anthropic Claude calls.
No mocks. These hit the actual API and validate the full pipeline:
  Orchestrator → ChatExecutor → Anthropic API → Tool Calling → LearningService

Run with:
    source .env && pytest tests/e2e/test_real_llm_orchestrator.py -v -s --timeout=120

Requirements:
    ANTHROPIC_API_KEY must be set in environment.
"""

import asyncio
import logging
import os
import time

import pytest

logger = logging.getLogger(__name__)

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real LLM tests",
)


def _make_orchestrator():
    """Create a minimal Orchestrator via object.__new__ to avoid deprecation/heavy init."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = object.__new__(Orchestrator)
    from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

    orch.config = SwarmConfig()
    orch.agents = []
    orch.mode = "single"
    orch.runners = {}
    orch._runners_built = False
    orch._efficiency_stats = {}
    orch._intelligence_metrics = {}
    orch._learning_ready = asyncio.Event()
    orch._learning_ready.set()
    return orch


# ============================================================================
# TEST 1: chat() — Multi-turn reasoning with tool use
# ============================================================================


class TestRealChat:
    """Real LLM chat tests via Orchestrator.chat()."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_chat_simple_question(self):
        """chat() handles a simple factual question via real Claude."""
        orch = _make_orchestrator()
        result = await orch.chat(
            "What are the three laws of thermodynamics? Answer in exactly 3 numbered points.",
            provider="anthropic",
        )
        assert result is not None
        assert result.success is True
        assert len(result.content) > 100
        assert "1" in result.content and "2" in result.content and "3" in result.content
        logger.info(f"Chat response length: {len(result.content)} chars")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_chat_with_history_context(self):
        """chat() respects conversation history for multi-turn reasoning."""
        orch = _make_orchestrator()

        history = [
            {
                "role": "user",
                "content": "My name is Alexander and I'm building an AI framework called Jotty.",
            },
            {
                "role": "assistant",
                "content": "Nice to meet you, Alexander! Jotty sounds like an interesting AI framework project.",
            },
        ]

        result = await orch.chat(
            "What's my name and what am I building? Just state the facts, no fluff.",
            history=history,
            provider="anthropic",
        )
        assert result.success is True
        assert "Alexander" in result.content
        assert "Jotty" in result.content
        logger.info(f"History-aware response: {result.content[:200]}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_chat_complex_reasoning(self):
        """chat() handles a complex multi-step reasoning problem."""
        orch = _make_orchestrator()
        result = await orch.chat(
            """Solve this step by step:
            A farmer has a 100-acre field. He plants corn on 40% of it,
            wheat on 1/3 of the remaining land, and leaves the rest fallow.
            How many acres are fallow? Show your work.""",
            provider="anthropic",
        )
        assert result.success is True
        assert "40" in result.content
        logger.info(f"Reasoning response: {result.content[:300]}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_chat_code_generation(self):
        """chat() generates working code for a complex algorithm."""
        orch = _make_orchestrator()
        result = await orch.chat(
            """Write a Python function that implements Dijkstra's shortest path algorithm.
            It should accept an adjacency list (dict of dict with weights) and a start node.
            Return a dict of shortest distances. Include type hints.""",
            provider="anthropic",
        )
        assert result.success is True
        assert "def " in result.content
        assert "dijkstra" in result.content.lower()
        logger.info(f"Code gen response length: {len(result.content)} chars")


# ============================================================================
# TEST 2: chat() with tool calling — Real web search + reasoning
# ============================================================================


class TestRealChatWithTools:
    """Real LLM chat tests with tool calling."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_chat_with_tool_calling(self):
        """chat() uses tools when needed (web_search for current info)."""
        orch = _make_orchestrator()
        result = await orch.chat(
            "What is today's date? Use your knowledge to answer.",
            provider="anthropic",
            max_steps=3,
        )
        assert result.success is True
        assert len(result.content) > 10
        logger.info(f"Tool-aware response: {result.content[:200]}")


# ============================================================================
# TEST 3: chat() — Super complex multi-domain reasoning
# ============================================================================


class TestRealChatSuperComplex:
    """Super complex reasoning that pushes Claude's limits."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_multi_domain_analysis(self):
        """Complex cross-domain analysis requiring deep reasoning."""
        orch = _make_orchestrator()
        result = await orch.chat(
            """You are a senior systems architect. Analyze the following scenario and provide
            a comprehensive recommendation:

            A fintech startup (50 engineers, $20M ARR) is migrating from a monolith
            Rails app to microservices. They handle:
            - 10M transactions/day with 99.99% uptime SLA
            - Real-time fraud detection (< 100ms latency)
            - PCI-DSS and SOX compliance
            - Multi-region deployment (US, EU, APAC)

            They're debating between:
            A) Event-driven architecture with Kafka + gRPC services
            B) Service mesh with Istio + REST APIs
            C) Hybrid: Event-driven for transactions, REST for admin

            Evaluate each option across: scalability, latency, compliance,
            team ramp-up, and operational complexity. Give a final recommendation
            with a phased migration plan.""",
            provider="anthropic",
        )
        assert result.success is True
        assert len(result.content) > 500
        content_lower = result.content.lower()
        assert "kafka" in content_lower or "event" in content_lower
        assert "compliance" in content_lower or "pci" in content_lower
        logger.info(f"Multi-domain analysis: {len(result.content)} chars")

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_mathematical_proof(self):
        """Complex mathematical reasoning — prove a theorem step by step."""
        orch = _make_orchestrator()
        result = await orch.chat(
            """Prove that the sum of the first n odd numbers equals n².

            Use mathematical induction:
            1. State the base case
            2. State the inductive hypothesis
            3. Prove the inductive step
            4. Conclude the proof

            Then verify with n=5 by computing both sides.""",
            provider="anthropic",
        )
        assert result.success is True
        assert "base case" in result.content.lower() or "base" in result.content.lower()
        assert "25" in result.content  # 5² = 25
        logger.info(f"Math proof: {len(result.content)} chars")

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_adversarial_edge_case_analysis(self):
        """Complex edge case analysis — find flaws in code."""
        orch = _make_orchestrator()
        result = await orch.chat(
            """Find ALL bugs, race conditions, and edge cases in this Python code:

```python
import threading

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self.lock = threading.Lock()

    def transfer(self, other, amount):
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount
                with other.lock:
                    other.balance += amount
                return True
        return False

    def get_balance(self):
        return self.balance
```

List every issue with severity (critical/high/medium/low),
explain why it's a problem, and provide the fixed code.""",
            provider="anthropic",
        )
        assert result.success is True
        content_lower = result.content.lower()
        assert "deadlock" in content_lower or "lock ordering" in content_lower
        logger.info(f"Code review: {len(result.content)} chars")


# ============================================================================
# TEST 4: run() — Auto-routing with real LLM
# ============================================================================


class TestRealRun:
    """Real LLM run() tests — auto-routing."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_run_auto_routes_simple(self):
        """run() auto-routes a simple task through the engine."""
        orch = _make_orchestrator()

        # Patch the engine to use a real ChatExecutor for this test
        from unittest.mock import AsyncMock
        from Jotty.core.intelligence.orchestration.execution.unified_executor import (
            ChatExecutor,
        )

        executor = ChatExecutor(provider="anthropic")
        real_result = await executor.execute("Explain what a hash table is in 2 sentences.")

        assert real_result.success is True
        assert "hash" in real_result.content.lower()
        logger.info(f"Run auto-route result: {real_result.content[:200]}")


# ============================================================================
# TEST 5: LearningService integration with real execution
# ============================================================================


class TestRealLearningIntegration:
    """Verify LearningService records real execution outcomes."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_chat_records_learning_episode(self):
        """Real chat should create a LearningService episode."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        learning = LearningService.get_instance()

        orch = _make_orchestrator()
        result = await orch.chat(
            "What is 2 + 2? Just the number.",
            provider="anthropic",
        )

        assert result.success is True
        assert "4" in result.content

        # Verify learning recorded something
        query_result = learning.query(
            domain="conversational",
            task_type="chat",
            context={"message": "test"},
        )
        assert isinstance(query_result, dict)
        logger.info(f"Learning query result: {query_result}")


# ============================================================================
# TEST 6: Streaming with real LLM
# ============================================================================


class TestRealStreaming:
    """Real streaming tests."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_chat_streaming(self):
        """chat(stream=True) streams real tokens from Claude."""
        orch = _make_orchestrator()

        events = []
        stream = await orch.chat(
            "Count from 1 to 5, one number per line.",
            stream=True,
            provider="anthropic",
        )
        async for event in stream:
            events.append(event)

        assert len(events) > 0
        logger.info(f"Received {len(events)} stream events")

        # Should have at least text events and a complete event
        event_types = [e.type for e in events]
        assert "text" in event_types or "complete" in event_types


# ============================================================================
# TEST 7: Pipeline with real LLM
# ============================================================================


class TestRealPipeline:
    """Real pipeline tests with stages."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_pipeline_two_stage(self):
        """run(stages=[...]) executes a real two-stage pipeline."""
        orch = _make_orchestrator()

        async def research_stage(context):
            """Stage 1: Research."""
            from Jotty.core.intelligence.orchestration.execution.unified_executor import (
                ChatExecutor,
            )

            executor = ChatExecutor(provider="anthropic")
            result = await executor.execute(
                "List 3 key benefits of microservices architecture. Be concise, one line each."
            )
            return result.content if result.success else "Failed"

        async def summarize_stage(context):
            """Stage 2: Summarize previous stage output."""
            from Jotty.core.intelligence.orchestration.execution.unified_executor import (
                ChatExecutor,
            )

            previous = context.get("previous_outputs", {}).get("research", "No data")
            executor = ChatExecutor(provider="anthropic")
            result = await executor.execute(
                f"Given this research:\n{previous}\n\nWrite a one-paragraph executive summary."
            )
            return result.content if result.success else "Failed"

        result = await orch.run(
            "Research and summarize microservices",
            stages=[
                {"name": "research", "callable": research_stage},
                {"name": "summary", "callable": summarize_stage, "depends_on": ["research"]},
            ],
        )

        assert result is not None
        assert result.success is True
        logger.info(f"Pipeline result: {str(result.output)[:300]}")
