"""
Integration tests for Multi-Agent System (MAS) swarm paths.

These tests make REAL LLM API calls and validate end-to-end execution.
Run with: pytest tests/integration/test_mas_swarms.py -v -m requires_llm

Skipped by default in CI unless ANTHROPIC_API_KEY is set.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

requires_llm = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real LLM tests",
)


def _extract_content(result) -> str:
    """Extract best text content from any result type."""
    if result is None:
        return ""
    content = ""
    inner = getattr(result, "output", None)
    if inner and not isinstance(inner, str):
        for attr in ("final_output", "output", "content", "code", "report"):
            val = getattr(inner, attr, None)
            if val and isinstance(val, str) and len(val) > len(content):
                content = val
    for attr in ("final_output", "output", "content", "code", "report", "analysis"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and len(val) > len(content):
            content = val
    out_dict = getattr(result, "output", None)
    if isinstance(out_dict, dict):
        for key in ("code", "report", "content"):
            val = out_dict.get(key, "")
            if isinstance(val, str) and len(val) > len(content):
                content = val
    if not content:
        s = str(result)
        if len(s) > 50:
            content = s
    return content


# ─── Orchestrator tests ──────────────────────────────────────────────────────


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_orchestrator_chat():
    """Orchestrator.chat() returns substantive response."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = Orchestrator()
    result = await orch.chat("What are the SOLID principles? Give a one-sentence summary of each.")
    content = _extract_content(result)
    assert len(content) > 200, f"Chat response too short: {len(content)} chars"


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_orchestrator_run_auto_detect():
    """orch.run() without explicit swarm routes and produces output."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = Orchestrator()
    result = await orch.run(
        "Explain the CAP theorem with a concrete example for each trade-off.",
        learn=True,
    )
    content = _extract_content(result)
    assert len(content) > 200, f"Auto-detect response too short: {len(content)} chars"


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_orchestrator_run_coding_swarm():
    """orch.run(swarm=CodingSwarm) produces code output."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

    orch = Orchestrator()
    swarm = CodingSwarm(CodingConfig())
    result = await orch.run(
        "Build a Python stack class with push, pop, peek, and is_empty methods. Include 3 tests.",
        swarm=swarm,
        learn=True,
    )
    content = _extract_content(result)
    assert len(content) > 200, f"CodingSwarm response too short: {len(content)} chars"
    assert getattr(result, "success", False), "CodingSwarm should report success"


# ─── Direct swarm tests ──────────────────────────────────────────────────────


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_review_swarm_direct():
    """ReviewSwarm.execute() reviews code and returns findings."""
    from Jotty.core.intelligence.orchestration.swarms.templates.review_swarm import ReviewSwarm

    swarm = ReviewSwarm()
    code = 'def add(a, b):\n    return eval(f"{a}+{b}")\n'
    result = await swarm.execute(task=code, context="Production code", language="python")
    content = _extract_content(result)
    assert len(content) > 100, f"ReviewSwarm response too short: {len(content)} chars"


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_testing_swarm_direct():
    """TestingSwarm.execute() generates test code."""
    from Jotty.core.intelligence.orchestration.swarms.templates.testing_swarm import TestingSwarm

    swarm = TestingSwarm()
    code = """
class Counter:
    def __init__(self):
        self._count = 0
    def increment(self):
        self._count += 1
    def decrement(self):
        self._count = max(0, self._count - 1)
    def value(self) -> int:
        return self._count
"""
    result = await swarm.execute(task=code, language="python")
    content = _extract_content(result)
    assert len(content) > 100, f"TestingSwarm response too short: {len(content)} chars"


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_devops_swarm_direct():
    """DevOpsSwarm.execute() produces infrastructure design."""
    from Jotty.core.intelligence.orchestration.swarms.templates.devops_swarm import DevOpsSwarm

    swarm = DevOpsSwarm()
    result = await swarm.execute(
        task="simple-api",
        app_type="api",
        language="python",
        requirements="REST API with PostgreSQL and Redis",
        scale="small",
    )
    content = _extract_content(result)
    assert len(content) > 200, f"DevOpsSwarm response too short: {len(content)} chars"


@requires_llm
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_idea_writer_swarm_direct():
    """IdeaWriterSwarm.execute() produces written content."""
    from Jotty.core.intelligence.orchestration.swarms.templates.idea_writer_swarm import (
        IdeaWriterSwarm,
    )

    swarm = IdeaWriterSwarm()
    result = await swarm.execute(task="Why developers should learn about observability")
    content = _extract_content(result)
    assert len(content) > 300, f"IdeaWriterSwarm response too short: {len(content)} chars"


# ─── Infrastructure tests (no LLM needed) ────────────────────────────────────


@pytest.mark.unit
def test_circuit_breaker_opens_on_threshold():
    """Circuit breaker opens after failure_threshold failures."""
    from Jotty.core.infrastructure.utils.provider_health import ProviderHealthManager

    health = ProviderHealthManager()
    assert health.is_healthy("test_x")
    health.record_failure("test_x", Exception("err"))
    health.record_failure("test_x", Exception("err"))
    assert not health.is_healthy("test_x")

    stats = health.get_stats()["test_x"]
    assert stats["backoff_multiplier"] == 1
    assert stats["current_timeout"] == 60.0


@pytest.mark.unit
def test_model_max_output_tokens():
    """get_max_output_tokens returns correct limits per model."""
    from Jotty.core.infrastructure.foundation.config_defaults import get_max_output_tokens

    assert get_max_output_tokens("haiku") == 4096
    assert get_max_output_tokens("claude-3-haiku-20240307") == 4096
    assert get_max_output_tokens("sonnet") == 8192
    assert get_max_output_tokens("anthropic/claude-3-haiku-20240307") == 4096
    assert get_max_output_tokens("gpt-4o") == 16384
    assert get_max_output_tokens("unknown-model") == 8192
    assert get_max_output_tokens(None) == 8192


@pytest.mark.unit
def test_learning_service_record_and_retrieve():
    """LearningService can record and retrieve episodes."""
    from Jotty.core.intelligence.learning.learning_service import LearningService

    svc = LearningService.get_instance()
    episode_id = svc.record(
        unit_name="test_agent",
        unit_type="agent",
        domain="test",
        task_type="unit_test",
        context={"goal": "test recording"},
        action={"tool": "none"},
        outcome={"result": "ok"},
        success=True,
        quality=0.9,
        execution_time=1.0,
    )
    assert episode_id and isinstance(episode_id, str)

    ctx = svc.build_context_string(domain="test", task_type="unit_test", goal="test")
    # Context may or may not return data depending on episode count, but should not crash
    assert isinstance(ctx, str)
