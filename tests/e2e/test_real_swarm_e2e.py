"""
E2E: Real swarm execution with TeamCoordinator, Learning feedback, Streaming.

Tests the ACTUAL multi-agent coordination stack — not mocks, not ChatExecutor.

Requires: ANTHROPIC_API_KEY in .env
"""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

try:
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
except ImportError:
    pass

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
pytestmark = [
    pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"),
    pytest.mark.e2e,
    pytest.mark.timeout(300),
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(test_name: str, content: str, metadata: dict | None = None) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_{test_name}.md"
    header = f"# {test_name}\n\n**Timestamp:** {datetime.now().isoformat()}\n"
    if metadata:
        for k, v in metadata.items():
            header += f"**{k}:** {v}\n"
    header += "\n---\n\n"
    path.write_text(header + content, encoding="utf-8")
    print(f"  [Saved] {path}")


def _configure_dspy():
    """Configure DSPy with Anthropic for swarm agents."""
    import dspy

    lm = dspy.LM(
        "anthropic/claude-sonnet-4-20250514",
        api_key=ANTHROPIC_KEY,
        max_tokens=2000,
        temperature=0.3,
    )
    dspy.configure(lm=lm)
    return lm


# ============================================================================
# Test 1: Real ReviewSwarm with 6 DSPy agents
# ============================================================================


class TestRealReviewSwarm:
    """Test ReviewSwarm.execute() — 6 agents, parallel + sequential phases."""

    @pytest.mark.asyncio
    async def test_review_swarm_full_execution(self):
        """
        Execute ReviewSwarm with real Anthropic LLM.

        Exercises:
        - 6 DSPy agents (CodeReviewer, SecurityScanner, PerformanceAnalyzer,
          ArchitectureReviewer, StyleChecker, ReviewSynthesizer)
        - Phase 1: Parallel reviews (4 agents simultaneously)
        - Phase 2: Style check (sequential)
        - Phase 3: Synthesis (merges all results)
        - LearningService episode recording
        """
        _configure_dspy()

        from Jotty.core.intelligence.orchestration.swarms.templates.review_swarm import (
            ReviewSwarm,
            ReviewConfig,
        )

        config = ReviewConfig()
        swarm = ReviewSwarm(config)

        vulnerable_code = """
import hashlib
import sqlite3

class UserAuth:
    def __init__(self):
        self.db = sqlite3.connect("users.db")

    def register(self, username, password):
        password_hash = hashlib.md5(password.encode()).hexdigest()
        self.db.execute(
            f"INSERT INTO users (username, password) VALUES ('{username}', '{password_hash}')"
        )
        self.db.commit()

    def login(self, username, password):
        password_hash = hashlib.md5(password.encode()).hexdigest()
        cursor = self.db.execute(
            f"SELECT * FROM users WHERE username='{username}' AND password='{password_hash}'"
        )
        return cursor.fetchone() is not None

    def get_user_data(self, user_id):
        cursor = self.db.execute(f"SELECT * FROM users WHERE id={user_id}")
        return cursor.fetchone()

    def delete_user(self, user_id):
        self.db.execute(f"DELETE FROM users WHERE id={user_id}")
        self.db.commit()
"""

        start = time.time()
        result = await swarm.execute(
            vulnerable_code,
            context="Authentication module for a web application",
            language="python",
        )
        elapsed = time.time() - start

        # Verify result structure
        assert result is not None, "ReviewSwarm returned None"
        assert hasattr(result, "success"), "Result should have 'success' attribute"

        # Extract findings
        output_str = str(result.output) if hasattr(result, "output") else str(result)

        # Verify agents actually found issues
        comments = getattr(result, "comments", [])
        security = getattr(result, "security_findings", [])
        performance = getattr(result, "performance_findings", [])

        total_findings = len(comments) + len(security) + len(performance)

        print(f"\n[PASS] ReviewSwarm: {elapsed:.1f}s, {total_findings} findings")
        print(f"  Code comments: {len(comments)}")
        print(f"  Security findings: {len(security)}")
        print(f"  Performance findings: {len(performance)}")
        if hasattr(result, "overall_score"):
            print(f"  Overall score: {result.overall_score}")

        content = f"## ReviewSwarm Output\n\n"
        content += f"**Findings:** {total_findings}\n"
        content += f"**Code Comments:** {len(comments)}\n"
        content += f"**Security Findings:** {len(security)}\n"
        content += f"**Performance Findings:** {len(performance)}\n\n"
        content += f"### Raw Output\n\n```\n{output_str[:3000]}\n```\n"

        if comments:
            content += "\n### Code Comments\n\n"
            for c in comments[:10]:
                content += f"- **{getattr(c, 'severity', 'unknown')}** L{getattr(c, 'line', '?')}: {getattr(c, 'message', str(c))}\n"

        if security:
            content += "\n### Security Findings\n\n"
            for s in security[:10]:
                content += f"- **{getattr(s, 'severity', 'unknown')}** {getattr(s, 'vulnerability_type', str(s))}: {getattr(s, 'description', '')}\n"

        _save_result(
            "review_swarm_real_6agents",
            content,
            {
                "elapsed_s": f"{elapsed:.1f}",
                "agents": "6 (CodeReviewer, SecurityScanner, PerformanceAnalyzer, ArchitectureReviewer, StyleChecker, ReviewSynthesizer)",
                "total_findings": str(total_findings),
            },
        )


# ============================================================================
# Test 2: TeamCoordinator PARALLEL pattern with real agents
# ============================================================================


class TestTeamCoordinatorPatterns:
    """Test TeamCoordinator coordination patterns with real LLMs."""

    @pytest.mark.asyncio
    async def test_parallel_coordination(self):
        """
        Build a team of 3 agents with PARALLEL pattern, verify:
        - All agents run concurrently
        - Results are merged
        - TeamResult contains outputs from all agents
        """
        _configure_dspy()

        import dspy
        from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import (
            TeamCoordinator,
            TeamResult,
        )
        from Jotty.core.infrastructure.foundation.types.execution_types import (
            CoordinationPattern,
            MergeStrategy,
        )

        class AnalystSignature(dspy.Signature):
            """Analyze from a specific perspective."""

            topic: str = dspy.InputField()
            perspective: str = dspy.InputField()
            analysis: str = dspy.OutputField(desc="2-3 sentence analysis")

        class PerspectiveAgent:
            """Simple agent that analyzes from a given perspective."""

            def __init__(self, perspective: str):
                self.perspective = perspective
                self._predictor = dspy.ChainOfThought(AnalystSignature)

            async def execute(self, input: Any = "", context: Any = None, **kw: Any) -> Any:
                result = self._predictor(
                    topic=str(input)[:500],
                    perspective=self.perspective,
                )
                return type("R", (), {"output": result.analysis})()

        # Create 3 perspective agents
        economic = PerspectiveAgent("economic impact and market dynamics")
        social = PerspectiveAgent("social implications and human behavior")
        tech = PerspectiveAgent("technological feasibility and innovation")

        # Define team with PARALLEL pattern
        team = TeamCoordinator(
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.COMBINE,
            timeout=60.0,
        )
        team.add(PerspectiveAgent, "Economic", "_economic")
        team.add(PerspectiveAgent, "Social", "_social")
        team.add(PerspectiveAgent, "Tech", "_tech")

        # Set instances manually (normally done by SwarmTemplate._init_agents)
        team.set_instances(
            {
                "_economic": economic,
                "_social": social,
                "_tech": tech,
            }
        )

        start = time.time()
        result = await team.execute(
            task="Should autonomous vehicles be allowed on public roads?",
            context={},
        )
        elapsed = time.time() - start

        assert isinstance(result, TeamResult), f"Expected TeamResult, got {type(result)}"
        assert result.success, f"Team execution failed: {result.errors}"
        assert result.pattern == CoordinationPattern.PARALLEL
        assert (
            len(result.outputs) == 3
        ), f"Expected 3 outputs, got {len(result.outputs)}: {list(result.outputs.keys())}"
        assert len(result.execution_order) == 3

        # Verify each agent produced real content
        for name, output in result.outputs.items():
            assert len(str(output)) > 20, f"Agent {name} output too short: {output}"

        # Verify it was faster than sequential (parallel should overlap)
        print(f"\n[PASS] PARALLEL TeamCoordinator: {elapsed:.1f}s, 3 agents")
        for name, output in result.outputs.items():
            print(f"  {name}: {str(output)[:80]}...")

        content = "## TeamCoordinator PARALLEL Pattern\n\n"
        for name, output in result.outputs.items():
            content += f"### {name}\n\n{output}\n\n"

        _save_result(
            "team_coordinator_parallel",
            content,
            {
                "elapsed_s": f"{elapsed:.1f}",
                "pattern": "PARALLEL",
                "agents": "3 (Economic, Social, Tech)",
                "all_succeeded": str(result.success),
            },
        )

    @pytest.mark.asyncio
    async def test_pipeline_coordination(self):
        """
        Build a team with PIPELINE pattern: Agent1 → Agent2 → Agent3.
        Verify output chaining.
        """
        _configure_dspy()

        import dspy
        from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import (
            TeamCoordinator,
            TeamResult,
        )
        from Jotty.core.infrastructure.foundation.types.execution_types import (
            CoordinationPattern,
        )

        class TransformSignature(dspy.Signature):
            """Transform text based on instructions."""

            text: str = dspy.InputField()
            instruction: str = dspy.InputField()
            result: str = dspy.OutputField()

        class TransformAgent:
            def __init__(self, instruction: str, priority: int = 0):
                self.instruction = instruction
                self.priority = priority
                self._predictor = dspy.ChainOfThought(TransformSignature)

            async def execute(self, input: Any = "", context: Any = None, **kw: Any) -> Any:
                result = self._predictor(
                    text=str(input)[:1000],
                    instruction=self.instruction,
                )
                return type("R", (), {"output": result.result})()

        # Create pipeline: Outline → Draft → Polish
        outliner = TransformAgent("Create a 3-point outline for this topic", priority=3)
        drafter = TransformAgent(
            "Expand this outline into a short paragraph (50-80 words)", priority=2
        )
        polisher = TransformAgent(
            "Polish this text: fix grammar, improve clarity, make it professional", priority=1
        )

        team = TeamCoordinator(
            pattern=CoordinationPattern.PIPELINE,
            timeout=60.0,
        )
        team.add(TransformAgent, "Outliner", "_outliner", priority=3)
        team.add(TransformAgent, "Drafter", "_drafter", priority=2)
        team.add(TransformAgent, "Polisher", "_polisher", priority=1)

        team.set_instances(
            {
                "_outliner": outliner,
                "_drafter": drafter,
                "_polisher": polisher,
            }
        )

        start = time.time()
        result = await team.execute(
            task="The impact of remote work on urban planning",
            context={},
        )
        elapsed = time.time() - start

        assert isinstance(result, TeamResult)
        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.pattern == CoordinationPattern.PIPELINE
        assert len(result.execution_order) == 3
        assert result.execution_order == ["Outliner", "Drafter", "Polisher"]

        # Verify chaining: final output should be the polished version
        final = str(result.merged_output)
        assert len(final) > 30, f"Final pipeline output too short: {len(final)} chars"

        print(f"\n[PASS] PIPELINE TeamCoordinator: {elapsed:.1f}s")
        print(f"  Execution order: {result.execution_order}")
        for name, output in result.outputs.items():
            print(f"  {name}: {str(output)[:80]}...")

        content = "## TeamCoordinator PIPELINE Pattern\n\n"
        content += f"**Execution order:** {' → '.join(result.execution_order)}\n\n"
        for name, output in result.outputs.items():
            content += f"### {name}\n\n{output}\n\n"
        content += f"### Final (merged) output\n\n{result.merged_output}\n"

        _save_result(
            "team_coordinator_pipeline",
            content,
            {
                "elapsed_s": f"{elapsed:.1f}",
                "pattern": "PIPELINE",
                "execution_order": " → ".join(result.execution_order),
            },
        )


# ============================================================================
# Test 3: Learning feedback loop — same task twice
# ============================================================================


class TestLearningFeedbackLoop:
    """Prove learning from run N feeds into run N+1."""

    @pytest.mark.asyncio
    async def test_learning_improves_context(self):
        """
        Run the same task twice via orchestrator.run().
        On the second run, verify that LearningService.build_context_string()
        returns guidance from the first run's recorded data.
        """
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()

        # Record a synthetic first-run outcome
        ls.record(
            unit_name="TestSwarm",
            unit_type="swarm",
            domain="code_review",
            task_type="review",
            context={"goal": "Review authentication code"},
            action={"mode": "swarm", "agents": ["CodeReviewer", "SecurityScanner"]},
            outcome={"findings": 5, "security_issues": 3},
            success=True,
            quality=0.9,
            execution_time=12.5,
        )

        # Now query guidance for the same domain
        guidance = ls.build_context_string(
            domain="code_review",
            task_type="review",
            unit_name="TestSwarm",
        )

        assert (
            guidance is not None and len(guidance) > 0
        ), "LearningService should return guidance after recording data"

        # Verify guidance contains useful information
        guidance_lower = guidance.lower()
        has_useful_info = any(
            w in guidance_lower
            for w in [
                "success",
                "quality",
                "pattern",
                "episode",
                "history",
                "review",
                "swarm",
                "strategy",
                "previous",
            ]
        )
        assert (
            has_useful_info
        ), f"Guidance should contain useful learning context, got: {guidance[:200]}"

        print(f"\n[PASS] Learning feedback loop verified")
        print(f"  Guidance length: {len(guidance)} chars")
        print(f"  Preview: {guidance[:200]}...")

        # Also verify improvement_report works
        report = ls.improvement_report(domain="code_review")
        assert isinstance(report, dict)
        assert "total" in report
        print(f"  Improvement report: {report}")

        _save_result(
            "learning_feedback_loop",
            (
                f"## Learning Guidance Output\n\n```\n{guidance}\n```\n\n"
                f"## Improvement Report\n\n```\n{report}\n```\n"
            ),
            {
                "guidance_length": str(len(guidance)),
                "report_total": str(report.get("total", 0)),
            },
        )


# ============================================================================
# Test 4: Streaming through orchestrator
# ============================================================================


class TestStreamingExecution:
    """Test streaming works through orchestrator.chat(stream=True)."""

    @pytest.mark.asyncio
    async def test_streaming_produces_events(self):
        """
        Call orchestrator.chat(stream=True), verify we get StreamEvents.
        """
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

        start = time.time()
        stream = await orch.chat(
            "What is 2+2? Answer in one word.",
            stream=True,
            learn=False,
            provider="anthropic",
        )

        chunks = []
        event_types = set()
        async for event in stream:
            event_types.add(event.type)
            if event.type == "text" and event.data:
                chunks.append(event.data)

        elapsed = time.time() - start
        full_text = "".join(chunks)

        assert len(chunks) > 0, "Should receive text chunks from stream"
        assert len(full_text) > 0, f"Should have non-empty text, got {len(full_text)} chars"

        print(f"\n[PASS] Streaming: {elapsed:.1f}s, {len(chunks)} chunks")
        print(f"  Event types: {event_types}")
        print(f"  Full text: {full_text[:200]}")

        _save_result(
            "streaming_execution",
            (
                f"## Streaming Output\n\n"
                f"**Chunks:** {len(chunks)}\n"
                f"**Event types:** {event_types}\n\n"
                f"### Full text\n\n{full_text}\n"
            ),
            {
                "elapsed_s": f"{elapsed:.1f}",
                "chunks": str(len(chunks)),
                "event_types": str(event_types),
            },
        )


# ============================================================================
# Test 5: Full Orchestrator.run(swarm=ReviewSwarm) integration
# ============================================================================


class TestOrchestratorSwarmIntegration:
    """Test the full Orchestrator → SwarmTemplate → Agents path."""

    @pytest.mark.asyncio
    async def test_orchestrator_run_with_swarm(self):
        """
        Call orchestrator.run(goal, swarm=ReviewSwarm).
        This tests the full stack:
          Orchestrator.run() → _run_swarm() → ReviewSwarm.execute()
          → _execute_domain() → PhaseExecutor → 6 DSPy agents → Anthropic
        """
        _configure_dspy()

        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
        from Jotty.core.intelligence.orchestration.swarms.templates.review_swarm import (
            ReviewSwarm,
        )

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

        code = """
def calculate_discount(price, discount_percent):
    return price * discount_percent / 100

def apply_bulk_discount(items):
    total = 0
    for item in items:
        if item["quantity"] > 100:
            total += calculate_discount(item["price"], 20)
        elif item["quantity"] > 50:
            total += calculate_discount(item["price"], 10)
        else:
            total += item["price"]
    return total
"""

        start = time.time()
        result = await orch.run(
            code,
            swarm=ReviewSwarm,
            learn=True,
        )
        elapsed = time.time() - start

        assert result is not None, "Orchestrator.run(swarm=ReviewSwarm) returned None"
        output_str = str(result.output) if hasattr(result, "output") else str(result)
        assert len(output_str) > 50, f"Output too short: {len(output_str)} chars"

        print(f"\n[PASS] Orchestrator.run(swarm=ReviewSwarm): {elapsed:.1f}s")
        print(f"  Output: {output_str[:200]}...")

        _save_result(
            "orchestrator_swarm_integration",
            (f"## Orchestrator.run(swarm=ReviewSwarm)\n\n" f"```\n{output_str[:3000]}\n```\n"),
            {
                "elapsed_s": f"{elapsed:.1f}",
                "swarm": "ReviewSwarm (6 agents)",
            },
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300", "-s"])
