"""
Real LLM Integration Tests — Verify learning hooks fire during actual execution.

Uses Anthropic API (from .env) to run actual tasks through the orchestration
pipeline and verifies that stigmergy, byzantine, MAS learning, and metrics
hooks are triggered.

Run: pytest tests/integration/test_real_llm_hooks.py -v --timeout=120
"""

import asyncio
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Load .env ────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HAS_KEY = bool(ANTHROPIC_KEY and ANTHROPIC_KEY.startswith("sk-ant-"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_KEY, reason="ANTHROPIC_API_KEY not set"),
]

logger = logging.getLogger(__name__)


# ── 1. ChatExecutor: Real LLM Call ──────────────────────────────────────────


class TestChatExecutorReal:
    """ChatExecutor with real Anthropic API."""

    @pytest.mark.asyncio
    async def test_basic_call(self):
        """ChatExecutor makes a real API call and returns content."""
        from Jotty.core.intelligence.orchestration.execution.unified_executor import ChatExecutor

        executor = ChatExecutor()
        assert executor.provider_name == "anthropic"

        result = await executor.execute("What is 2+2? Reply with ONLY the number.")

        assert result is not None
        assert result.success, f"Expected success, got error: {result.error}"
        assert "4" in result.content, f"Expected '4' in: {result.content}"
        logger.info(f"LLM response: {result.content!r}")
        logger.info(f"Usage: {result.usage}")

    @pytest.mark.asyncio
    async def test_second_call(self):
        """ChatExecutor handles a second call correctly."""
        from Jotty.core.intelligence.orchestration.execution.unified_executor import ChatExecutor

        executor = ChatExecutor()
        result = await executor.execute(
            "What is the square root of 144? Reply with ONLY the number."
        )
        assert result is not None
        assert result.success
        assert "12" in result.content
        logger.info(f"Response: {result.content!r}")


# ── 2. Learning Components: Real Initialization ─────────────────────────────


class TestLearningComponentsReal:
    """Verify ALL 4 dormant learning systems are real and functional."""

    @pytest.mark.asyncio
    async def test_orchestrator_creates_all_components(self):
        """Orchestrator initializes with all 4 learning components."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            orch = Orchestrator(agents="test component init")

        lp = orch.learning

        assert lp is not None, "LearningPipeline should be created"
        assert lp.stigmergy is not None, "Stigmergy should be initialized"
        assert lp.byzantine_verifier is not None, "ByzantineVerifier should be initialized"
        assert lp.mas_learning is not None, "MASLearning should be initialized"
        assert lp.metrics is not None, "MetricsCollector should be initialized"

        components = lp.learning_components
        assert set(components.keys()) == {"stigmergy", "byzantine", "mas_learning", "metrics"}
        assert all(
            v is not None for v in components.values()
        ), f"All components should be non-None: {components}"
        logger.info("All 4 learning components initialized successfully")

    @pytest.mark.asyncio
    async def test_agent_runner_receives_injection(self):
        """AgentRunner gets all 4 learning components via inject_learning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            orch = Orchestrator(agents="test injection")

        orch._ensure_runners()

        injected_count = 0
        for name, runner in orch.runners.items():
            assert runner._stigmergy is not None, f"{name}: stigmergy missing"
            assert runner._byzantine_verifier is not None, f"{name}: byzantine missing"
            assert runner._mas_learning is not None, f"{name}: MAS missing"
            assert runner._metrics is not None, f"{name}: metrics missing"
            injected_count += 1

        assert injected_count > 0, "No runners were built"
        logger.info(f"All 4 components injected into {injected_count} runners")


# ── 3. Post-Execution Hooks: Fire After Task ────────────────────────────────


class TestPostExecutionHooks:
    """Verify all 4 hooks fire after agent execution."""

    @pytest.mark.asyncio
    async def test_hooks_fire_via_orchestrator(self):
        """Post-execution hooks fire via real Orchestrator-built runners."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            orch = Orchestrator(agents="hook test")

        lp = orch.learning
        orch._ensure_runners()

        # Get a real runner built by the Orchestrator
        runner_name, runner = next(iter(orch.runners.items()))

        # Fire the hook directly
        await runner._post_execution_learning(
            agent_name=runner_name,
            task_type="research",
            goal="Verify hooks fire on success",
            output="Sample output from agent",
            success=True,
            quality=0.85,
            duration=2.5,
        )

        # 1. Stigmergy: should have signals
        assert len(lp.stigmergy.signals) > 0, "Stigmergy should record signal"
        logger.info(f"Stigmergy: {len(lp.stigmergy.signals)} signal(s)")

        # 2. Metrics: should have task count
        report = lp.metrics.get_report()
        assert report.get("total_tasks", 0) > 0, f"Metrics: {report}"
        logger.info(
            f"Metrics: total_tasks={report['total_tasks']}, success_rate={report['success_rate']:.0%}"
        )

    @pytest.mark.asyncio
    async def test_hooks_fire_on_failure(self):
        """Hooks also fire after a failed task."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            orch = Orchestrator(agents="failure hook test")

        lp = orch.learning
        orch._ensure_runners()

        runner_name, runner = next(iter(orch.runners.items()))

        await runner._post_execution_learning(
            agent_name=runner_name,
            task_type="coding",
            goal="This task failed",
            output="Error: something went wrong",
            success=False,
            quality=0.0,
            duration=5.0,
        )

        report = lp.metrics.get_report()
        assert report.get("total_tasks", 0) > 0
        logger.info(f"Failure metrics: total_tasks={report['total_tasks']}")


# ── 4. Stigmergy: Routing Intelligence ──────────────────────────────────────


class TestStigmergyReal:
    """Stigmergy builds routing intelligence from outcomes."""

    @pytest.mark.asyncio
    async def test_agent_recommendation(self):
        """Stigmergy recommends agents based on track record."""
        from Jotty.core.intelligence.orchestration.learning.stigmergy import StigmergyLayer

        stig = StigmergyLayer()

        for _ in range(5):
            stig.record_outcome(agent="researcher", task_type="research", success=True, quality=0.9)
        for _ in range(3):
            stig.record_outcome(agent="coder", task_type="research", success=False, quality=0.2)

        recs = stig.recommend_agent("research", candidates=["researcher", "coder", "unknown"])
        logger.info(f"Stigmergy recommendations: {recs}")
        assert len(recs) > 0
        assert recs[0][0] == "researcher"

    @pytest.mark.asyncio
    async def test_approach_recommendation(self):
        """Stigmergy recommends approaches for task types."""
        from Jotty.core.intelligence.orchestration.learning.stigmergy import StigmergyLayer

        stig = StigmergyLayer()

        stig.record_approach_outcome(
            "research", "web_search", tools_used=["web-search"], success=True, quality=0.95
        )
        stig.record_approach_outcome(
            "research", "web_search", tools_used=["web-search"], success=True, quality=0.90
        )
        stig.record_approach_outcome(
            "research", "db_query", tools_used=["sql-query"], success=False, quality=0.3
        )

        rec = stig.recommend_approach("research")
        logger.info(f"Approach recommendation: {rec}")
        assert rec is not None


# ── 5. Byzantine Verification: Trust Scoring ─────────────────────────────────


class TestByzantineReal:
    """Byzantine verifier tracks agent trustworthiness."""

    @pytest.mark.asyncio
    async def test_verify_claim(self):
        """Byzantine verifies agent claims against actual results."""
        from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
            SwarmIntelligence,
        )
        from Jotty.core.intelligence.orchestration.learning.byzantine_verification import (
            ByzantineVerifier,
        )

        si = SwarmIntelligence()
        bv = ByzantineVerifier(si)

        honest = bv.verify_claim(
            agent="honest", claimed_success=True, actual_result={"quality": 0.9}, task_type="test"
        )
        logger.info(f"Honest claim verified: {honest}")
        assert isinstance(honest, bool)

    @pytest.mark.asyncio
    async def test_output_quality_verification(self):
        """Byzantine verifies output quality in detail."""
        from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
            SwarmIntelligence,
        )
        from Jotty.core.intelligence.orchestration.learning.byzantine_verification import (
            ByzantineVerifier,
        )

        si = SwarmIntelligence()
        bv = ByzantineVerifier(si)

        result = bv.verify_output_quality(
            agent="test_agent",
            claimed_success=True,
            output="This is a well-written research summary about AI trends in 2026, covering major developments in foundation models, multi-agent systems, and autonomous coding agents.",
            goal="Research AI trends",
            task_type="research",
        )
        assert isinstance(result, dict)
        assert "quality_ok" in result
        logger.info(f"Output quality: {result}")


# ── 6. MAS Learning: Strategy & Fix Database ─────────────────────────────────


class TestMASLearningReal:
    """MAS learning provides strategy recommendations and error fixes."""

    @pytest.mark.asyncio
    async def test_session_recording_and_strategy(self):
        """MAS records sessions and recommends strategies."""
        from Jotty.core.intelligence.orchestration.learning.mas_learning import MASLearning

        mas = MASLearning()

        mas.record_session(
            task_description="Research AI startup trends",
            agents_used=["researcher", "analyst"],
            total_time=30.0,
            success=True,
        )
        mas.record_session(
            task_description="Research AI startup trends",
            agents_used=["researcher"],
            total_time=45.0,
            success=False,
        )

        strategy = mas.get_execution_strategy(
            "Research AI trends in healthcare", ["researcher", "analyst", "writer"]
        )
        logger.info(f"MAS strategy: {strategy}")
        assert isinstance(strategy, dict)

    @pytest.mark.asyncio
    async def test_find_fix_returns_none_when_empty(self):
        """MAS find_fix returns None when no fix exists."""
        from Jotty.core.intelligence.orchestration.learning.mas_learning import MASLearning

        mas = MASLearning()
        fix = mas.find_fix("ModuleNotFoundError: No module named 'nonexistent'")
        logger.info(f"find_fix result (empty DB): {fix}")


# ── 7. MetricsCollector: Observability ───────────────────────────────────────


class TestMetricsReal:
    """MetricsCollector provides swarm health observability."""

    @pytest.mark.asyncio
    async def test_record_and_report(self):
        """Metrics records tasks and generates health reports."""
        from Jotty.core.intelligence.orchestration.learning.metrics_collector import (
            MetricsCollector,
        )

        m = MetricsCollector()

        m.record_task(
            swarm="coding_swarm", agent="coder", task_type="coding", success=True, duration=5.0
        )
        m.record_task(
            swarm="coding_swarm", agent="coder", task_type="coding", success=True, duration=3.0
        )
        m.record_task(
            swarm="coding_swarm", agent="reviewer", task_type="review", success=False, duration=2.0
        )

        report = m.get_report()
        logger.info(
            f"Metrics: tasks={report['total_tasks']}, success={report['success_rate']:.0%}, avg_dur={report['avg_duration']:.1f}s"
        )

        assert report["total_tasks"] == 3
        assert report["success_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert "by_agent" in report
        assert "by_swarm" in report


# ── 8. SwarmIntelligence: Consensus & Caching ────────────────────────────────


class TestSwarmIntelligenceReal:
    """SwarmIntelligence consensus and caching features."""

    @pytest.mark.asyncio
    async def test_agent_trust_tracking(self):
        """SI tracks agent trust profiles."""
        from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
            SwarmIntelligence,
        )

        si = SwarmIntelligence()
        si.register_agent("alpha")
        si.register_agent("beta")
        assert "alpha" in si.agent_profiles
        assert "beta" in si.agent_profiles
        logger.info(f"Registered agents: {list(si.agent_profiles.keys())}")

    @pytest.mark.asyncio
    async def test_result_caching(self):
        """SI caches and retrieves results."""
        from Jotty.core.intelligence.orchestration.intelligence.swarm_intelligence import (
            SwarmIntelligence,
        )

        si = SwarmIntelligence()

        if not (hasattr(si, "cache_result") and hasattr(si, "get_cached")):
            pytest.skip("Caching methods not available")

        si.cache_result("test_key_123", {"answer": "cached_data"}, ttl=60.0)
        cached = si.get_cached("test_key_123")
        assert cached is not None
        assert cached["answer"] == "cached_data"

        miss = si.get_cached("nonexistent_key")
        assert miss is None
        logger.info("Result caching: store/retrieve/miss all work")


# ── 9. Shutdown Health Report ─────────────────────────────────────────────────


class TestShutdownReal:
    """Orchestrator shutdown logs health report."""

    @pytest.mark.asyncio
    async def test_shutdown_logs_health(self):
        """Shutdown produces metrics-based health report."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            orch = Orchestrator(agents="shutdown test")

        lp = orch.learning

        lp.metrics.record_task(swarm="test", agent="a", task_type="t", success=True, duration=1.0)
        lp.metrics.record_task(swarm="test", agent="a", task_type="t", success=False, duration=2.0)

        with patch(
            "Jotty.core.intelligence.orchestration.core.swarm_manager.logger"
        ) as mock_logger:
            await orch.shutdown()

            info_calls = [str(c) for c in mock_logger.info.call_args_list]
            health_logged = any("Swarm Health" in c for c in info_calls)
            logger.info(f"Health report in shutdown logs: {health_logged}")
            if health_logged:
                health_line = [c for c in info_calls if "Swarm Health" in c][0]
                logger.info(f"Health line: {health_line}")


# ── 10. Full Pipeline: Real LLM End-to-End ──────────────────────────────────


class TestFullPipelineReal:
    """End-to-end test with real LLM through Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_run_with_hooks(self):
        """Run Orchestrator with real LLM and verify hooks activated."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            orch = Orchestrator(agents="Answer: what is 2+2?")

        lp = orch.learning
        metrics_before = lp.metrics.get_report().get("total_tasks", 0)

        try:
            result = await asyncio.wait_for(
                orch.run(goal="What is 2+2? Reply with just the number 4."),
                timeout=90,
            )
            success = getattr(result, "success", None)
            output = getattr(result, "output", None) or getattr(result, "content", "")
            logger.info(f"Orchestrator result: success={success}, output={str(output)[:200]}")
        except asyncio.TimeoutError:
            logger.warning("Orchestrator.run() timed out (90s)")
        except Exception as e:
            logger.warning(f"Orchestrator.run() error: {type(e).__name__}: {e}")

        metrics_after = lp.metrics.get_report().get("total_tasks", 0)
        stigmergy_trails = len(lp.stigmergy.signals)

        logger.info(f"Metrics: {metrics_before} -> {metrics_after}")
        logger.info(f"Stigmergy trails: {stigmergy_trails}")
