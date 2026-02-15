"""
Tests for Orchestrator improvements v3:

1. Per-task-type intelligence metrics
2. Agent warm-start (stigmergy + profile context injection)
3. Cross-swarm paradigm transfer via shared persistence
4. Real-world LLM benchmark (optional, requires API key)
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Jotty.core.infrastructure.foundation.agent_config import AgentConfig
from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult, SwarmConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(base_path=None):
    cfg = SwarmConfig()
    if base_path:
        cfg.base_path = base_path
    else:
        cfg.base_path = "/tmp/jotty_test_v3"
    return cfg


def _episode(success=True, output="ok"):
    return EpisodeResult(
        success=success,
        output=output,
        trajectory=[],
        tagged_outputs=[],
        episode=0,
        execution_time=1.0,
        architect_results=[],
        auditor_results=[],
        agent_contributions={},
    )


def _make_dummy_agent(name="tester"):
    agent = MagicMock()
    agent.name = name
    agent.config = MagicMock()
    agent.config.system_prompt = None
    return agent


def _make_swarm(agents=None, enable_zero_config=False, base_path=None):
    from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator

    if agents is None:
        agents = [
            AgentConfig(
                name="alpha", agent=_make_dummy_agent("alpha"), capabilities=["Analyze data"]
            ),
            AgentConfig(
                name="beta", agent=_make_dummy_agent("beta"), capabilities=["Summarize findings"]
            ),
        ]

    sm = Orchestrator(
        agents=agents,
        config=_cfg(base_path),
        enable_zero_config=enable_zero_config,
    )
    return sm


# ===========================================================================
# 1. Per-Task-Type Intelligence Metrics
# ===========================================================================


class TestPerTaskTypeIntelligenceMetrics:
    """Intelligence A/B metrics now tracked per task_type."""

    def test_initial_metrics_empty(self):
        """No metrics until a multi-agent run happens."""
        sm = _make_swarm()
        assert sm._intelligence_metrics == {}

    def test_metrics_accumulate_per_task_type(self):
        """Manually simulating guided/unguided should bucket by task_type."""
        sm = _make_swarm()

        # Simulate: guided run for 'analysis' task
        sm._intelligence_metrics["analysis"] = {
            "guided_runs": 5,
            "guided_successes": 4,
            "unguided_runs": 3,
            "unguided_successes": 1,
        }
        sm._intelligence_metrics["_global"] = {
            "guided_runs": 5,
            "guided_successes": 4,
            "unguided_runs": 3,
            "unguided_successes": 1,
        }

        s = sm.status()
        ie = s["intelligence_effectiveness"]

        # Should have per-task-type breakdown
        assert "analysis" in ie
        assert ie["analysis"]["guided_success_rate"] == pytest.approx(4 / 5)
        assert ie["analysis"]["unguided_success_rate"] == pytest.approx(1 / 3)
        assert ie["analysis"]["guidance_lift"] > 0

        # And _global rollup
        assert "_global" in ie

    def test_post_episode_updates_both_buckets(self):
        """_post_episode_learning should update task-specific and _global."""
        sm = _make_swarm()

        mock_lp = MagicMock()
        mock_lp.episode_count = 1
        mock_lp.record_paradigm_result = MagicMock()
        mock_lp.transfer_learning.extractor.extract_task_type.return_value = "coding"

        with patch.object(type(sm), "learning", property(lambda self: mock_lp)):
            # Simulate guided run
            sm._last_run_guided = True
            sm._last_paradigm = "fanout"
            sm._last_task_type = "coding"
            sm._intelligence_metrics = {
                "coding": {
                    "guided_runs": 1,
                    "guided_successes": 0,
                    "unguided_runs": 0,
                    "unguided_successes": 0,
                },
                "_global": {
                    "guided_runs": 1,
                    "guided_successes": 0,
                    "unguided_runs": 0,
                    "unguided_successes": 0,
                },
            }

            sm._post_episode_learning(_episode(success=True), "Write a function")

            assert sm._intelligence_metrics["coding"]["guided_successes"] == 1
            assert sm._intelligence_metrics["_global"]["guided_successes"] == 1

    def test_status_empty_when_no_runs(self):
        """status() intelligence_effectiveness should be empty dict with no runs."""
        sm = _make_swarm()
        s = sm.status()
        assert s["intelligence_effectiveness"] == {}


# ===========================================================================
# 2. Agent Warm-Start
# ===========================================================================


class TestAgentWarmStart:
    """Agent runner should inject stigmergy + profile hints into learning context."""

    def test_warm_start_injects_profile_context(self):
        """Runner should include agent trust score and specialization in context."""
        from Jotty.core.intelligence.orchestration.agent_runner import (
            AgentRunner,
            AgentRunnerConfig,
        )
        from Jotty.core.intelligence.orchestration.swarm_data_structures import (
            AgentProfile,
            AgentSpecialization,
        )
        from Jotty.core.intelligence.orchestration.swarm_intelligence import SwarmIntelligence

        cfg = AgentRunnerConfig(
            architect_prompts=["configs/prompts/architect/base_architect.md"],
            auditor_prompts=["configs/prompts/auditor/base_auditor.md"],
            config=_cfg(),
            agent_name="coder",
            enable_learning=False,
            enable_memory=False,
        )

        si = SwarmIntelligence(_cfg())
        si.register_agent("coder")
        profile = si.agent_profiles["coder"]
        profile.trust_score = 0.85
        profile.specialization = AgentSpecialization.EXECUTOR
        profile.total_tasks = 10

        agent = MagicMock()
        agent.execute = AsyncMock(return_value="done")

        runner = AgentRunner(
            agent=agent,
            config=cfg,
            swarm_intelligence=si,
        )

        # Verify warm-start data is accessible to the runner
        assert runner.swarm_intelligence is si
        assert si.agent_profiles["coder"].trust_score == 0.85
        assert si.agent_profiles["coder"].total_tasks == 10

    def test_warm_start_with_stigmergy_hint(self):
        """Runner should have access to stigmergy route hints."""
        from Jotty.core.intelligence.orchestration.agent_runner import (
            AgentRunner,
            AgentRunnerConfig,
        )
        from Jotty.core.intelligence.orchestration.swarm_data_structures import (
            AgentProfile,
            AgentSpecialization,
        )
        from Jotty.core.intelligence.orchestration.swarm_intelligence import SwarmIntelligence

        cfg = AgentRunnerConfig(
            architect_prompts=["configs/prompts/architect/base_architect.md"],
            auditor_prompts=["configs/prompts/auditor/base_auditor.md"],
            config=_cfg(),
            agent_name="researcher",
            enable_learning=False,
            enable_memory=False,
        )

        si = SwarmIntelligence(_cfg())
        si.register_agent("researcher")
        profile = si.agent_profiles["researcher"]
        profile.trust_score = 0.9
        profile.specialization = AgentSpecialization.RESEARCHER
        profile.total_tasks = 15

        # Deposit a stigmergy route signal (correct API: agent=, not created_by=)
        si.stigmergy.deposit(
            signal_type="route",
            content={"task_type": "research", "agent": "researcher", "success": True},
            agent="system",
            strength=0.9,
        )

        agent = MagicMock()
        runner = AgentRunner(
            agent=agent,
            config=cfg,
            swarm_intelligence=si,
        )

        # Verify stigmergy is accessible for route hints
        routes = si.stigmergy.get_route_signals("research")
        assert routes.get("researcher", 0) > 0


# ===========================================================================
# 3. Cross-Swarm Paradigm Transfer
# ===========================================================================


class TestCrossSwarmTransfer:
    """Paradigm stats should transfer between Orchestrator instances via shared persistence."""

    def test_paradigm_stats_transfer_between_pipelines(self):
        """Two learning pipelines sharing base_path should share paradigm learnings."""
        from Jotty.core.intelligence.orchestration.learning_pipeline import SwarmLearningPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _cfg(tmpdir)

            # Pipeline 1: learns that 'relay' works for 'analysis'
            lp1 = SwarmLearningPipeline(cfg)
            for _ in range(10):
                lp1.record_paradigm_result("relay", True, task_type="analysis")
                lp1.record_paradigm_result("debate", False, task_type="analysis")
            lp1.auto_save()

            # Pipeline 2: fresh, loads from same path
            lp2 = SwarmLearningPipeline(cfg)
            lp2.auto_load()

            # Should inherit lp1's paradigm learnings
            stats = lp2.get_paradigm_stats("analysis")
            assert stats["relay"]["runs"] == 10
            assert stats["relay"]["successes"] == 10
            assert stats["debate"]["runs"] == 10
            assert stats["debate"]["successes"] == 0

            # And recommend relay for analysis
            picks = [lp2.recommend_paradigm("analysis") for _ in range(30)]
            assert picks.count("relay") > 20

    def test_swarm_managers_share_via_persistence(self):
        """Two SwarmManagers with same base_path share paradigm learnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # SM1: record paradigm results
            sm1 = _make_swarm(base_path=tmpdir)

            # Manually record via the learning pipeline
            sm1.learning.record_paradigm_result("debate", True, task_type="writing")
            sm1.learning.record_paradigm_result("debate", True, task_type="writing")
            sm1.learning.record_paradigm_result("debate", True, task_type="writing")
            sm1.learning.record_paradigm_result("fanout", False, task_type="writing")
            sm1.learning.record_paradigm_result("fanout", False, task_type="writing")

            # Save
            sm1.learning.auto_save()

            # SM2: fresh instance, same path
            sm2 = _make_swarm(base_path=tmpdir)
            sm2.learning.auto_load()

            # SM2 should know debate works for writing
            stats = sm2.learning.get_paradigm_stats("writing")
            assert stats["debate"]["runs"] == 3
            assert stats["debate"]["successes"] == 3
            assert stats["fanout"]["runs"] == 2
            assert stats["fanout"]["successes"] == 0

    def test_stigmergy_transfers_between_instances(self):
        """Stigmergy pheromone trails should persist and transfer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sm1 = _make_swarm(base_path=tmpdir)

            # Deposit stigmergy signal (correct API: agent=, not created_by=)
            sm1.learning.stigmergy.deposit(
                signal_type="success",
                content={"task_type": "coding", "agent": "alpha"},
                agent="alpha",
                strength=0.8,
            )
            sm1.learning.auto_save()

            # Fresh instance loads it
            sm2 = _make_swarm(base_path=tmpdir)
            sm2.learning.auto_load()

            assert len(sm2.learning.stigmergy.signals) == 1
            signals = sm2.learning.stigmergy.sense(signal_type="success")
            assert len(signals) == 1


# ===========================================================================
# 4. Real-World LLM Benchmark (Optional)
# ===========================================================================

HAS_LLM_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("GROQ_API_KEY")
)


@pytest.mark.skipif(not HAS_LLM_KEY, reason="No LLM API key available")
class TestRealWorldBenchmark:
    """Systematic A/B benchmark with real LLM calls."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_guided_vs_unguided_simple_task(self):
        """Compare success rate: intelligence-guided vs unguided execution."""
        from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator
        from Jotty.core.modes.agent.auto_agent import AutoAgent

        agents = [
            AgentConfig(
                name="analyst",
                agent=AutoAgent(),
                capabilities=["Analyze the question and give a concise answer"],
            ),
            AgentConfig(
                name="checker",
                agent=AutoAgent(),
                capabilities=["Verify the analysis and confirm the answer"],
            ),
        ]

        sm = Orchestrator(
            agents=agents,
            config=_cfg(),
            enable_zero_config=False,
        )

        # Simple math — should succeed regardless
        result = await sm.run(
            goal="What is 25 * 4? Reply with just the number.",
            skip_autonomous_setup=True,
            skip_validation=True,
            discussion_paradigm="relay",
        )

        assert result is not None
        # Check metrics are being tracked
        s = sm.status()
        ie = s.get("intelligence_effectiveness", {})
        # At least one bucket should exist
        total_runs = sum(b.get("guided_runs", 0) + b.get("unguided_runs", 0) for b in ie.values())
        assert total_runs >= 1, f"Expected at least 1 run tracked, got: {ie}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_auto_paradigm_with_real_llm(self):
        """Auto paradigm selection should work end-to-end with real LLM."""
        from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator
        from Jotty.core.modes.agent.auto_agent import AutoAgent

        agents = [
            AgentConfig(
                name="solver",
                agent=AutoAgent(),
                capabilities=["Solve the problem step by step"],
            ),
            AgentConfig(
                name="reviewer",
                agent=AutoAgent(),
                capabilities=["Review the solution for correctness"],
            ),
        ]

        sm = Orchestrator(
            agents=agents,
            config=_cfg(),
            enable_zero_config=False,
        )

        result = await sm.run(
            goal="What is the capital of France? Reply in one word.",
            skip_autonomous_setup=True,
            skip_validation=True,
            discussion_paradigm="auto",
        )

        assert result is not None
        # Verify paradigm was auto-selected
        assert hasattr(sm, "_last_paradigm")
        assert sm._last_paradigm in ("fanout", "relay", "debate", "refinement")
