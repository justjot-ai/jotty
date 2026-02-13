"""
Tests for Orchestrator improvements:

1. Background training daemon (start/stop/status)
2. Intelligence effectiveness A/B metrics
3. Auto paradigm selection (Thompson sampling)
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg():
    cfg = SwarmConfig()
    cfg.base_path = "/tmp/jotty_test_improvements"
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
    """Create a minimal agent object."""
    agent = MagicMock()
    agent.name = name
    agent.config = MagicMock()
    agent.config.system_prompt = None
    return agent


def _make_swarm(agents=None, enable_zero_config=False):
    """Create a Orchestrator with dummy agents."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator

    if agents is None:
        agents = [
            AgentConfig(name="alpha", agent=_make_dummy_agent("alpha"), capabilities=["Analyze data"]),
            AgentConfig(name="beta", agent=_make_dummy_agent("beta"), capabilities=["Summarize"]),
        ]

    sm = Orchestrator(
        agents=agents,
        config=_cfg(),
        enable_zero_config=enable_zero_config,
    )
    return sm


# ===========================================================================
# 1. Background Training Daemon
# ===========================================================================

class TestTrainingDaemon:
    """Tests for start_training_daemon / stop_training_daemon / training_daemon_status."""

    def test_daemon_status_when_idle(self):
        """Daemon status should show 'not running' before start."""
        sm = _make_swarm()
        status = sm.training_daemon_status()
        assert status['running'] is False
        assert status['completed'] == 0
        assert status['success_rate'] == 0.0

    @pytest.mark.asyncio
    async def test_daemon_starts_and_runs(self):
        """Daemon should start as a background task and complete."""
        sm = _make_swarm()

        # Mock learning to provide 2 training tasks then empty
        tasks = [
            MagicMock(description="Train task 1", difficulty=0.5),
            MagicMock(description="Train task 2", difficulty=0.6),
        ]
        task_iter = iter(tasks)

        def pop_task():
            try:
                return next(task_iter)
            except StopIteration:
                return None

        sm._lazy_learning = True  # Mark as initialized

        # Mock the learning pipeline
        mock_lp = MagicMock()
        mock_lp.pop_training_task = pop_task
        mock_lp.pending_training_count.return_value = 0
        mock_lp.adaptive_learning.state.is_converging = False
        mock_lp.adaptive_learning.should_stop_early.return_value = False

        # Mock the run method to return success
        async def mock_run(**kwargs):
            return _episode(success=True, output="trained")

        with patch.object(sm, '_lazy_learning', mock_lp), \
             patch.object(type(sm), 'learning', property(lambda self: mock_lp)), \
             patch.object(sm, 'run', side_effect=mock_run):

            started = sm.start_training_daemon(max_tasks=5, interval_seconds=0)
            assert started is True

            # Verify it's running
            status = sm.training_daemon_status()
            assert status['running'] is True

            # Wait for it to complete
            await sm._training_daemon

            status = sm.training_daemon_status()
            assert status['running'] is False
            assert status['completed'] == 2
            assert status['succeeded'] == 2
            assert status['success_rate'] == 1.0

    @pytest.mark.asyncio
    async def test_daemon_cannot_start_twice(self):
        """Starting daemon twice should return False."""
        sm = _make_swarm()

        # Create a daemon that runs forever
        async def _long_running():
            await asyncio.sleep(100)

        sm._training_daemon = asyncio.ensure_future(_long_running())

        started = sm.start_training_daemon()
        assert started is False

        # Cleanup
        sm._training_daemon.cancel()
        try:
            await sm._training_daemon
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_daemon_stop(self):
        """Stop should cancel the running daemon."""
        sm = _make_swarm()

        async def _long_running():
            await asyncio.sleep(100)

        sm._training_daemon = asyncio.ensure_future(_long_running())

        stopped = sm.stop_training_daemon()
        assert stopped is True

        # Already stopped
        await asyncio.sleep(0.1)
        stopped_again = sm.stop_training_daemon()
        assert stopped_again is False

    def test_daemon_in_status(self):
        """status() should include training_daemon info."""
        sm = _make_swarm()
        s = sm.status()
        assert 'training_daemon' in s
        assert s['training_daemon']['running'] is False


# ===========================================================================
# 2. Intelligence Effectiveness A/B Metrics
# ===========================================================================

class TestIntelligenceMetrics:
    """Tests for intelligence effectiveness A/B tracking (per-task-type)."""

    def test_initial_metrics_empty(self):
        """Metrics should start empty (no task types yet)."""
        sm = _make_swarm()
        assert sm._intelligence_metrics == {}

    def test_metrics_in_status(self):
        """status() should include intelligence_effectiveness."""
        sm = _make_swarm()
        s = sm.status()
        assert 'intelligence_effectiveness' in s
        ie = s['intelligence_effectiveness']
        assert ie == {}  # No runs yet

    def test_guided_run_tracking(self):
        """Simulating a guided run should increment counters per task_type."""
        sm = _make_swarm()

        # Simulate intelligence guidance applied for 'analysis' tasks
        sm._intelligence_metrics['analysis'] = {
            'guided_runs': 3, 'guided_successes': 2,
            'unguided_runs': 5, 'unguided_successes': 2,
        }
        sm._intelligence_metrics['_global'] = {
            'guided_runs': 3, 'guided_successes': 2,
            'unguided_runs': 5, 'unguided_successes': 2,
        }

        s = sm.status()
        ie = s['intelligence_effectiveness']
        assert ie['analysis']['guided_success_rate'] == pytest.approx(2 / 3)
        assert ie['analysis']['unguided_success_rate'] == pytest.approx(2 / 5)
        lift = ie['analysis']['guidance_lift']
        assert lift > 0  # Guided is better in this case

    def test_post_episode_records_success(self):
        """_post_episode_learning should credit guided/unguided success."""
        sm = _make_swarm()

        # Mock the learning pipeline
        mock_lp = MagicMock()
        mock_lp.episode_count = 1
        mock_lp.record_paradigm_result = MagicMock()
        mock_lp.transfer_learning.extractor.extract_task_type.return_value = 'coding'

        with patch.object(type(sm), 'learning', property(lambda self: mock_lp)):
            # Pre-populate buckets (as _execute_multi_agent would)
            sm._intelligence_metrics = {
                'coding': {
                    'guided_runs': 1, 'guided_successes': 0,
                    'unguided_runs': 0, 'unguided_successes': 0,
                },
                '_global': {
                    'guided_runs': 1, 'guided_successes': 0,
                    'unguided_runs': 0, 'unguided_successes': 0,
                },
            }

            # Simulate guided run success
            sm._last_run_guided = True
            sm._last_task_type = 'coding'
            sm._last_paradigm = 'fanout'
            sm._post_episode_learning(_episode(success=True), "test goal")
            assert sm._intelligence_metrics['coding']['guided_successes'] == 1
            assert sm._intelligence_metrics['_global']['guided_successes'] == 1

            # Add unguided bucket
            sm._intelligence_metrics['coding']['unguided_runs'] = 1
            sm._intelligence_metrics['_global']['unguided_runs'] = 1

            # Simulate unguided run success
            sm._last_run_guided = False
            sm._post_episode_learning(_episode(success=True), "test goal 2")
            assert sm._intelligence_metrics['coding']['unguided_successes'] == 1

            # Failed run should NOT increment successes
            sm._last_run_guided = True
            sm._post_episode_learning(_episode(success=False), "test goal 3")
            assert sm._intelligence_metrics['coding']['guided_successes'] == 1  # unchanged


# ===========================================================================
# 3. Auto Paradigm Selection
# ===========================================================================

class TestAutoParadigmSelection:
    """Tests for Thompson-sampling paradigm auto-selection."""

    def test_recommend_paradigm_no_data(self):
        """With no history, should return a valid paradigm."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())
        paradigm = lp.recommend_paradigm()
        assert paradigm in ('fanout', 'relay', 'debate', 'refinement')

    def test_recommend_favors_successful_paradigm(self):
        """After recording successes, the winning paradigm should be favored."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())

        # Simulate: 'relay' always succeeds, others always fail
        for _ in range(20):
            lp.record_paradigm_result('relay', True)
            lp.record_paradigm_result('fanout', False)
            lp.record_paradigm_result('debate', False)
            lp.record_paradigm_result('refinement', False)

        # With 20 successes vs 0, relay should win most of the time
        selections = [lp.recommend_paradigm() for _ in range(50)]
        relay_count = selections.count('relay')
        assert relay_count > 30, f"Expected relay to dominate, got {relay_count}/50"

    def test_record_paradigm_result(self):
        """record_paradigm_result should update stats correctly."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())

        lp.record_paradigm_result('debate', True)
        lp.record_paradigm_result('debate', True)
        lp.record_paradigm_result('debate', False)

        # Default task_type is '_global'
        stats = lp.get_paradigm_stats('_global')
        assert stats['debate']['runs'] == 3
        assert stats['debate']['successes'] == 2
        assert stats['debate']['success_rate'] == pytest.approx(2 / 3)

    def test_record_with_task_type(self):
        """Recording with task_type should update both task-specific and _global."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())

        lp.record_paradigm_result('relay', True, task_type='analysis')
        lp.record_paradigm_result('relay', True, task_type='analysis')
        lp.record_paradigm_result('debate', True, task_type='writing')

        # Task-specific stats
        analysis_stats = lp.get_paradigm_stats('analysis')
        assert analysis_stats['relay']['runs'] == 2
        assert analysis_stats['relay']['successes'] == 2

        writing_stats = lp.get_paradigm_stats('writing')
        assert writing_stats['debate']['runs'] == 1

        # _global should have all 3
        global_stats = lp.get_paradigm_stats('_global')
        assert global_stats['relay']['runs'] == 2
        assert global_stats['debate']['runs'] == 1

    def test_recommend_uses_task_type(self):
        """recommend_paradigm should prefer task-specific data when available."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())

        # 'relay' dominates for 'analysis' tasks
        for _ in range(20):
            lp.record_paradigm_result('relay', True, task_type='analysis')
            lp.record_paradigm_result('debate', False, task_type='analysis')
            lp.record_paradigm_result('fanout', False, task_type='analysis')

        # 'debate' dominates for 'writing' tasks
        for _ in range(20):
            lp.record_paradigm_result('debate', True, task_type='writing')
            lp.record_paradigm_result('relay', False, task_type='writing')
            lp.record_paradigm_result('fanout', False, task_type='writing')

        # For analysis, relay should dominate
        analysis_picks = [lp.recommend_paradigm('analysis') for _ in range(50)]
        assert analysis_picks.count('relay') > 30, (
            f"Expected relay for analysis, got {analysis_picks.count('relay')}/50"
        )

        # For writing, debate should dominate
        writing_picks = [lp.recommend_paradigm('writing') for _ in range(50)]
        assert writing_picks.count('debate') > 30, (
            f"Expected debate for writing, got {writing_picks.count('debate')}/50"
        )

    def test_record_unknown_paradigm(self):
        """Recording a new paradigm should auto-create its entry."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())

        lp.record_paradigm_result('custom_paradigm', True)
        assert lp._paradigm_stats['_global']['custom_paradigm']['runs'] == 1

    def test_paradigm_stats_persistence(self):
        """Paradigm stats should survive save/load cycle."""
        import tempfile
        import json
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline

        cfg = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.base_path = tmpdir
            lp1 = SwarmLearningPipeline(cfg)

            # Record some paradigm results with task types
            lp1.record_paradigm_result('relay', True, task_type='analysis')
            lp1.record_paradigm_result('relay', True, task_type='analysis')
            lp1.record_paradigm_result('debate', False, task_type='writing')
            lp1.auto_save()

            # Load in a fresh pipeline
            lp2 = SwarmLearningPipeline(cfg)
            lp2.auto_load()

            # Task-specific survived
            a_stats = lp2.get_paradigm_stats('analysis')
            assert a_stats['relay']['runs'] == 2
            assert a_stats['relay']['successes'] == 2

            w_stats = lp2.get_paradigm_stats('writing')
            assert w_stats['debate']['runs'] == 1
            assert w_stats['debate']['successes'] == 0

            # Global survived
            g_stats = lp2.get_paradigm_stats('_global')
            assert g_stats['relay']['runs'] == 2
            assert g_stats['debate']['runs'] == 1

    def test_backward_compat_old_format(self):
        """Loading old flat-format paradigm_stats should auto-migrate to nested."""
        import tempfile
        import json
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline

        cfg = _cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.base_path = tmpdir

            # Write old-format stigmergy file (signals must be a dict, not list)
            old_data = {
                'signals': {},
                'decay_rate': 0.1,
                'paradigm_stats': {
                    'fanout': {'runs': 5, 'successes': 3},
                    'relay': {'runs': 2, 'successes': 2},
                    'debate': {'runs': 0, 'successes': 0},
                    'refinement': {'runs': 0, 'successes': 0},
                },
            }
            stig_path = tmpdir + '/stigmergy.json'
            with open(stig_path, 'w') as f:
                json.dump(old_data, f)

            lp = SwarmLearningPipeline(cfg)
            lp.auto_load()

            # Old format should be migrated under '_global'
            g_stats = lp.get_paradigm_stats('_global')
            assert g_stats['fanout']['runs'] == 5
            assert g_stats['relay']['runs'] == 2

    @pytest.mark.asyncio
    async def test_auto_paradigm_dispatch(self):
        """discussion_paradigm='auto' should pick and dispatch to a paradigm."""
        sm = _make_swarm()

        # Mock the learning pipeline to recommend 'relay'
        mock_lp = MagicMock()
        mock_lp.recommend_paradigm.return_value = 'relay'
        mock_lp.transfer_learning.extractor.extract_task_type.return_value = 'analysis'
        mock_lp.is_agent_trusted.return_value = True
        mock_lp.stigmergy.get_route_signals.return_value = {}
        mock_lp.episode_count = 0
        mock_lp.record_paradigm_result = MagicMock()

        # Mock _paradigm_relay to verify it's called
        relay_called = False
        original_relay = sm._paradigm_relay

        async def mock_relay(goal, **kwargs):
            nonlocal relay_called
            relay_called = True
            return _episode(success=True, output="relayed")

        sm._ensure_runners()

        with patch.object(type(sm), 'learning', property(lambda self: mock_lp)), \
             patch.object(sm, '_paradigm_relay', side_effect=mock_relay):

            result = await sm._execute_multi_agent(
                "Test auto paradigm", discussion_paradigm='auto'
            )

            assert relay_called, "Expected _paradigm_relay to be called via auto selection"
            assert sm._last_paradigm == 'relay'

    def test_get_paradigm_stats_initial(self):
        """Fresh pipeline should have empty stats."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())
        stats = lp.get_paradigm_stats()
        # No data recorded yet — empty dict
        assert stats == {}

    def test_recommend_falls_back_to_global(self):
        """With <5 task-specific runs, should fall back to _global stats."""
        from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
        lp = SwarmLearningPipeline(_cfg())

        # Only 2 task-specific runs (below threshold of 5)
        lp.record_paradigm_result('relay', True, task_type='rare_task')
        lp.record_paradigm_result('relay', True, task_type='rare_task')

        # 20 global runs strongly favoring debate
        for _ in range(20):
            lp.record_paradigm_result('debate', True)
            lp.record_paradigm_result('relay', False)

        # Should use _global (debate-heavy), not 'rare_task' (relay-heavy but sparse)
        picks = [lp.recommend_paradigm('rare_task') for _ in range(50)]
        debate_count = picks.count('debate')
        assert debate_count > 25, (
            f"Expected _global fallback to favor debate, got {debate_count}/50"
        )
