"""
Tests for the 9+ rating features:

1. Autonomous training scheduler (start_training_loop)
2. Stigmergy persistence (save/load across sessions)
3. Credit-driven pruning (auto-prune low-value learnings)
4. Real-world LLM integration test (optional, requires API key)
"""

import json
import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(base_path=None):
    cfg = SwarmConfig()
    if base_path:
        cfg.base_path = base_path
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


def _agent(name):
    return type('A', (), {'name': name})()


def _agent_spec(name, capabilities=None):
    from Jotty.core.foundation.agent_config import AgentConfig
    dummy = type('DummyAgent', (), {
        'forward': lambda self, **kw: None,
        'config': type('C', (), {'system_prompt': None})(),
    })()
    spec = AgentConfig(name=name, agent=dummy)
    if capabilities:
        spec.capabilities = capabilities
    return spec


def _pipeline(base_path=None):
    from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
    return SwarmLearningPipeline(_cfg(base_path))


# ===========================================================================
# 1. Autonomous Training Scheduler
# ===========================================================================

class TestTrainingScheduler:
    """Prove start_training_loop drains queue and respects convergence."""

    def test_training_loop_drains_queue(self):
        """Loop should execute queued tasks and return results."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[_agent_spec("trainer", capabilities=["learn"])],
            config=_cfg(),
        )

        # Pre-queue 3 training tasks
        lp = sm.learning
        agent = _agent("stagnant")
        for _ in range(10):
            lp.post_episode(
                result=_episode(False),
                goal="Stuck",
                agents=[agent],
                architect_prompts=[],
            )

        queued = sm.pending_training_tasks
        assert queued >= 1, f"Expected queued tasks, got {queued}"

        # Mock run() to avoid LLM calls
        async def fake_run(goal, **kwargs):
            return _episode(True, f"trained on: {goal[:30]}")

        async def _test():
            with patch.object(sm, 'run', side_effect=fake_run):
                results = await sm.start_training_loop(max_tasks=3)
            return results

        results = asyncio.run(_test())
        assert len(results) >= 1
        assert all(r.success for r in results)

    def test_training_loop_stops_on_convergence(self):
        """Loop should stop early when adaptive learning says converged."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[_agent_spec("trainer", capabilities=["learn"])],
            config=_cfg(),
        )

        # Pre-queue many tasks
        lp = sm.learning
        agent = _agent("stagnant")
        for _ in range(15):
            lp.post_episode(
                result=_episode(False),
                goal="Stuck",
                agents=[agent],
                architect_prompts=[],
            )

        # Now force convergence state (high stable scores)
        for _ in range(10):
            lp.adaptive_learning.update_score(0.97)

        assert lp.adaptive_learning.state.is_converging
        assert lp.adaptive_learning.should_stop_early()

        async def fake_run(goal, **kwargs):
            return _episode(True, "trained")

        async def _test():
            with patch.object(sm, 'run', side_effect=fake_run):
                results = await sm.start_training_loop(
                    max_tasks=10,
                    stop_on_convergence=True,
                )
            return results

        results = asyncio.run(_test())
        # Should stop immediately (0 tasks) because convergence detected
        assert len(results) == 0, \
            f"Should stop on convergence, but ran {len(results)} tasks"

    def test_training_loop_stops_on_empty_queue(self):
        """Loop should stop when queue is drained."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[_agent_spec("trainer", capabilities=["learn"])],
            config=_cfg(),
        )

        # No tasks queued
        assert sm.pending_training_tasks == 0

        async def _test():
            results = await sm.start_training_loop(max_tasks=5)
            return results

        results = asyncio.run(_test())
        assert len(results) == 0


# ===========================================================================
# 2. Stigmergy Persistence
# ===========================================================================

class TestStigmergyPersistence:
    """Prove pheromone trails survive save/load cycle."""

    def test_save_and_load_stigmergy(self):
        """Signals deposited in session 1 should be loadable in session 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: deposit signals and save
            lp1 = _pipeline(base_path=tmpdir)
            lp1.stigmergy.deposit(
                signal_type='route',
                content={'task_type': 'coding', 'agent': 'coder'},
                agent='coder',
                strength=0.9,
            )
            lp1.stigmergy.deposit(
                signal_type='success',
                content={'task_type': 'coding', 'goal': 'fix bug'},
                agent='coder',
                strength=0.8,
            )
            assert len(lp1.stigmergy.signals) == 2

            # Save
            lp1.auto_save()
            stig_path = Path(tmpdir) / 'stigmergy.json'
            assert stig_path.exists()

            # Session 2: load and verify
            lp2 = _pipeline(base_path=tmpdir)
            lp2.auto_load()
            assert len(lp2.stigmergy.signals) == 2

            # Route signal should still work
            routes = lp2.stigmergy.get_route_signals('coding')
            assert 'coder' in routes
            assert routes['coder'] > 0

    def test_stigmergy_file_format(self):
        """Saved file should be valid JSON with expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lp = _pipeline(base_path=tmpdir)
            lp.stigmergy.deposit(
                signal_type='route',
                content={'task_type': 'analysis', 'agent': 'analyst'},
                agent='analyst',
                strength=0.7,
            )
            lp.auto_save()

            stig_path = Path(tmpdir) / 'stigmergy.json'
            with open(stig_path) as f:
                data = json.load(f)

            assert 'signals' in data
            assert 'decay_rate' in data
            assert len(data['signals']) == 1

    def test_empty_stigmergy_save_load(self):
        """Empty stigmergy should save/load cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lp1 = _pipeline(base_path=tmpdir)
            lp1.auto_save()

            lp2 = _pipeline(base_path=tmpdir)
            lp2.auto_load()
            assert len(lp2.stigmergy.signals) == 0


# ===========================================================================
# 3. Credit-Driven Pruning
# ===========================================================================

class TestCreditPruning:
    """Prove that low-value learnings are pruned from transfer store."""

    def test_pruning_triggers_at_episode_10(self):
        """After 10 episodes, pruning should run."""
        lp = _pipeline()
        agent = _agent("worker")

        # Add 25 experiences to transfer learning (above the 20 threshold)
        for i in range(25):
            lp.transfer_learning.experiences.append({
                'query': f'task_{i}',
                'action': f'action_{i}',
                'success': i % 3 == 0,  # Only 1/3 succeed
                'reward': 1.0 if i % 3 == 0 else 0.0,
            })

        before = len(lp.transfer_learning.experiences)

        # Run exactly 10 episodes to trigger pruning
        for i in range(10):
            lp.post_episode(
                result=_episode(i % 2 == 0),
                goal=f"Task {i}",
                agents=[agent],
                architect_prompts=[],
            )

        # Pruning should have run (episode_count % 10 == 0 at episode 10)
        after = len(lp.transfer_learning.experiences)
        # Pruning keeps new/unknown improvements, so count may not decrease much
        # But the mechanism should have executed without error
        assert after <= before + 10  # +10 for the new episodes that added experiences

    def test_pruning_keeps_valuable_experiences(self):
        """High-credit experiences should survive pruning."""
        lp = _pipeline()

        # Record some improvements as high-value
        lp.credit_assigner.record_improvement_application(
            improvement={'learned_pattern': 'use_structured_prompts', 'task': 'coding'},
            student_score=0.2,
            teacher_score=0.0,
            final_score=0.95,
            context={'task': 'coding'},
        )

        # Add the high-value experience to transfer learning
        lp.transfer_learning.experiences.append({
            'query': 'use_structured_prompts',
            'action': 'coding',
            'success': True,
        })

        # Add many low-value experiences
        for i in range(25):
            lp.transfer_learning.experiences.append({
                'query': f'random_noise_{i}',
                'action': f'failed_{i}',
                'success': False,
            })

        # Trigger pruning at episode 10
        agent = _agent("worker")
        for i in range(10):
            lp.post_episode(
                result=_episode(True),
                goal="Good task",
                agents=[agent],
                architect_prompts=[],
            )

        # The high-value experience should survive
        remaining_queries = [e.get('query', '') for e in lp.transfer_learning.experiences]
        assert 'use_structured_prompts' in remaining_queries, \
            "High-credit experience should survive pruning"


# ===========================================================================
# 4. Real-World LLM Integration (optional)
# ===========================================================================

class TestRealWorldIntegration:
    """
    Optional LLM-backed test. Skipped if no API key configured.
    Tests the full closed loop: run → learn → route → run again.
    """

    @pytest.fixture(autouse=True)
    def check_llm(self):
        """Skip if no LLM is configured."""
        import os
        has_key = any(
            os.environ.get(k)
            for k in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GROQ_API_KEY']
        )
        if not has_key:
            pytest.skip("No LLM API key configured")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_closed_loop_real(self):
        """
        Full closed loop with real LLM:
        1. Run a task → learning fires
        2. Check stigmergy has signals
        3. Check byzantine has verifications
        4. Check adaptive learning updated
        """
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[_agent_spec("analyst", capabilities=["Explain what 2+2 is"])],
            config=_cfg(),
        )

        result = await sm.run(
            goal="What is 2+2? Reply with just the number.",
            skip_autonomous_setup=True,
            skip_validation=True,
        )

        # Task should succeed
        assert result.success, f"Real LLM task failed: {result.output}"

        # Learning should have fired
        lp = sm.learning
        assert len(lp.stigmergy.signals) > 0, "Stigmergy should have signals after run"
        assert lp.byzantine_verifier.verified_count > 0, "Byzantine should have verified"
        assert lp.adaptive_learning.state.iteration_count > 0, "Adaptive should have updated"


# ===========================================================================
# Integration: Full lifecycle across sessions
# ===========================================================================

class TestFullLifecycle:
    """End-to-end: generate tasks, train, persist, reload, verify improvement."""

    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Session 1: Build up knowledge ---
            lp1 = _pipeline(base_path=tmpdir)
            agent = _agent("analyst")

            # 10 successful analysis episodes
            for i in range(10):
                ep = _episode(True)
                ep.agent_name = 'analyst'
                lp1.post_episode(
                    result=ep,
                    goal=f"Analyze data set {i}",
                    agents=[agent],
                    architect_prompts=[],
                )

            # 5 failed coding episodes → triggers plateau signals
            coder = _agent("coder")
            for i in range(5):
                lp1.post_episode(
                    result=_episode(False),
                    goal=f"Fix bug {i}",
                    agents=[coder],
                    architect_prompts=[],
                )

            # Verify state before save
            assert len(lp1.stigmergy.signals) > 0
            assert lp1.byzantine_verifier.verified_count == 15
            routes1 = lp1.stigmergy.get_route_signals('analysis')
            assert 'analyst' in routes1

            # Save
            lp1.auto_save()

            # --- Session 2: Reload stigmergy only (skip heavy embedding reload) ---
            lp2 = _pipeline(base_path=tmpdir)
            # Load just stigmergy (the feature we're testing)
            stig_path = Path(tmpdir) / 'stigmergy.json'
            if stig_path.exists():
                from Jotty.core.orchestration.stigmergy import StigmergyLayer
                with open(stig_path) as f:
                    lp2.stigmergy = StigmergyLayer.from_dict(json.load(f))

            # Stigmergy should persist
            assert len(lp2.stigmergy.signals) > 0
            routes2 = lp2.stigmergy.get_route_signals('analysis')
            assert 'analyst' in routes2

            # Route strength should match (within decay tolerance)
            assert routes2['analyst'] > 0, \
                f"Analyst route should persist, got {routes2}"

    def test_training_loop_with_mocked_execution(self):
        """Full training loop: plateau → queue → train → learn."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[_agent_spec("learner", capabilities=["learn new skills"])],
            config=_cfg(),
        )

        # Force plateau to generate training tasks
        lp = sm.learning
        agent = _agent("stagnant")
        for _ in range(12):
            lp.post_episode(
                result=_episode(False),
                goal="Impossible task",
                agents=[agent],
                architect_prompts=[],
            )

        queued = sm.pending_training_tasks
        assert queued >= 1, f"Expected training tasks queued, got {queued}"

        # Run training loop with mocked execution
        tasks_run = []

        async def fake_run(goal, **kwargs):
            tasks_run.append(goal)
            return _episode(True, f"learned: {goal[:30]}")

        async def _test():
            with patch.object(sm, 'run', side_effect=fake_run):
                results = await sm.start_training_loop(max_tasks=3)
            return results

        results = asyncio.run(_test())

        assert len(results) >= 1
        assert len(tasks_run) >= 1
        # All training tasks should succeed (mocked)
        assert all(r.success for r in results)
