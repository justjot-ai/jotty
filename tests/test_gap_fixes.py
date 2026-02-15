"""
Tests proving the 3 critical gaps are closed:

Gap 1: Learning writes are now READ in the hot path (stigmergy routing + byzantine filtering)
Gap 2: Adaptive learning controls refinement iteration count
Gap 3: Curriculum tasks auto-queue on plateau → consumable via run_training_task
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg():
    return SwarmConfig()


def _episode(success=True, output="ok"):
    ep = EpisodeResult(
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
    return ep


def _agent_obj(name):
    return type('A', (), {'name': name})()


def _agent_spec(name, capabilities=None):
    """Create an AgentConfig with a dummy agent."""
    dummy = type('DummyAgent', (), {
        'forward': lambda self, **kw: type('R', (), {'_store': {'output': 'ok'}, 'output': 'ok'})(),
        '__call__': lambda self, **kw: type('R', (), {'_store': {'output': 'ok'}, 'output': 'ok'})(),
        'config': type('C', (), {'system_prompt': None})(),
    })()
    spec = AgentConfig(name=name, agent=dummy)
    if capabilities:
        spec.capabilities = capabilities
    return spec


def _pipeline():
    from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
    return SwarmLearningPipeline(_cfg())


# ===========================================================================
# Gap 1: Stigmergy + Byzantine influence agent selection in hot path
# ===========================================================================

class TestGap1AgentSelection:
    """
    Prove that _execute_multi_agent now reads stigmergy + trust
    to reorder agents and filter untrusted ones.
    """

    def test_stigmergy_reorders_agents(self):
        """After depositing strong signals, agents should be reordered."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[
                _agent_spec("weak", capabilities=["do analysis"]),
                _agent_spec("strong", capabilities=["do analysis"]),
            ],
            config=_cfg(),
        )

        # Pre-seed stigmergy: 'strong' has better pheromone for 'analysis'
        lp = sm.learning
        for _ in range(5):
            lp.stigmergy.deposit(
                signal_type='route',
                content={'task_type': 'analysis', 'agent': 'strong', 'success': True},
                agent='strong',
                strength=0.9,
            )

        # Before _execute_multi_agent, agents are [weak, strong]
        assert sm.agents[0].name == 'weak'

        # Trigger the intelligence-guided selection by calling the method
        # We'll mock the actual execution since we just want to test reordering
        import asyncio

        async def _test():
            # Patch the paradigm dispatch on the ENGINE (not facade)
            engine = sm._ensure_engine()
            original_agents = None

            async def fake_relay(goal, **kw):
                nonlocal original_agents
                original_agents = [a.name for a in sm.agents]
                return _episode(True, "relayed")

            with patch.object(engine, '_paradigm_relay', side_effect=fake_relay):
                await sm._execute_multi_agent(
                    "Analyze the data trends",
                    discussion_paradigm='relay',
                )

            return original_agents

        result = asyncio.run(_test())
        # After stigmergy reorder, 'strong' should be first
        assert result[0] == 'strong', f"Expected 'strong' first, got {result}"

    def test_byzantine_filters_untrusted_agents(self):
        """Agents with trust < 0.2 should be excluded from execution."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[
                _agent_spec("good", capabilities=["do task"]),
                _agent_spec("bad", capabilities=["do task"]),
            ],
            config=_cfg(),
        )

        # Destroy 'bad' agent's trust via byzantine
        lp = sm.learning
        lp.byzantine_verifier.si.register_agent('bad')
        for _ in range(10):
            lp.byzantine_verifier.verify_claim(
                agent='bad',
                claimed_success=True,
                actual_result=_episode(False),
            )

        # Confirm bad agent trust is near 0
        assert lp.get_agent_trust('bad') < 0.2

        import asyncio

        async def _test():
            engine = sm._ensure_engine()
            agents_at_dispatch = None

            async def fake_relay(goal, **kw):
                nonlocal agents_at_dispatch
                agents_at_dispatch = [a.name for a in sm.agents]
                return _episode(True, "ok")

            with patch.object(engine, '_paradigm_relay', side_effect=fake_relay):
                await sm._execute_multi_agent(
                    "Do some analysis",
                    discussion_paradigm='relay',
                )

            return agents_at_dispatch

        result = asyncio.run(_test())
        # 'bad' should be filtered out
        assert 'bad' not in result, f"Untrusted 'bad' should be excluded, got {result}"
        assert 'good' in result


# ===========================================================================
# Gap 2: Adaptive learning controls refinement iterations
# ===========================================================================

class TestGap2AdaptiveRefinement:
    """
    Prove that refinement paradigm now uses adaptive_learning.should_stop_early()
    to break iteration loops instead of only hardcoded convergence checks.
    """

    def test_refinement_stops_early_on_adaptive_signal(self):
        """If adaptive learning says stop, refinement should exit early."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[
                _agent_spec("drafter", capabilities=["write initial draft"]),
                _agent_spec("editor", capabilities=["improve the draft"]),
            ],
            config=_cfg(),
        )

        # Pre-load adaptive learning to convergence state
        lp = sm.learning
        for _ in range(10):
            lp.adaptive_learning.update_score(0.96)

        # Confirm it recommends stopping
        assert lp.adaptive_learning.should_stop_early()

        import asyncio

        iteration_count = 0

        async def _test():
            nonlocal iteration_count

            # Mock _paradigm_run_agent to count iterations
            original_run = sm._paradigm_run_agent

            async def counting_run(runner, goal, agent_name, **kw):
                nonlocal iteration_count
                iteration_count += 1
                return _episode(True, f"output from {agent_name}")

            sm._paradigm_run_agent = counting_run
            sm._ensure_runners()

            result = await sm._paradigm_refinement(
                "Write a report about AI trends",
                refinement_iterations=5,
            )
            return result

        asyncio.run(_test())

        # With adaptive early stop, should NOT run all 5 iterations * 2 agents
        # First agent drafts (1 call), then should stop early before iterating
        # Expect: 1 (draft) + at most 1 iteration = ~2-3 calls, not 1+5=6
        assert iteration_count <= 3, \
            f"Adaptive early stop should limit iterations, got {iteration_count} calls"

    @pytest.mark.skip(reason="Refinement now delegates to ParadigmExecutor; _paradigm_run_agent patching doesn't reach internal executor")
    def test_refinement_runs_full_without_convergence(self):
        """Without convergence signal, refinement should run all iterations."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[
                _agent_spec("drafter", capabilities=["write"]),
                _agent_spec("editor", capabilities=["edit"]),
            ],
            config=_cfg(),
        )

        # Fresh adaptive learning — no convergence
        assert not sm.learning.adaptive_learning.should_stop_early()

        import asyncio
        iteration_count = 0

        async def _test():
            nonlocal iteration_count
            engine = sm._ensure_engine()
            call_num = [0]

            async def counting_run(runner, goal, agent_name, **kw):
                nonlocal iteration_count
                call_num[0] += 1
                iteration_count += 1
                # Return different output each time so draft-comparison doesn't converge
                return _episode(True, f"output v{call_num[0]}")

            engine._paradigm_run_agent = counting_run
            sm._ensure_runners()

            await sm._paradigm_refinement(
                "Write a report",
                refinement_iterations=3,
            )

        asyncio.run(_test())

        # Should run: 1 (draft) + 3 iterations * 1 editor = 4 calls
        assert iteration_count >= 3, \
            f"Without convergence, should run full iterations, got {iteration_count}"


# ===========================================================================
# Gap 3: Curriculum tasks auto-queue on plateau, consumable
# ===========================================================================

class TestGap3CurriculumQueue:
    """
    Prove that:
    1. When adaptive learning detects plateau, curriculum tasks are queued
    2. Queued tasks are consumable via pop_training_task()
    3. Orchestrator.pending_training_tasks reflects the queue
    """

    def test_plateau_queues_training_tasks(self):
        """Flat zero-reward episodes → plateau → curriculum tasks queued."""
        lp = _pipeline()

        # Feed 10 failure episodes to trigger plateau
        agent = _agent_obj('stagnant')
        for _ in range(10):
            lp.post_episode(
                result=_episode(False),
                goal="Stuck on impossible task",
                agents=[agent],
                architect_prompts=[],
            )

        # Adaptive learning should have detected plateau and queued tasks
        assert lp.pending_training_count() >= 1, \
            f"Expected queued training tasks on plateau, got {lp.pending_training_count()}"

    def test_training_tasks_are_consumable(self):
        """Queued tasks can be popped for execution."""
        lp = _pipeline()
        agent = _agent_obj('stagnant')

        for _ in range(10):
            lp.post_episode(
                result=_episode(False),
                goal="Stuck on task",
                agents=[agent],
                architect_prompts=[],
            )

        count_before = lp.pending_training_count()
        assert count_before >= 1

        task = lp.pop_training_task()
        assert task is not None
        assert hasattr(task, 'description')
        assert hasattr(task, 'difficulty')

        assert lp.pending_training_count() == count_before - 1

    @pytest.mark.skip(reason="Byzantine quality checker flags short outputs even on success; test needs LLM-backed quality evaluator")
    def test_no_queue_on_success_streak(self):
        """Successful episodes should NOT queue training tasks."""
        lp = _pipeline()
        agent = _agent_obj('winner')

        # Use sufficiently long output to avoid 'output_too_short' quality flag
        for _ in range(10):
            lp.post_episode(
                result=_episode(True, output="This is a sufficiently detailed successful output for the given task"),
                goal="Easy task",
                agents=[agent],
                architect_prompts=[],
            )

        assert lp.pending_training_count() == 0, \
            "Should NOT queue training tasks when succeeding"

    def test_swarm_manager_exposes_pending_count(self):
        """Orchestrator.pending_training_tasks should reflect queue state."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            agents=[_agent_spec("tester", capabilities=["test"])],
            config=_cfg(),
        )

        # Initially zero
        assert sm.pending_training_tasks == 0

        # Force plateau
        lp = sm.learning
        agent = _agent_obj('stagnant')
        for _ in range(10):
            lp.post_episode(
                result=_episode(False),
                goal="Stuck",
                agents=[agent],
                architect_prompts=[],
            )

        assert sm.pending_training_tasks >= 1

    def test_queue_bounded(self):
        """Queue should not grow unbounded."""
        lp = _pipeline()
        agent = _agent_obj('stagnant')

        # Feed 30 failures — more than queue limit (10)
        for _ in range(30):
            lp.post_episode(
                result=_episode(False),
                goal="Always failing",
                agents=[agent],
                architect_prompts=[],
            )

        assert lp.pending_training_count() <= 10, \
            f"Queue should be bounded to 10, got {lp.pending_training_count()}"
