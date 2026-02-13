"""
Tests for wiring the 6 previously-dormant modules into SwarmLearningPipeline.

Modules:
1. stigmergy.py      — pheromone-based agent routing
2. byzantine_verification.py — trust scoring / inconsistency detection
3. credit_assignment.py      — improvement credit tracking
4. adaptive_learning.py      — dynamic learning rate + exploration
5. curriculum_generator.py   — DrZero self-generated training tasks
6. sandbox_manager.py        — sandboxed code execution in SwarmTerminal
"""

import asyncio
import pytest

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> SwarmConfig:
    return SwarmConfig()


def _make_pipeline():
    from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
    return SwarmLearningPipeline(_make_config())


def _make_episode(success: bool = True, output: str = "done") -> EpisodeResult:
    return EpisodeResult(
        success=success,
        output=output,
        trajectory=[],
        tagged_outputs=[],
        episode=0,
        execution_time=0.5,
        architect_results=[],
        auditor_results=[],
        agent_contributions={},
    )


# ===========================================================================
# 1. Stigmergy
# ===========================================================================

class TestStigmergyWiring:
    """Test that stigmergy is initialized and wired into post_episode."""

    def test_stigmergy_initialized(self):
        lp = _make_pipeline()
        assert hasattr(lp, 'stigmergy')
        assert len(lp.stigmergy.signals) == 0

    def test_post_episode_deposits_success_signal(self):
        lp = _make_pipeline()
        result = _make_episode(success=True)
        lp.post_episode(
            result=result,
            goal="Analyze quarterly earnings report",
            agents=[type('A', (), {'name': 'analyst'})()],
            architect_prompts=[],
        )
        signals = lp.stigmergy.sense(signal_type='success')
        assert len(signals) >= 1
        assert signals[0].created_by == 'analyst'

    def test_post_episode_deposits_route_signal(self):
        lp = _make_pipeline()
        result = _make_episode(success=True)
        lp.post_episode(
            result=result,
            goal="Search the web for data",
            agents=[type('A', (), {'name': 'researcher'})()],
            architect_prompts=[],
        )
        routes = lp.stigmergy.sense(signal_type='route')
        assert len(routes) >= 1

    def test_post_episode_deposits_warning_on_failure(self):
        lp = _make_pipeline()
        result = _make_episode(success=False)
        lp.post_episode(
            result=result,
            goal="Deploy the app",
            agents=[type('A', (), {'name': 'deployer'})()],
            architect_prompts=[],
        )
        warnings = lp.stigmergy.sense(signal_type='warning')
        assert len(warnings) >= 1

    def test_get_stigmergy_route(self):
        lp = _make_pipeline()
        # Deposit a route signal manually
        lp.stigmergy.deposit(
            signal_type='route',
            content={'task_type': 'analysis', 'agent': 'analyst'},
            agent='analyst',
            strength=0.9,
        )
        best = lp.get_stigmergy_route('analysis')
        assert best == 'analyst'

    def test_get_stigmergy_route_missing(self):
        lp = _make_pipeline()
        assert lp.get_stigmergy_route('nonexistent') is None


# ===========================================================================
# 2. Byzantine Verification
# ===========================================================================

class TestByzantineWiring:
    """Test byzantine verification is initialized and wired."""

    def test_byzantine_initialized(self):
        lp = _make_pipeline()
        assert hasattr(lp, 'byzantine_verifier')
        assert lp.byzantine_verifier.verified_count == 0

    def test_post_episode_runs_verification(self):
        lp = _make_pipeline()
        result = _make_episode(success=True)
        lp.post_episode(
            result=result,
            goal="Write a poem",
            agents=[type('A', (), {'name': 'poet'})()],
            architect_prompts=[],
        )
        assert lp.byzantine_verifier.verified_count >= 1

    def test_trust_query(self):
        lp = _make_pipeline()
        # Unknown agent → full trust
        assert lp.get_agent_trust('unknown_agent') == 1.0
        assert lp.is_agent_trusted('unknown_agent')

    def test_inconsistency_lowers_trust(self):
        lp = _make_pipeline()
        # Simulate agent claiming success but result failed
        failed_result = _make_episode(success=False)
        # Register agent first
        lp.byzantine_verifier.si.register_agent('liar')
        # Verify claim: agent claims success, but actual is failure
        consistent = lp.byzantine_verifier.verify_claim(
            agent='liar',
            claimed_success=True,
            actual_result=failed_result,
        )
        assert not consistent
        assert lp.get_agent_trust('liar') < 1.0
        assert lp.byzantine_verifier.inconsistent_count == 1


# ===========================================================================
# 3. Credit Assignment
# ===========================================================================

class TestCreditAssignmentWiring:
    """Test credit assignment is initialized and wired."""

    def test_credit_initialized(self):
        lp = _make_pipeline()
        assert hasattr(lp, 'credit_assigner')
        stats = lp.get_credit_stats()
        assert stats['total_improvements'] == 0

    def test_post_episode_records_credit(self):
        lp = _make_pipeline()
        result = _make_episode(success=True)
        lp.post_episode(
            result=result,
            goal="Optimize SQL query performance",
            agents=[type('A', (), {'name': 'optimizer'})()],
            architect_prompts=[],
        )
        stats = lp.get_credit_stats()
        assert stats['total_improvements'] >= 1
        assert stats['total_applications'] >= 1

    def test_multiple_episodes_accumulate(self):
        lp = _make_pipeline()
        agent = type('A', (), {'name': 'worker'})()
        for i in range(3):
            result = _make_episode(success=(i % 2 == 0))
            lp.post_episode(
                result=result,
                goal=f"Task {i}",
                agents=[agent],
                architect_prompts=[],
            )
        stats = lp.get_credit_stats()
        assert stats['total_applications'] >= 3


# ===========================================================================
# 4. Adaptive Learning
# ===========================================================================

class TestAdaptiveLearningWiring:
    """Test adaptive learning is initialized and wired."""

    def test_adaptive_initialized(self):
        lp = _make_pipeline()
        assert hasattr(lp, 'adaptive_learning')
        state = lp.get_learning_state()
        assert state['learning_rate'] == 1.0

    def test_post_episode_updates_learning_rate(self):
        lp = _make_pipeline()
        agent = type('A', (), {'name': 'learner'})()
        for _ in range(5):
            result = _make_episode(success=True)
            lp.post_episode(
                result=result,
                goal="Test task",
                agents=[agent],
                architect_prompts=[],
            )
        state = lp.get_learning_state()
        # After 5 successful episodes, rate should have adjusted
        assert state['learning_rate'] != 1.0 or state['improvement_velocity'] != 0.0

    def test_plateau_detection(self):
        lp = _make_pipeline()
        agent = type('A', (), {'name': 'learner'})()
        # Feed identical scores to trigger plateau
        for _ in range(8):
            result = _make_episode(success=False)
            lp.post_episode(
                result=result,
                goal="Stagnant task",
                agents=[agent],
                architect_prompts=[],
            )
        state = lp.get_learning_state()
        # Should detect plateau after many zero-reward episodes
        assert state['is_plateau'] or state['learning_rate'] > 1.0


# ===========================================================================
# 5. Curriculum Generator
# ===========================================================================

class TestCurriculumWiring:
    """Test curriculum generator is initialized and wired."""

    def test_curriculum_initialized(self):
        lp = _make_pipeline()
        assert hasattr(lp, 'curriculum_generator')

    def test_generate_training_tasks(self):
        lp = _make_pipeline()
        tasks = lp.generate_training_tasks(count=2)
        assert isinstance(tasks, list)
        assert len(tasks) == 2
        # Each task should be a SyntheticTask
        for t in tasks:
            assert hasattr(t, 'description')
            assert hasattr(t, 'difficulty')
            assert 0.0 <= t.difficulty <= 1.0

    def test_generate_with_profiles(self):
        from Jotty.core.orchestration.swarm_data_structures import AgentProfile
        lp = _make_pipeline()
        profiles = {
            'coder': AgentProfile(agent_name='coder'),
        }
        tasks = lp.generate_training_tasks(agent_profiles=profiles, count=1)
        assert len(tasks) >= 1


# ===========================================================================
# 6. Sandbox Manager in SwarmTerminal
# ===========================================================================

class TestSandboxWiring:
    """Test sandbox manager is wired into SwarmTerminal."""

    def test_sandbox_attached(self):
        from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
        terminal = SwarmTerminal()
        assert hasattr(terminal, '_sandbox')
        # SandboxManager should be available (subprocess fallback always works)
        assert terminal._sandbox is not None

    @pytest.mark.asyncio
    async def test_execute_sandboxed_basic(self):
        from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
        terminal = SwarmTerminal()
        result = await terminal.execute_sandboxed(
            code="print('hello from sandbox')",
            language="python",
        )
        assert result.success
        assert 'hello from sandbox' in result.output

    @pytest.mark.asyncio
    async def test_execute_sandboxed_error(self):
        from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
        terminal = SwarmTerminal()
        result = await terminal.execute_sandboxed(
            code="raise ValueError('test error')",
            language="python",
        )
        # Should fail gracefully
        assert not result.success or 'test error' in (result.error or '')


# ===========================================================================
# Integration: Full post_episode runs all 6 modules
# ===========================================================================

class TestFullIntegration:
    """Verify a single post_episode call activates all 6 dormant modules."""

    def test_all_modules_fire(self):
        lp = _make_pipeline()
        agent = type('A', (), {'name': 'integrator'})()
        result = _make_episode(success=True)

        lp.post_episode(
            result=result,
            goal="Build a REST API with authentication",
            agents=[agent],
            architect_prompts=[],
        )

        # 1. Stigmergy deposited signals
        assert len(lp.stigmergy.signals) > 0

        # 2. Byzantine verified at least once
        assert lp.byzantine_verifier.verified_count >= 1

        # 3. Credit assignment recorded
        stats = lp.get_credit_stats()
        assert stats['total_improvements'] >= 1

        # 4. Adaptive learning updated
        state = lp.get_learning_state()
        assert lp.adaptive_learning.state.iteration_count >= 1

        # 5. Curriculum generator available
        tasks = lp.generate_training_tasks(count=1)
        assert len(tasks) >= 1

    def test_swarm_manager_status_includes_dormant_stats(self):
        """Orchestrator.status() should include stats from wired modules."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator
        from Jotty.core.foundation.agent_config import AgentConfig

        # Create a minimal AgentConfig with a dummy agent
        dummy_agent = type('DummyAgent', (), {'forward': lambda self, **kw: None})()
        spec = AgentConfig(name="tester", agent=dummy_agent)

        sm = Orchestrator(
            agents=[spec],
            config=_make_config(),
        )
        # Force learning pipeline init by accessing it
        _ = sm.learning
        status = sm.status()

        assert 'learning' in status
        learning = status['learning']
        assert 'stigmergy_signals' in learning
        assert 'byzantine_verifications' in learning
        assert 'credit_stats' in learning
        assert 'adaptive_learning' in learning
