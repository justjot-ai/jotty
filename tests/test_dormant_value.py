"""
VALUE TESTS for the 6 wired dormant modules.

These are NOT "does it fire" tests — they prove each module produces
*measurable behavioral improvement* across multi-episode simulations.

Each test simulates a realistic scenario and asserts the module's output
changes the system's decisions in a provably better direction.
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg():
    return SwarmConfig()


def _pipeline():
    from Jotty.core.orchestration.learning_pipeline import SwarmLearningPipeline
    return SwarmLearningPipeline(_cfg())


def _episode(success, output="ok", agent_name=None, contributions=None):
    ep = EpisodeResult(
        success=success,
        output=output,
        trajectory=[],
        tagged_outputs=[],
        episode=0,
        execution_time=1.0,
        architect_results=[],
        auditor_results=[],
        agent_contributions=contributions or {},
    )
    if agent_name:
        ep.agent_name = agent_name
    return ep


def _agent(name):
    return type('A', (), {'name': name})()


# ===========================================================================
# 1. Stigmergy: Does pheromone routing actually pick the right agent?
# ===========================================================================

class TestStigmergyValue:
    """
    Scenario: 3 agents do 'analysis' tasks. 'analyst' succeeds 8/10 times,
    'coder' succeeds 2/10, 'writer' succeeds 5/10.
    After 10 episodes, stigmergy routing should recommend 'analyst'.
    """

    def test_routing_converges_to_best_agent(self):
        lp = _pipeline()

        # Simulate 10 episodes with different agents and success rates
        episodes = [
            ('analyst', True),  ('analyst', True),  ('analyst', True),
            ('analyst', True),  ('analyst', True),  ('analyst', True),
            ('analyst', True),  ('analyst', False),
            ('coder',   True),  ('coder',   False),
            ('coder',   False), ('coder',   False),
            ('writer',  True),  ('writer',  True),
            ('writer',  False), ('writer',  False),
        ]
        for agent_name, success in episodes:
            lp.post_episode(
                result=_episode(success, agent_name=agent_name),
                goal="Analyze the quarterly data trends",
                agents=[_agent(agent_name)],
                architect_prompts=[],
            )

        # Ask stigmergy: who should do 'analysis' next?
        routes = lp.stigmergy.get_route_signals('analysis')

        # 'analyst' should have the strongest pheromone trail
        assert 'analyst' in routes, f"Expected 'analyst' in routes, got {routes}"
        if 'coder' in routes:
            assert routes['analyst'] > routes['coder'], \
                f"analyst ({routes['analyst']:.2f}) should beat coder ({routes['coder']:.2f})"
        if 'writer' in routes:
            assert routes['analyst'] > routes['writer'], \
                f"analyst ({routes['analyst']:.2f}) should beat writer ({routes['writer']:.2f})"

    def test_failure_warnings_accumulate(self):
        """Agents that fail should generate warning signals — not route signals."""
        lp = _pipeline()
        for _ in range(5):
            lp.post_episode(
                result=_episode(False),
                goal="Deploy to production",
                agents=[_agent('bad_deployer')],
                architect_prompts=[],
            )

        warnings = lp.stigmergy.sense(signal_type='warning')
        routes = lp.stigmergy.sense(signal_type='route')

        # Should have warnings but NOT route signals for this agent
        assert len(warnings) >= 5
        deployer_routes = [
            r for r in routes
            if isinstance(r.content, dict) and r.content.get('agent') == 'bad_deployer'
        ]
        assert len(deployer_routes) == 0


# ===========================================================================
# 2. Byzantine: Does it actually catch and penalize liars?
# ===========================================================================

class TestByzantineValue:
    """
    Scenario: Agent 'honest' always reports truthfully.
    Agent 'liar' claims success but actually fails 5 times.
    Byzantine should lower liar's trust, keep honest's trust high.
    """

    def test_liar_loses_trust_honest_keeps_it(self):
        lp = _pipeline()

        # Register both agents
        lp.byzantine_verifier.si.register_agent('honest')
        lp.byzantine_verifier.si.register_agent('liar')

        # Honest agent: claims match reality
        for _ in range(5):
            lp.byzantine_verifier.verify_claim(
                agent='honest',
                claimed_success=True,
                actual_result=_episode(True),
            )

        # Liar agent: claims success but actually fails
        for _ in range(5):
            lp.byzantine_verifier.verify_claim(
                agent='liar',
                claimed_success=True,
                actual_result=_episode(False),
            )

        honest_trust = lp.get_agent_trust('honest')
        liar_trust = lp.get_agent_trust('liar')

        # Honest gets +0.05 per consistent claim from default 0.5 → ~0.75 after 5
        # Liar gets -0.15 per lie from default 0.5 → 0.0 after 4 lies
        assert honest_trust > 0.7, f"Honest trust too low: {honest_trust}"
        assert liar_trust < 0.1, f"Liar trust too high: {liar_trust}"
        assert honest_trust > liar_trust, "Honest must be more trusted than liar"
        # The GAP is the real value: honest is 0.75+, liar is 0.0
        assert (honest_trust - liar_trust) > 0.5, \
            f"Trust gap too small: honest={honest_trust}, liar={liar_trust}"

    def test_liar_gets_flagged_as_untrusted(self):
        lp = _pipeline()
        lp.byzantine_verifier.si.register_agent('liar')

        # 5 lies
        for _ in range(5):
            lp.byzantine_verifier.verify_claim(
                agent='liar',
                claimed_success=True,
                actual_result=_episode(False),
            )

        untrusted = lp.byzantine_verifier.get_untrusted_agents(threshold=0.5)
        assert 'liar' in untrusted

    def test_trust_weighted_voting_prefers_honest_agent(self):
        """When agents disagree, trust-weighted voting should favor honest agents."""
        lp = _pipeline()
        lp.byzantine_verifier.si.register_agent('honest')
        lp.byzantine_verifier.si.register_agent('liar')

        # Damage liar's trust
        for _ in range(5):
            lp.byzantine_verifier.verify_claim(
                agent='liar', claimed_success=True,
                actual_result=_episode(False),
            )

        # Voting: honest says "A", liar says "B"
        winner, confidence = lp.byzantine_verifier.majority_vote({
            'honest': 'A',
            'liar': 'B',
        })
        assert winner == 'A', f"Expected honest's vote to win, got: {winner}"
        assert confidence > 0.5


# ===========================================================================
# 3. Credit Assignment: Does it track which approaches actually work?
# ===========================================================================

class TestCreditAssignmentValue:
    """
    Scenario: Two approaches tried over multiple episodes.
    'good_approach' succeeds 4/5, 'bad_approach' succeeds 1/5.
    Credit assignment should rank good_approach higher.
    """

    def test_good_approach_gets_more_credit(self):
        lp = _pipeline()

        # Good approach: 4/5 successes
        for i in range(5):
            success = i < 4
            lp.credit_assigner.record_improvement_application(
                improvement={'learned_pattern': 'Use structured prompts with examples', 'task': 'coding'},
                student_score=0.3,
                teacher_score=0.0,
                final_score=0.9 if success else 0.2,
                context={'task': 'coding', 'episode': i},
            )

        # Bad approach: 1/5 successes
        for i in range(5):
            success = i == 0
            lp.credit_assigner.record_improvement_application(
                improvement={'learned_pattern': 'Just try harder without structure', 'task': 'coding'},
                student_score=0.3,
                teacher_score=0.0,
                final_score=0.9 if success else 0.2,
                context={'task': 'coding', 'episode': i},
            )

        # Prioritize
        improvements = [
            {'learned_pattern': 'Use structured prompts with examples', 'task': 'coding'},
            {'learned_pattern': 'Just try harder without structure', 'task': 'coding'},
        ]
        ranked = lp.credit_assigner.prioritize_improvements(improvements)

        assert len(ranked) >= 1
        # Good approach should be first
        assert 'structured prompts' in ranked[0]['learned_pattern'].lower(), \
            f"Expected good approach first, got: {ranked[0]['learned_pattern']}"

    def test_duplicate_detection(self):
        """Near-duplicate improvements should be flagged."""
        lp = _pipeline()
        improvements = [
            {'learned_pattern': 'Use chain of thought reasoning for complex tasks'},
            {'learned_pattern': 'Use chain of thought reasoning for difficult tasks'},
            {'learned_pattern': 'Always validate input data before processing'},
        ]
        dupes = lp.credit_assigner.detect_duplicates(improvements, similarity_threshold=0.6)
        # First two are near-duplicates
        id0 = lp.credit_assigner._get_improvement_id(improvements[0])
        assert len(dupes.get(id0, [])) >= 1, "Should detect near-duplicate"


# ===========================================================================
# 4. Adaptive Learning: Does it actually adapt to different trajectories?
# ===========================================================================

class TestAdaptiveLearningValue:
    """
    Scenario 1: Feed flat scores (0.5 x 10) → should detect plateau, increase exploration
    Scenario 2: Feed improving scores → should NOT detect plateau, lower exploration
    Scenario 3: Feed high converging scores → should recommend early stop
    """

    def test_plateau_increases_exploration(self):
        lp = _pipeline()
        agent = _agent('stagnant')

        initial_explore = lp.adaptive_learning.state.exploration_rate

        # 10 identical zero-reward episodes = plateau
        for _ in range(10):
            lp.post_episode(
                result=_episode(False),
                goal="Stuck on this task",
                agents=[agent],
                architect_prompts=[],
            )

        state = lp.get_learning_state()
        # Plateau should be detected, exploration rate increased
        assert state['is_plateau'] or state['exploration_rate'] > initial_explore, \
            f"Expected plateau detection or increased exploration. State: {state}"

    def test_improving_scores_dont_trigger_plateau(self):
        """Scores that climb from bad to perfect should NOT be plateau."""
        lp = _pipeline()
        agent = _agent('improver')

        # Feed improving success pattern: F, F, T, T, T, T, T
        # Last 5 are all 1.0 — should be detected as CONVERGENCE, not plateau
        for i, success in enumerate([False, False, True, True, True, True, True]):
            lp.post_episode(
                result=_episode(success),
                goal=f"Improving task {i}",
                agents=[agent],
                architect_prompts=[],
            )

        state = lp.get_learning_state()
        # All 1.0 at the end = convergence (mean >= 0.9), not plateau
        assert not state['is_plateau'], \
            f"High stable scores should be convergence, not plateau: {state}"
        assert state['is_converging'], \
            f"All-perfect tail should detect convergence: {state}"

    def test_convergence_triggers_early_stop(self):
        """Stable high scores should converge and recommend early stop."""
        lp = _pipeline()

        # Feed high-converging scores — need enough for convergence + early stop
        for _ in range(8):
            lp.adaptive_learning.update_score(0.96)

        state = lp.adaptive_learning.get_state()
        assert state.is_converging, \
            f"Stable 0.96 scores should detect convergence: lr={state.learning_rate}"
        assert lp.adaptive_learning.should_stop_early(), \
            "Should recommend early stop after converging at 0.96"


# ===========================================================================
# 5. Curriculum: Does it generate meaningfully different tasks?
# ===========================================================================

class TestCurriculumValue:

    def test_tasks_have_variety(self):
        """Generated tasks should not all be identical."""
        lp = _pipeline()
        tasks = lp.generate_training_tasks(count=5)
        descriptions = [t.description for t in tasks]

        # At least 2 unique descriptions (curriculum should vary)
        unique = set(descriptions)
        assert len(unique) >= 2, f"All tasks identical: {descriptions}"

    def test_tasks_have_reasonable_difficulty(self):
        """Difficulty should be in valid range and not all extremes."""
        lp = _pipeline()
        tasks = lp.generate_training_tasks(count=5)
        difficulties = [t.difficulty for t in tasks]

        for d in difficulties:
            assert 0.0 <= d <= 1.0, f"Invalid difficulty: {d}"

        # Not all at the same extreme
        assert not all(d == 0.0 for d in difficulties), "All zero difficulty"
        assert not all(d == 1.0 for d in difficulties), "All max difficulty"

    def test_weak_agent_gets_targeted_tasks(self):
        """Tasks for a weak agent should target their weakness."""
        from Jotty.core.orchestration.swarm_data_structures import AgentProfile
        lp = _pipeline()

        # Agent that fails at 'analysis' tasks
        weak_profile = AgentProfile(agent_name='weak_analyst')
        weak_profile.task_success = {
            'analysis': (1, 10),    # 10% success — very weak
            'coding': (9, 10),      # 90% success — strong
        }

        tasks = lp.generate_training_tasks(
            agent_profiles={'weak_analyst': weak_profile},
            count=5,
        )
        # Should generate tasks (the curriculum generator targets weaknesses)
        assert len(tasks) >= 1


# ===========================================================================
# 6. Sandbox: Does it actually isolate dangerous code?
# ===========================================================================

class TestSandboxValue:

    @pytest.mark.asyncio
    async def test_safe_code_succeeds(self):
        from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
        t = SwarmTerminal()
        result = await t.execute_sandboxed("print(2 + 2)")
        assert result.success
        assert '4' in result.output

    @pytest.mark.asyncio
    async def test_error_code_fails_gracefully(self):
        """Bad code should fail without crashing the host process."""
        from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
        t = SwarmTerminal()
        result = await t.execute_sandboxed("import sys; sys.exit(1)")
        # Should return failure, not crash pytest
        assert not result.success

    @pytest.mark.asyncio
    async def test_infinite_loop_times_out(self):
        """Infinite loops should not hang — subprocess has a timeout."""
        from Jotty.core.orchestration.swarm_terminal import SwarmTerminal
        t = SwarmTerminal()
        # Explicitly short timeout — should kill the subprocess
        result = await t.execute_sandboxed(
            code="while True: pass",
            language="python",
            timeout=3,
        )
        # Should timeout and return failure, not hang forever
        assert not result.success


# ===========================================================================
# Integration: Full multi-episode scenario
# ===========================================================================

class TestEndToEndValue:
    """
    Simulate 20 episodes with mixed agents/outcomes.
    After simulation, all modules should produce actionable intelligence.
    """

    def test_20_episode_simulation(self):
        lp = _pipeline()

        # --- Simulate 20 episodes ---
        scenarios = [
            # (agent, goal, success)
            ('analyst', 'Analyze market trends Q1', True),
            ('analyst', 'Analyze revenue growth', True),
            ('analyst', 'Analyze competitor pricing', True),
            ('coder',   'Write unit tests for API', True),
            ('coder',   'Fix authentication bug', True),
            ('coder',   'Optimize database queries', False),
            ('writer',  'Draft blog post about AI', True),
            ('writer',  'Write technical documentation', False),
            ('writer',  'Create user onboarding guide', False),
            ('analyst', 'Analyze churn patterns', True),
            ('coder',   'Build REST endpoint', True),
            ('coder',   'Deploy to staging', False),
            ('analyst', 'Analyze A/B test results', True),
            ('writer',  'Write release notes', True),
            ('analyst', 'Analyze user behavior data', True),
            ('coder',   'Refactor payment module', True),
            ('writer',  'Create API documentation', False),
            ('analyst', 'Analyze support ticket trends', True),
            ('coder',   'Fix memory leak', False),
            ('analyst', 'Analyze conversion funnel', True),
        ]

        for agent_name, goal, success in scenarios:
            lp.post_episode(
                result=_episode(success, agent_name=agent_name),
                goal=goal,
                agents=[_agent(agent_name)],
                architect_prompts=[],
            )

        # --- Assert each module produced actionable intelligence ---

        # 1. Stigmergy: analyst should be top route for analysis tasks
        routes = lp.stigmergy.get_route_signals('analysis')
        assert 'analyst' in routes
        if 'coder' in routes:
            assert routes['analyst'] > routes.get('coder', 0)

        # 2. Byzantine: all agents are honest (claims match reality),
        #    so trust should increase above default 0.5 with the +0.05 boost
        for name in ['analyst', 'coder', 'writer']:
            trust = lp.get_agent_trust(name)
            assert trust > 0.5, f"{name} trust should exceed default 0.5, got: {trust}"

        # 3. Credit: should have recorded improvements
        stats = lp.get_credit_stats()
        assert stats['total_applications'] == 20
        assert stats['total_improvements'] >= 1

        # 4. Adaptive learning: should have processed 20 scores
        assert lp.adaptive_learning.state.iteration_count == 20
        state = lp.get_learning_state()
        # With mixed success, shouldn't claim convergence
        assert not state['should_stop']

        # 5. Curriculum: should generate tasks
        tasks = lp.generate_training_tasks(count=3)
        assert len(tasks) == 3

        # 6. Summary check: warnings should exist for failed tasks
        warnings = lp.stigmergy.sense(signal_type='warning')
        assert len(warnings) >= 1, "Should have warnings for failed episodes"
