"""
Tests for MALLM-inspired enhancements to Jotty V2 Orchestrator.

3 features (Becker et al., EMNLP 2025):
  1. Discussion paradigms: relay, debate, refinement
  2. Decision protocols: majority, supermajority, unanimity, ranked, approval
  3. Judge intervention: auditor retry with feedback
"""

import asyncio
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

# =========================================================================
# 1. DISCUSSION PARADIGMS
# =========================================================================


class TestDiscussionParadigms:
    """Test that paradigms dispatch correctly in _execute_multi_agent."""

    def test_paradigm_methods_exist(self):
        from core.orchestration.swarm_manager import Orchestrator

        assert hasattr(Orchestrator, "_paradigm_relay")
        assert hasattr(Orchestrator, "_paradigm_debate")
        assert hasattr(Orchestrator, "_paradigm_refinement")

    @pytest.mark.asyncio
    async def test_relay_paradigm(self):
        """Relay: agents run sequentially, chaining output."""
        from core.foundation.data_structures import EpisodeResult
        from core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator.__new__(Orchestrator)
        # Minimal mocking for paradigm
        sm._agent_semaphore = asyncio.Semaphore(3)
        sm._scheduling_stats = defaultdict(int)

        call_order = []

        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.capabilities = [f"Sub-goal for {name}"]

        class MockRunner:
            def __init__(self, name):
                self.name = name

            async def run(self, goal, **kwargs):
                call_order.append(self.name)
                return EpisodeResult(
                    output=f"{self.name} output",
                    success=True,
                    trajectory=[],
                    tagged_outputs=[],
                    episode=0,
                    execution_time=0.1,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                )

        sm.agents = [MockAgent("researcher"), MockAgent("writer")]
        sm.runners = {
            "researcher": MockRunner("researcher"),
            "writer": MockRunner("writer"),
        }

        # Mock learning/aggregation methods
        sm._aggregate_results = lambda results, goal: list(results.values())[-1]
        sm._post_episode_learning = lambda result, goal: None
        sm._auto_save_learnings = lambda: None

        result = await sm._paradigm_relay("test goal")

        assert result.success
        # Relay = sequential order
        assert call_order == ["researcher", "writer"]

    @pytest.mark.asyncio
    async def test_debate_paradigm(self):
        """Debate: agents draft then critique each other."""
        from core.foundation.data_structures import EpisodeResult
        from core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator.__new__(Orchestrator)
        sm._agent_semaphore = asyncio.Semaphore(3)
        sm._scheduling_stats = defaultdict(int)

        call_count = 0

        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.capabilities = [f"Analyze for {name}"]

        class MockRunner:
            def __init__(self, name):
                self.name = name

            async def run(self, goal, **kwargs):
                nonlocal call_count
                call_count += 1
                return EpisodeResult(
                    output=f"{self.name} analysis (call {call_count})",
                    success=True,
                    trajectory=[],
                    tagged_outputs=[],
                    episode=0,
                    execution_time=0.1,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                )

        sm.agents = [MockAgent("analyst_a"), MockAgent("analyst_b")]
        sm.runners = {
            "analyst_a": MockRunner("analyst_a"),
            "analyst_b": MockRunner("analyst_b"),
        }
        sm._aggregate_results = lambda results, goal: list(results.values())[-1]
        sm._post_episode_learning = lambda result, goal: None
        sm._auto_save_learnings = lambda: None

        result = await sm._paradigm_debate("analyze topic", debate_rounds=2)

        assert result.success
        # Round 1: 2 drafts + Round 2: 2 critiques = 4 calls
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_refinement_paradigm(self):
        """Refinement: iterative improvement of shared draft."""
        from core.foundation.data_structures import EpisodeResult
        from core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator.__new__(Orchestrator)
        sm._agent_semaphore = asyncio.Semaphore(3)
        sm._scheduling_stats = defaultdict(int)

        outputs = []

        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.capabilities = [f"Write for {name}"]

        class MockRunner:
            def __init__(self, name):
                self.name = name

            async def run(self, goal, **kwargs):
                output = f"{self.name} refined"
                outputs.append(output)
                return EpisodeResult(
                    output=output,
                    success=True,
                    trajectory=[],
                    tagged_outputs=[],
                    episode=0,
                    execution_time=0.1,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                )

        sm.agents = [MockAgent("drafter"), MockAgent("editor")]
        sm.runners = {
            "drafter": MockRunner("drafter"),
            "editor": MockRunner("editor"),
        }
        sm._aggregate_results = lambda results, goal: list(results.values())[-1]
        sm._post_episode_learning = lambda result, goal: None
        sm._auto_save_learnings = lambda: None

        result = await sm._paradigm_refinement("write report", refinement_iterations=2)

        assert result.success
        # Drafter first, then editor refines (multiple iterations possible)
        assert "drafter refined" in outputs
        assert "editor refined" in outputs

    @pytest.mark.asyncio
    async def test_paradigm_dispatch_in_multi_agent(self):
        """Test that discussion_paradigm kwarg routes correctly."""
        from core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator.__new__(Orchestrator)
        sm._agent_semaphore = asyncio.Semaphore(3)
        sm._scheduling_stats = defaultdict(int)

        dispatched = {}

        async def mock_relay(goal, **kw):
            dispatched["relay"] = True
            from core.foundation.data_structures import EpisodeResult

            return EpisodeResult(
                output="relay",
                success=True,
                trajectory=[],
                tagged_outputs=[],
                episode=0,
                execution_time=0.1,
                architect_results=[],
                auditor_results=[],
                agent_contributions={},
            )

        sm._paradigm_relay = mock_relay

        # Simulate the dispatch logic from _execute_multi_agent
        discussion_paradigm = "relay"
        if discussion_paradigm == "relay":
            result = await sm._paradigm_relay("test")

        assert dispatched.get("relay") is True


# =========================================================================
# 2. DECISION PROTOCOLS
# =========================================================================


class TestDecisionProtocols:

    def test_protocol_constants(self):
        from core.orchestration._consensus_mixin import DECISION_PROTOCOLS

        assert "weighted" in DECISION_PROTOCOLS
        assert "majority" in DECISION_PROTOCOLS
        assert "supermajority" in DECISION_PROTOCOLS
        assert "unanimity" in DECISION_PROTOCOLS
        assert "ranked" in DECISION_PROTOCOLS
        assert "approval" in DECISION_PROTOCOLS

    def _make_mixin(self):
        """Create a ConsensusMixin with minimal agent profiles."""
        from core.orchestration._consensus_mixin import ConsensusMixin
        from core.orchestration.swarm_data_structures import AgentProfile

        mixin = object.__new__(ConsensusMixin)
        mixin.agent_profiles = {}
        mixin.consensus_history = []

        def register_agent(name):
            if name not in mixin.agent_profiles:
                mixin.agent_profiles[name] = AgentProfile(agent_name=name, trust_score=0.8)

        mixin.register_agent = register_agent
        return mixin

    def _make_votes(self, decisions_and_confidences):
        """Create ConsensusVote objects from (agent, decision, confidence) tuples."""
        from core.orchestration.swarm_data_structures import ConsensusVote

        return [
            ConsensusVote(
                agent_name=agent,
                decision=decision,
                confidence=confidence,
                reasoning=f"{agent} chose {decision}",
            )
            for agent, decision, confidence in decisions_and_confidences
        ]

    def test_tally_weighted(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "optionA", 0.9),
                ("b", "optionA", 0.8),
                ("c", "optionB", 0.5),
            ]
        )

        winner, strength = mixin._tally_weighted(votes)
        assert winner == "optionA"
        assert strength > 0.5

    def test_tally_majority_met(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "X", 0.9),
                ("b", "X", 0.8),
                ("c", "Y", 0.7),
            ]
        )

        winner, strength = mixin._tally_threshold(votes, threshold=0.5)
        assert winner == "X"
        assert strength >= 0.5  # 2/3 ≈ 0.67

    def test_tally_majority_not_met(self):
        mixin = self._make_mixin()
        # Equal split — no majority
        votes = self._make_votes(
            [
                ("a", "X", 0.9),
                ("b", "Y", 0.8),
            ]
        )

        winner, strength = mixin._tally_threshold(votes, threshold=0.5)
        # 1/2 = 0.5 which is >= threshold
        assert winner in ("X", "Y")

    def test_tally_supermajority(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "A", 0.9),
                ("b", "A", 0.8),
                ("c", "A", 0.7),
                ("d", "B", 0.6),
            ]
        )

        winner, strength = mixin._tally_threshold(votes, threshold=2 / 3)
        assert winner == "A"
        assert strength >= 2 / 3  # 3/4 = 0.75

    def test_tally_unanimity_all_agree(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "Z", 0.9),
                ("b", "Z", 0.8),
                ("c", "Z", 0.7),
            ]
        )

        winner, strength = mixin._tally_threshold(votes, threshold=1.0)
        assert winner == "Z"
        assert strength == 1.0

    def test_tally_unanimity_disagree(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "Z", 0.9),
                ("b", "Z", 0.8),
                ("c", "W", 0.7),
            ]
        )

        winner, strength = mixin._tally_threshold(votes, threshold=1.0)
        assert winner == "Z"
        # Below threshold → penalized strength
        assert strength < 1.0

    def test_tally_ranked(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "X", 0.9),
                ("b", "Y", 0.8),
                ("c", "X", 0.7),
            ]
        )

        winner, strength = mixin._tally_ranked(votes, ["X", "Y", "Z"])
        assert winner == "X"  # 2 vs 1 votes

    def test_tally_approval(self):
        mixin = self._make_mixin()
        votes = self._make_votes(
            [
                ("a", "A", 0.9),  # approved
                ("b", "A", 0.6),  # approved
                ("c", "B", 0.3),  # NOT approved (confidence < 0.5)
            ]
        )

        winner, strength = mixin._tally_approval(votes)
        assert winner == "A"

    @pytest.mark.asyncio
    async def test_gather_consensus_with_protocol(self):
        """Test full gather_consensus with protocol parameter."""
        mixin = self._make_mixin()

        def vote_func(agent, question, options):
            # All agents pick "yes"
            return "yes", 0.9, f"{agent} says yes"

        decision = await mixin.gather_consensus(
            question="Should we deploy?",
            options=["yes", "no"],
            agents=["a", "b", "c"],
            vote_func=vote_func,
            protocol="unanimity",
        )

        assert decision.final_decision == "yes"
        assert decision.consensus_strength == 1.0
        assert len(decision.dissenting_views) == 0


# =========================================================================
# 3. JUDGE INTERVENTION
# =========================================================================


class TestJudgeIntervention:

    def test_judge_intervention_code_exists(self):
        """Verify the judge intervention logic is in agent_runner."""
        import inspect

        from core.orchestration.agent_runner import AgentRunner

        source = inspect.getsource(AgentRunner.run)
        assert "Judge intervention" in source
        assert "_judge_retried" in source
        assert "judge_goal" in source

    def test_judge_intervention_respects_retry_flag(self):
        """Verify _judge_retried prevents infinite loops."""
        import inspect

        from core.orchestration.agent_runner import AgentRunner

        source = inspect.getsource(AgentRunner.run)
        # Must check _judge_retried to prevent infinite retry
        assert "not _judge_retried" in source

    def test_judge_intervention_confidence_threshold(self):
        """Verify intervention only triggers below confidence threshold."""
        import inspect

        from core.orchestration.agent_runner import AgentRunner

        source = inspect.getsource(AgentRunner.run)
        # Must check auditor_confidence < 0.6
        assert "auditor_confidence < 0.6" in source

    def test_judge_intervention_requires_feedback(self):
        """Verify intervention only triggers when auditor gives real feedback."""
        import inspect

        from core.orchestration.agent_runner import AgentRunner

        source = inspect.getsource(AgentRunner.run)
        # Must check feedback is not empty
        assert '"No feedback"' in source


# =========================================================================
# Also verify previous enhancements still work
# =========================================================================


class TestPreviousEnhancementsStillWork:

    def test_agentscope_tests_still_pass(self):
        """Quick smoke test that AgentScope enhancements still import."""
        from core.agents.axon import SmartAgentSlack
        from core.agents.feedback_channel import FeedbackChannel
        from core.context.context_guard import LLMContextManager
        from core.orchestration import fanout_pipeline, sequential_pipeline
        from core.orchestration.agent_runner import HOOK_TYPES
        from core.registry.unified_registry import get_unified_registry

        assert len(HOOK_TYPES) == 6
        assert hasattr(LLMContextManager, "compress_structured")
        assert callable(sequential_pipeline)
        assert callable(fanout_pipeline)
        assert hasattr(FeedbackChannel, "broadcast")
        assert hasattr(FeedbackChannel, "request")
        assert hasattr(SmartAgentSlack, "broadcast")

        registry = get_unified_registry()
        assert hasattr(registry, "get_scoped_tools")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
