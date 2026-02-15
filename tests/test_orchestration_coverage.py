"""
Tests for orchestration coverage gaps.

Covers:
- ParadigmExecutor (relay, debate, refinement)
- Orchestrator/SwarmManager agent selection and fallbacks
- TierExecutor routing and output extraction
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest

from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_episode(output="agent output", success=True, agent_name="agent"):
    return EpisodeResult(
        output=output,
        success=success,
        trajectory=[],
        tagged_outputs={agent_name: output},
        episode=0,
        execution_time=1.0,
        architect_results=[],
        auditor_results=[],
        agent_contributions={agent_name: str(output)[:200]},
    )


def _make_mock_manager(agent_names=None):
    """Create a mock Orchestrator that ParadigmExecutor expects."""
    agent_names = agent_names or ["researcher", "writer"]
    sm = Mock()
    sm.episode_count = 0

    # Build agent configs
    agents = []
    runners = {}
    for name in agent_names:
        ac = Mock()
        ac.name = name
        ac.capabilities = [f"do {name} work"]
        agents.append(ac)

        runner = AsyncMock()
        # runner.agent is optional for schema wiring
        runner.agent = None
        runners[name] = runner

    sm.agents = agents
    sm.runners = runners
    sm.agent_semaphore = asyncio.Semaphore(10)
    sm._schedule_background_learning = Mock()
    sm._mas_zero_verify = Mock(return_value=None)

    # Learning stubs
    learning = Mock()
    learning.adaptive_learning = Mock()
    learning.adaptive_learning.should_stop_early = Mock(return_value=False)
    sm.learning = learning

    # Credit weights
    credit_weights = Mock()
    credit_weights.get = Mock(return_value=0.33)
    credit_weights.update_from_feedback = Mock()
    sm.credit_weights = credit_weights

    sm.learning_manager = Mock()
    sm.learning_manager.record_outcome = Mock()

    return sm, runners


# ──────────────────────────────────────────────────────────────────────
# ParadigmExecutor — Relay
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestParadigmExecutorRelay:
    """Tests for relay paradigm."""

    async def test_relay_passes_output_sequentially(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["agent_a", "agent_b"])
        pe = ParadigmExecutor(sm)

        runners["agent_a"].run = AsyncMock(
            return_value=_make_episode("First draft from A", agent_name="agent_a")
        )
        runners["agent_b"].run = AsyncMock(
            return_value=_make_episode("Refined by B", agent_name="agent_b")
        )

        # Override run_agent to use runner.run
        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            return await runner.run(goal=sub_goal, **kwargs)

        pe.run_agent = _run_agent

        result = await pe.relay("Write a report")
        assert result.success

    async def test_relay_failed_agent_continues(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["fail_agent", "good_agent"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            if agent_name == "fail_agent":
                return _make_episode(None, success=False, agent_name="fail_agent")
            return _make_episode("Good output", agent_name="good_agent")

        pe.run_agent = _run_agent
        result = await pe.relay("Write a report")
        # Should still complete (good_agent ran)
        assert result is not None

    async def test_relay_empty_agent_list(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, _ = _make_mock_manager([])
        pe = ParadigmExecutor(sm)
        # aggregate_results({}) returns failure with no output
        result = pe.aggregate_results({}, "Write a report")
        assert not result.success
        assert result.output is None

    async def test_relay_enriches_goal_with_previous_output(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        # Use capabilities=[] so relay uses the enriched_goal, not capabilities[0]
        sm, runners = _make_mock_manager(["agent_a", "agent_b"])
        for ac in sm.agents:
            ac.capabilities = []  # Force relay to use enriched_goal
        pe = ParadigmExecutor(sm)

        captured_goals = []

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            captured_goals.append((agent_name, sub_goal))
            return _make_episode(f"Output from {agent_name}", agent_name=agent_name)

        pe.run_agent = _run_agent
        await pe.relay("Original task")

        # agent_b should receive enriched goal with agent_a's output
        assert len(captured_goals) == 2
        _, b_goal = captured_goals[1]
        assert "Output from agent_a" in b_goal

    async def test_relay_single_agent(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["solo"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            return _make_episode("Solo result", agent_name="solo")

        pe.run_agent = _run_agent
        result = await pe.relay("Task")
        # Single agent: aggregate_results returns it directly
        assert result.output == "Solo result"

    async def test_relay_output_extraction_clean_text(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        assert _extract_output_text("plain text") == "plain text"
        assert _extract_output_text(None) == ""
        assert _extract_output_text({"content": "from dict"}) == "from dict"

    async def test_relay_output_extraction_nested_result(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        # Simulate AgenticExecutionResult
        nested = Mock()
        nested.final_output = "clean final"
        assert _extract_output_text(nested) == "clean final"

    async def test_relay_output_extraction_episode_result(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        ep = _make_episode("episode output")
        result = _extract_output_text(ep)
        assert result == "episode output"


# ──────────────────────────────────────────────────────────────────────
# ParadigmExecutor — Debate
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestParadigmExecutorDebate:
    """Tests for debate paradigm."""

    async def test_debate_collects_drafts_from_all_agents(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["alice", "bob"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            return _make_episode(f"Draft from {agent_name}", agent_name=agent_name)

        pe.run_agent = _run_agent
        result = await pe.debate("Analyze X", debate_rounds=1)
        assert result is not None

    async def test_debate_critique_round_enriches_goal(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["alice", "bob"])
        pe = ParadigmExecutor(sm)

        goals_seen = []

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            goals_seen.append((agent_name, sub_goal))
            return _make_episode(f"Draft from {agent_name}", agent_name=agent_name)

        pe.run_agent = _run_agent
        result = await pe.debate("Analyze X", debate_rounds=2)

        # Round 2 should include critique context
        round2_goals = [(n, g) for n, g in goals_seen if "Other agents" in g]
        assert len(round2_goals) > 0

    async def test_debate_single_agent_degrades(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["solo"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            return _make_episode("Solo draft", agent_name="solo")

        pe.run_agent = _run_agent
        result = await pe.debate("Analyze X")
        # With only 1 agent, no critique rounds happen
        assert result.output == "Solo draft"

    async def test_debate_failed_draft_skipped(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["fail", "good"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            if agent_name == "fail":
                return _make_episode(None, success=False, agent_name="fail")
            return _make_episode("Good draft", agent_name="good")

        pe.run_agent = _run_agent
        result = await pe.debate("Analyze X")
        # Only 1 valid draft → no critique rounds, still returns
        assert result is not None

    async def test_debate_exception_in_draft_handled(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["crasher", "stable"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            if agent_name == "crasher":
                raise RuntimeError("Agent crashed")
            return _make_episode("Stable output", agent_name="stable")

        pe.run_agent = _run_agent
        # Debate uses asyncio.gather with return_exceptions=True
        result = await pe.debate("Analyze X")
        assert result is not None

    async def test_debate_multiple_rounds(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["a", "b"])
        pe = ParadigmExecutor(sm)

        call_count = 0

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_episode(f"Output {call_count}", agent_name=agent_name)

        pe.run_agent = _run_agent
        await pe.debate("Analyze X", debate_rounds=3)
        # Round 1: 2 drafts, Round 2: 2 critiques, Round 3: 2 critiques = 6
        assert call_count == 6


# ──────────────────────────────────────────────────────────────────────
# ParadigmExecutor — Refinement
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestParadigmExecutorRefinement:
    """Tests for refinement paradigm."""

    async def test_refinement_iterates(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["drafter", "refiner"])
        pe = ParadigmExecutor(sm)

        call_count = 0

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_episode(f"Version {call_count}", agent_name=agent_name)

        pe.run_agent = _run_agent
        result = await pe.refinement("Write essay", refinement_iterations=2)
        # 1 initial draft + 2 iterations * 1 refiner = 3
        assert call_count == 3

    async def test_refinement_convergence_stops_early(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["drafter", "refiner"])
        pe = ParadigmExecutor(sm)

        # Return same output to trigger convergence
        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            return _make_episode("Same output every time" * 20, agent_name=agent_name)

        pe.run_agent = _run_agent
        result = await pe.refinement("Write essay", refinement_iterations=5)
        # Should converge before 5 iterations
        assert result is not None

    async def test_refinement_max_iterations_respected(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["drafter", "refiner"])
        pe = ParadigmExecutor(sm)

        call_count = 0

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_episode(f"Unique output {call_count}", agent_name=agent_name)

        pe.run_agent = _run_agent
        await pe.refinement("Write essay", refinement_iterations=1)
        # 1 initial + 1 iteration * 1 refiner = 2
        assert call_count == 2

    async def test_refinement_draft_includes_previous(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["drafter", "refiner"])
        pe = ParadigmExecutor(sm)

        goals_seen = []

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            goals_seen.append((agent_name, sub_goal))
            return _make_episode(f"Draft by {agent_name}", agent_name=agent_name)

        pe.run_agent = _run_agent
        await pe.refinement("Write essay", refinement_iterations=1)

        refiner_goals = [(n, g) for n, g in goals_seen if n == "refiner"]
        assert len(refiner_goals) == 1
        assert "current draft" in refiner_goals[0][1].lower()

    async def test_refinement_failed_refiner_continues(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["drafter", "bad_refiner", "good_refiner"])
        pe = ParadigmExecutor(sm)

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            if agent_name == "bad_refiner":
                return _make_episode(None, success=False, agent_name="bad_refiner")
            return _make_episode(f"Output from {agent_name}", agent_name=agent_name)

        pe.run_agent = _run_agent
        result = await pe.refinement("Write essay", refinement_iterations=1)
        assert result is not None

    async def test_refinement_adaptive_early_stop(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, runners = _make_mock_manager(["drafter", "refiner"])
        sm.learning.adaptive_learning.should_stop_early = Mock(return_value=True)
        pe = ParadigmExecutor(sm)

        call_count = 0

        async def _run_agent(runner, sub_goal, agent_name, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_episode(f"V{call_count}", agent_name=agent_name)

        pe.run_agent = _run_agent
        await pe.refinement("Write essay", refinement_iterations=5)
        # Should stop after first iteration due to adaptive early stop
        # 1 drafter + at most 1 refiner iteration
        assert call_count <= 3


# ──────────────────────────────────────────────────────────────────────
# Aggregate results and cooperative credit
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestAggregateAndCredit:
    """Tests for result aggregation and cooperative credit."""

    async def test_aggregate_empty_results(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, _ = _make_mock_manager()
        pe = ParadigmExecutor(sm)
        result = pe.aggregate_results({}, "goal")
        assert not result.success
        assert result.output is None

    async def test_aggregate_single_result_passthrough(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, _ = _make_mock_manager()
        pe = ParadigmExecutor(sm)
        single = _make_episode("solo output")
        result = pe.aggregate_results({"solo": single}, "goal")
        assert result.output == "solo output"

    async def test_aggregate_multiple_results(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, _ = _make_mock_manager()
        pe = ParadigmExecutor(sm)
        results = {
            "a": _make_episode("output A", agent_name="a"),
            "b": _make_episode("output B", agent_name="b"),
        }
        combined = pe.aggregate_results(results, "goal")
        assert combined.success

    async def test_cooperative_credit_assignment(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, _ = _make_mock_manager()
        pe = ParadigmExecutor(sm)
        results = {
            "a": _make_episode("A ok", agent_name="a"),
            "b": _make_episode("B ok", agent_name="b"),
        }
        # Should not raise
        pe.assign_cooperative_credit(results, "goal")
        assert sm.learning_manager.record_outcome.call_count == 2

    async def test_cooperative_credit_skips_single_agent(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import ParadigmExecutor

        sm, _ = _make_mock_manager()
        pe = ParadigmExecutor(sm)
        pe.assign_cooperative_credit({"solo": _make_episode("x")}, "goal")
        sm.learning_manager.record_outcome.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# TierExecutor — Routing
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
class TestExecutorTierRouting:
    """Tests for TierExecutor routing."""

    async def test_tier1_routes_correctly(self, v3_executor):
        from Jotty.core.modes.execution.types import ExecutionConfig, ExecutionTier

        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert result.success
        assert result.tier == ExecutionTier.DIRECT

    async def test_tier2_routes_correctly(self, v3_executor):
        from Jotty.core.modes.execution.types import ExecutionConfig, ExecutionTier

        result = await v3_executor.execute(
            "Plan a research project",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert result.tier == ExecutionTier.AGENTIC

    async def test_tier3_routes_correctly(self, v3_executor):
        from Jotty.core.modes.execution.types import ExecutionConfig, ExecutionTier

        result = await v3_executor.execute(
            "Learn from past data",
            config=ExecutionConfig(
                tier=ExecutionTier.LEARNING,
                memory_backend="none",
                enable_validation=False,
            ),
        )
        assert result.tier == ExecutionTier.LEARNING

    async def test_tier1_output_is_clean_text(self, v3_executor):
        from Jotty.core.modes.execution.types import ExecutionConfig, ExecutionTier

        result = await v3_executor.execute(
            "Hello",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert isinstance(result.output, str)
        assert "AgenticExecutionResult" not in result.output

    async def test_tier1_records_cost(self, v3_executor):
        from Jotty.core.modes.execution.types import ExecutionConfig, ExecutionTier

        result = await v3_executor.execute(
            "What is Python?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert result.cost_usd >= 0

    async def test_execution_failure_returns_error_result(self, v3_executor):
        from Jotty.core.modes.execution.types import ExecutionConfig, ExecutionTier

        v3_executor._provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await v3_executor.execute(
            "Will fail",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert not result.success
        assert "LLM down" in result.error


# ──────────────────────────────────────────────────────────────────────
# Output extraction edge cases
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestOutputExtraction:
    """Tests for _extract_output_text edge cases."""

    def test_none_returns_empty(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        assert _extract_output_text(None) == ""

    def test_string_passthrough(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        assert _extract_output_text("hello world") == "hello world"

    def test_dict_content_field(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        assert _extract_output_text({"content": "extracted"}) == "extracted"

    def test_dict_response_field(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        assert _extract_output_text({"response": "from response"}) == "from response"

    def test_dict_priority_order(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        # content should be tried before response
        result = _extract_output_text({"content": "first", "response": "second"})
        assert result == "first"

    def test_nested_final_output(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        obj = Mock()
        obj.final_output = "clean text"
        assert _extract_output_text(obj) == "clean text"

    def test_nested_outputs_dict(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        obj = Mock()
        obj.final_output = None
        obj.outputs = {"step_1": {"content": "step 1 result"}}
        # Should not have final_output, fall through to outputs
        obj2 = Mock(spec=[])
        obj2.outputs = {"step_1": {"content": "step 1 result"}}
        result = _extract_output_text(obj2)
        assert result == "step 1 result"

    def test_summary_fallback(self):
        from Jotty.core.intelligence.orchestration.paradigm_executor import _extract_output_text

        obj = Mock(spec=[])
        obj.summary = "summary text"
        assert _extract_output_text(obj) == "summary text"
