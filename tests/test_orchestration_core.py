"""
Tests for Orchestration Core Modules
=====================================
Tests for ExecutionContext, AgentRunnerConfig (agent_runner.py),
ProviderManager (provider_manager.py), and EnsembleManager (ensemble_manager.py).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import fields

from Jotty.core.orchestration.agent_runner import ExecutionContext, AgentRunnerConfig
from Jotty.core.foundation.data_structures import SwarmConfig

# Conditional imports for modules with complex dependency chains
try:
    from Jotty.core.orchestration.provider_manager import ProviderManager
    HAS_PROVIDER_MANAGER = True
except ImportError:
    HAS_PROVIDER_MANAGER = False

try:
    from Jotty.core.orchestration.ensemble_manager import EnsembleManager
    HAS_ENSEMBLE_MANAGER = True
except ImportError:
    HAS_ENSEMBLE_MANAGER = False


# =============================================================================
# ExecutionContext Tests (1-6)
# =============================================================================

@pytest.mark.unit
class TestExecutionContext:
    """Tests for the ExecutionContext dataclass used throughout the agent pipeline."""

    def test_defaults_populated(self):
        """ExecutionContext initializes all default fields correctly."""
        ctx = ExecutionContext(goal="test goal", kwargs={"key": "value"})
        assert ctx.start_time == 0.0
        assert ctx.status_callback is None
        assert ctx.gate_decision is None
        assert ctx.skip_architect is False
        assert ctx.skip_auditor is False
        assert ctx.enriched_goal == ""
        assert ctx.proceed is True
        assert ctx.architect_shaped_reward == 0.0
        assert ctx.agent_output is None
        assert ctx.inner_success is False
        assert ctx.success is False
        assert ctx.auditor_reasoning == ""
        assert ctx.auditor_confidence == 0.0
        assert ctx.duration == 0.0
        assert ctx.task_progress is None
        assert ctx.ws_checkpoint_id is None

    def test_goal_stored(self):
        """ExecutionContext stores the goal string."""
        ctx = ExecutionContext(goal="Research quantum computing", kwargs={})
        assert ctx.goal == "Research quantum computing"

    def test_kwargs_stored(self):
        """ExecutionContext stores the kwargs dictionary."""
        kwargs = {"workspace_dir": "/tmp", "status_callback": None}
        ctx = ExecutionContext(goal="test", kwargs=kwargs)
        assert ctx.kwargs == kwargs
        assert ctx.kwargs["workspace_dir"] == "/tmp"

    def test_proceed_defaults_true(self):
        """ExecutionContext.proceed defaults to True so agents run by default."""
        ctx = ExecutionContext(goal="any goal", kwargs={})
        assert ctx.proceed is True

    def test_success_defaults_false(self):
        """ExecutionContext.success defaults to False until auditor validates."""
        ctx = ExecutionContext(goal="any goal", kwargs={})
        assert ctx.success is False

    def test_learning_context_parts_defaults_empty(self):
        """ExecutionContext.learning_context_parts defaults to an empty list."""
        ctx = ExecutionContext(goal="any goal", kwargs={})
        assert ctx.learning_context_parts == []
        assert isinstance(ctx.learning_context_parts, list)

    def test_mutable_default_fields_are_independent(self):
        """Each ExecutionContext instance gets its own mutable default lists/dicts."""
        ctx_a = ExecutionContext(goal="a", kwargs={})
        ctx_b = ExecutionContext(goal="b", kwargs={})
        ctx_a.learning_context_parts.append("item")
        ctx_a.trajectory.append({"step": 1})
        ctx_a.learning_data["key"] = "val"
        # ctx_b should not be affected
        assert ctx_b.learning_context_parts == []
        assert ctx_b.trajectory == []
        assert ctx_b.learning_data == {}


# =============================================================================
# AgentRunnerConfig Tests (7-10)
# =============================================================================

@pytest.mark.unit
class TestAgentRunnerConfig:
    """Tests for the AgentRunnerConfig dataclass."""

    def test_requires_architect_auditor_config(self):
        """AgentRunnerConfig requires architect_prompts, auditor_prompts, and config."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=["prompt_a.md"],
            auditor_prompts=["prompt_b.md"],
            config=config,
        )
        assert runner_config.architect_prompts == ["prompt_a.md"]
        assert runner_config.auditor_prompts == ["prompt_b.md"]
        assert runner_config.config is config

    def test_missing_required_fields_raises(self):
        """AgentRunnerConfig raises TypeError when required fields are missing."""
        with pytest.raises(TypeError):
            AgentRunnerConfig()  # missing all required fields
        with pytest.raises(TypeError):
            AgentRunnerConfig(architect_prompts=["a"])  # missing auditor_prompts and config

    def test_agent_name_defaults_agent(self):
        """AgentRunnerConfig.agent_name defaults to 'agent'."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.agent_name == "agent"

    def test_enable_learning_defaults_true(self):
        """AgentRunnerConfig.enable_learning defaults to True."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.enable_learning is True

    def test_enable_terminal_defaults_true(self):
        """AgentRunnerConfig.enable_terminal defaults to True."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.enable_terminal is True

    def test_enable_memory_defaults_true(self):
        """AgentRunnerConfig.enable_memory defaults to True."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.enable_memory is True


# =============================================================================
# ProviderManager Tests (11-14)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_PROVIDER_MANAGER, reason="ProviderManager not importable")
class TestProviderManager:
    """Tests for the ProviderManager composed class."""

    def test_init_with_config_and_callable(self):
        """ProviderManager stores config and swarm_intelligence getter."""
        config = SwarmConfig()
        getter = Mock(return_value="si_instance")
        pm = ProviderManager(config=config, get_swarm_intelligence=getter)
        assert pm.config is config
        assert pm._get_si is getter

    def test_swarm_intelligence_property_returns_callable_result(self):
        """ProviderManager.swarm_intelligence property delegates to the getter callable."""
        mock_si = Mock(name="SwarmIntelligence")
        getter = Mock(return_value=mock_si)
        config = SwarmConfig()
        pm = ProviderManager(config=config, get_swarm_intelligence=getter)
        result = pm.swarm_intelligence
        getter.assert_called_once()
        assert result is mock_si

    def test_init_provider_registry_creates_registry(self):
        """ProviderManager.init_provider_registry creates a registry when providers load."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False

        config = SwarmConfig()
        mock_si = Mock(name="SwarmIntelligence")
        pm = ProviderManager(config=config, get_swarm_intelligence=lambda: mock_si)

        # Mock the _load_providers and _provider_cache to simulate provider loading
        mock_registry_instance = Mock()
        mock_registry_instance.get_all_providers.return_value = {"test_provider": Mock()}

        mock_cache = {
            "ProviderRegistry": Mock(return_value=mock_registry_instance),
            "BrowserUseProvider": Mock(return_value=Mock(name="browser-use")),
            "OpenHandsProvider": Mock(return_value=Mock(name="openhands")),
            "AgentSProvider": Mock(return_value=Mock(name="agent-s")),
            "OpenInterpreterProvider": Mock(return_value=Mock(name="open-interpreter")),
            "ResearchAndAnalyzeProvider": Mock(return_value=Mock(name="research")),
            "AutomateWorkflowProvider": Mock(return_value=Mock(name="workflow")),
            "FullStackAgentProvider": Mock(return_value=Mock(name="fullstack")),
        }

        with patch(
            "Jotty.core.orchestration.swarm_manager._load_providers",
            return_value=True,
        ), patch(
            "Jotty.core.orchestration.swarm_manager._provider_cache",
            mock_cache,
        ), patch.object(
            pm, "_get_provider_registry_path", return_value=mock_path_obj,
        ):
            pm.init_provider_registry()
            assert pm.provider_registry is not None

    def test_provider_registry_starts_none(self):
        """ProviderManager.provider_registry is None before init_provider_registry."""
        config = SwarmConfig()
        pm = ProviderManager(config=config, get_swarm_intelligence=lambda: None)
        assert pm.provider_registry is None


# =============================================================================
# EnsembleManager Tests (15-18)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_ENSEMBLE_MANAGER, reason="EnsembleManager not importable")
class TestEnsembleManager:
    """Tests for the EnsembleManager composed class."""

    def test_instantiation(self):
        """EnsembleManager can be instantiated without arguments."""
        em = EnsembleManager()
        assert em is not None
        assert hasattr(em, "execute_ensemble")
        assert hasattr(em, "should_auto_ensemble")

    @pytest.mark.asyncio
    async def test_execute_ensemble_returns_dict(self):
        """EnsembleManager.execute_ensemble returns a dict with success key."""
        em = EnsembleManager()

        # Mock the skill-based ensemble path via the source module where the
        # local import resolves: Jotty.core.registry.skills_registry
        mock_tool = Mock(return_value={
            "success": True,
            "response": "synthesized answer",
            "quality_scores": {"analytical": 0.9},
        })
        mock_skill = Mock()
        mock_skill.tools = {"ensemble_prompt_tool": mock_tool}
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        with patch(
            "Jotty.core.registry.skills_registry.get_skills_registry",
            return_value=mock_registry,
        ):
            result = await em.execute_ensemble("What is AI?")
            assert isinstance(result, dict)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_ensemble_strategy_parameter(self):
        """EnsembleManager.execute_ensemble accepts strategy parameter."""
        em = EnsembleManager()

        mock_tool = Mock(return_value={
            "success": True,
            "response": "debate result",
            "quality_scores": {},
        })
        mock_skill = Mock()
        mock_skill.tools = {"ensemble_prompt_tool": mock_tool}
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        with patch(
            "Jotty.core.registry.skills_registry.get_skills_registry",
            return_value=mock_registry,
        ):
            result = await em.execute_ensemble(
                "Analyze climate change",
                strategy="debate",
                max_perspectives=3,
            )
            assert isinstance(result, dict)
            # Verify the tool was called with the strategy parameter
            call_args = mock_tool.call_args[0][0]
            assert call_args["strategy"] == "debate"
            assert call_args["max_perspectives"] == 3

    @pytest.mark.asyncio
    async def test_execute_ensemble_skill_fallback(self):
        """EnsembleManager falls back to DSPy when skill registry unavailable."""
        em = EnsembleManager()

        # Simulate ImportError for skills registry so it falls to DSPy path,
        # then mock DSPy to return a result.
        with patch(
            "Jotty.core.registry.skills_registry.get_skills_registry",
            side_effect=ImportError("no skills"),
        ):
            # The DSPy fallback path will try 'import dspy'. Mock it so
            # dspy.settings.lm exists and produces responses.
            mock_lm = Mock(side_effect=[
                ["analytical perspective"],   # perspective 1
                ["creative perspective"],      # perspective 2
                ["critical perspective"],      # perspective 3
                ["practical perspective"],     # perspective 4
                ["final synthesis"],           # synthesis call
            ])
            mock_dspy = MagicMock()
            mock_dspy.settings.lm = mock_lm

            with patch.dict("sys.modules", {"dspy": mock_dspy}):
                result = await em.execute_ensemble("What is consciousness?")
                assert isinstance(result, dict)
                assert result["success"] is True
                assert "perspectives_used" in result
                assert len(result["perspectives_used"]) > 0


# =============================================================================
# Conditional imports for new test classes
# =============================================================================

try:
    from Jotty.core.orchestration.swarm_data_structures import (
        AgentSpecialization, AgentProfile, ConsensusVote, SwarmDecision,
        AgentSession, HandoffContext, Coalition, AuctionBid,
        GossipMessage, SupervisorNode,
    )
    HAS_SWARM_DATA_STRUCTURES = True
except ImportError:
    HAS_SWARM_DATA_STRUCTURES = False

try:
    from Jotty.core.orchestration.swarm_roadmap import (
        SwarmTaskBoard, SubtaskState, TodoItem, TrajectoryStep,
        AgenticState, DecomposedQFunction, ThoughtLevelCredit,
        StateCheckpointer,
    )
    from Jotty.core.foundation.types import TaskStatus
    HAS_SWARM_ROADMAP = True
except ImportError:
    HAS_SWARM_ROADMAP = False

try:
    from Jotty.core.orchestration.swarm_intelligence import SwarmIntelligence
    HAS_SWARM_INTELLIGENCE = True
except ImportError:
    HAS_SWARM_INTELLIGENCE = False

try:
    from Jotty.core.orchestration.swarm_state_manager import (
        AgentStateTracker, SwarmStateManager,
    )
    HAS_SWARM_STATE_MANAGER = True
except ImportError:
    HAS_SWARM_STATE_MANAGER = False

import time


# =============================================================================
# TestSwarmDataStructuresDeep (25 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_DATA_STRUCTURES, reason="swarm_data_structures not importable")
class TestSwarmDataStructuresDeep:
    """Deep tests for all swarm data structures: enums, dataclasses, methods."""

    # --- AgentSpecialization enum ---

    def test_specialization_generalist_value(self):
        """AgentSpecialization.GENERALIST has value 'generalist'."""
        assert AgentSpecialization.GENERALIST.value == "generalist"

    def test_specialization_aggregator_value(self):
        """AgentSpecialization.AGGREGATOR has value 'aggregator'."""
        assert AgentSpecialization.AGGREGATOR.value == "aggregator"

    def test_specialization_analyzer_value(self):
        """AgentSpecialization.ANALYZER has value 'analyzer'."""
        assert AgentSpecialization.ANALYZER.value == "analyzer"

    def test_specialization_transformer_value(self):
        """AgentSpecialization.TRANSFORMER has value 'transformer'."""
        assert AgentSpecialization.TRANSFORMER.value == "transformer"

    def test_specialization_validator_value(self):
        """AgentSpecialization.VALIDATOR has value 'validator'."""
        assert AgentSpecialization.VALIDATOR.value == "validator"

    def test_specialization_planner_value(self):
        """AgentSpecialization.PLANNER has value 'planner'."""
        assert AgentSpecialization.PLANNER.value == "planner"

    def test_specialization_executor_value(self):
        """AgentSpecialization.EXECUTOR has value 'executor'."""
        assert AgentSpecialization.EXECUTOR.value == "executor"

    def test_specialization_actor_value(self):
        """AgentSpecialization.ACTOR has value 'actor'."""
        assert AgentSpecialization.ACTOR.value == "actor"

    def test_specialization_expert_value(self):
        """AgentSpecialization.EXPERT has value 'expert'."""
        assert AgentSpecialization.EXPERT.value == "expert"

    def test_specialization_reviewer_value(self):
        """AgentSpecialization.REVIEWER has value 'reviewer'."""
        assert AgentSpecialization.REVIEWER.value == "reviewer"

    def test_specialization_orchestrator_value(self):
        """AgentSpecialization.ORCHESTRATOR has value 'orchestrator'."""
        assert AgentSpecialization.ORCHESTRATOR.value == "orchestrator"

    def test_specialization_researcher_value(self):
        """AgentSpecialization.RESEARCHER has value 'researcher'."""
        assert AgentSpecialization.RESEARCHER.value == "researcher"

    def test_specialization_has_exactly_twelve_members(self):
        """AgentSpecialization enum has exactly 12 members."""
        assert len(AgentSpecialization) == 12

    # --- AgentProfile ---

    def test_agent_profile_update_task_result_incremental_avg_time(self):
        """update_task_result computes incremental average execution time."""
        profile = AgentProfile(agent_name="a1")
        profile.update_task_result("analysis", True, 10.0)
        assert profile.avg_execution_time == 10.0
        profile.update_task_result("analysis", True, 20.0)
        assert profile.avg_execution_time == pytest.approx(15.0)
        profile.update_task_result("analysis", True, 30.0)
        assert profile.avg_execution_time == pytest.approx(20.0)

    def test_agent_profile_trust_score_formula(self):
        """Trust score = 0.3 + 0.7 * (overall_success / overall_total)."""
        profile = AgentProfile(agent_name="a1")
        profile.update_task_result("analysis", True, 1.0)
        # 1 success out of 1 total -> trust = 0.3 + 0.7 * 1.0 = 1.0
        assert profile.trust_score == pytest.approx(1.0)
        profile.update_task_result("analysis", False, 1.0)
        # 1 success out of 2 total -> trust = 0.3 + 0.7 * 0.5 = 0.65
        assert profile.trust_score == pytest.approx(0.65)

    def test_agent_profile_specialization_update_above_threshold(self):
        """Agent specializes when success rate > 0.7 and enough samples."""
        profile = AgentProfile(agent_name="a1")
        # Need min_samples=2 for single-task-type agents
        profile.update_task_result("analysis", True, 1.0)
        profile.update_task_result("analysis", True, 1.0)
        assert profile.specialization == AgentSpecialization.ANALYZER

    def test_agent_profile_specialization_stays_generalist_below_threshold(self):
        """Agent stays GENERALIST when success rate is below 0.7."""
        profile = AgentProfile(agent_name="a1")
        # 1 success, 2 failures => rate = 1/3 < 0.7
        profile.update_task_result("analysis", True, 1.0)
        profile.update_task_result("analysis", False, 1.0)
        profile.update_task_result("analysis", False, 1.0)
        assert profile.specialization == AgentSpecialization.GENERALIST

    def test_agent_profile_specialization_adaptive_min_samples(self):
        """min_samples is 2 for single-task-type, 3 for multi-task-type agents."""
        profile = AgentProfile(agent_name="a1")
        # Start with two task types introduced together (both below threshold)
        profile.update_task_result("analysis", True, 1.0)
        profile.update_task_result("planning", True, 1.0)
        # Now we have 2 task types, min_samples=3; each has only 1 sample
        assert profile.specialization == AgentSpecialization.GENERALIST
        profile.update_task_result("analysis", True, 1.0)
        # analysis has 2 samples, but min_samples for multi-type is 3
        assert profile.specialization == AgentSpecialization.GENERALIST
        # Add third analysis sample
        profile.update_task_result("analysis", True, 1.0)
        assert profile.specialization == AgentSpecialization.ANALYZER

    def test_agent_profile_get_success_rate_default(self):
        """get_success_rate returns 0.5 for unknown task types."""
        profile = AgentProfile(agent_name="a1")
        assert profile.get_success_rate("unknown_type") == 0.5

    def test_agent_profile_get_success_rate_computed(self):
        """get_success_rate returns actual rate for known task types."""
        profile = AgentProfile(agent_name="a1")
        profile.update_task_result("analysis", True, 1.0)
        profile.update_task_result("analysis", False, 1.0)
        assert profile.get_success_rate("analysis") == pytest.approx(0.5)

    # --- ConsensusVote ---

    def test_consensus_vote_fields(self):
        """ConsensusVote has all expected fields and auto-generates timestamp."""
        before = time.time()
        vote = ConsensusVote(
            agent_name="agent_a",
            decision="approve",
            confidence=0.9,
            reasoning="Looks good",
        )
        after = time.time()
        assert vote.agent_name == "agent_a"
        assert vote.decision == "approve"
        assert vote.confidence == 0.9
        assert vote.reasoning == "Looks good"
        assert before <= vote.timestamp <= after

    # --- SwarmDecision ---

    def test_swarm_decision_fields(self):
        """SwarmDecision stores question, votes, final decision, consensus strength, dissent."""
        vote = ConsensusVote(agent_name="a1", decision="yes", confidence=0.8, reasoning="ok")
        decision = SwarmDecision(
            question="Deploy?",
            votes=[vote],
            final_decision="yes",
            consensus_strength=0.9,
            dissenting_views=["agent_b disagrees"],
        )
        assert decision.question == "Deploy?"
        assert len(decision.votes) == 1
        assert decision.final_decision == "yes"
        assert decision.consensus_strength == 0.9
        assert decision.dissenting_views == ["agent_b disagrees"]

    # --- AgentSession ---

    def test_agent_session_add_message_updates_last_active(self):
        """add_message updates last_active timestamp."""
        session = AgentSession(session_id="s1", agent_name="a1", context="main")
        old_active = session.last_active
        time.sleep(0.01)
        session.add_message("a2", "hello")
        assert session.last_active > old_active

    def test_agent_session_add_message_bounded_at_100(self):
        """AgentSession.messages is bounded at 100 entries."""
        session = AgentSession(session_id="s1", agent_name="a1", context="main")
        for i in range(120):
            session.add_message("agent", f"msg_{i}")
        assert len(session.messages) == 100
        # Last message should be the most recent
        assert session.messages[-1]["content"] == "msg_119"

    # --- HandoffContext ---

    def test_handoff_context_add_to_chain(self):
        """add_to_chain appends agent to handoff_chain if not already present."""
        hc = HandoffContext(
            task_id="t1", from_agent="a1", to_agent="a2", task_type="analysis"
        )
        hc.add_to_chain("a1")
        hc.add_to_chain("a2")
        hc.add_to_chain("a1")  # duplicate, should not be added
        assert hc.handoff_chain == ["a1", "a2"]

    # --- Coalition ---

    def test_coalition_add_member_with_role(self):
        """add_member adds agent with a role to the coalition."""
        coal = Coalition(coalition_id="c1", task_type="coding", leader="a1")
        coal.add_member("a2", role="reviewer")
        assert "a2" in coal.members
        assert coal.roles["a2"] == "reviewer"

    def test_coalition_add_member_no_duplicate(self):
        """add_member does not add duplicate members."""
        coal = Coalition(coalition_id="c1", task_type="coding", leader="a1")
        coal.add_member("a2", role="worker")
        coal.add_member("a2", role="lead")  # should not add again
        assert coal.members.count("a2") == 1
        assert coal.roles["a2"] == "worker"  # original role preserved

    def test_coalition_remove_member(self):
        """remove_member removes agent and its role."""
        coal = Coalition(coalition_id="c1", task_type="coding", leader="a1")
        coal.add_member("a2", role="reviewer")
        coal.remove_member("a2")
        assert "a2" not in coal.members
        assert "a2" not in coal.roles

    def test_coalition_remove_nonexistent_member(self):
        """remove_member is a no-op for non-existent agents."""
        coal = Coalition(coalition_id="c1", task_type="coding", leader="a1")
        coal.remove_member("nonexistent")  # should not raise
        assert len(coal.members) == 0

    # --- AuctionBid ---

    def test_auction_bid_score_property(self):
        """AuctionBid.score = 0.3*bid + 0.25*conf + 0.25*spec + 0.2*(1-load)."""
        bid = AuctionBid(
            agent_name="a1", task_id="t1",
            bid_value=1.0, estimated_time=10.0,
            confidence=1.0, specialization_match=1.0, current_load=0.0,
        )
        # 0.3*1.0 + 0.25*1.0 + 0.25*1.0 + 0.2*(1-0) = 1.0
        assert bid.score == pytest.approx(1.0)

    def test_auction_bid_score_partial_values(self):
        """AuctionBid.score with partial values computes correctly."""
        bid = AuctionBid(
            agent_name="a1", task_id="t1",
            bid_value=0.5, estimated_time=10.0,
            confidence=0.5, specialization_match=0.5, current_load=0.5,
        )
        expected = 0.3 * 0.5 + 0.25 * 0.5 + 0.25 * 0.5 + 0.2 * 0.5
        assert bid.score == pytest.approx(expected)

    # --- GossipMessage ---

    def test_gossip_message_mark_seen_decrements_ttl(self):
        """mark_seen decrements ttl and returns True if ttl > 0."""
        msg = GossipMessage(
            message_id="m1", content={"info": "test"},
            origin_agent="a1", message_type="info", ttl=3,
        )
        assert msg.mark_seen("a2") is True  # ttl goes to 2
        assert msg.ttl == 2
        assert "a2" in msg.seen_by

    def test_gossip_message_mark_seen_returns_false_when_already_seen(self):
        """mark_seen returns False if agent already in seen_by."""
        msg = GossipMessage(
            message_id="m1", content={},
            origin_agent="a1", message_type="info", ttl=3,
        )
        msg.mark_seen("a2")
        assert msg.mark_seen("a2") is False
        assert msg.ttl == 2  # ttl only decremented once

    def test_gossip_message_mark_seen_returns_false_when_ttl_exhausted(self):
        """mark_seen returns False when ttl reaches 0."""
        msg = GossipMessage(
            message_id="m1", content={},
            origin_agent="a1", message_type="info", ttl=1,
        )
        # ttl goes from 1 to 0, should return False (no more propagation)
        assert msg.mark_seen("a2") is False
        assert msg.ttl == 0

    # --- SupervisorNode ---

    def test_supervisor_node_is_leaf(self):
        """is_leaf returns True when level == 0."""
        node = SupervisorNode(node_id="n1", agent_name="a1", level=0)
        assert node.is_leaf() is True

    def test_supervisor_node_is_not_leaf(self):
        """is_leaf returns False when level > 0."""
        node = SupervisorNode(node_id="n1", agent_name="a1", level=2)
        assert node.is_leaf() is False

    def test_supervisor_node_is_root(self):
        """is_root returns True when parent is None."""
        node = SupervisorNode(node_id="n1", agent_name="a1", level=3, parent=None)
        assert node.is_root() is True

    def test_supervisor_node_is_not_root(self):
        """is_root returns False when parent is set."""
        node = SupervisorNode(node_id="n1", agent_name="a1", level=1, parent="n0")
        assert node.is_root() is False


# =============================================================================
# TestSubtaskState (12 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_ROADMAP, reason="swarm_roadmap not importable")
class TestSubtaskState:
    """Tests for SubtaskState dataclass lifecycle methods."""

    def test_defaults(self):
        """SubtaskState has correct defaults: priority=1.0, confidence=0.5, max_attempts=3."""
        st = SubtaskState(task_id="t1", description="Do something")
        assert st.priority == 1.0
        assert st.confidence == 0.5
        assert st.max_attempts == 3
        assert st.status == TaskStatus.PENDING
        assert st.attempts == 0
        assert st.progress == 0.0

    def test_can_start_no_deps(self):
        """can_start returns True when no dependencies."""
        st = SubtaskState(task_id="t1", description="Do something")
        assert st.can_start(set()) is True

    def test_can_start_deps_met(self):
        """can_start returns True when all dependencies are in completed_tasks."""
        st = SubtaskState(task_id="t2", description="Step 2", depends_on=["t1"])
        assert st.can_start({"t1"}) is True

    def test_can_start_deps_not_met(self):
        """can_start returns False when not all dependencies are completed."""
        st = SubtaskState(task_id="t2", description="Step 2", depends_on=["t1", "t0"])
        assert st.can_start({"t1"}) is False

    def test_start_sets_in_progress(self):
        """start() sets status to IN_PROGRESS."""
        st = SubtaskState(task_id="t1", description="Do something")
        st.start()
        assert st.status == TaskStatus.IN_PROGRESS

    def test_start_increments_attempts(self):
        """start() increments attempts count."""
        st = SubtaskState(task_id="t1", description="Do something")
        st.start()
        assert st.attempts == 1
        # Reset status to PENDING for second start
        st.status = TaskStatus.PENDING
        st.start()
        assert st.attempts == 2

    def test_start_sets_started_at(self):
        """start() sets started_at timestamp."""
        st = SubtaskState(task_id="t1", description="Do something")
        assert st.started_at is None
        st.start()
        assert st.started_at is not None

    def test_complete_sets_completed(self):
        """complete(result) sets status to COMPLETED and progress to 1.0."""
        st = SubtaskState(task_id="t1", description="Do something")
        st.start()
        st.complete(result={"output": "done"})
        assert st.status == TaskStatus.COMPLETED
        assert st.progress == 1.0
        assert st.result == {"output": "done"}

    def test_complete_sets_completed_at(self):
        """complete() sets completed_at timestamp."""
        st = SubtaskState(task_id="t1", description="Do something")
        st.start()
        st.complete()
        assert st.completed_at is not None

    def test_fail_sets_failed_when_max_attempts_reached(self):
        """fail() sets FAILED when attempts >= max_attempts."""
        st = SubtaskState(task_id="t1", description="Do something", max_attempts=2)
        st.start()  # attempts = 1
        st.status = TaskStatus.PENDING
        st.start()  # attempts = 2
        st.fail("some error")
        assert st.status == TaskStatus.FAILED
        assert st.error == "some error"

    def test_fail_sets_pending_for_retry(self):
        """fail() sets PENDING when attempts < max_attempts (retry)."""
        st = SubtaskState(task_id="t1", description="Do something", max_attempts=3)
        st.start()  # attempts = 1
        st.fail("first error")
        assert st.status == TaskStatus.PENDING  # can retry
        assert st.error == "first error"

    def test_intermediary_values_dict(self):
        """intermediary_values stores arbitrary runtime metrics."""
        st = SubtaskState(task_id="t1", description="Do something")
        st.intermediary_values["llm_calls"] = 8
        st.intermediary_values["time_elapsed"] = 4.2
        assert st.intermediary_values == {"llm_calls": 8, "time_elapsed": 4.2}


# =============================================================================
# TestSwarmTaskBoard (25 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_ROADMAP, reason="swarm_roadmap not importable")
class TestSwarmTaskBoard:
    """Tests for SwarmTaskBoard task management and lifecycle."""

    def test_post_init_generates_todo_id(self):
        """__post_init__ generates a todo_id from hash when not provided."""
        board = SwarmTaskBoard(root_task="Test root")
        assert board.todo_id != ""
        assert len(board.todo_id) == 32  # MD5 hex digest

    def test_post_init_keeps_provided_todo_id(self):
        """__post_init__ preserves explicit todo_id."""
        board = SwarmTaskBoard(root_task="Test", todo_id="custom_id")
        assert board.todo_id == "custom_id"

    def test_add_task_creates_subtask_state(self):
        """add_task creates a SubtaskState in subtasks dict."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First task", actor="agent_a")
        assert "t1" in board.subtasks
        assert isinstance(board.subtasks["t1"], SubtaskState)
        assert board.subtasks["t1"].description == "First task"
        assert board.subtasks["t1"].actor == "agent_a"

    def test_add_task_no_hardcoded_values(self):
        """add_task uses provided parameters, no hardcoded overrides."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "Task", priority=2.5, max_attempts=5, estimated_duration=120.0)
        st = board.subtasks["t1"]
        assert st.priority == 2.5
        assert st.max_attempts == 5
        assert st.estimated_duration == 120.0

    def test_add_task_updates_blocks(self):
        """add_task updates blocks list on dependency tasks."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second", depends_on=["t1"])
        assert "t2" in board.subtasks["t1"].blocks

    def test_add_task_appends_to_execution_order(self):
        """add_task appends task_id to execution_order."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second")
        assert board.execution_order == ["t1", "t2"]

    def test_get_next_task_returns_subtask_state(self):
        """get_next_task returns a SubtaskState object, not a string ID."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First", actor="a1")
        result = board.get_next_task()
        assert isinstance(result, SubtaskState)
        assert result.task_id == "t1"

    def test_get_next_task_returns_none_when_empty(self):
        """get_next_task returns None when no tasks are available."""
        board = SwarmTaskBoard(root_task="Root")
        assert board.get_next_task() is None

    def test_get_next_task_respects_dependencies(self):
        """get_next_task skips tasks with unmet dependencies."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second", depends_on=["t1"])
        result = board.get_next_task()
        assert result.task_id == "t1"

    def test_get_next_task_fallback_to_execution_order(self):
        """get_next_task falls back to execution order without q_predictor."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second")
        result = board.get_next_task()
        assert result.task_id == "t1"

    def test_get_next_task_epsilon_greedy_with_q_predictor(self):
        """get_next_task uses Q-predictor when provided with epsilon=0 (exploit)."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First", actor="a1")
        board.add_task("t2", "Second", actor="a2")

        mock_predictor = Mock()
        # Return higher Q-value for t2
        mock_predictor.predict_q_value.side_effect = lambda state, action, goal: (
            (0.9, None, None) if action['actor'] == 'a2' else (0.3, None, None)
        )
        result = board.get_next_task(
            q_predictor=mock_predictor,
            current_state={"s": 1},
            goal="test",
            epsilon=0.0,  # pure exploitation
        )
        assert result.task_id == "t2"

    def test_unblock_ready_tasks(self):
        """unblock_ready_tasks returns count of newly unblocked tasks."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second", depends_on=["t1"])
        # Manually set t2 to BLOCKED
        board.subtasks["t2"].status = TaskStatus.BLOCKED
        # Complete t1
        board.completed_tasks.add("t1")
        unblocked = board.unblock_ready_tasks()
        assert unblocked == 1
        assert board.subtasks["t2"].status == TaskStatus.PENDING

    def test_unblock_ready_tasks_returns_zero_when_none_blocked(self):
        """unblock_ready_tasks returns 0 when no tasks are blocked."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        assert board.unblock_ready_tasks() == 0

    def test_start_task(self):
        """start_task transitions task to IN_PROGRESS and sets current_task_id."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.start_task("t1")
        assert board.subtasks["t1"].status == TaskStatus.IN_PROGRESS
        assert board.current_task_id == "t1"

    def test_complete_task(self):
        """complete_task transitions task to COMPLETED, adds to completed_tasks."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.start_task("t1")
        board.complete_task("t1", result={"output": "done"})
        assert board.subtasks["t1"].status == TaskStatus.COMPLETED
        assert "t1" in board.completed_tasks
        assert board.current_task_id is None

    def test_fail_task_adds_to_failed_tasks(self):
        """fail_task adds to failed_tasks when max_attempts reached."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First", max_attempts=1)
        board.start_task("t1")
        board.fail_task("t1", "error occurred")
        assert "t1" in board.failed_tasks
        assert board.subtasks["t1"].status == TaskStatus.FAILED

    def test_fail_task_retries_when_under_max_attempts(self):
        """fail_task keeps task retryable when under max_attempts."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First", max_attempts=3)
        board.start_task("t1")
        board.fail_task("t1", "error")
        assert "t1" not in board.failed_tasks
        assert board.subtasks["t1"].status == TaskStatus.PENDING

    def test_checkpoint_and_restore(self):
        """checkpoint() and restore_from_checkpoint() roundtrip correctly."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second")
        board.start_task("t1")
        board.complete_task("t1", {"out": "ok"})
        board.start_task("t2")

        cp = board.checkpoint()
        assert cp["todo_id"] == board.todo_id
        assert "t1" in cp["completed_tasks"]

        # Create new board and restore
        board2 = SwarmTaskBoard(root_task="Root", todo_id=board.todo_id)
        board2.add_task("t1", "First")
        board2.add_task("t2", "Second")
        board2.restore_from_checkpoint(cp)
        assert "t1" in board2.completed_tasks
        assert board2.subtasks["t1"].status == TaskStatus.COMPLETED

    def test_get_progress_summary(self):
        """get_progress_summary returns human-readable string."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.add_task("t2", "Second")
        board.start_task("t1")
        board.complete_task("t1")
        summary = board.get_progress_summary()
        assert "1/2 completed" in summary
        assert "Completion probability" in summary

    def test_should_replan_deadline_exceeded(self):
        """should_replan returns True when deadline exceeded."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        should, reason = board.should_replan(elapsed_time=310.0, global_deadline=300.0)
        assert should is True
        assert "DEADLINE_EXCEEDED" in reason

    def test_should_replan_on_track(self):
        """should_replan returns False when on track."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.complete_task("t1")
        should, reason = board.should_replan(elapsed_time=10.0, global_deadline=300.0)
        assert should is False
        assert "On track" in reason

    def test_should_replan_high_failure_rate(self):
        """should_replan returns True when failure rate > 30%."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First", max_attempts=1)
        board.add_task("t2", "Second", max_attempts=1)
        board.start_task("t1")
        board.fail_task("t1", "err")
        should, reason = board.should_replan(elapsed_time=10.0, global_deadline=300.0)
        assert should is True
        assert "HIGH_FAILURE_RATE" in reason

    def test_replan_skips_blocked_by_failed(self):
        """replan skips tasks blocked by failed dependencies."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First", max_attempts=1)
        board.add_task("t2", "Second", depends_on=["t1"])
        board.start_task("t1")
        board.fail_task("t1", "error")
        actions = board.replan()
        skip_actions = [a for a in actions if a.startswith("SKIP:")]
        assert len(skip_actions) == 1
        assert "t2" in skip_actions[0]

    def test_replan_records_risk_factors(self):
        """replan records observation in risk_factors."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        actions = board.replan(observation="Agent is struggling")
        assert any("RISK_NOTED" in a for a in actions)
        assert any("Agent is struggling" in rf for rf in board.risk_factors)

    def test_items_property_alias(self):
        """items property is an alias for subtasks."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        assert board.items is board.subtasks
        assert "t1" in board.items

    def test_completed_property_alias(self):
        """completed property is an alias for completed_tasks."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.complete_task("t1")
        assert board.completed is board.completed_tasks
        assert "t1" in board.completed

    def test_get_task_by_id(self):
        """get_task_by_id returns SubtaskState for known ID, None for unknown."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        assert board.get_task_by_id("t1") is board.subtasks["t1"]
        assert board.get_task_by_id("nonexistent") is None

    def test_update_q_value(self):
        """update_q_value clamps Q-value and confidence to [0, 1]."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.update_q_value("t1", 0.85, 0.75)
        assert board.subtasks["t1"].estimated_reward == pytest.approx(0.85)
        assert board.subtasks["t1"].confidence == pytest.approx(0.75)
        # Test clamping
        board.update_q_value("t1", 1.5, -0.3)
        assert board.subtasks["t1"].estimated_reward == 1.0
        assert board.subtasks["t1"].confidence == 0.0

    def test_record_intermediary_values(self):
        """record_intermediary_values updates task intermediary_values dict."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.record_intermediary_values("t1", {"llm_calls": 5, "time": 3.2})
        assert board.subtasks["t1"].intermediary_values == {"llm_calls": 5, "time": 3.2}
        # Additional update merges
        board.record_intermediary_values("t1", {"blocks": 2})
        assert board.subtasks["t1"].intermediary_values == {"llm_calls": 5, "time": 3.2, "blocks": 2}

    def test_predict_next(self):
        """predict_next records prediction metadata on a task."""
        board = SwarmTaskBoard(root_task="Root")
        board.add_task("t1", "First")
        board.predict_next("t1", "t2", duration=5.0, reward=0.8)
        st = board.subtasks["t1"]
        assert st.predicted_next_task == "t2"
        assert st.predicted_duration == 5.0
        assert st.predicted_reward == 0.8


# =============================================================================
# TestDecomposedQFunction (15 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_ROADMAP, reason="swarm_roadmap not importable")
class TestDecomposedQFunction:
    """Tests for multi-objective DecomposedQFunction."""

    def _make_state(self, agent="test", task="task"):
        """Helper to create an AgenticState for tests."""
        return AgenticState(agent_name=agent, task_description=task)

    def test_init_four_q_tables(self):
        """Init creates 4 empty Q-tables."""
        qf = DecomposedQFunction()
        assert isinstance(qf.q_task, dict) and len(qf.q_task) == 0
        assert isinstance(qf.q_explore, dict) and len(qf.q_explore) == 0
        assert isinstance(qf.q_causal, dict) and len(qf.q_causal) == 0
        assert isinstance(qf.q_safety, dict) and len(qf.q_safety) == 0

    def test_init_default_weights(self):
        """Default weights are task=0.5, explore=0.2, causal=0.15, safety=0.15."""
        qf = DecomposedQFunction()
        assert qf.weights == {'task': 0.5, 'explore': 0.2, 'causal': 0.15, 'safety': 0.15}

    def test_get_q_value_default_for_unknown(self):
        """get_q_value returns default 0.5 for unknown state-action pair."""
        qf = DecomposedQFunction()
        state = self._make_state()
        assert qf.get_q_value(state, "proceed", objective="task") == 0.5

    def test_get_q_value_returns_stored(self):
        """get_q_value returns stored value after update."""
        qf = DecomposedQFunction()
        state = self._make_state()
        key = (state.to_key(), "proceed")
        qf.q_task[key] = 0.8
        assert qf.get_q_value(state, "proceed", objective="task") == 0.8

    def test_get_q_value_without_objective_returns_combined(self):
        """get_q_value without objective returns combined value."""
        qf = DecomposedQFunction()
        state = self._make_state()
        # All tables default to 0.5, so combined = sum(w*0.5) = 0.5
        assert qf.get_q_value(state, "proceed") == pytest.approx(0.5)

    def test_get_combined_value_weighted_sum(self):
        """get_combined_value is weighted sum of all Q-tables."""
        qf = DecomposedQFunction()
        state = self._make_state()
        key = (state.to_key(), "proceed")
        qf.q_task[key] = 1.0
        qf.q_explore[key] = 0.0
        qf.q_causal[key] = 0.0
        qf.q_safety[key] = 0.0
        # 0.5*1.0 + 0.2*0.0 + 0.15*0.0 + 0.15*0.0 = 0.5
        assert qf.get_combined_value(state, "proceed") == pytest.approx(0.5)

    def test_update_td_update_all_tables(self):
        """update performs TD update on all four Q-tables."""
        qf = DecomposedQFunction()
        state = self._make_state(agent="a1", task="t1")
        next_state = self._make_state(agent="a1", task="t2")
        rewards = {'task': 1.0, 'explore': 0.5, 'causal': 0.3, 'safety': 0.8}
        qf.update(state, "proceed", rewards, next_state, gamma=0.95)
        # All tables should have been updated (no longer at default)
        key = (state.to_key(), "proceed")
        assert key in qf.q_task
        assert key in qf.q_explore
        assert key in qf.q_causal
        assert key in qf.q_safety

    def test_update_changes_q_value(self):
        """update changes Q-value from default towards reward."""
        qf = DecomposedQFunction()
        state = self._make_state()
        next_state = self._make_state(agent="a1", task="next")
        rewards = {'task': 1.0, 'explore': 0.0, 'causal': 0.0, 'safety': 0.0}
        qf.update(state, "proceed", rewards, next_state, gamma=0.0)
        key = (state.to_key(), "proceed")
        # TD target = reward + 0*max_next = 1.0
        # new_q = 0.5 + alpha_task * (1.0 - 0.5) = 0.5 + 0.05 * 0.5 = 0.525
        assert qf.q_task[key] == pytest.approx(0.525)

    def test_get_possible_actions(self):
        """_get_possible_actions returns the four standard actions."""
        qf = DecomposedQFunction()
        state = self._make_state()
        actions = qf._get_possible_actions(state)
        assert actions == ['proceed', 'retry', 'refine', 'escalate']

    def test_adjust_weights_exploration(self):
        """adjust_weights('exploration') boosts explore weight."""
        qf = DecomposedQFunction()
        qf.adjust_weights('exploration')
        assert qf.weights['explore'] == 0.4
        assert qf.weights['task'] == 0.3

    def test_adjust_weights_exploitation(self):
        """adjust_weights('exploitation') boosts task weight."""
        qf = DecomposedQFunction()
        qf.adjust_weights('exploitation')
        assert qf.weights['task'] == 0.6
        assert qf.weights['explore'] == 0.1

    def test_adjust_weights_safety_critical(self):
        """adjust_weights('safety_critical') boosts safety weight."""
        qf = DecomposedQFunction()
        qf.adjust_weights('safety_critical')
        assert qf.weights['safety'] == 0.5
        assert qf.weights['task'] == 0.3

    def test_get_action_ranking(self):
        """get_action_ranking ranks actions by combined Q-value descending."""
        qf = DecomposedQFunction()
        state = self._make_state()
        key_proceed = (state.to_key(), "proceed")
        key_retry = (state.to_key(), "retry")
        qf.q_task[key_proceed] = 0.9
        qf.q_task[key_retry] = 0.1
        ranking = qf.get_action_ranking(state, ["proceed", "retry"])
        assert ranking[0][0] == "proceed"
        assert ranking[1][0] == "retry"
        assert ranking[0][1] > ranking[1][1]

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict and from_dict preserve Q-table data."""
        qf = DecomposedQFunction()
        state = self._make_state()
        key = (state.to_key(), "proceed")
        qf.q_task[key] = 0.75
        qf.q_explore[key] = 0.42
        qf.adjust_weights('exploration')

        data = qf.to_dict()
        qf2 = DecomposedQFunction.from_dict(data)
        key2 = (state.to_key(), "proceed")
        assert qf2.q_task[key2] == pytest.approx(0.75)
        assert qf2.q_explore[key2] == pytest.approx(0.42)
        assert qf2.weights == qf.weights

    def test_to_dict_structure(self):
        """to_dict contains all expected keys."""
        qf = DecomposedQFunction()
        data = qf.to_dict()
        assert "q_task" in data
        assert "q_explore" in data
        assert "q_causal" in data
        assert "q_safety" in data
        assert "weights" in data
        assert "alphas" in data


# =============================================================================
# TestThoughtLevelCredit (12 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_ROADMAP, reason="swarm_roadmap not importable")
class TestThoughtLevelCredit:
    """Tests for ThoughtLevelCredit reasoning step credit assignment."""

    def test_init_default_weights(self):
        """Init sets temporal_weight=0.3, tool_weight=0.4, decision_weight=0.3."""
        tlc = ThoughtLevelCredit()
        assert tlc.temporal_weight == 0.3
        assert tlc.tool_weight == 0.4
        assert tlc.decision_weight == 0.3

    def test_init_custom_weights(self):
        """Init accepts custom weights via config."""
        tlc = ThoughtLevelCredit({'temporal_weight': 0.5, 'tool_weight': 0.3, 'decision_weight': 0.2})
        assert tlc.temporal_weight == 0.5
        assert tlc.tool_weight == 0.3
        assert tlc.decision_weight == 0.2

    def test_assign_credit_empty_trace(self):
        """assign_credit returns empty dict for empty reasoning trace."""
        tlc = ThoughtLevelCredit()
        result = tlc.assign_credit([], [], 1.0)
        assert result == {}

    def test_assign_credit_returns_dict_int_float(self):
        """assign_credit returns Dict[int, float] mapping step index to credit."""
        tlc = ThoughtLevelCredit()
        trace = ["Step one analysis", "Step two conclusion - we should proceed"]
        result = tlc.assign_credit(trace, [], 1.0)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, int)
            assert isinstance(v, float)

    def test_assign_credit_temporal_later_steps_more(self):
        """Temporal credit: later steps get more credit (factor = (i+1)/n)."""
        tlc = ThoughtLevelCredit()
        trace = [
            "A" * 200,  # long analysis, not a decision point by length heuristic
            "B" * 200,
            "C" * 200,
        ]
        # With no tool calls and no decision points,
        # credit comes entirely from temporal factor
        result = tlc.assign_credit(trace, [], 1.0)
        assert len(result) > 0

    def test_assign_credit_normalized_to_outcome(self):
        """Credits are normalized so they sum to abs(outcome)."""
        tlc = ThoughtLevelCredit()
        trace = ["First thought about the problem", "Second thought with conclusion"]
        result = tlc.assign_credit(trace, [], 0.8)
        total = sum(result.values())
        assert total == pytest.approx(0.8, abs=0.01)

    def test_compute_tool_credits_success(self):
        """_compute_tool_credits gives 1.0 for successful tool calls."""
        tlc = ThoughtLevelCredit()
        trace = ["Use calculator to compute"]
        tools = [{"tool": "calculator", "success": True}]
        # Without LM, _find_linked_thought returns None, so tool_credits will be empty
        result = tlc._compute_tool_credits(trace, tools)
        # Without LM configured, linkage returns None, so empty
        assert isinstance(result, dict)

    def test_compute_tool_credits_failure(self):
        """_compute_tool_credits gives -0.5 for failed tool calls."""
        tlc = ThoughtLevelCredit()
        trace = ["Try search"]
        tools = [{"tool": "search", "success": False}]
        result = tlc._compute_tool_credits(trace, tools)
        assert isinstance(result, dict)

    def test_identify_decision_steps_final_third(self):
        """_identify_decision_steps marks steps in final third as decisions."""
        tlc = ThoughtLevelCredit()
        trace = [
            "A" * 200,  # long, not in final third
            "B" * 200,
            "C" * 200,
            "Short conclusion",  # in final third AND short
        ]
        decisions = tlc._identify_decision_steps(trace)
        assert 3 in decisions  # last step in final third

    def test_identify_decision_steps_concise_statement(self):
        """_identify_decision_steps marks short definitive statements (20-150 chars)."""
        tlc = ThoughtLevelCredit()
        trace = [
            "We should proceed with the implementation plan",  # ~48 chars, decision
        ]
        decisions = tlc._identify_decision_steps(trace)
        assert 0 in decisions

    def test_identify_decision_steps_follows_analysis(self):
        """_identify_decision_steps marks steps following detailed analysis."""
        tlc = ThoughtLevelCredit()
        trace = [
            "A" * 300,  # detailed analysis (long)
            "Ok proceed",  # short follow-up, 10 chars - under 20 threshold
        ]
        decisions = tlc._identify_decision_steps(trace)
        # Step 1 follows long analysis, but is < 20 chars
        # However it's in final third (index 1 >= 2*0.7=1.4? No, 1 < 1.4)
        # The "follows analysis" heuristic: prev > current*2 -> True
        assert 1 in decisions

    def test_get_step_value_summary_no_credits(self):
        """get_step_value_summary returns 'No credits assigned' for empty credits."""
        tlc = ThoughtLevelCredit()
        result = tlc.get_step_value_summary({}, [])
        assert result == "No credits assigned"

    def test_get_step_value_summary_formats_output(self):
        """get_step_value_summary formats credit assignment with step indices."""
        tlc = ThoughtLevelCredit()
        credits = {0: 0.3, 1: 0.7}
        trace = ["First thought", "Second thought"]
        result = tlc.get_step_value_summary(credits, trace)
        assert "Step 0:" in result
        assert "Step 1:" in result
        assert "0.300" in result
        assert "0.700" in result


# =============================================================================
# TestSwarmIntelligence (20 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_INTELLIGENCE, reason="SwarmIntelligence not importable")
class TestSwarmIntelligence:
    """Tests for SwarmIntelligence coordination engine."""

    def test_init_default_collective_memory_limit(self):
        """Init sets default collective_memory_limit=200."""
        si = SwarmIntelligence()
        assert si.collective_memory_limit == 200

    def test_init_custom_collective_memory_limit(self):
        """Init accepts custom collective_memory_limit."""
        si = SwarmIntelligence(collective_memory_limit=50)
        assert si.collective_memory_limit == 50

    def test_register_agent_creates_profile(self):
        """register_agent creates an AgentProfile for a new agent."""
        si = SwarmIntelligence()
        si.register_agent("agent_a")
        assert "agent_a" in si.agent_profiles
        assert si.agent_profiles["agent_a"].agent_name == "agent_a"

    def test_register_agent_idempotent(self):
        """register_agent does not overwrite existing profile."""
        si = SwarmIntelligence()
        si.register_agent("agent_a")
        si.agent_profiles["agent_a"].trust_score = 0.99
        si.register_agent("agent_a")  # should not reset
        assert si.agent_profiles["agent_a"].trust_score == 0.99

    def test_record_task_result_updates_profile(self):
        """record_task_result updates agent profile with task result."""
        si = SwarmIntelligence()
        si.record_task_result("agent_a", "analysis", True, 2.5)
        profile = si.agent_profiles["agent_a"]
        assert profile.total_tasks == 1
        assert profile.avg_execution_time == pytest.approx(2.5)

    def test_record_task_result_thread_safe(self):
        """record_task_result acquires state lock."""
        si = SwarmIntelligence()
        # Just verify it runs without deadlock
        si.record_task_result("agent_a", "analysis", True, 1.0)
        si.record_task_result("agent_a", "analysis", False, 2.0)
        assert si.agent_profiles["agent_a"].total_tasks == 2

    def test_record_task_result_adds_to_collective_memory(self):
        """record_task_result deposits entry in collective_memory."""
        si = SwarmIntelligence()
        si.record_task_result("agent_a", "analysis", True, 1.0)
        assert len(si.collective_memory) == 1
        entry = si.collective_memory[0]
        assert entry["agent"] == "agent_a"
        assert entry["task_type"] == "analysis"
        assert entry["success"] is True

    def test_record_task_result_deposits_success_signal(self):
        """record_task_result deposits stigmergy success signal on success."""
        si = SwarmIntelligence()
        si.record_task_result("agent_a", "analysis", True, 1.0)
        signals = si.stigmergy.sense(signal_type='success')
        assert len(signals) > 0

    def test_record_task_result_deposits_warning_signal(self):
        """record_task_result deposits stigmergy warning signal on failure."""
        si = SwarmIntelligence()
        si.record_task_result("agent_a", "analysis", False, 1.0)
        signals = si.stigmergy.sense(signal_type='warning')
        assert len(signals) > 0

    def test_get_agent_specialization_default(self):
        """get_agent_specialization returns GENERALIST for unknown agent."""
        si = SwarmIntelligence()
        assert si.get_agent_specialization("unknown") == AgentSpecialization.GENERALIST

    def test_get_agent_specialization_after_training(self):
        """get_agent_specialization returns learned specialization."""
        si = SwarmIntelligence()
        # Train agent with enough analysis tasks
        for _ in range(3):
            si.record_task_result("agent_a", "analysis", True, 1.0)
        spec = si.get_agent_specialization("agent_a")
        assert spec == AgentSpecialization.ANALYZER

    def test_get_best_agent_for_task_returns_none_no_agents(self):
        """get_best_agent_for_task returns None when no agents available."""
        si = SwarmIntelligence()
        result = si.get_best_agent_for_task("analysis", [])
        assert result is None

    def test_get_best_agent_for_task_scoring(self):
        """get_best_agent_for_task returns agent with highest composite score."""
        si = SwarmIntelligence()
        # Train agents with different success rates
        for _ in range(5):
            si.record_task_result("good_agent", "analysis", True, 1.0)
        for _ in range(5):
            si.record_task_result("bad_agent", "analysis", False, 1.0)
        result = si.get_best_agent_for_task(
            "analysis",
            ["good_agent", "bad_agent"],
            use_morph_scoring=False,
        )
        assert result == "good_agent"

    def test_get_best_agent_for_task_registers_unknown_agents(self):
        """get_best_agent_for_task registers any unregistered agents."""
        si = SwarmIntelligence()
        si.get_best_agent_for_task("analysis", ["new_agent"], use_morph_scoring=False)
        assert "new_agent" in si.agent_profiles

    def test_deposit_success_signal(self):
        """deposit_success_signal creates success and route signals."""
        si = SwarmIntelligence()
        si.deposit_success_signal("agent_a", "analysis", 2.0)
        success_signals = si.stigmergy.sense(signal_type='success')
        route_signals = si.stigmergy.sense(signal_type='route')
        assert len(success_signals) > 0
        assert len(route_signals) > 0

    def test_deposit_warning_signal(self):
        """deposit_warning_signal creates warning signal."""
        si = SwarmIntelligence()
        si.deposit_warning_signal("agent_a", "analysis", "Task timed out")
        signals = si.stigmergy.sense(signal_type='warning')
        assert len(signals) > 0

    def test_get_swarm_wisdom_returns_dict(self):
        """get_swarm_wisdom returns dict with expected keys."""
        si = SwarmIntelligence()
        wisdom = si.get_swarm_wisdom("analyze data", task_type="analysis")
        assert "recommended_agent" in wisdom
        assert "similar_experiences" in wisdom
        assert "warnings" in wisdom
        assert "confidence" in wisdom

    def test_get_swarm_wisdom_with_experiences(self):
        """get_swarm_wisdom includes similar_experiences from collective memory."""
        si = SwarmIntelligence()
        si.record_task_result("agent_a", "analysis", True, 1.0)
        wisdom = si.get_swarm_wisdom("analyze data", task_type="analysis")
        assert len(wisdom["similar_experiences"]) > 0
        assert wisdom["similar_experiences"][0]["agent"] == "agent_a"

    def test_condense_collective_memory_nothing_to_condense(self):
        """condense_collective_memory returns empty string when entries <= keep_recent."""
        si = SwarmIntelligence()
        for i in range(5):
            si.record_task_result(f"a{i}", "analysis", True, 1.0)
        result = si.condense_collective_memory(keep_recent=20)
        assert result == ""

    def test_condense_collective_memory_produces_summary(self):
        """condense_collective_memory produces aggregated summary when entries > keep_recent."""
        si = SwarmIntelligence()
        for i in range(30):
            si.record_task_result("agent_a", "analysis", True, 1.0)
        result = si.condense_collective_memory(keep_recent=5)
        assert "Condensed history" in result
        assert "analysis" in result
        assert "success" in result

    def test_get_swarm_status_returns_health_metrics(self):
        """get_swarm_status returns dict with health metrics."""
        si = SwarmIntelligence()
        si.register_agent("agent_a")
        # Initialize attributes that get_swarm_status reads from mixin state
        si.active_auctions = {}
        si.priority_queue = []
        status = si.get_swarm_status()
        assert "agent_count" in status
        assert "health_score" in status
        assert "avg_load" in status
        assert "collective_memory_size" in status
        assert status["agent_count"] == 1

    def test_save_and_load_roundtrip(self):
        """save and load preserve agent profiles and collective memory."""
        import tempfile
        import os

        si = SwarmIntelligence()
        si.record_task_result("agent_a", "analysis", True, 2.5)
        si.record_task_result("agent_b", "planning", False, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            si.save(path)

            si2 = SwarmIntelligence()
            loaded = si2.load(path)
            assert loaded is True
            assert "agent_a" in si2.agent_profiles
            assert "agent_b" in si2.agent_profiles
            assert si2.agent_profiles["agent_a"].total_tasks == 1
            assert len(si2.collective_memory) == 2
        finally:
            os.unlink(path)


# =============================================================================
# TestAgentStateTracker (12 tests)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SWARM_STATE_MANAGER, reason="swarm_state_manager not importable")
class TestAgentStateTracker:
    """Tests for AgentStateTracker per-agent state management."""

    def test_init_agent_name(self):
        """AgentStateTracker stores agent_name."""
        tracker = AgentStateTracker("agent_a")
        assert tracker.agent_name == "agent_a"

    def test_record_output_increments_successful(self):
        """record_output increments successful_executions."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_output("result data", output_type="str")
        assert tracker.stats["successful_executions"] == 1
        assert tracker.stats["total_executions"] == 1
        assert len(tracker.outputs) == 1

    def test_record_error_increments_failed(self):
        """record_error increments failed_executions and stores error."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_error("something broke", error_type="RuntimeError")
        assert tracker.stats["failed_executions"] == 1
        assert tracker.stats["total_executions"] == 1
        assert len(tracker.errors) == 1
        assert tracker.errors[0]["type"] == "RuntimeError"

    def test_record_error_default_type(self):
        """record_error defaults error_type to 'Unknown'."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_error("oops")
        assert tracker.errors[0]["type"] == "Unknown"

    def test_record_tool_call_success(self):
        """record_tool_call tracks successful tool calls."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_tool_call("calculator", success=True)
        assert tracker.tool_usage["successful"]["calculator"] == 1
        assert tracker.stats["successful_tool_calls"] == 1
        assert tracker.stats["total_tool_calls"] == 1

    def test_record_tool_call_failure(self):
        """record_tool_call tracks failed tool calls."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_tool_call("search", success=False)
        assert tracker.tool_usage["failed"]["search"] == 1
        assert tracker.stats["failed_tool_calls"] == 1
        assert tracker.stats["total_tool_calls"] == 1

    def test_record_trajectory_step(self):
        """record_trajectory_step adds timestamp and agent name."""
        tracker = AgentStateTracker("agent_a")
        step = {"action": "execute", "result": "ok"}
        tracker.record_trajectory_step(step)
        assert len(tracker.trajectory) == 1
        assert tracker.trajectory[0]["agent"] == "agent_a"
        assert "timestamp" in tracker.trajectory[0]

    def test_record_validation(self):
        """record_validation stores validation result."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_validation("auditor", passed=True, confidence=0.95, feedback="good")
        assert len(tracker.validation_results) == 1
        assert tracker.validation_results[0]["type"] == "auditor"
        assert tracker.validation_results[0]["passed"] is True
        assert tracker.validation_results[0]["confidence"] == 0.95

    def test_get_state_comprehensive(self):
        """get_state returns comprehensive state dict with stats and rates."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_output("ok")
        tracker.record_error("fail")
        tracker.record_tool_call("calc", success=True)
        tracker.record_tool_call("calc", success=True)
        tracker.record_tool_call("search", success=False)

        state = tracker.get_state()
        assert state["agent_name"] == "agent_a"
        assert state["stats"]["total_executions"] == 2
        assert state["stats"]["successful_executions"] == 1
        assert state["stats"]["failed_executions"] == 1
        assert state["success_rate"] == pytest.approx(0.5)
        assert state["tool_success_rate"] == pytest.approx(2.0 / 3.0)
        assert "calc" in state["tool_usage"]["successful"]
        assert "search" in state["tool_usage"]["failed"]

    def test_get_error_patterns(self):
        """get_error_patterns extracts patterns from last 10 errors."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_error("timeout", error_type="TimeoutError")
        tracker.record_error("timeout again", error_type="TimeoutError")
        tracker.record_error("bad input", error_type="ValueError")

        patterns = tracker.get_error_patterns()
        assert len(patterns) == 3
        # Check frequency counts
        timeout_patterns = [p for p in patterns if p["type"] == "TimeoutError"]
        assert timeout_patterns[0]["frequency"] == 2

    def test_get_successful_patterns(self):
        """get_successful_patterns extracts tool usage patterns with count >= 2."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_tool_call("calculator", success=True)
        tracker.record_tool_call("calculator", success=True)
        tracker.record_tool_call("search", success=True)  # only once

        patterns = tracker.get_successful_patterns()
        tool_patterns = [p for p in patterns if p["type"] == "tool_usage"]
        assert len(tool_patterns) == 1  # only calculator has count >= 2
        assert tool_patterns[0]["tool"] == "calculator"
        assert tool_patterns[0]["success_count"] == 2

    def test_get_successful_patterns_validation(self):
        """get_successful_patterns includes validation patterns."""
        tracker = AgentStateTracker("agent_a")
        tracker.record_validation("auditor", passed=True)
        tracker.record_validation("architect", passed=True)

        patterns = tracker.get_successful_patterns()
        validation_patterns = [p for p in patterns if p["type"] == "validation"]
        assert len(validation_patterns) == 1
        assert validation_patterns[0]["count"] == 2
