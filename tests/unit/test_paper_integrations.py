"""
Tests for Research Paper Integrations
======================================

Covers all 6 paper integrations:
- Phase 0: GroupedValueBaseline tier/agent tracking
- #1: SkillsBench skill cap
- #2: MAPLE async learning (hot/cold split)
- #3: Delegation overhead floor
- #4: CogRouter learned tier routing
- #5: Team of Thoughts agent proficiency filtering
- #6: LCM hierarchical compression
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Phase 0: GroupedValueBaseline tier/agent tracking
# =============================================================================


class TestGroupedValueBaselineTierAgent:
    """Phase 0: tier_baselines, tier_counts, agent_baselines, agent_counts."""

    @pytest.fixture
    def baseline(self):
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        return GroupedValueBaseline()

    def test_update_tier_and_get_success(self, baseline):
        """update_tier records data, get_tier_success retrieves it."""
        assert baseline.get_tier_success("DIRECT", "research") is None

        baseline.update_tier("DIRECT", "research", 1.0)
        result = baseline.get_tier_success("DIRECT", "research")
        assert result is not None
        success_rate, count = result
        assert count == 1
        assert success_rate > 0.0

    def test_update_agent_and_get_proficiency(self, baseline):
        """update_agent records data, get_agent_proficiency retrieves it."""
        assert baseline.get_agent_proficiency("coder", "coding") is None

        baseline.update_agent("coder", "coding", 0.8)
        result = baseline.get_agent_proficiency("coder", "coding")
        assert result is not None
        proficiency, count = result
        assert count == 1
        assert proficiency > 0.0

    def test_tier_ema_updates(self, baseline):
        """Multiple updates converge via EMA."""
        for _ in range(10):
            baseline.update_tier("AGENTIC", "analysis", 1.0)
        result = baseline.get_tier_success("AGENTIC", "analysis")
        assert result is not None
        success_rate, count = result
        assert count == 10
        assert success_rate > 0.8  # EMA converges toward 1.0

    def test_agent_ema_updates(self, baseline):
        """Multiple agent updates converge via EMA."""
        for _ in range(10):
            baseline.update_agent("researcher", "research", 0.0)
        result = baseline.get_agent_proficiency("researcher", "research")
        assert result is not None
        proficiency, count = result
        assert count == 10
        assert proficiency < 0.2  # EMA converges toward 0.0

    def test_serialization_roundtrip(self, baseline):
        """to_dict/from_dict preserves tier and agent data."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        baseline.update_tier("DIRECT", "research", 0.9)
        baseline.update_tier("AGENTIC", "coding", 0.7)
        baseline.update_agent("coder", "coding", 0.85)
        baseline.update_agent("researcher", "research", 0.6)

        data = baseline.to_dict()
        restored = GroupedValueBaseline.from_dict(data)

        # Tier data preserved
        assert restored.get_tier_success("DIRECT", "research") is not None
        assert restored.get_tier_success("AGENTIC", "coding") is not None

        # Agent data preserved
        assert restored.get_agent_proficiency("coder", "coding") is not None
        assert restored.get_agent_proficiency("researcher", "research") is not None

        # Values match
        orig_tier = baseline.get_tier_success("DIRECT", "research")
        rest_tier = restored.get_tier_success("DIRECT", "research")
        assert orig_tier[0] == pytest.approx(rest_tier[0])

    def test_serialization_backward_compat(self, baseline):
        """from_dict handles missing tier/agent fields gracefully."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        old_data = {
            "group_baselines": {"test": 0.5},
            "group_counts": {"test": 3},
            "domain_baselines": {},
            "ema_alpha": 0.1,
            # No tier_baselines, tier_counts, agent_baselines, agent_counts
        }
        restored = GroupedValueBaseline.from_dict(old_data)
        assert restored.tier_baselines == {}
        assert restored.tier_counts == {}
        assert restored.agent_baselines == {}
        assert restored.agent_counts == {}


# =============================================================================
# Integration 1: SkillsBench Skill Cap
# =============================================================================


class TestSkillsBenchSkillCap:
    """#1: skill cap enforced after injection, essential skills survive cut."""

    def test_enforce_skill_cap_trims_excess(self):
        """Skills exceeding cap are trimmed, keeping highest-ranked."""
        from Jotty.core.intelligence.reasoning.executors.skill_plan_executor import (
            SkillPlanExecutor,
        )

        executor = SkillPlanExecutor.__new__(SkillPlanExecutor)

        skills = [
            {"name": "skill_a", "rank": 1},
            {"name": "skill_b", "rank": 2},
            {"name": "skill_c", "rank": 3},
            {"name": "skill_d", "rank": 4},
            {"name": "skill_e", "rank": 5},
        ]
        essential_names = set()

        result = executor._enforce_skill_cap(skills, essential_names, max_skills=3)
        assert len(result) == 3

    def test_enforce_skill_cap_preserves_essential(self):
        """Essential skills survive the cap even if low-ranked."""
        from Jotty.core.intelligence.reasoning.executors.skill_plan_executor import (
            SkillPlanExecutor,
        )

        executor = SkillPlanExecutor.__new__(SkillPlanExecutor)

        skills = [
            {"name": "skill_a", "rank": 1},
            {"name": "skill_b", "rank": 2},
            {"name": "essential_1", "rank": 5},
            {"name": "skill_d", "rank": 3},
            {"name": "essential_2", "rank": 6},
        ]
        essential_names = {"essential_1", "essential_2"}

        result = executor._enforce_skill_cap(skills, essential_names, max_skills=3)
        result_names = {s["name"] for s in result}

        # Both essential skills must survive
        assert "essential_1" in result_names
        assert "essential_2" in result_names

    def test_enforce_skill_cap_noop_under_limit(self):
        """No change when already under cap."""
        from Jotty.core.intelligence.reasoning.executors.skill_plan_executor import (
            SkillPlanExecutor,
        )

        executor = SkillPlanExecutor.__new__(SkillPlanExecutor)

        skills = [{"name": "a"}, {"name": "b"}]
        result = executor._enforce_skill_cap(skills, set(), max_skills=3)
        assert len(result) == 2

    def test_select_skills_default_max_is_3(self):
        """select_skills signature defaults to max_skills=3."""
        import inspect

        from Jotty.core.intelligence.reasoning.executors.skill_plan_executor import (
            SkillPlanExecutor,
        )

        sig = inspect.signature(SkillPlanExecutor.select_skills)
        assert sig.parameters["max_skills"].default == 3


# =============================================================================
# Integration 2: MAPLE Async Learning
# =============================================================================


class TestMAPLEAsyncLearning:
    """#2: hot-path completes before cold-path, errors don't propagate, sync fallback."""

    def test_bg_learning_tasks_exists(self):
        """_bg_learning_tasks class attribute exists on SwarmLearningMixin."""
        from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
            SwarmLearningMixin,
        )

        assert hasattr(SwarmLearningMixin, "_bg_learning_tasks")

    def test_schedule_deferred_method_exists(self):
        """_schedule_deferred_learning method exists."""
        from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
            SwarmLearningMixin,
        )

        assert hasattr(SwarmLearningMixin, "_schedule_deferred_learning")

    def test_deferred_post_learning_method_exists(self):
        """_deferred_post_learning async method exists."""
        from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
            SwarmLearningMixin,
        )

        assert hasattr(SwarmLearningMixin, "_deferred_post_learning")
        assert asyncio.iscoroutinefunction(SwarmLearningMixin._deferred_post_learning)


# =============================================================================
# Integration 3: Delegation Overhead Floor
# =============================================================================


class TestDelegationOverheadFloor:
    """#3: short tasks -> DIRECT, complex tasks -> not forced."""

    @pytest.fixture
    def detector(self):
        from Jotty.core.intelligence.orchestration.execution.tier_detector import (
            TierDetector,
        )

        return TierDetector()

    def test_short_trivial_task_is_direct(self, detector):
        """Short trivial tasks below delegation floor -> DIRECT."""
        from Jotty.core.intelligence.orchestration.execution.types import ExecutionTier

        tier = detector.detect("What is GDP?")
        assert tier == ExecutionTier.DIRECT

    def test_complex_keyword_not_forced_direct(self, detector):
        """Tasks with complex keywords bypass the delegation floor."""
        # "analyze" is in _COMPLEX_KEYWORDS, so _below_delegation_floor returns False.
        # However, if the task is also short, _is_simple_query may trigger first.
        # Use a longer phrase without direct indicators to test the floor path.
        assert detector._below_delegation_floor("analyze the performance trends carefully") is False
        assert detector._below_delegation_floor("please research the topic") is False

    def test_below_delegation_floor_method(self, detector):
        """_below_delegation_floor returns True for trivial tasks."""
        assert detector._below_delegation_floor("hello world") is True
        assert detector._below_delegation_floor("analyze complex system") is False

    def test_multi_step_not_below_floor(self, detector):
        """Multi-step indicators prevent below-floor classification."""
        assert detector._below_delegation_floor("first do x and then do y") is False

    def test_long_task_not_below_floor(self, detector):
        """Tasks longer than 15 words are not below the delegation floor."""
        long_task = " ".join(["word"] * 20)
        assert detector._below_delegation_floor(long_task) is False


# =============================================================================
# Integration 4: CogRouter Learned Tier Routing
# =============================================================================


class TestCogRouterLearnedTierRouting:
    """#4: learned tier returned when data sufficient, None when insufficient."""

    @pytest.fixture
    def baseline(self):
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

        return GroupedValueBaseline()

    def test_consult_tier_history_returns_none_without_baseline(self):
        """No baseline -> None."""
        from Jotty.core.intelligence.orchestration.execution.tier_detector import (
            TierDetector,
        )

        td = TierDetector()
        result = td._consult_tier_history("research something")
        assert result is None

    def test_consult_tier_history_returns_none_insufficient_data(self, baseline):
        """Insufficient data (count < 5) -> None."""
        from Jotty.core.intelligence.orchestration.execution.tier_detector import (
            TierDetector,
        )

        # Only 4 samples, need 5 (CogRouter is primary, higher threshold)
        for _ in range(4):
            baseline.update_tier("RESEARCH", "research", 1.0)

        td = TierDetector(grouped_baseline=baseline)
        result = td._consult_tier_history("research something")
        assert result is None

    def test_consult_tier_history_returns_best_tier(self, baseline):
        """With sufficient data, returns tier with highest success."""
        from Jotty.core.intelligence.orchestration.execution.tier_detector import (
            TierDetector,
        )
        from Jotty.core.intelligence.orchestration.execution.types import ExecutionTier

        # RESEARCH tier has 5 samples, high success
        for _ in range(5):
            baseline.update_tier("RESEARCH", "research", 0.9)

        # DIRECT tier has 5 samples, low success
        for _ in range(5):
            baseline.update_tier("DIRECT", "research", 0.3)

        td = TierDetector(grouped_baseline=baseline)
        result = td._consult_tier_history("research something")
        assert result is not None
        tier, confidence = result
        assert tier == ExecutionTier.RESEARCH
        assert confidence == 0.85

    def test_consult_tier_history_low_success_ignored(self, baseline):
        """Tiers with success <= 0.6 are ignored."""
        from Jotty.core.intelligence.orchestration.execution.tier_detector import (
            TierDetector,
        )

        for _ in range(5):
            baseline.update_tier("AGENTIC", "general", 0.4)

        td = TierDetector(grouped_baseline=baseline)
        result = td._consult_tier_history("do something")
        assert result is None

    def test_task_type_inference(self):
        """_infer_task_type_simple categorizes by keywords."""
        from Jotty.core.intelligence.orchestration.execution.tier_detector import (
            TierDetector,
        )

        td = TierDetector()
        assert td._infer_task_type_simple("research ai trends") == "research"
        assert td._infer_task_type_simple("create a website") == "creation"
        assert td._infer_task_type_simple("analyze the data") == "analysis"
        assert td._infer_task_type_simple("code a function") == "coding"
        assert td._infer_task_type_simple("hello world") == "general"


# =============================================================================
# Integration 5: Team of Thoughts Agent Proficiency
# =============================================================================


class TestTeamOfThoughtsAgentProficiency:
    """#5: low-proficiency agents filtered, minimum 2 preserved, unknown agents kept."""

    @pytest.fixture
    def executor(self):
        from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
            ParadigmExecutor,
        )

        # Create with a mock manager
        manager = MagicMock()
        manager.learning = None  # No learning -> no baseline
        return ParadigmExecutor(manager)

    def test_no_baseline_returns_all_agents(self, executor):
        """Without baseline, all agents are returned."""
        agents = [
            SimpleNamespace(name="a"),
            SimpleNamespace(name="b"),
            SimpleNamespace(name="c"),
        ]
        result = executor._filter_by_proficiency(agents, "test goal")
        assert len(result) == 3

    def test_low_proficiency_agents_filtered(self):
        """Agents below 0.3 proficiency are dropped."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline
        from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
            ParadigmExecutor,
        )

        baseline = GroupedValueBaseline()
        # Agent "bad" has very low proficiency
        for _ in range(5):
            baseline.update_agent("bad", "general", 0.0)
        # Agents "good1" and "good2" have high proficiency
        for _ in range(5):
            baseline.update_agent("good1", "general", 0.9)
            baseline.update_agent("good2", "general", 0.8)

        # Mock manager with learning -> td_learner -> grouped_baseline
        td_learner = SimpleNamespace(grouped_baseline=baseline)
        learning = SimpleNamespace(td_learner=td_learner)
        manager = MagicMock()
        manager.learning = learning

        executor = ParadigmExecutor(manager)
        agents = [
            SimpleNamespace(name="good1"),
            SimpleNamespace(name="good2"),
            SimpleNamespace(name="bad"),
        ]
        result = executor._filter_by_proficiency(agents, "do something")
        result_names = [a.name for a in result]

        assert "good1" in result_names
        assert "good2" in result_names
        # "bad" should be filtered out
        assert "bad" not in result_names

    def test_minimum_agents_preserved(self):
        """At least min_agents (2) are always kept, even if all are low."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline
        from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
            ParadigmExecutor,
        )

        baseline = GroupedValueBaseline()
        # All agents low proficiency
        for _ in range(5):
            baseline.update_agent("low1", "general", 0.1)
            baseline.update_agent("low2", "general", 0.1)
            baseline.update_agent("low3", "general", 0.1)

        td_learner = SimpleNamespace(grouped_baseline=baseline)
        learning = SimpleNamespace(td_learner=td_learner)
        manager = MagicMock()
        manager.learning = learning

        executor = ParadigmExecutor(manager)
        agents = [
            SimpleNamespace(name="low1"),
            SimpleNamespace(name="low2"),
            SimpleNamespace(name="low3"),
        ]
        result = executor._filter_by_proficiency(agents, "do something", min_agents=2)
        assert len(result) >= 2

    def test_unknown_agents_kept(self):
        """Unknown agents get benefit of the doubt (0.5 > 0.3 threshold)."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline
        from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
            ParadigmExecutor,
        )

        baseline = GroupedValueBaseline()
        for _ in range(5):
            baseline.update_agent("known_good", "general", 0.9)

        td_learner = SimpleNamespace(grouped_baseline=baseline)
        learning = SimpleNamespace(td_learner=td_learner)
        manager = MagicMock()
        manager.learning = learning

        executor = ParadigmExecutor(manager)
        agents = [
            SimpleNamespace(name="known_good"),
            SimpleNamespace(name="unknown_agent"),
        ]
        result = executor._filter_by_proficiency(agents, "do something")
        result_names = [a.name for a in result]

        assert "unknown_agent" in result_names

    def test_record_agent_outcomes(self):
        """_record_agent_outcomes writes to baseline."""
        from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline
        from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
            ParadigmExecutor,
        )

        baseline = GroupedValueBaseline()
        td_learner = SimpleNamespace(grouped_baseline=baseline)
        learning = SimpleNamespace(td_learner=td_learner)
        manager = MagicMock()
        manager.learning = learning

        executor = ParadigmExecutor(manager)

        results = {
            "agent_a": SimpleNamespace(success=True),
            "agent_b": SimpleNamespace(success=False),
        }
        executor._record_agent_outcomes(results, "research task")

        # Check that data was recorded
        assert baseline.get_agent_proficiency("agent_a", "research") is not None
        assert baseline.get_agent_proficiency("agent_b", "research") is not None


# =============================================================================
# Integration 6: LCM Hierarchical Compression
# =============================================================================


class TestLCMHierarchicalCompression:
    """#6: hierarchical_compress stores original, returns summary, retrieve works."""

    def test_hierarchical_compress_stores_original(self):
        """Original text is stored in content_store."""
        from Jotty.core.infrastructure.context.utils import hierarchical_compress

        store = {}
        original = "This is the first line.\nSecond line.\nThird line."
        hierarchical_compress(original, "chunk_1", store)

        assert "chunk_1" in store
        assert store["chunk_1"] == original

    def test_hierarchical_compress_returns_summary(self):
        """Summary includes first line, stats, and retrieval pointer."""
        from Jotty.core.infrastructure.context.utils import hierarchical_compress

        store = {}
        text = "Important finding about AI.\nMore details here.\nConclusion."
        summary = hierarchical_compress(text, "c42", store)

        assert "Important finding about AI." in summary
        assert "retrieve:c42" in summary
        assert "3 lines" in summary

    def test_hierarchical_compress_empty_text(self):
        """Empty text returns empty."""
        from Jotty.core.infrastructure.context.utils import hierarchical_compress

        store = {}
        result = hierarchical_compress("", "empty", store)
        assert result == ""

    def test_context_manager_content_store_exists(self):
        """SmartContextManager has _content_store dict."""
        from Jotty.core.infrastructure.context.context_manager import (
            SmartContextManager,
        )

        mgr = SmartContextManager()
        assert hasattr(mgr, "_content_store")
        assert isinstance(mgr._content_store, dict)

    def test_retrieve_full_content(self):
        """retrieve_full_content returns stored original."""
        from Jotty.core.infrastructure.context.context_manager import (
            SmartContextManager,
        )

        mgr = SmartContextManager()
        mgr._content_store["test_id"] = "full original content"
        assert mgr.retrieve_full_content("test_id") == "full original content"
        assert mgr.retrieve_full_content("nonexistent") is None

    def test_clear_content_store(self):
        """clear_content_store empties the store."""
        from Jotty.core.infrastructure.context.context_manager import (
            SmartContextManager,
        )

        mgr = SmartContextManager()
        mgr._content_store["a"] = "text"
        mgr._content_store["b"] = "text"
        mgr.clear_content_store()
        assert len(mgr._content_store) == 0

    def test_compress_parts_uses_hierarchical(self):
        """compress_parts stores originals via hierarchical compression."""
        from Jotty.core.infrastructure.context.context_manager import (
            SmartContextManager,
        )

        mgr = SmartContextManager()

        # Create parts that exceed the budget to trigger compression
        part_a = "A" * 5000
        part_b = "B" * 5000
        result = mgr.compress_parts([part_a, part_b], max_total_chars=200)

        # Should have compressed
        assert len(result) == 2
        assert sum(len(p) for p in result) < 10000

        # Originals should be in the store
        assert len(mgr._content_store) == 2

        # Each stored value is an original part
        stored_values = list(mgr._content_store.values())
        assert part_a in stored_values
        assert part_b in stored_values

    def test_compress_parts_noop_under_budget(self):
        """compress_parts returns unchanged when under budget."""
        from Jotty.core.infrastructure.context.context_manager import (
            SmartContextManager,
        )

        mgr = SmartContextManager()
        parts = ["short", "text"]
        result = mgr.compress_parts(parts, max_total_chars=1000)
        assert result == parts
        assert len(mgr._content_store) == 0
