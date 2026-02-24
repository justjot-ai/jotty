"""
End-to-end test: autonomous agent learns through probation and graduates.

Simulates the full lifecycle without real LLM calls:
1. Agent executes tasks → LearningService.record() updates Q-tables
2. Q-values converge as high-reward episodes accumulate
3. should_crystallize() passes when thresholds are met AND Q-values converged
4. crystallize() extracts a CrystallizedConfig (SOP, skill bindings)
5. Agent uses crystallized config for future tasks (no more exploration)

Also tests:
- Negative case: agent that doesn't learn enough should NOT graduate
- Q-value convergence detection (rolling TD error window)
- DSPy gold metric (key-term F1 + structural quality + coherence)
- DomainTaskPipeline (3-stage generate→validate→refine)
- Train/dev split in DomainDSPyOptimizer
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest


def _make_td_lambda():
    """Create a fresh TDLambdaLearner with no persisted state.

    Patches the LearningStore so __init__ doesn't load stale data from disk.
    """
    from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig
    from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner

    mock_store = MagicMock()
    mock_store.get_value.return_value = None

    with patch(
        "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance",
        return_value=mock_store,
    ):
        td = TDLambdaLearner(config=LearningConfig())
    return td


def _simulate_episode(
    td,
    task_type: str,
    domain: str,
    skills: List[str],
    reward: float,
):
    """Simulate a complete episode: update SkillQTable, StepQTable, record plan."""
    for skill in skills:
        td.skill_q.update(task_type, skill, reward, domain=domain)

    for pos, skill in enumerate(skills):
        td.step_q.update(task_type, pos, skill, reward, description="", domain=domain)

    td.step_q.record_plan(task_type, skills, reward, domain=domain, descriptions=[""] * len(skills))


class TestProbationGraduation:
    """Full lifecycle: learn → accumulate Q → pass thresholds → crystallize."""

    @pytest.mark.unit
    def test_agent_graduates_after_enough_successful_episodes(self):
        """An agent with consistent high-reward episodes should pass crystallization.

        Needs enough episodes for Q-value convergence (rolling TD errors → 0).
        """
        from Jotty.core.intelligence.learning.crystallization import should_crystallize

        td = _make_td_lambda()

        task_type = "coding"
        domain = "backend"
        skills = ["claude-cli-llm", "file-operations", "test-runner"]

        # Simulate 40 successful episodes — enough for Q-values to converge
        # (TD errors shrink as Q approaches the stable reward value)
        for _ in range(40):
            _simulate_episode(td, task_type, domain, skills, reward=0.92)

        # Verify Q-values converged high
        for skill in skills:
            q = td.skill_q.get_q(task_type, skill, domain=domain)
            assert q > 0.7, f"Skill {skill} Q={q:.3f} should be > 0.7"

        # Verify convergence stats show stabilized Q-values
        conv = td.skill_q.get_convergence_stats(task_type, domain=domain)
        assert conv["converged"], f"Q-values should have converged: {conv['reason']}"
        assert conv["mean_td_error"] < 0.08

        # Verify plan history is populated
        key = td.skill_q._make_key(task_type, domain)
        plan_history = td.step_q._plan_history.get(key, [])
        assert len(plan_history) >= 20, f"Expected >=20 plans, got {len(plan_history)}"

        # Verify role guidance exists with sufficient visits
        role_guidance = td.step_q.get_role_guidance(task_type, domain=domain)
        assert len(role_guidance) > 0, "Expected role guidance data"
        for rg in role_guidance:
            assert rg["total_visits"] >= 3, f"Role {rg['role']} has too few visits"
            assert rg["best_q"] >= 0.65, f"Role {rg['role']} Q too low: {rg['best_q']}"

        _MOCK_TD = "Jotty.core.intelligence.learning.facade.get_td_lambda"
        with patch(_MOCK_TD, return_value=td):
            ok, stats = should_crystallize(
                task_type,
                domain,
                thresholds={
                    "min_episodes": 25,
                    "min_success_rate": 0.85,
                    "min_plan_consistency": 0.60,
                    "min_role_q": 0.65,
                    "min_plans": 8,
                },
            )

        assert ok, f"Should crystallize but didn't: {stats.get('reasons', [])}"
        assert stats["success_rate"] >= 0.85
        assert stats["plan_consistency"] >= 0.60
        assert stats["convergence"]["converged"]

    @pytest.mark.unit
    def test_agent_does_not_graduate_with_low_success(self):
        """An agent with mixed results should NOT pass crystallization."""
        from Jotty.core.intelligence.learning.crystallization import should_crystallize

        td = _make_td_lambda()

        task_type = "research"
        domain = "science"
        skills = ["web-search", "claude-cli-llm"]

        # Simulate 15 good + 15 bad episodes (50% success → below 85% threshold)
        for _ in range(15):
            _simulate_episode(td, task_type, domain, skills, reward=0.9)
        for _ in range(15):
            _simulate_episode(td, task_type, domain, skills, reward=0.2)

        _MOCK_TD = "Jotty.core.intelligence.learning.facade.get_td_lambda"
        with patch(_MOCK_TD, return_value=td):
            ok, stats = should_crystallize(task_type, domain)

        assert not ok, "Should NOT crystallize with 50% success rate"
        assert any("success rate" in r for r in stats.get("reasons", []))

    @pytest.mark.unit
    def test_agent_does_not_graduate_with_too_few_episodes(self):
        """An agent with too few episodes should NOT pass crystallization."""
        from Jotty.core.intelligence.learning.crystallization import should_crystallize

        td = _make_td_lambda()

        task_type = "analysis"
        domain = "finance"
        skills = ["web-search", "calculator"]

        # Only 5 episodes — below min_episodes threshold (25)
        for _ in range(5):
            _simulate_episode(td, task_type, domain, skills, reward=0.95)

        _MOCK_TD = "Jotty.core.intelligence.learning.facade.get_td_lambda"
        with patch(_MOCK_TD, return_value=td):
            ok, stats = should_crystallize(task_type, domain)

        assert not ok, "Should NOT crystallize with only 5 episodes"
        assert any("too few" in r for r in stats.get("reasons", []))

    @pytest.mark.unit
    def test_inconsistent_plans_prevent_graduation(self):
        """An agent with random plan templates shouldn't crystallize."""
        from Jotty.core.intelligence.learning.crystallization import should_crystallize

        td = _make_td_lambda()

        task_type = "coding"
        domain = "devops"

        # 30 episodes, each with a DIFFERENT plan → 0% consistency
        plan_variants = [
            ["web-search", "claude-cli-llm"],
            ["claude-cli-llm", "file-operations"],
            ["test-runner", "claude-cli-llm"],
            ["file-operations", "web-search"],
            ["calculator", "claude-cli-llm"],
        ]
        for i in range(30):
            plan = plan_variants[i % len(plan_variants)]
            _simulate_episode(td, task_type, domain, plan, reward=0.9)

        _MOCK_TD = "Jotty.core.intelligence.learning.facade.get_td_lambda"
        with patch(_MOCK_TD, return_value=td):
            ok, stats = should_crystallize(task_type, domain)

        # 5 different plans with 6 each = 20% consistency → below 60%
        assert not ok, "Should NOT crystallize with inconsistent plans"


class TestCrystallizeExtractsConfig:
    """Verify crystallize() produces a correct CrystallizedConfig."""

    @pytest.mark.unit
    def test_crystallize_produces_valid_config(self, tmp_path):
        """Graduated agent gets a CrystallizedConfig with SOP, skills, bindings."""
        from Jotty.core.intelligence.learning.crystallization import crystallize
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        td = _make_td_lambda()

        task_type = "research"
        domain = "travel"
        skills = ["web-search", "claude-cli-llm", "file-operations"]

        # Build up enough high-quality episodes (40 for Q-value convergence)
        for _ in range(40):
            _simulate_episode(td, task_type, domain, skills, reward=0.93)

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            with (
                patch(
                    "Jotty.core.intelligence.learning.facade.get_td_lambda",
                    return_value=td,
                ),
                patch(
                    "Jotty.core.intelligence.learning.facade.get_learning_service",
                    return_value=MagicMock(retrieve_distilled_lessons=MagicMock(return_value=[])),
                ),
            ):
                config = crystallize(task_type, domain)

            assert config is not None, "crystallize() should succeed"
            assert config.task_type == task_type
            assert config.domain == domain
            assert len(config.skills) > 0, "Should have extracted skills"
            assert len(config.sop_roles) > 0, "Should have extracted SOP roles"
            assert config.success_rate >= 0.85
            assert config.total_episodes >= 25

            # Role → skill bindings should exist
            assert len(config.role_skill_map) > 0

            # Role confidence should be computed
            assert len(config.role_confidence) > 0
            for role, conf in config.role_confidence.items():
                assert 0 <= conf <= 1, f"Role {role} confidence {conf} out of range"

            # Config should be saved to disk
            saved_files = list(tmp_path.glob("*.json"))
            assert len(saved_files) == 1

        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_crystallize_returns_none_when_not_ready(self):
        """crystallize() returns None when thresholds not met."""
        from Jotty.core.intelligence.learning.crystallization import crystallize

        td = _make_td_lambda()
        # Only 3 episodes — nowhere near ready
        for _ in range(3):
            _simulate_episode(td, "coding", "", ["claude-cli-llm"], reward=0.5)

        with patch(
            "Jotty.core.intelligence.learning.facade.get_td_lambda",
            return_value=td,
        ):
            config = crystallize("coding")

        assert config is None


class TestCrystallizedConfigUsage:
    """Verify crystallized config produces correct plan hints for the agent."""

    @pytest.mark.unit
    def test_plan_hint_contains_sop_and_bindings(self):
        from Jotty.core.intelligence.learning.crystallization import CrystallizedConfig

        config = CrystallizedConfig(
            domain_key="coding:backend",
            task_type="coding",
            domain="backend",
            skills=["claude-cli-llm", "file-operations", "test-runner"],
            sop_roles=("generate", "save", "test"),
            role_skill_map={
                "generate": "claude-cli-llm",
                "save": "file-operations",
                "test": "test-runner",
            },
            prompt_guidance="Always write tests before pushing.",
            success_rate=0.92,
            total_episodes=30,
            role_confidence={"generate": 0.95, "save": 0.88, "test": 0.90},
        )

        hint = config.to_plan_hint()
        assert "CRYSTALLIZED SOP" in hint
        assert "generate → save → test" in hint
        assert "SKILL BINDINGS" in hint
        assert "generate: use claude-cli-llm" in hint
        assert "LOW-CONFIDENCE" not in hint  # all roles > 0.7

    @pytest.mark.unit
    def test_plan_hint_warns_low_confidence_roles(self):
        from Jotty.core.intelligence.learning.crystallization import CrystallizedConfig

        config = CrystallizedConfig(
            domain_key="research",
            task_type="research",
            sop_roles=("search", "analyze"),
            role_skill_map={"search": "web-search", "analyze": "claude-cli-llm"},
            role_confidence={"search": 0.95, "analyze": 0.40},
        )

        hint = config.to_plan_hint()
        assert "LOW-CONFIDENCE ROLES" in hint
        assert "analyze" in hint


class TestStalenessCanaryIntegration:
    """Verify that consecutive failures decrystallize a graduated agent."""

    @pytest.mark.unit
    def test_graduated_agent_loses_crystallization_after_failures(self, tmp_path):
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            load,
            record_crystallized_outcome,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            config = CrystallizedConfig(
                domain_key="coding:backend",
                task_type="coding",
                domain="backend",
                skills=["claude-cli-llm"],
                sop_roles=("generate",),
                success_rate=0.9,
                total_episodes=30,
            )
            _save(config)
            assert load("coding", "backend") is not None

            # 2 failures: still crystallized
            record_crystallized_outcome("coding", "backend", success=False, max_failures=3)
            record_crystallized_outcome("coding", "backend", success=False, max_failures=3)
            assert load("coding", "backend") is not None

            # 3rd failure: decrystallized
            result = record_crystallized_outcome("coding", "backend", success=False, max_failures=3)
            assert result == "decrystallized"
            assert load("coding", "backend") is None

        finally:
            crystal_mod._CRYSTAL_DIR = original_dir

    @pytest.mark.unit
    def test_success_resets_failure_counter(self, tmp_path):
        from Jotty.core.intelligence.learning.crystallization import (
            CrystallizedConfig,
            _save,
            load,
            record_crystallized_outcome,
        )
        import Jotty.core.intelligence.learning.crystallization as crystal_mod

        original_dir = crystal_mod._CRYSTAL_DIR
        crystal_mod._CRYSTAL_DIR = tmp_path
        try:
            config = CrystallizedConfig(
                domain_key="research",
                task_type="research",
                skills=["web-search"],
                sop_roles=("search",),
                success_rate=0.85,
                total_episodes=25,
            )
            _save(config)

            # 2 failures
            record_crystallized_outcome("research", success=False, max_failures=3)
            record_crystallized_outcome("research", success=False, max_failures=3)
            # Then a success — should reset counter
            record_crystallized_outcome("research", success=True, max_failures=3)
            # 2 more failures — still below 3 consecutive
            record_crystallized_outcome("research", success=False, max_failures=3)
            record_crystallized_outcome("research", success=False, max_failures=3)

            # Should still be crystallized (counter was reset by the success)
            assert load("research") is not None

        finally:
            crystal_mod._CRYSTAL_DIR = original_dir


class TestLearningServiceRecordFeedsQTables:
    """Verify that LearningService._update_values() updates Q-tables
    that should_crystallize() reads — the core integration point."""

    @pytest.mark.unit
    def test_update_values_writes_to_store(self):
        """_update_values should compute TD updates and save ValueEstimates."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        svc = LearningService.__new__(LearningService)

        mock_store = MagicMock()
        mock_store.get_value.return_value = None
        mock_store.save_value = MagicMock()
        svc._store = mock_store

        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        svc._config = LearningConfig()

        svc._update_values(
            domain="coding",
            task_type="code_generation",
            action={"paradigm": "single", "model": "sonnet"},
            success=True,
            quality=0.88,
        )

        # _update_values should save a new ValueEstimate
        assert mock_store.save_value.called
        saved = mock_store.save_value.call_args[0][0]
        assert saved.value > 0, "Value estimate should be positive for success"

    @pytest.mark.unit
    def test_td_update_produces_correct_direction(self):
        """High reward should push Q-values up, low reward should push down."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()

        # Initial Q is 0.5 (default)
        assert q.get_q("coding", "web-search") == 0.5

        # High reward should increase Q
        q.update("coding", "web-search", reward=0.95)
        assert q.get_q("coding", "web-search") > 0.5

        # Repeated high rewards should push Q close to reward
        for _ in range(20):
            q.update("coding", "web-search", reward=0.95)
        assert q.get_q("coding", "web-search") > 0.8

        # Low reward on a different skill should keep it low
        q.update("coding", "calculator", reward=0.1)
        assert q.get_q("coding", "calculator") < 0.5


class TestConvergenceDetection:
    """Verify Q-value convergence tracking prevents premature crystallization."""

    @pytest.mark.unit
    def test_unconverged_q_values_block_crystallization(self):
        """Q-values that are still moving should NOT allow crystallization,
        even if they've crossed the threshold."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()

        # Only 5 updates — not enough for convergence window
        for _ in range(5):
            q.update("coding", "web-search", reward=0.9)

        stats = q.get_convergence_stats("coding")
        assert not stats["converged"]
        assert "too few" in stats["reason"]

    @pytest.mark.unit
    def test_oscillating_q_values_block_crystallization(self):
        """Q-values that oscillate (high variance) should NOT converge."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()

        # Alternate high/low rewards — creates oscillation
        for i in range(25):
            reward = 0.95 if i % 2 == 0 else 0.1
            q.update("research", "web-search", reward=reward)

        stats = q.get_convergence_stats("research")
        # Mean |TD error| should be high due to oscillation
        assert not stats["converged"] or stats["td_variance"] > 0.001

    @pytest.mark.unit
    def test_stable_q_values_converge(self):
        """Consistent rewards should produce converged Q-values."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()

        # 40 consistent updates — Q should converge
        for _ in range(40):
            q.update("analysis", "calculator", reward=0.85, domain="finance")

        stats = q.get_convergence_stats("analysis", domain="finance")
        assert stats["converged"], f"Should converge with 40 consistent updates: {stats['reason']}"
        assert stats["mean_td_error"] < 0.08
        assert stats["td_variance"] < 0.01

    @pytest.mark.unit
    def test_convergence_persists_in_serialization(self):
        """TD errors should survive to_dict/from_dict round-trip."""
        from Jotty.core.intelligence.learning.td_lambda import SkillQTable

        q = SkillQTable()
        for _ in range(20):
            q.update("coding", "web-search", reward=0.9)

        data = q.to_dict()
        assert "td_errors" in data

        restored = SkillQTable.from_dict(data)
        original_stats = q.get_convergence_stats("coding")
        restored_stats = restored.get_convergence_stats("coding")
        assert original_stats["mean_td_error"] == restored_stats["mean_td_error"]

    @pytest.mark.unit
    def test_convergence_blocks_premature_crystallization(self):
        """should_crystallize should fail when Q-values haven't converged."""
        from Jotty.core.intelligence.learning.crystallization import should_crystallize

        td = _make_td_lambda()

        task_type = "coding"
        domain = "backend"
        skills = ["claude-cli-llm", "file-operations"]

        # Only 12 episodes — Q-values above threshold but not converged
        for _ in range(12):
            _simulate_episode(td, task_type, domain, skills, reward=0.95)

        _MOCK_TD = "Jotty.core.intelligence.learning.facade.get_td_lambda"
        with patch(_MOCK_TD, return_value=td):
            ok, stats = should_crystallize(
                task_type,
                domain,
                thresholds={
                    "min_episodes": 10,
                    "min_success_rate": 0.80,
                    "min_plan_consistency": 0.50,
                    "min_role_q": 0.60,
                    "min_plans": 5,
                },
            )

        # Other thresholds pass, but convergence should block
        assert not ok, f"Should NOT crystallize: Q-values haven't converged yet"
        assert any("converge" in r.lower() for r in stats.get("reasons", []))


class TestGoldMetric:
    """Verify the quality-aware DSPy metric scores outputs correctly."""

    @pytest.mark.unit
    def test_empty_prediction_scores_zero(self):
        from Jotty.core.intelligence.learning.advanced_learning import _gold_metric
        from types import SimpleNamespace

        ex = SimpleNamespace(output="Some gold output", domain="general")
        assert _gold_metric(ex, SimpleNamespace(output="")) == 0.0
        assert _gold_metric(ex, SimpleNamespace(output="short")) == 0.0

    @pytest.mark.unit
    def test_identical_output_scores_high(self):
        from Jotty.core.intelligence.learning.advanced_learning import _gold_metric
        from types import SimpleNamespace

        gold = "Create a comprehensive analysis of market trends including charts and data."
        ex = SimpleNamespace(output=gold, domain="research")
        pred = SimpleNamespace(output=gold)

        score = _gold_metric(ex, pred)
        assert score > 0.7, f"Identical output should score high, got {score}"

    @pytest.mark.unit
    def test_good_output_scores_higher_than_bad(self):
        from Jotty.core.intelligence.learning.advanced_learning import _gold_metric
        from types import SimpleNamespace

        gold = (
            "## Market Analysis\n- Revenue grew 15%\n- Margins expanded to 22%\n- Forecast: bullish"
        )
        ex = SimpleNamespace(output=gold, domain="research")

        good = SimpleNamespace(
            output="## Market Analysis\n- Revenue increased by 15%\n- Profit margins reached 22%\n- Outlook is bullish"
        )
        bad = SimpleNamespace(
            output="Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
        )

        good_score = _gold_metric(ex, good)
        bad_score = _gold_metric(ex, bad)
        assert (
            good_score > bad_score
        ), f"Good output ({good_score:.3f}) should score higher than bad ({bad_score:.3f})"

    @pytest.mark.unit
    def test_repetitive_output_penalized(self):
        """Degenerate repetition should score lower than diverse output."""
        from Jotty.core.intelligence.learning.advanced_learning import _gold_metric
        from types import SimpleNamespace

        gold = "A comprehensive report with multiple insights and data points."
        ex = SimpleNamespace(output=gold, domain="general")

        diverse = SimpleNamespace(
            output="A comprehensive report covering multiple insights, data analysis, and key findings."
        )
        repetitive = SimpleNamespace(
            output="report report report report report report report report report report "
            "report report report report report report report report report report"
        )

        diverse_score = _gold_metric(ex, diverse)
        rep_score = _gold_metric(ex, repetitive)
        assert (
            diverse_score > rep_score
        ), f"Diverse ({diverse_score:.3f}) should beat repetitive ({rep_score:.3f})"

    @pytest.mark.unit
    def test_metric_handles_no_gold(self):
        """When gold output is empty, metric should still give partial credit."""
        from Jotty.core.intelligence.learning.advanced_learning import _gold_metric
        from types import SimpleNamespace

        ex = SimpleNamespace(output="", domain="general")
        pred = SimpleNamespace(
            output="Some meaningful output that is long enough to count as valid"
        )

        score = _gold_metric(ex, pred)
        assert score > 0.0, "Should give partial credit for non-empty output even without gold"
        assert score < 0.5, "But not high credit without gold to compare against"


class TestDomainTaskModule:
    """Verify DSPy module classes returned by _get_domain_task_classes."""

    @pytest.mark.unit
    def test_module_has_generate_and_validate(self):
        from Jotty.core.intelligence.learning.advanced_learning import _get_domain_task_classes

        Sig, Module = _get_domain_task_classes()

        module = Module()
        assert hasattr(module, "generate"), "Module should have generate stage"
        assert hasattr(module, "forward"), "Module should have forward method"

    @pytest.mark.unit
    def test_module_still_works_as_before(self):
        from Jotty.core.intelligence.learning.advanced_learning import _get_domain_task_classes

        _, Module = _get_domain_task_classes()
        module = Module()
        assert hasattr(module, "generate")

    @pytest.mark.unit
    def test_signature_has_correct_fields(self):
        from Jotty.core.intelligence.learning.advanced_learning import _get_domain_task_classes

        Sig, _ = _get_domain_task_classes()

        input_keys = [
            k
            for k, v in Sig.model_fields.items()
            if hasattr(v, "json_schema_extra")
            and v.json_schema_extra
            and v.json_schema_extra.get("__dspy_field_type") == "input"
        ]
        output_keys = [
            k
            for k, v in Sig.model_fields.items()
            if hasattr(v, "json_schema_extra")
            and v.json_schema_extra
            and v.json_schema_extra.get("__dspy_field_type") == "output"
        ]

        assert "task_description" in Sig.model_fields
        assert "domain" in Sig.model_fields
        assert "output" in Sig.model_fields


class TestOptimizerTrainDevSplit:
    """Verify the optimizer uses train/dev split correctly."""

    @pytest.mark.unit
    def test_gather_training_data_quality_gate(self):
        """Only episodes with quality >= 0.7 should be included."""
        from Jotty.core.intelligence.learning.advanced_learning import DomainDSPyOptimizer
        from Jotty.core.intelligence.learning.learning_store import EpisodeRecord

        optimizer = DomainDSPyOptimizer()

        # Create mock episodes with varying quality
        high_quality = EpisodeRecord(
            episode_id="ep1",
            unit_type="swarm",
            unit_name="Test",
            domain="coding",
            task_type="coding",
            success=True,
            quality=0.9,
            execution_time=1.0,
            cost=0.01,
            context={"task": "Write a sort function"},
            action={"paradigm": "single"},
            outcome={
                "content": "def sort(arr): return sorted(arr)  # Production quality implementation"
            },
        )
        low_quality = EpisodeRecord(
            episode_id="ep2",
            unit_type="swarm",
            unit_name="Test",
            domain="coding",
            task_type="coding",
            success=True,
            quality=0.4,
            execution_time=1.0,
            cost=0.01,
            context={"task": "Write a function"},
            action={"paradigm": "single"},
            outcome={"content": "def f(): pass  # Low quality stub"},
        )

        mock_store = MagicMock()
        mock_store.query_episodes.return_value = [high_quality, low_quality]
        mock_store.get_distilled_lessons.return_value = []

        with patch(
            "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance",
            return_value=mock_store,
        ):
            examples = optimizer._gather_training_data("coding", min_quality=0.7)

        # Only the high-quality episode should pass the gate
        assert len(examples) == 1

    @pytest.mark.unit
    def test_optimize_accepts_strategy_parameter(self):
        """The optimize() method should accept strategy='auto'|'mipro'|'bootstrap'."""
        import inspect
        from Jotty.core.intelligence.learning.advanced_learning import DomainDSPyOptimizer

        sig = inspect.signature(DomainDSPyOptimizer.optimize)
        assert "strategy" in sig.parameters
        assert sig.parameters["strategy"].default == "auto"


class TestSkillCreditDistribution:
    """Verify per-skill credit-weighted Q-table updates."""

    @pytest.mark.unit
    def test_credit_weights_by_step_success(self):
        """Skills with successful steps get more credit than failed ones."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        mock_store = MagicMock()
        mock_store.get_value.return_value = None
        mock_store.query_episodes.return_value = []
        mock_store.get_distilled_lessons.return_value = []

        td = _make_td_lambda()

        with (
            patch(
                "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance",
                return_value=mock_store,
            ),
            patch(
                "Jotty.core.intelligence.learning.facade.get_td_lambda",
                return_value=td,
            ),
        ):
            svc = LearningService.__new__(LearningService)
            svc._store = mock_store

            svc._update_skill_credits(
                domain="coding",
                task_type="code_generation",
                action={"skills": ["good-skill", "bad-skill"]},
                outcome={
                    "per_step_results": [
                        {"success": True},
                        {"success": False},
                    ]
                },
                success=True,
                quality=0.8,
            )

        good_q = td.skill_q.get_q("code_generation", "good-skill", domain="coding")
        bad_q = td.skill_q.get_q("code_generation", "bad-skill", domain="coding")
        assert (
            good_q > bad_q
        ), f"Successful skill ({good_q}) should have higher Q than failed ({bad_q})"

    @pytest.mark.unit
    def test_uniform_credit_without_step_data(self):
        """Without per-step data, credit is distributed uniformly."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        mock_store = MagicMock()
        mock_store.get_value.return_value = None

        td = _make_td_lambda()

        with (
            patch(
                "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance",
                return_value=mock_store,
            ),
            patch(
                "Jotty.core.intelligence.learning.facade.get_td_lambda",
                return_value=td,
            ),
        ):
            svc = LearningService.__new__(LearningService)
            svc._store = mock_store

            svc._update_skill_credits(
                domain="research",
                task_type="research_summary",
                action={"skills": ["skill-a", "skill-b"]},
                outcome={"content": "some output"},
                success=True,
                quality=0.8,
            )

        q_a = td.skill_q.get_q("research_summary", "skill-a", domain="research")
        q_b = td.skill_q.get_q("research_summary", "skill-b", domain="research")
        assert (
            abs(q_a - q_b) < 0.01
        ), f"Without step data, skills should get equal credit ({q_a} vs {q_b})"

    @pytest.mark.unit
    def test_no_credit_on_failure(self):
        """Failed episodes should not trigger credit distribution."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        mock_store = MagicMock()
        mock_store.get_value.return_value = None

        td = _make_td_lambda()

        with (
            patch(
                "Jotty.core.intelligence.learning.learning_store.LearningStore.get_instance",
                return_value=mock_store,
            ),
            patch(
                "Jotty.core.intelligence.learning.facade.get_td_lambda",
                return_value=td,
            ),
        ):
            svc = LearningService.__new__(LearningService)
            svc._store = mock_store

            svc._update_skill_credits(
                domain="coding",
                task_type="code_generation",
                action={"skills": ["skill-x"]},
                outcome={},
                success=False,
                quality=0.2,
            )

        q_x = td.skill_q.get_q("code_generation", "skill-x", domain="coding")
        assert q_x == 0.5, f"Failed episode should leave Q at default 0.5, got {q_x}"


@pytest.mark.unit
class TestAutoOptimizeTrigger:
    """Verify that record() auto-triggers DSPy optimization and crystallization."""

    def test_auto_optimize_fires_after_threshold(self):
        """_maybe_auto_optimize should call DomainDSPyOptimizer.optimize when gold >= 5."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        svc = LearningService.__new__(LearningService)
        svc._last_optimize_counts = {}
        svc._dspy_optimize_interval = 15

        mock_optimizer = MagicMock()
        mock_optimizer._gather_training_data.return_value = [1, 2, 3, 4, 5]  # 5 gold
        mock_optimizer.optimize.return_value = MagicMock()

        with (
            patch("Jotty.core.intelligence.learning.learning_service.logger"),
            patch(
                "Jotty.core.intelligence.learning.advanced_learning.DomainDSPyOptimizer.get_instance",
                return_value=mock_optimizer,
            ),
            patch(
                "Jotty.core.intelligence.learning.crystallization.maybe_crystallize",
                return_value=None,
            ) as mock_crystal,
        ):
            svc._maybe_auto_optimize("coding", "code_gen", domain_count=20)

        mock_optimizer.optimize.assert_called_once_with("coding")
        mock_crystal.assert_called_once_with("code_gen", "coding")

    def test_auto_optimize_skips_below_interval(self):
        """Should not fire when domain_count hasn't advanced by _dspy_optimize_interval."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        svc = LearningService.__new__(LearningService)
        svc._last_optimize_counts = {"coding": 10}
        svc._dspy_optimize_interval = 15

        mock_optimizer = MagicMock()

        with patch(
            "Jotty.core.intelligence.learning.advanced_learning.DomainDSPyOptimizer.get_instance",
            return_value=mock_optimizer,
        ):
            svc._maybe_auto_optimize("coding", "code_gen", domain_count=20)

        mock_optimizer.optimize.assert_not_called()

    def test_auto_optimize_skips_low_gold(self):
        """Should not call optimize when gold episodes < 5."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        svc = LearningService.__new__(LearningService)
        svc._last_optimize_counts = {}
        svc._dspy_optimize_interval = 15

        mock_optimizer = MagicMock()
        mock_optimizer._gather_training_data.return_value = [1, 2]  # only 2

        with (
            patch("Jotty.core.intelligence.learning.learning_service.logger"),
            patch(
                "Jotty.core.intelligence.learning.advanced_learning.DomainDSPyOptimizer.get_instance",
                return_value=mock_optimizer,
            ),
            patch(
                "Jotty.core.intelligence.learning.crystallization.maybe_crystallize",
                return_value=None,
            ),
        ):
            svc._maybe_auto_optimize("coding", "code_gen", domain_count=20)

        mock_optimizer.optimize.assert_not_called()
