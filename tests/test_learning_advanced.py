"""
Tests for advanced learning modules:
- transfer_learning.py (AbstractPattern, RoleProfile, MetaPattern, SemanticEmbedder)
- shaped_rewards.py (RewardCondition)
- reasoning_credit.py (ReasoningCreditAssigner)
"""

import time
from dataclasses import fields
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    from Jotty.core.intelligence.learning.transfer_learning import (
        AbstractPattern,
        MetaPattern,
        RoleProfile,
        SemanticEmbedder,
    )

    HAS_TRANSFER = True
except ImportError:
    HAS_TRANSFER = False

try:
    from Jotty.core.intelligence.learning.shaped_rewards import RewardCondition

    HAS_REWARDS = True
except ImportError:
    HAS_REWARDS = False

try:
    from Jotty.core.intelligence.learning.reasoning_credit import ReasoningCreditAssigner

    HAS_CREDIT = True
except ImportError:
    HAS_CREDIT = False


# =============================================================================
# AbstractPattern Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestAbstractPattern:
    """Tests for the AbstractPattern dataclass."""

    def test_success_rate_with_counts(self):
        """AbstractPattern.success_rate returns correct ratio when counts are present."""
        pattern = AbstractPattern(
            pattern_id="test_001",
            level="task",
            pattern_type="aggregation",
            description="Test pattern",
            success_count=7,
            failure_count=3,
        )
        assert pattern.success_rate == pytest.approx(0.7)

    def test_success_rate_zero_counts(self):
        """AbstractPattern.success_rate returns 0.5 when no counts recorded."""
        pattern = AbstractPattern(
            pattern_id="test_002",
            level="error",
            pattern_type="TIMEOUT",
            description="No attempts yet",
        )
        # Source code returns 0.5 as the default when total == 0
        assert pattern.success_rate == pytest.approx(0.5)

    def test_avg_reward_calculation(self):
        """AbstractPattern.avg_reward computes total_reward / total_count."""
        pattern = AbstractPattern(
            pattern_id="test_003",
            level="workflow",
            pattern_type="RETRY_STRATEGY",
            description="Reward test",
            success_count=4,
            failure_count=1,
            total_reward=2.5,
        )
        # avg_reward = 2.5 / (4 + 1) = 0.5
        assert pattern.avg_reward == pytest.approx(0.5)

    def test_defaults(self):
        """AbstractPattern defaults: success_count=0, failure_count=0, total_reward=0.0, contexts=[]."""
        pattern = AbstractPattern(
            pattern_id="test_004",
            level="meta",
            pattern_type="GENERAL",
            description="Default values check",
        )
        assert pattern.success_count == 0
        assert pattern.failure_count == 0
        assert pattern.total_reward == 0.0
        assert pattern.contexts == []
        assert isinstance(pattern.created_at, float)
        assert isinstance(pattern.last_used, float)


# =============================================================================
# RoleProfile Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestRoleProfile:
    """Tests for the RoleProfile dataclass."""

    def test_defaults(self):
        """RoleProfile defaults: cooperation_score=0.5, avg_execution_time=0.0."""
        profile = RoleProfile(role="sql_generator")
        assert profile.cooperation_score == pytest.approx(0.5)
        assert profile.avg_execution_time == pytest.approx(0.0)
        assert profile.strengths == []
        assert profile.weaknesses == []
        assert profile.success_by_task_type == {}

    def test_stores_strengths_and_weaknesses(self):
        """RoleProfile correctly stores provided strengths and weaknesses."""
        profile = RoleProfile(
            role="validator",
            strengths=["analysis", "filtering"],
            weaknesses=["generation"],
        )
        assert "analysis" in profile.strengths
        assert "filtering" in profile.strengths
        assert "generation" in profile.weaknesses
        assert profile.role == "validator"


# =============================================================================
# MetaPattern Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestMetaPattern:
    """Tests for the MetaPattern dataclass."""

    def test_defaults(self):
        """MetaPattern defaults: success_rate=0.5, applications=0."""
        meta = MetaPattern(
            pattern_id="meta_001",
            trigger="3+ failures",
            strategy="Change approach",
        )
        assert meta.success_rate == pytest.approx(0.5)
        assert meta.applications == 0

    def test_stores_trigger_and_strategy(self):
        """MetaPattern correctly stores trigger and strategy strings."""
        meta = MetaPattern(
            pattern_id="meta_002",
            trigger="confidence < 0.5",
            strategy="Gather more context before proceeding",
        )
        assert meta.trigger == "confidence < 0.5"
        assert meta.strategy == "Gather more context before proceeding"
        assert meta.pattern_id == "meta_002"


# =============================================================================
# SemanticEmbedder Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestSemanticEmbedder:
    """Tests for the SemanticEmbedder class."""

    def test_init_with_embeddings_false(self):
        """SemanticEmbedder with use_embeddings=False does not load model."""
        embedder = SemanticEmbedder(use_embeddings=False)
        assert embedder.model is None
        assert embedder._use_embeddings is False
        assert embedder._model_loaded is False
        assert embedder.cache == {}

    def test_fallback_when_sentence_transformers_unavailable(self):
        """SemanticEmbedder falls back gracefully when sentence_transformers cannot be imported."""
        embedder = SemanticEmbedder(use_embeddings=False)
        # Force model load attempt -- since use_embeddings=False, it skips loading
        embedder._ensure_model()
        assert embedder.model is None
        assert embedder._model_loaded is True

        # Fallback embed should still work (word frequency vector)
        result = embedder._fallback_embed("test query for embeddings")
        assert isinstance(result, dict)
        assert len(result) > 0


# =============================================================================
# RewardCondition Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_REWARDS, reason="shaped_rewards module not available")
class TestRewardCondition:
    """Tests for the RewardCondition dataclass."""

    def test_defaults(self):
        """RewardCondition defaults: check_after='any', one_time=True, triggered=False."""
        cond = RewardCondition(
            name="test_condition",
            description="A test condition",
            reward_value=0.1,
        )
        assert cond.check_after == "any"
        assert cond.one_time is True
        assert cond.triggered is False
        assert cond.triggered_at is None

    def test_to_dict_returns_correct_keys(self):
        """RewardCondition.to_dict() returns dict with expected keys."""
        cond = RewardCondition(
            name="goal_achieved",
            description="Final goal achieved",
            reward_value=0.5,
        )
        result = cond.to_dict()
        expected_keys = {
            "name",
            "description",
            "reward",
            "check_after",
            "one_time",
            "triggered",
            "count",
        }
        assert set(result.keys()) == expected_keys
        assert result["name"] == "goal_achieved"
        assert result["description"] == "Final goal achieved"
        assert result["reward"] == 0.5
        assert result["check_after"] == "any"
        assert result["one_time"] is True
        assert result["triggered"] is False
        assert result["count"] == 0

    def test_custom_values_stored(self):
        """RewardCondition stores custom values for check_after and one_time."""
        cond = RewardCondition(
            name="tool_call_success",
            description="Tool returned valid result",
            reward_value=0.15,
            check_after="tool_call",
            one_time=False,
        )
        assert cond.name == "tool_call_success"
        assert cond.reward_value == pytest.approx(0.15)
        assert cond.check_after == "tool_call"
        assert cond.one_time is False

    def test_trigger_count_starts_at_zero(self):
        """RewardCondition.trigger_count defaults to 0."""
        cond = RewardCondition(
            name="partial_output",
            description="Agent produced some output",
            reward_value=0.1,
        )
        assert cond.trigger_count == 0


# =============================================================================
# ReasoningCreditAssigner Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CREDIT, reason="reasoning_credit module not available")
class TestReasoningCreditAssigner:
    """Tests for the ReasoningCreditAssigner class."""

    def _make_mock_config(self):
        """Create a mock SwarmConfig with reasoning_weight and evidence_weight."""
        config = Mock()
        config.reasoning_weight = 0.3
        config.evidence_weight = 0.2
        return config

    def _make_mock_validation_result(
        self,
        agent_name,
        confidence=0.8,
        should_proceed=True,
        is_valid=True,
        reasoning="Valid output",
        tool_calls=None,
    ):
        """Create a mock ValidationResult for testing."""
        result = Mock()
        result.agent_name = agent_name
        result.confidence = confidence
        result.should_proceed = should_proceed
        result.is_valid = is_valid
        result.reasoning = reasoning
        result.tool_calls = tool_calls or []
        return result

    def test_instantiation_with_mock_config(self):
        """ReasoningCreditAssigner can be instantiated with a mock SwarmConfig."""
        config = self._make_mock_config()
        assigner = ReasoningCreditAssigner(config)
        assert assigner.config is config
        assert assigner.reasoning_weight == 0.3
        assert assigner.evidence_weight == 0.2

    def test_analyze_contributions_returns_dict(self):
        """ReasoningCreditAssigner.analyze_contributions returns a dict keyed by agent name."""
        config = self._make_mock_config()
        assigner = ReasoningCreditAssigner(config)

        architect_result = self._make_mock_validation_result(
            agent_name="architect_1",
            confidence=0.85,
            should_proceed=True,
            reasoning="The query structure looks correct for aggregation.",
        )
        auditor_result = self._make_mock_validation_result(
            agent_name="auditor_1",
            confidence=0.9,
            is_valid=True,
            reasoning="Results validated against expected schema.",
        )

        trajectory = [
            {"actor": "architect_1", "action": "approve", "result": "proceed"},
            {"actor": "actor_1", "action": "execute", "result": "success"},
            {"actor": "auditor_1", "action": "validate", "result": "valid"},
        ]

        contributions = assigner.analyze_contributions(
            success=True,
            architect_results=[architect_result],
            auditor_results=[auditor_result],
            actor_succeeded=True,
            trajectory=trajectory,
        )

        assert isinstance(contributions, dict)
        assert "architect_1" in contributions
        assert "auditor_1" in contributions
