"""
Tests for advanced learning modules:
- shaped_rewards.py (RewardCondition)
- algorithmic_credit.py (AlgorithmicCreditAssigner)
"""

import pytest

try:
    from Jotty.core.intelligence.learning.shaped_rewards import RewardCondition

    HAS_REWARDS = True
except ImportError:
    HAS_REWARDS = False

try:
    from Jotty.core.intelligence.learning.algorithmic_credit import AlgorithmicCreditAssigner

    HAS_CREDIT = True
except ImportError:
    HAS_CREDIT = False


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
        assert result["reward"] == 0.5

    def test_custom_values_stored(self):
        """RewardCondition stores custom values for check_after and one_time."""
        cond = RewardCondition(
            name="tool_call_success",
            description="Tool returned valid result",
            reward_value=0.15,
            check_after="tool_call",
            one_time=False,
        )
        assert cond.check_after == "tool_call"
        assert cond.one_time is False

    def test_trigger_count_starts_at_zero(self):
        cond = RewardCondition(
            name="partial_output",
            description="Agent produced some output",
            reward_value=0.1,
        )
        assert cond.trigger_count == 0


# =============================================================================
# AlgorithmicCreditAssigner Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CREDIT, reason="algorithmic_credit module not available")
class TestAlgorithmicCreditAssigner:
    """Tests for the AlgorithmicCreditAssigner class."""

    def test_instantiation(self):
        assigner = AlgorithmicCreditAssigner()
        assert assigner is not None
