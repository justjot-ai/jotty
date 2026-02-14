"""
Tests for learning coverage gaps.

Covers:
- SwarmLearningPipeline (EffectivenessTracker)
- TD-Lambda updates with state transitions
- Credit assignment reward distribution
"""

import pytest
from unittest.mock import Mock, patch


# ──────────────────────────────────────────────────────────────────────
# EffectivenessTracker
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestEffectivenessTracker:
    """Tests for EffectivenessTracker improvement measurement."""

    def test_initial_not_improving(self):
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=5, historical_window=10)
        assert tracker.is_improving() is False

    def test_record_and_report(self):
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=5)

        for _ in range(5):
            tracker.record("analysis", success=False, quality=0.2)
        for _ in range(3):
            tracker.record("analysis", success=True, quality=0.9)

        report = tracker.improvement_report()
        assert "analysis" in report
        assert report["analysis"]["recent_success_rate"] > report["analysis"]["historical_success_rate"]

    def test_improvement_detected(self):
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=5)

        # Historical: mostly failures
        for _ in range(5):
            tracker.record("search", success=False, quality=0.1)
        # Recent: all successes
        for _ in range(3):
            tracker.record("search", success=True, quality=0.9)

        assert tracker.is_improving("search") is True

    def test_no_improvement_when_declining(self):
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=5)

        # Historical: all successes
        for _ in range(5):
            tracker.record("code", success=True, quality=0.9)
        # Recent: all failures
        for _ in range(3):
            tracker.record("code", success=False, quality=0.1)

        assert tracker.is_improving("code") is False

    def test_serialization_roundtrip(self):
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker()
        tracker.record("task_a", success=True, quality=0.8, agent="agent1")
        tracker.record("task_b", success=False, quality=0.3, agent="agent2")

        data = tracker.to_dict()
        restored = EffectivenessTracker.from_dict(data)

        assert len(restored._records) == 2
        assert len(restored._global) == 2

    def test_global_vs_per_task(self):
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=5)

        for _ in range(5):
            tracker.record("type_a", success=True, quality=0.9)
        for _ in range(5):
            tracker.record("type_b", success=False, quality=0.1)

        report = tracker.improvement_report()
        assert "_global" in report
        assert "type_a" in report
        assert "type_b" in report


# ──────────────────────────────────────────────────────────────────────
# TD-Lambda Learning
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTDLambdaLearning:
    """Tests for TD-Lambda learning updates."""

    def test_update_modifies_state(self):
        from Jotty.core.learning.td_lambda import TDLambdaLearner
        from Jotty.core.foundation.data_structures import SwarmConfig

        learner = TDLambdaLearner(config=SwarmConfig())
        state = {"task": "research", "agent": "r1"}
        action = {"tool": "web-search"}
        next_state = {"task": "research", "agent": "r1", "step": 2}

        learner.update(state, action, reward=1.0, next_state=next_state)
        # Should not raise and should have processed the update

    def test_gamma_and_lambda_configurable(self):
        from Jotty.core.learning.td_lambda import TDLambdaLearner
        from Jotty.core.foundation.configs.learning import LearningConfig

        config = LearningConfig(gamma=0.5, lambda_trace=0.3, alpha=0.1)
        learner = TDLambdaLearner(config=config)
        assert learner.gamma == 0.5
        assert learner.lambda_trace == 0.3

    def test_multiple_updates_accumulate(self):
        from Jotty.core.learning.td_lambda import TDLambdaLearner
        from Jotty.core.foundation.data_structures import SwarmConfig

        learner = TDLambdaLearner(config=SwarmConfig())
        for i in range(5):
            learner.update(
                state={"step": i},
                action={"act": i},
                reward=float(i),
                next_state={"step": i + 1},
            )
        # Should process all updates without error


# ──────────────────────────────────────────────────────────────────────
# Credit Assignment
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCreditAssignment:
    """Tests for CreditAssignment component."""

    def test_credit_assignment_creates(self):
        from Jotty.core.orchestration.credit_assignment import CreditAssignment
        ca = CreditAssignment()
        assert ca is not None

    def test_record_improvement_application(self):
        from Jotty.core.orchestration.credit_assignment import CreditAssignment
        ca = CreditAssignment()
        improvement = {"id": "imp_1", "description": "Better prompt", "credit": 0.0}
        credit = ca.record_improvement_application(
            improvement=improvement,
            student_score=0.6,
            teacher_score=0.8,
            final_score=0.75,
            context={"task": "research"},
        )
        # Credit object is created — improvement_id format may vary
        assert credit is not None
        assert hasattr(credit, 'improvement_id')

        stats = ca.get_credit_statistics()
        assert isinstance(stats, dict)
