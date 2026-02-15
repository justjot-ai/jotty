"""
Tests for the Learning Subsystem Facade (Phase 2a).

Verifies each learning accessor returns the correct type.
All tests use mocks where needed and run offline.
"""

import pytest


@pytest.mark.unit
class TestLearningFacade:
    """Tests for learning facade accessor functions."""

    def test_get_learning_system_returns_learning_manager(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.intelligence.learning.facade import get_learning_system

        config = SwarmConfig()
        manager = get_learning_system(config)
        from Jotty.core.intelligence.learning.learning_coordinator import LearningManager

        assert isinstance(manager, LearningManager)

    def test_get_learning_system_default_config(self):
        from Jotty.core.intelligence.learning.facade import get_learning_system

        manager = get_learning_system()
        from Jotty.core.intelligence.learning.learning_coordinator import LearningManager

        assert isinstance(manager, LearningManager)

    def test_get_td_lambda_returns_learner(self):
        from Jotty.core.intelligence.learning.facade import get_td_lambda

        learner = get_td_lambda()
        from Jotty.core.intelligence.learning.learning import TDLambdaLearner

        assert isinstance(learner, TDLambdaLearner)

    def test_get_credit_assigner_returns_assigner(self):
        from Jotty.core.intelligence.learning.facade import get_credit_assigner

        assigner = get_credit_assigner()
        from Jotty.core.intelligence.learning.learning import ReasoningCreditAssigner

        assert isinstance(assigner, ReasoningCreditAssigner)

    def test_get_offline_learner_returns_dict(self):
        from Jotty.core.intelligence.learning.facade import get_offline_learner

        result = get_offline_learner()
        assert isinstance(result, dict)
        assert "offline_learner" in result
        assert "counterfactual" in result
        assert "pattern_discovery" in result

    def test_get_offline_learner_types(self):
        from Jotty.core.intelligence.learning.facade import get_offline_learner
        from Jotty.core.intelligence.learning.offline_learning import (
            CounterfactualLearner,
            OfflineLearner,
            PatternDiscovery,
        )

        result = get_offline_learner()
        assert isinstance(result["offline_learner"], OfflineLearner)
        assert isinstance(result["counterfactual"], CounterfactualLearner)
        assert isinstance(result["pattern_discovery"], PatternDiscovery)

    def test_get_reward_manager_returns_manager(self):
        from Jotty.core.intelligence.learning.facade import get_reward_manager
        from Jotty.core.intelligence.learning.shaped_rewards import ShapedRewardManager

        manager = get_reward_manager()
        assert isinstance(manager, ShapedRewardManager)

    def test_get_cooperative_agents_returns_classes(self):
        from Jotty.core.intelligence.learning.facade import get_cooperative_agents
        from Jotty.core.intelligence.learning.predictive_cooperation import (
            NashBargainingSolver,
            PredictiveCooperativeAgent,
        )

        result = get_cooperative_agents()
        assert result["predictive_agent"] is PredictiveCooperativeAgent
        assert result["nash_solver"] is NashBargainingSolver

    def test_list_components_returns_dict(self):
        from Jotty.core.intelligence.learning.facade import list_components

        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.intelligence.learning.facade import list_components

        components = list_components()
        expected = [
            "LearningManager",
            "TDLambdaLearner",
            "ReasoningCreditAssigner",
            "OfflineLearner",
            "ShapedRewardManager",
            "PredictiveCooperativeAgent",
            "NashBargainingSolver",
        ]
        for name in expected:
            assert name in components, f"Missing component: {name}"

    def test_list_components_values_are_strings(self):
        from Jotty.core.intelligence.learning.facade import list_components

        for name, desc in list_components().items():
            assert isinstance(desc, str), f"{name} description is not a string"
            assert len(desc) > 0, f"{name} has empty description"


@pytest.mark.unit
class TestLearningFacadeFromInit:
    """Test facade functions are accessible from __init__."""

    def test_import_get_learning_system(self):
        from Jotty.core.intelligence.learning import get_learning_system

        assert callable(get_learning_system)

    def test_import_get_td_lambda(self):
        from Jotty.core.intelligence.learning import get_td_lambda

        assert callable(get_td_lambda)

    def test_import_get_credit_assigner(self):
        from Jotty.core.intelligence.learning import get_credit_assigner

        assert callable(get_credit_assigner)
