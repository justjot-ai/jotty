"""
Tests for the Learning Subsystem Facade.

Verifies each learning accessor returns the correct type.
All tests use mocks where needed and run offline.
"""

import pytest


@pytest.mark.unit
class TestLearningFacade:
    """Tests for learning facade accessor functions."""

    def test_get_learning_system_returns_learning_service(self):
        from Jotty.core.intelligence.learning.facade import get_learning_system
        from Jotty.core.intelligence.learning.learning_service import LearningService

        service = get_learning_system()
        assert isinstance(service, LearningService)

    def test_get_td_lambda_returns_learner(self):
        from Jotty.core.intelligence.learning.facade import get_td_lambda
        from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner

        learner = get_td_lambda()
        assert isinstance(learner, TDLambdaLearner)

    def test_get_credit_assigner_returns_assigner(self):
        from Jotty.core.intelligence.learning.facade import get_credit_assigner
        from Jotty.core.intelligence.learning.algorithmic_credit import AlgorithmicCreditAssigner

        assigner = get_credit_assigner()
        assert isinstance(assigner, AlgorithmicCreditAssigner)

    def test_get_reward_manager_returns_manager(self):
        from Jotty.core.intelligence.learning.facade import get_reward_manager
        from Jotty.core.intelligence.learning.shaped_rewards import ShapedRewardManager

        manager = get_reward_manager()
        assert isinstance(manager, ShapedRewardManager)

    def test_list_components_returns_dict(self):
        from Jotty.core.intelligence.learning.facade import list_components

        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.intelligence.learning.facade import list_components

        components = list_components()
        expected = [
            "LearningService",
            "TDLambdaLearner",
            "AlgorithmicCreditAssigner",
            "ShapedRewardManager",
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
