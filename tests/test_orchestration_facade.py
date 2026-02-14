"""
Tests for the Orchestration & Intelligence Subsystem Facade (Phase 2e).

Verifies each orchestration accessor returns the correct type.
All tests run offline.
"""

import pytest


@pytest.mark.unit
class TestOrchestrationFacade:
    """Tests for orchestration facade accessor functions."""

    def test_get_swarm_intelligence_returns_class(self):
        from Jotty.core.orchestration.facade import get_swarm_intelligence
        from Jotty.core.orchestration.swarm_intelligence import SwarmIntelligence
        result = get_swarm_intelligence()
        assert result is SwarmIntelligence

    def test_get_paradigm_executor_returns_class(self):
        from Jotty.core.orchestration.facade import get_paradigm_executor
        from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
        result = get_paradigm_executor()
        assert result is ParadigmExecutor

    def test_get_training_daemon_returns_class(self):
        from Jotty.core.orchestration.facade import get_training_daemon
        from Jotty.core.orchestration.training_daemon import TrainingDaemon
        result = get_training_daemon()
        assert result is TrainingDaemon

    def test_get_ensemble_manager_returns_instance(self):
        from Jotty.core.orchestration.facade import get_ensemble_manager
        from Jotty.core.orchestration.ensemble_manager import EnsembleManager
        result = get_ensemble_manager()
        assert isinstance(result, EnsembleManager)

    def test_get_provider_manager_returns_instance(self):
        from Jotty.core.orchestration.facade import get_provider_manager
        from Jotty.core.orchestration.provider_manager import ProviderManager
        result = get_provider_manager()
        assert isinstance(result, ProviderManager)

    def test_get_model_tier_router_returns_class(self):
        from Jotty.core.orchestration.facade import get_model_tier_router
        from Jotty.core.orchestration.model_tier_router import ModelTierRouter
        result = get_model_tier_router()
        assert result is ModelTierRouter

    def test_get_swarm_router_returns_instance(self):
        from Jotty.core.orchestration.facade import get_swarm_router
        from Jotty.core.orchestration.swarm_router import SwarmRouter
        result = get_swarm_router()
        assert isinstance(result, SwarmRouter)

    def test_list_components_returns_dict(self):
        from Jotty.core.orchestration.facade import list_components
        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.orchestration.facade import list_components
        components = list_components()
        expected = [
            "Orchestrator",
            "SwarmIntelligence",
            "ParadigmExecutor",
            "TrainingDaemon",
            "EnsembleManager",
            "ProviderManager",
            "ModelTierRouter",
            "SwarmRouter",
        ]
        for name in expected:
            assert name in components, f"Missing component: {name}"

    def test_list_components_values_are_strings(self):
        from Jotty.core.orchestration.facade import list_components
        for name, desc in list_components().items():
            assert isinstance(desc, str)
            assert len(desc) > 0


@pytest.mark.unit
class TestOrchestrationFacadeFromInit:
    """Test facade functions are accessible from __init__."""

    def test_import_get_swarm_intelligence(self):
        from Jotty.core.orchestration import get_swarm_intelligence
        assert callable(get_swarm_intelligence)

    def test_import_get_ensemble_manager(self):
        from Jotty.core.orchestration import get_ensemble_manager
        assert callable(get_ensemble_manager)

    def test_import_get_swarm_router(self):
        from Jotty.core.orchestration import get_swarm_router
        assert callable(get_swarm_router)
