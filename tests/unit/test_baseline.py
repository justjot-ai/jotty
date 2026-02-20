"""
Baseline Test Suite - Verify Nothing Breaks During Refactoring
================================================================

Simple tests that verify core functionality works.
These tests establish a baseline before refactoring.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestCoreImports:
    """Verify all core imports work."""

    def test_can_import_core_module(self):
        """Test that core module can be imported."""
        import core

        assert core is not None

    def test_can_import_swarm_config(self):
        """Test SwarmConfig import."""
        from core import SwarmConfig

        assert SwarmConfig is not None

    def test_can_import_jotty_config(self):
        """Test SwarmConfig import."""
        from core import SwarmConfig

        assert SwarmConfig is not None

    def test_can_import_agent_spec(self):
        """Test AgentConfig import."""
        from core import AgentConfig

        assert AgentConfig is not None

    def test_agent_config_from_foundation(self):
        """Test AgentConfig from foundation module."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        assert AgentConfig is not None

    def test_can_import_swarm_manager(self):
        """Test Orchestrator import (V2 main orchestrator)."""
        from core import Orchestrator

        assert Orchestrator is not None

    def test_can_import_jotty_core(self):
        """Test Orchestrator import."""
        from core import Orchestrator

        assert Orchestrator is not None


@pytest.mark.unit
class TestMemoryImports:
    """Verify memory module imports work."""

    def test_can_import_memory_facade(self):
        """Test memory facade import."""
        from Jotty.core.intelligence.memory.facade import get_memory_system

        assert get_memory_system is not None


@pytest.mark.unit
class TestLearningImports:
    """Verify learning module imports work."""

    def test_can_import_td_lambda_learner(self):
        """Test TDLambdaLearner import."""
        from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner

        assert TDLambdaLearner is not None

    def test_can_import_learning_service(self):
        """Test LearningService import."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        assert LearningService is not None


@pytest.mark.unit
class TestBasicInstantiation:
    """Verify basic objects can be created."""

    def test_can_create_swarm_config(self):
        """Test SwarmConfig creation."""
        from core import SwarmConfig

        config = SwarmConfig()
        assert config is not None
        assert config.gamma == 0.99
        assert config.lambda_trace == 0.95

    def test_can_create_learning_service(self):
        """Test LearningService creation."""
        from Jotty.core.intelligence.learning.learning_service import LearningService

        service = LearningService.get_instance()
        assert service is not None

    def test_backward_compat_jotty_config_works(self):
        """Test that old SwarmConfig name still works."""
        from core import SwarmConfig

        config = SwarmConfig()
        assert config is not None
        assert hasattr(config, "gamma")


@pytest.mark.integration
class TestHelloWorld:
    """Integration test using hello world pattern."""

    def test_simple_dspy_agent(self):
        """Test a simple DSPy agent works."""
        import dspy

        class HelloAgent(dspy.Module):
            def forward(self, task):
                return f"Hello! Task: {task}"

        agent = HelloAgent()
        result = agent.forward(task="test")
        assert "Hello" in result
        assert "test" in result


def run_baseline_tests():
    """Run all baseline tests."""
    print("=" * 70)
    print("BASELINE TEST SUITE - PRE-REFACTORING")
    print("=" * 70)

    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    return exit_code


if __name__ == "__main__":
    exit_code = run_baseline_tests()
    sys.exit(exit_code)
