"""
Baseline Test Suite - Verify Nothing Breaks During Refactoring
================================================================

Simple tests that verify core functionality works.
These tests establish a baseline before refactoring.
"""

import pytest
import sys
from pathlib import Path

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

    def test_can_import_jotty_config_backward_compat(self):
        """Test JottyConfig backward compatibility."""
        from core import JottyConfig, SwarmConfig
        assert JottyConfig == SwarmConfig

    def test_can_import_agent_spec(self):
        """Test AgentSpec import."""
        from core import AgentSpec
        assert AgentSpec is not None

    def test_can_import_agent_config_backward_compat(self):
        """Test AgentConfig backward compatibility."""
        from core import AgentConfig, AgentSpec
        assert AgentConfig == AgentSpec

    def test_can_import_conductor(self):
        """Test Conductor import."""
        from core import Conductor
        assert Conductor is not None

    def test_can_import_jotty_core(self):
        """Test JottyCore import."""
        from core import JottyCore
        assert JottyCore is not None


@pytest.mark.unit
class TestMemoryImports:
    """Verify memory module imports work."""

    def test_can_import_hierarchical_memory(self):
        """Test HierarchicalMemory import."""
        from core.memory import HierarchicalMemory
        assert HierarchicalMemory is not None

    def test_can_import_simple_brain(self):
        """Test SimpleBrain import."""
        from core.memory import SimpleBrain
        assert SimpleBrain is not None

    def test_can_import_brain_inspired_memory_manager(self):
        """Test BrainInspiredMemoryManager import."""
        from core.memory import BrainInspiredMemoryManager
        assert BrainInspiredMemoryManager is not None

    def test_can_import_consolidation_engine_components(self):
        """Test consolidation engine imports."""
        from core.memory import (
            BrainMode,
            HippocampalExtractor,
            SharpWaveRippleConsolidator
        )
        assert BrainMode is not None
        assert HippocampalExtractor is not None
        assert SharpWaveRippleConsolidator is not None


@pytest.mark.unit
class TestLearningImports:
    """Verify learning module imports work."""

    def test_can_import_td_lambda_learner(self):
        """Test TDLambdaLearner import."""
        from core.learning.learning import TDLambdaLearner
        assert TDLambdaLearner is not None

    def test_can_import_llm_q_predictor(self):
        """Test LLMQPredictor import."""
        from core.learning.q_learning import LLMQPredictor
        assert LLMQPredictor is not None


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

    def test_can_create_simple_brain(self):
        """Test SimpleBrain creation."""
        from core.memory import SimpleBrain
        brain = SimpleBrain()
        assert brain is not None

    def test_backward_compat_jotty_config_works(self):
        """Test that old JottyConfig name still works."""
        from core import JottyConfig
        config = JottyConfig()
        assert config is not None
        assert hasattr(config, 'gamma')


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
    print("="*70)
    print("BASELINE TEST SUITE - PRE-REFACTORING")
    print("="*70)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_baseline_tests()
    sys.exit(exit_code)
