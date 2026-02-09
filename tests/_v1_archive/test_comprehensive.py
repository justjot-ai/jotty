"""
Comprehensive Test Suite for Jotty Framework
=============================================

Tests all core components to ensure nothing breaks during refactoring.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestImports:
    """Test all core imports work."""

    def test_core_imports(self):
        """Test main core module imports."""
        from core import (
            SwarmConfig, JottyConfig,
            AgentSpec, AgentConfig,
            Conductor, JottyCore,
            SimpleBrain, BrainMode
        )
        assert SwarmConfig is not None
        assert JottyConfig == SwarmConfig  # Backward compat
        assert AgentSpec is not None
        assert AgentConfig == AgentSpec  # Backward compat
        assert Conductor is not None
        assert JottyCore is not None
        assert SimpleBrain is not None
        assert BrainMode is not None

    def test_memory_imports(self):
        """Test memory module imports."""
        from core.memory import (
            BrainInspiredMemoryManager,
            EpisodicMemory,
            SemanticPattern,
            HippocampalExtractor,
            SharpWaveRippleConsolidator,
            HierarchicalMemory,
            LLMRAGRetriever
        )
        assert BrainInspiredMemoryManager is not None
        assert EpisodicMemory is not None
        assert SemanticPattern is not None
        assert HippocampalExtractor is not None
        assert SharpWaveRippleConsolidator is not None
        assert HierarchicalMemory is not None
        assert LLMRAGRetriever is not None

    def test_learning_imports(self):
        """Test learning module imports."""
        from core.learning.learning import (
            TDLambdaLearner,
            AdaptiveLearningRate,
            IntermediateRewardCalculator
        )
        from core.learning.q_learning import LLMQPredictor
        assert TDLambdaLearner is not None
        assert AdaptiveLearningRate is not None
        assert IntermediateRewardCalculator is not None
        assert LLMQPredictor is not None

    def test_orchestration_imports(self):
        """Test orchestration module imports."""
        from core.orchestration.conductor import Conductor
        from core.orchestration.jotty_core import JottyCore
        from core.orchestration.roadmap import MarkovianTODO
        assert Conductor is not None
        assert JottyCore is not None
        assert MarkovianTODO is not None

    def test_foundation_imports(self):
        """Test foundation module imports."""
        from core.foundation.data_structures import (
            SwarmConfig,
            MemoryEntry,
            EpisodeResult,
            ValidationResult
        )
        from core.foundation.agent_config import AgentSpec
        assert SwarmConfig is not None
        assert MemoryEntry is not None
        assert EpisodeResult is not None
        assert ValidationResult is not None
        assert AgentSpec is not None


@pytest.mark.unit
class TestConfiguration:
    """Test configuration objects."""

    def test_swarm_config_creation(self):
        """Test SwarmConfig can be created with defaults."""
        from core import SwarmConfig
        config = SwarmConfig()
        assert config is not None
        assert config.alpha is not None
        assert config.gamma is not None
        assert config.lambda_ is not None

    def test_backward_compat_jotty_config(self):
        """Test JottyConfig backward compatibility."""
        from core import JottyConfig, SwarmConfig
        config = JottyConfig()
        assert isinstance(config, SwarmConfig)

    def test_agent_spec_creation(self):
        """Test AgentSpec can be created."""
        from core import AgentSpec
        import dspy

        class DummyAgent(dspy.Module):
            def forward(self, input):
                return "dummy"

        spec = AgentSpec(
            name="TestAgent",
            agent=DummyAgent(),
            architect_prompts=["Plan this"],
            auditor_prompts=["Validate this"]
        )
        assert spec.name == "TestAgent"
        assert spec.architect_prompts == ["Plan this"]
        assert spec.auditor_prompts == ["Validate this"]

    def test_backward_compat_agent_config(self):
        """Test AgentConfig backward compatibility."""
        from core import AgentConfig, AgentSpec
        import dspy

        class DummyAgent(dspy.Module):
            def forward(self, input):
                return "dummy"

        config = AgentConfig(
            name="TestAgent",
            agent=DummyAgent(),
            architect_prompts=[],
            auditor_prompts=[]
        )
        assert isinstance(config, AgentSpec)


@pytest.mark.unit
class TestMemory:
    """Test memory systems."""

    def test_hierarchical_memory_creation(self):
        """Test HierarchicalMemory can be created."""
        from core.memory import HierarchicalMemory
        from core import SwarmConfig

        config = SwarmConfig()
        memory = HierarchicalMemory(config)
        assert memory is not None

    def test_simple_brain_creation(self):
        """Test SimpleBrain can be created."""
        from core import SimpleBrain

        brain = SimpleBrain()
        assert brain is not None

    def test_brain_inspired_memory_manager_creation(self):
        """Test BrainInspiredMemoryManager can be created."""
        from core.memory import BrainInspiredMemoryManager

        manager = BrainInspiredMemoryManager()
        assert manager is not None
        assert manager.episodic_memory == []
        assert manager.semantic_memory == []


@pytest.mark.unit
class TestLearning:
    """Test learning systems."""

    def test_td_lambda_learner_creation(self):
        """Test TDLambdaLearner can be created."""
        from core.learning.learning import TDLambdaLearner
        from core import SwarmConfig

        config = SwarmConfig()
        learner = TDLambdaLearner(config)
        assert learner is not None

    def test_adaptive_learning_rate_creation(self):
        """Test AdaptiveLearningRate can be created."""
        from core.learning.learning import AdaptiveLearningRate

        alr = AdaptiveLearningRate()
        assert alr is not None


@pytest.mark.unit
class TestOrchestration:
    """Test orchestration components."""

    def test_conductor_creation(self):
        """Test Conductor can be created."""
        from core import Conductor, AgentSpec, SwarmConfig
        import dspy

        class DummyAgent(dspy.Module):
            def forward(self, input):
                return "dummy"

        agents = [
            AgentSpec(
                name="TestAgent",
                agent=DummyAgent(),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]
        config = SwarmConfig()

        conductor = Conductor(actors=agents, config=config)
        assert conductor is not None

    def test_markovian_todo_creation(self):
        """Test MarkovianTODO can be created."""
        from core.orchestration.roadmap import MarkovianTODO

        todo = MarkovianTODO(main_goal="Test goal")
        assert todo is not None
        assert todo.main_goal == "Test goal"


@pytest.mark.unit
class TestDataStructures:
    """Test data structures."""

    def test_memory_entry_creation(self):
        """Test MemoryEntry can be created."""
        from core.foundation.data_structures import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            content="Test content",
            level=MemoryLevel.EPISODIC,
            timestamp=1234567890.0
        )
        assert entry.content == "Test content"
        assert entry.level == MemoryLevel.EPISODIC

    def test_episode_result_creation(self):
        """Test EpisodeResult can be created."""
        from core.foundation.data_structures import EpisodeResult

        result = EpisodeResult(
            goal="Test goal",
            output="Test output",
            reward=1.0,
            metadata={}
        )
        assert result.goal == "Test goal"
        assert result.reward == 1.0


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""

    def test_simple_agent_execution(self):
        """Test simple agent can execute."""
        import dspy
        from core import AgentSpec

        class HelloAgent(dspy.Module):
            def forward(self, task):
                return f"Hello! Task: {task}"

        agent = HelloAgent()
        result = agent.forward(task="test")
        assert "Hello" in result
        assert "test" in result

    def test_create_conductor_convenience_function(self):
        """Test create_conductor convenience function."""
        from core import create_conductor, AgentSpec, SwarmConfig
        import dspy

        class DummyAgent(dspy.Module):
            def forward(self, input):
                return "dummy"

        agents = [
            AgentSpec(
                name="TestAgent",
                agent=DummyAgent(),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]

        conductor = create_conductor(agents=agents)
        assert conductor is not None


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print("COMPREHENSIVE TEST SUITE - JOTTY FRAMEWORK")
    print("="*70)

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
