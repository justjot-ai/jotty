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
        from Jotty.core.foundation.agent_config import AgentConfig
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

    def test_can_import_hierarchical_memory(self):
        """Test SwarmMemory import."""
        from core.memory import SwarmMemory
        assert SwarmMemory is not None

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
        """Test that old SwarmConfig name still works."""
        from core import SwarmConfig
        config = SwarmConfig()
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


@pytest.mark.unit
class TestPlanDecomposition:
    """Test plan decomposition for comparison tasks."""

    def test_extract_comparison_entities_vs(self):
        """Test extracting entities from 'X vs Y' task."""
        from core.agents.agentic_planner import TaskPlanner
        planner = TaskPlanner()
        entities = planner._extract_comparison_entities(
            "Research Paytm vs PhonePe payment gateway comparison"
        )
        assert len(entities) >= 2
        assert any('paytm' in e.lower() for e in entities)
        assert any('phonepe' in e.lower() or 'phone' in e.lower() for e in entities)

    def test_extract_comparison_entities_three_way(self):
        """Test extracting three entities from 'X vs Y vs Z'."""
        from core.agents.agentic_planner import TaskPlanner
        planner = TaskPlanner()
        entities = planner._extract_comparison_entities(
            "Compare React vs Vue vs Angular"
        )
        assert len(entities) >= 3

    def test_extract_comparison_entities_no_comparison(self):
        """Non-comparison task returns empty list."""
        from core.agents.agentic_planner import TaskPlanner
        planner = TaskPlanner()
        entities = planner._extract_comparison_entities(
            "Research AI trends and create a PDF"
        )
        assert entities == []

    def test_maybe_decompose_plan_comparison_with_composite(self):
        """Comparison task with 1-step composite plan should be decomposed."""
        from core.agents.agentic_planner import TaskPlanner
        from core.agents.base.autonomous_agent import ExecutionStep

        planner = TaskPlanner()
        # Simulate a 1-step composite plan
        steps = [ExecutionStep(
            skill_name='search-summarize-pdf-telegram-v2',
            tool_name='search_summarize_pdf_telegram_tool',
            params={'topic': 'Paytm vs PhonePe'},
            description='Search and summarize',
        )]
        skills = [
            {'name': 'web-search', 'tools': [{'name': 'search_web_tool'}]},
            {'name': 'claude-cli-llm', 'tools': [{'name': 'generate_text_tool'}]},
            {'name': 'simple-pdf-generator', 'tools': [{'name': 'generate_pdf_tool'}]},
            {'name': 'telegram-sender', 'tools': [{'name': 'send_telegram_file_tool'}]},
            {'name': 'search-summarize-pdf-telegram-v2', 'tools': [{'name': 'search_summarize_pdf_telegram_tool'}]},
        ]
        task = "Research Paytm vs PhonePe payment gateway comparison, create PDF and send via Telegram"

        decomposed = planner._maybe_decompose_plan(steps, skills, task, 'comparison')
        assert decomposed is not None
        assert len(decomposed) > 2  # Should have at least: 2 searches + synthesis + PDF + telegram
        # First steps should be web-search
        assert decomposed[0].skill_name == 'web-search'
        assert decomposed[1].skill_name == 'web-search'
        # Should end with telegram-sender (optional)
        assert decomposed[-1].skill_name == 'telegram-sender'

    def test_maybe_decompose_plan_simple_task_unchanged(self):
        """Simple non-comparison task should NOT be decomposed."""
        from core.agents.agentic_planner import TaskPlanner
        from core.agents.base.autonomous_agent import ExecutionStep

        planner = TaskPlanner()
        steps = [ExecutionStep(
            skill_name='web-search',
            tool_name='search_web_tool',
            params={'query': 'Delhi weather'},
            description='Search weather',
        )]
        skills = [{'name': 'web-search', 'tools': [{'name': 'search_web_tool'}]}]
        task = "Delhi weather"

        decomposed = planner._maybe_decompose_plan(steps, skills, task, 'research')
        assert decomposed is None  # No decomposition needed

    def test_maybe_decompose_plan_already_multi_step(self):
        """Multi-step plan (3+ steps) should NOT be decomposed."""
        from core.agents.agentic_planner import TaskPlanner
        from core.agents.base.autonomous_agent import ExecutionStep

        planner = TaskPlanner()
        steps = [
            ExecutionStep(skill_name='web-search', tool_name='search_web_tool',
                         params={}, description='s1'),
            ExecutionStep(skill_name='claude-cli-llm', tool_name='generate_text_tool',
                         params={}, description='s2'),
            ExecutionStep(skill_name='telegram-sender', tool_name='send_telegram_file_tool',
                         params={}, description='s3'),
        ]
        skills = [{'name': 'web-search'}, {'name': 'claude-cli-llm'}, {'name': 'telegram-sender'}]
        task = "Compare React vs Vue and send via Telegram"

        decomposed = planner._maybe_decompose_plan(steps, skills, task, 'comparison')
        assert decomposed is None  # Already multi-step


@pytest.mark.unit
class TestAggregateResearch:
    """Test research output aggregation."""

    def test_aggregate_research_outputs(self):
        """Test aggregation of multiple research outputs."""
        from core.agents.base.skill_plan_executor import SkillPlanExecutor
        executor = SkillPlanExecutor.__new__(SkillPlanExecutor)
        
        outputs = {
            'research_0': {
                'success': True,
                'query': 'Paytm payment gateway',
                'results': [
                    {'title': 'Paytm for Business', 'snippet': 'Payment solutions', 'url': 'https://paytm.com'},
                ],
            },
            'research_1': {
                'success': True,
                'query': 'PhonePe payment gateway',
                'results': [
                    {'title': 'PhonePe Business', 'snippet': 'UPI payments', 'url': 'https://phonepe.com'},
                ],
            },
        }
        
        aggregated = executor._aggregate_research_outputs(outputs)
        assert 'Paytm' in aggregated
        assert 'PhonePe' in aggregated
        assert '## Research:' in aggregated

    def test_aggregate_empty_outputs(self):
        """Empty outputs returns empty string."""
        from core.agents.base.skill_plan_executor import SkillPlanExecutor
        executor = SkillPlanExecutor.__new__(SkillPlanExecutor)
        
        assert executor._aggregate_research_outputs({}) == ''


@pytest.mark.unit
class TestSkillLoadResilience:
    """Test skill loading resilience."""

    def test_get_failed_skills_returns_dict(self):
        """get_failed_skills should return a dict even when no failures."""
        from core.registry.skills_registry import get_skills_registry
        registry = get_skills_registry()
        failed = registry.get_failed_skills()
        assert isinstance(failed, dict)


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
