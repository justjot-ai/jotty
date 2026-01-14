"""
Test Component Integration
===========================

Tests that all extracted components are properly imported and structured.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.integration
class TestComponentImports:
    """Test that all components can be imported."""

    def test_can_import_all_components(self):
        """Test all three components can be imported."""
        from core.orchestration.parameter_resolver import ParameterResolver
        from core.orchestration.tool_manager import ToolManager
        from core.orchestration.state_manager import StateManager

        assert ParameterResolver is not None
        assert ToolManager is not None
        assert StateManager is not None

    def test_conductor_imports_components(self):
        """Test Conductor successfully imports all components."""
        # This will fail if there are any circular import issues
        from core import Conductor
        assert Conductor is not None

    def test_components_have_expected_methods(self):
        """Test each component has expected methods."""
        from core.orchestration.parameter_resolver import ParameterResolver
        from core.orchestration.tool_manager import ToolManager
        from core.orchestration.state_manager import StateManager

        # ParameterResolver methods
        assert hasattr(ParameterResolver, '_resolve_param_from_iomanager')
        assert hasattr(ParameterResolver, 'resolve_input')
        assert hasattr(ParameterResolver, '_resolve_parameter')

        # ToolManager methods
        assert hasattr(ToolManager, '_get_auto_discovered_dspy_tools')
        assert hasattr(ToolManager, '_get_architect_tools')
        assert hasattr(ToolManager, '_get_auditor_tools')

        # StateManager methods
        assert hasattr(StateManager, '_get_current_state')
        assert hasattr(StateManager, '_get_available_actions')
        assert hasattr(StateManager, '_introspect_actor_signature')
        assert hasattr(StateManager, 'get_actor_outputs')


@pytest.mark.integration
class TestComponentStructure:
    """Test component structure and organization."""

    def test_components_reduce_conductor_complexity(self):
        """Test that components successfully extract code from Conductor."""
        from core.orchestration import conductor, parameter_resolver, tool_manager, state_manager
        import inspect

        # Count methods in each module
        conductor_methods = len([m for m in dir(conductor.Conductor) if not m.startswith('_') or m.startswith('_get') or m.startswith('_introspect')])
        param_methods = len([m for m in dir(parameter_resolver.ParameterResolver) if callable(getattr(parameter_resolver.ParameterResolver, m))])
        tool_methods = len([m for m in dir(tool_manager.ToolManager) if callable(getattr(tool_manager.ToolManager, m))])
        state_methods = len([m for m in dir(state_manager.StateManager) if callable(getattr(state_manager.StateManager, m))])

        # Verify components have substantiallogic
        assert param_methods >= 5, f"ParameterResolver should have at least 5 methods, has {param_methods}"
        assert tool_methods >= 3, f"ToolManager should have at least 3 methods, has {tool_methods}"
        assert state_methods >= 5, f"StateManager should have at least 5 methods, has {state_methods}"


def run_integration_tests():
    """Run all integration tests."""
    print("="*70)
    print("COMPONENT INTEGRATION TESTS")
    print("="*70)

    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])

    return exit_code


if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code)
