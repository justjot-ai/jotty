"""
Phase 7 Refactoring Tests - Terminology Standardization
========================================================

Tests for Orchestrator → SingleAgentOrchestrator rename and
actor → agent terminology standardization.
"""

import sys
import os

# Add Jotty to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import warnings


def test_single_agent_orchestrator_import():
    """New class name imports successfully."""
    from core.orchestration import SingleAgentOrchestrator
    assert SingleAgentOrchestrator is not None
    print("✓ SingleAgentOrchestrator imports successfully")


def test_jotty_core_backward_compat():
    """Old Orchestrator name still works (deprecated alias)."""
    from core.orchestration import Orchestrator
    from core.orchestration import SingleAgentOrchestrator

    # Orchestrator is an alias for SingleAgentOrchestrator
    assert Orchestrator is SingleAgentOrchestrator
    print("✓ Orchestrator is alias for SingleAgentOrchestrator")


def test_jotty_core_module_import():
    """Old jotty_core module import still works."""
    from core.orchestration.jotty_core import Orchestrator
    from core.orchestration import SingleAgentOrchestrator

    assert Orchestrator is SingleAgentOrchestrator
    print("✓ jotty_core module import works (deprecated)")


def test_actor_parameter_backward_compat():
    """Old 'actor' parameter still works with deprecation warning."""
    from core.orchestration import SingleAgentOrchestrator
    import dspy

    # Create a simple agent
    agent = dspy.ChainOfThought("question -> answer")

    # Using old 'actor' parameter should work but warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        orch = SingleAgentOrchestrator(
            actor=agent,  # Old parameter name
            architect_prompts=[],
            auditor_prompts=[],
            architect_tools=[],
            auditor_tools=[]
        )

        # Should have deprecation warning
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("'actor' parameter is deprecated" in str(warning.message) for warning in w)

        # But should still work
        assert orch.agent is agent

    print("✓ 'actor' parameter works with deprecation warning")


def test_agent_parameter_new():
    """New 'agent' parameter works without warning."""
    from core.orchestration import SingleAgentOrchestrator
    import dspy

    # Create a simple agent
    agent = dspy.ChainOfThought("question -> answer")

    # Using new 'agent' parameter should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        orch = SingleAgentOrchestrator(
            agent=agent,  # New parameter name
            architect_prompts=[],
            auditor_prompts=[],
            architect_tools=[],
            auditor_tools=[]
        )

        # Filter for DeprecationWarnings only
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]

        # Should have no deprecation warnings
        assert len(deprecation_warnings) == 0
        assert orch.agent is agent

    print("✓ 'agent' parameter works without deprecation warning")


def test_instance_variable_name():
    """Internal instance variable is named 'agent'."""
    from core.orchestration import SingleAgentOrchestrator
    import dspy

    agent = dspy.ChainOfThought("question -> answer")

    orch = SingleAgentOrchestrator(
        agent=agent,
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[]
    )

    # Should have 'agent' attribute
    assert hasattr(orch, 'agent')
    assert orch.agent is agent

    print("✓ Instance variable is named 'agent'")


def test_package_exports():
    """Package __init__.py exports new names correctly."""
    from core.orchestration import SingleAgentOrchestrator, Orchestrator

    # Both should be available
    assert SingleAgentOrchestrator is not None
    assert Orchestrator is not None

    # Should be the same class
    assert Orchestrator is SingleAgentOrchestrator

    print("✓ Package exports both new and old names")


def test_orchestration_layer_imports():
    """Orchestration layer imports work correctly."""
    # New import path
    from core.orchestration.single_agent_orchestrator import SingleAgentOrchestrator

    # Old import path
    from core.orchestration.jotty_core import Orchestrator

    # Should be the same
    assert Orchestrator is SingleAgentOrchestrator

    print("✓ Both import paths work")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase 7 Refactoring Tests - Terminology Standardization")
    print("="*60 + "\n")

    test_single_agent_orchestrator_import()
    test_jotty_core_backward_compat()
    test_jotty_core_module_import()
    test_actor_parameter_backward_compat()
    test_agent_parameter_new()
    test_instance_variable_name()
    test_package_exports()
    test_orchestration_layer_imports()

    print("\n" + "="*60)
    print("✅ All Phase 7 tests passed!")
    print("="*60 + "\n")
