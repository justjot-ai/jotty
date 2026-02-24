"""
Phase 7 Refactoring Tests - Terminology Standardization
========================================================

Tests for Orchestrator → SingleAgentOrchestrator rename and
actor → agent terminology standardization.

NOTE: SingleAgentOrchestrator was never implemented. The architecture
evolved to use Orchestrator directly. These tests are skipped.
"""

import os
import sys

# Add Jotty to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import pytest

pytest.skip(
    "SingleAgentOrchestrator was never implemented; module removed in Feb 2026 restructure",
    allow_module_level=True,
)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phase 7 Refactoring Tests - Terminology Standardization")
    print("=" * 60 + "\n")

    test_single_agent_orchestrator_import()
    test_jotty_core_backward_compat()
    test_jotty_core_module_import()
    test_actor_parameter_backward_compat()
    test_agent_parameter_new()
    test_instance_variable_name()
    test_package_exports()
    test_orchestration_layer_imports()

    print("\n" + "=" * 60)
    print("✅ All Phase 7 tests passed!")
    print("=" * 60 + "\n")
