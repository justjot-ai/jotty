"""
Tests for the Context Subsystem Facade (Phase 2c).

Verifies each context accessor returns the correct type.
All tests run offline.
"""

import pytest


@pytest.mark.unit
class TestContextFacade:
    """Tests for context facade accessor functions."""

    def test_get_context_manager_returns_manager(self):
        from Jotty.core.infrastructure.context.context_manager import SmartContextManager
        from Jotty.core.infrastructure.context.facade import get_context_manager

        mgr = get_context_manager()
        assert isinstance(mgr, SmartContextManager)

    def test_get_context_guard_returns_guard(self):
        from Jotty.core.infrastructure.context.facade import get_context_guard
        from Jotty.core.infrastructure.context.global_context_guard import GlobalContextGuard

        guard = get_context_guard()
        assert isinstance(guard, GlobalContextGuard)

    def test_get_content_gate_returns_gate(self):
        from Jotty.core.infrastructure.context.content_gate import ContentGate
        from Jotty.core.infrastructure.context.facade import get_content_gate

        gate = get_content_gate()
        assert isinstance(gate, ContentGate)

    def test_list_components_returns_dict(self):
        from Jotty.core.infrastructure.context.facade import list_components

        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.infrastructure.context.facade import list_components

        components = list_components()
        expected = [
            "SmartContextManager",
            "GlobalContextGuard",
            "ContentGate",
            "LLMContextManager",
            "AgenticCompressor",
        ]
        for name in expected:
            assert name in components, f"Missing component: {name}"

    def test_list_components_values_are_strings(self):
        from Jotty.core.infrastructure.context.facade import list_components

        for name, desc in list_components().items():
            assert isinstance(desc, str)
            assert len(desc) > 0


@pytest.mark.unit
class TestContextFacadeFromInit:
    """Test facade functions are accessible from __init__."""

    def test_import_get_context_manager(self):
        from Jotty.core.infrastructure.context import get_context_manager

        assert callable(get_context_manager)

    def test_import_get_context_guard(self):
        from Jotty.core.infrastructure.context import get_context_guard

        assert callable(get_context_guard)

    def test_import_get_content_gate(self):
        from Jotty.core.infrastructure.context import get_content_gate

        assert callable(get_content_gate)
