"""
Tests for UI Registry Module
===============================

Tests the unified UI Registry system (ui_registry.py) which is the "Eyes"
of the Jotty framework -- handling all visual output and rendering components.

Covers:
- UIComponent dataclass: to_dict, to_widget_dict, to_agui_dict,
  to_json_serializable, has_adapters property
- UIRegistry: initialization, register, register_batch, register_from_widget,
  register_from_agui, get, get_all, get_by_category, get_by_client,
  get_by_content_type, get_with_adapters, get_categories, get_clients,
  list_types, convert_to_a2ui, convert_to_agui, convert_from_a2ui,
  convert_from_agui, to_api_response, export_for_remote_agent, clear,
  merge_from_widget_registry, merge_from_agui_registry
- Global functions: get_ui_registry, reset_ui_registry, compatibility aliases

All external dependencies (builtin widgets, supervisor widgets) are mocked.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing the module under test
try:
    from Jotty.core.capabilities.registry.ui_registry import (
        UIComponent,
        UIRegistry,
        get_ui_registry,
        reset_ui_registry,
        get_widget_registry_compat,
        get_agui_registry_compat,
        _load_builtin_components,
    )
    UI_REGISTRY_AVAILABLE = True
except ImportError:
    UI_REGISTRY_AVAILABLE = False


# =============================================================================
# UIComponent Dataclass Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIComponentDefaults:
    """Tests for UIComponent default values and construction."""

    def test_minimal_construction(self):
        """UIComponent with only required fields uses sensible defaults."""
        comp = UIComponent(
            component_type="test",
            label="Test",
            category="Testing",
        )
        assert comp.component_type == "test"
        assert comp.label == "Test"
        assert comp.category == "Testing"
        assert comp.icon == ""
        assert comp.description == ""
        assert comp.content_type == "text"
        assert comp.has_own_ui is False
        assert comp.content_schema == ""
        assert comp.client_id == "jotty"
        assert comp.version == "1.0.0"
        assert comp.bidirectional is False
        assert comp.example_input is None
        assert comp.example_output is None

    def test_full_construction(self):
        """UIComponent accepts all fields."""
        converter = lambda x: x
        comp = UIComponent(
            component_type="data-table",
            label="Data Table",
            category="Data",
            icon="T",
            description="Shows data",
            content_type="json",
            has_own_ui=True,
            content_schema='{"rows": []}',
            to_a2ui_func=converter,
            to_agui_func=converter,
            from_a2ui_func=converter,
            from_agui_func=converter,
            to_a2ui="code_str",
            to_agui="code_str",
            from_a2ui="code_str",
            from_agui="code_str",
            client_id="justjot",
            version="2.0.0",
            bidirectional=True,
            example_input='{"data": []}',
            example_output='[{"type": "table"}]',
        )
        assert comp.content_type == "json"
        assert comp.has_own_ui is True
        assert comp.client_id == "justjot"
        assert comp.bidirectional is True


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIComponentToDict:
    """Tests for UIComponent.to_dict serialization."""

    def test_to_dict_keys(self):
        """to_dict returns all expected keys."""
        comp = UIComponent(component_type="x", label="X", category="C")
        d = comp.to_dict()
        expected_keys = {
            'component_type', 'label', 'category', 'icon', 'description',
            'content_type', 'has_own_ui', 'content_schema', 'client_id',
            'version', 'bidirectional', 'has_to_a2ui', 'has_to_agui',
            'has_from_a2ui', 'has_from_agui',
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_has_adapter_flags(self):
        """to_dict reflects adapter availability."""
        comp = UIComponent(
            component_type="x", label="X", category="C",
            to_a2ui_func=lambda x: x,
        )
        d = comp.to_dict()
        assert d['has_to_a2ui'] is True
        assert d['has_to_agui'] is False

    def test_to_dict_serialized_adapter_flag(self):
        """to_dict detects serialized adapter strings."""
        comp = UIComponent(
            component_type="x", label="X", category="C",
            to_agui="function code",
        )
        d = comp.to_dict()
        assert d['has_to_agui'] is True


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIComponentWidgetDict:
    """Tests for UIComponent.to_widget_dict (legacy format)."""

    def test_to_widget_dict_keys(self):
        """to_widget_dict returns legacy widget format keys."""
        comp = UIComponent(
            component_type="chart",
            label="Chart",
            category="Viz",
            icon="C",
            description="Charts",
            has_own_ui=True,
            content_type="json",
            content_schema="{}",
        )
        d = comp.to_widget_dict()
        assert d['value'] == "chart"
        assert d['label'] == "Chart"
        assert d['icon'] == "C"
        assert d['hasOwnUI'] is True
        assert d['contentType'] == "json"
        assert d['contentSchema'] == "{}"


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIComponentAguiDict:
    """Tests for UIComponent.to_agui_dict (legacy format)."""

    def test_to_agui_dict_keys(self):
        """to_agui_dict returns legacy AGUI format keys."""
        comp = UIComponent(
            component_type="mermaid",
            label="Mermaid",
            category="Diagrams",
            description="Mermaid diagrams",
            bidirectional=True,
            client_id="justjot",
            version="1.5.0",
        )
        d = comp.to_agui_dict()
        assert d['section_type'] == "mermaid"
        assert d['label'] == "Mermaid"
        assert d['bidirectional'] is True
        assert d['client_id'] == "justjot"
        assert d['version'] == "1.5.0"


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIComponentJsonSerializable:
    """Tests for UIComponent.to_json_serializable."""

    def test_to_json_serializable_includes_adapter_code(self):
        """to_json_serializable includes serialized adapter code strings."""
        comp = UIComponent(
            component_type="x", label="X", category="C",
            to_a2ui="adapter code A2UI",
            from_agui="adapter code from AGUI",
        )
        d = comp.to_json_serializable()
        assert d['to_a2ui'] == "adapter code A2UI"
        assert d['from_agui'] == "adapter code from AGUI"
        assert d['to_agui'] is None

    def test_to_json_serializable_is_json_safe(self):
        """to_json_serializable output can be JSON-serialized."""
        comp = UIComponent(component_type="x", label="X", category="C")
        d = comp.to_json_serializable()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIComponentHasAdapters:
    """Tests for UIComponent.has_adapters property."""

    def test_no_adapters(self):
        """has_adapters is False when no adapters are set."""
        comp = UIComponent(component_type="x", label="X", category="C")
        assert comp.has_adapters is False

    def test_has_func_adapter(self):
        """has_adapters is True when a func adapter is set."""
        comp = UIComponent(
            component_type="x", label="X", category="C",
            to_a2ui_func=lambda x: x,
        )
        assert comp.has_adapters is True

    def test_has_string_adapter(self):
        """has_adapters is True when a serialized adapter is set."""
        comp = UIComponent(
            component_type="x", label="X", category="C",
            from_agui="code",
        )
        assert comp.has_adapters is True

    def test_has_mixed_adapters(self):
        """has_adapters is True with mixed func and string adapters."""
        comp = UIComponent(
            component_type="x", label="X", category="C",
            to_a2ui_func=lambda x: x,
            from_agui="code",
        )
        assert comp.has_adapters is True


# =============================================================================
# UIRegistry Initialization Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryInit:
    """Tests for UIRegistry initialization."""

    def test_init_creates_empty_registry(self):
        """UIRegistry starts with no components."""
        registry = UIRegistry()
        assert len(registry.get_all()) == 0
        assert registry.get_categories() == []
        assert registry.get_clients() == []
        assert registry.list_types() == []


# =============================================================================
# UIRegistry Registration Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryRegister:
    """Tests for UIRegistry.register method."""

    def test_register_basic_component(self):
        """register adds a component and returns it."""
        registry = UIRegistry()
        comp = registry.register(
            component_type="test",
            label="Test",
            category="Testing",
        )
        assert isinstance(comp, UIComponent)
        assert comp.component_type == "test"
        assert registry.get("test") is comp

    def test_register_updates_category_index(self):
        """register adds component to category index."""
        registry = UIRegistry()
        registry.register(component_type="a", label="A", category="Cat1")
        registry.register(component_type="b", label="B", category="Cat1")
        registry.register(component_type="c", label="C", category="Cat2")
        assert "Cat1" in registry.get_categories()
        assert "Cat2" in registry.get_categories()
        assert len(registry.get_by_category("Cat1")) == 2

    def test_register_updates_client_index(self):
        """register adds component to client index."""
        registry = UIRegistry()
        registry.register(component_type="a", label="A", category="C", client_id="app1")
        registry.register(component_type="b", label="B", category="C", client_id="app2")
        assert len(registry.get_by_client("app1")) == 1
        assert len(registry.get_by_client("app2")) == 1

    def test_register_updates_content_type_index(self):
        """register adds component to content_type index."""
        registry = UIRegistry()
        registry.register(component_type="t", label="T", category="C", content_type="json")
        registry.register(component_type="m", label="M", category="C", content_type="markdown")
        assert len(registry.get_by_content_type("json")) == 1
        assert len(registry.get_by_content_type("markdown")) == 1

    def test_register_overwrite_same_type(self):
        """register overwrites component with same component_type."""
        registry = UIRegistry()
        registry.register(component_type="x", label="Old", category="C")
        registry.register(component_type="x", label="New", category="C")
        comp = registry.get("x")
        assert comp.label == "New"

    def test_register_with_all_params(self):
        """register accepts all optional parameters."""
        registry = UIRegistry()
        converter = lambda x: x
        comp = registry.register(
            component_type="full",
            label="Full",
            category="All",
            icon="F",
            description="Full component",
            content_type="json",
            has_own_ui=True,
            content_schema="{}",
            to_a2ui_func=converter,
            to_agui_func=converter,
            from_a2ui_func=converter,
            from_agui_func=converter,
            to_a2ui="code",
            to_agui="code",
            from_a2ui="code",
            from_agui="code",
            client_id="test-client",
            version="3.0.0",
            bidirectional=True,
            example_input="in",
            example_output="out",
        )
        assert comp.has_own_ui is True
        assert comp.bidirectional is True
        assert comp.version == "3.0.0"

    def test_register_no_duplicate_index_entries(self):
        """register does not duplicate entries in indexes on re-register."""
        registry = UIRegistry()
        registry.register(component_type="x", label="X", category="C", client_id="app")
        registry.register(component_type="x", label="X2", category="C", client_id="app")
        assert len(registry.get_by_category("C")) == 1
        assert len(registry.get_by_client("app")) == 1


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryRegisterBatch:
    """Tests for UIRegistry.register_batch method."""

    def test_register_batch(self):
        """register_batch registers multiple components."""
        registry = UIRegistry()
        components = [
            {"component_type": "a", "label": "A", "category": "C1"},
            {"component_type": "b", "label": "B", "category": "C2"},
            {"component_type": "c", "label": "C", "category": "C1"},
        ]
        result = registry.register_batch(components)
        assert len(result) == 3
        assert all(isinstance(c, UIComponent) for c in result)
        assert len(registry.get_all()) == 3

    def test_register_batch_empty(self):
        """register_batch with empty list returns empty."""
        registry = UIRegistry()
        result = registry.register_batch([])
        assert result == []


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryRegisterFromWidget:
    """Tests for UIRegistry.register_from_widget (legacy WidgetSchema format)."""

    def test_register_from_widget(self):
        """register_from_widget maps legacy widget fields correctly."""
        registry = UIRegistry()
        comp = registry.register_from_widget(
            value="chart",
            label="Chart",
            icon="C",
            description="A chart",
            category="Viz",
            hasOwnUI=True,
            contentType="json",
            contentSchema="{}",
        )
        assert comp.component_type == "chart"
        assert comp.has_own_ui is True
        assert comp.content_type == "json"


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryRegisterFromAgui:
    """Tests for UIRegistry.register_from_agui (legacy AGUIComponentAdapter format)."""

    def test_register_from_agui(self):
        """register_from_agui maps legacy AGUI fields correctly."""
        registry = UIRegistry()
        comp = registry.register_from_agui(
            section_type="mermaid",
            label="Mermaid",
            category="Diagrams",
            description="Mermaid diagrams",
            bidirectional=True,
            content_type="text",
            client_id="justjot",
            version="2.0.0",
        )
        assert comp.component_type == "mermaid"
        assert comp.bidirectional is True
        assert comp.client_id == "justjot"


# =============================================================================
# UIRegistry Retrieval Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryRetrieval:
    """Tests for UIRegistry get/lookup methods."""

    def _make_registry(self):
        """Create a registry with a few pre-registered components."""
        reg = UIRegistry()
        reg.register(component_type="text", label="Text", category="Content", content_type="markdown", client_id="jotty")
        reg.register(component_type="code", label="Code", category="Content", content_type="code", client_id="jotty")
        reg.register(component_type="chart", label="Chart", category="Viz", content_type="json", client_id="justjot",
                     to_a2ui_func=lambda x: [{"type": "chart"}])
        return reg

    def test_get_existing(self):
        """get returns component by type."""
        reg = self._make_registry()
        comp = reg.get("text")
        assert comp is not None
        assert comp.label == "Text"

    def test_get_nonexistent(self):
        """get returns None for unknown type."""
        reg = self._make_registry()
        assert reg.get("nonexistent") is None

    def test_get_all(self):
        """get_all returns all registered components."""
        reg = self._make_registry()
        all_comps = reg.get_all()
        assert len(all_comps) == 3

    def test_get_by_category(self):
        """get_by_category returns components in category."""
        reg = self._make_registry()
        content_comps = reg.get_by_category("Content")
        assert len(content_comps) == 2
        types = [c.component_type for c in content_comps]
        assert "text" in types
        assert "code" in types

    def test_get_by_category_empty(self):
        """get_by_category returns empty for unknown category."""
        reg = self._make_registry()
        assert reg.get_by_category("Unknown") == []

    def test_get_by_client(self):
        """get_by_client returns components for specific client."""
        reg = self._make_registry()
        jotty_comps = reg.get_by_client("jotty")
        assert len(jotty_comps) == 2
        justjot_comps = reg.get_by_client("justjot")
        assert len(justjot_comps) == 1

    def test_get_by_content_type(self):
        """get_by_content_type returns components with matching content_type."""
        reg = self._make_registry()
        json_comps = reg.get_by_content_type("json")
        assert len(json_comps) == 1
        assert json_comps[0].component_type == "chart"

    def test_get_with_adapters(self):
        """get_with_adapters returns only components with adapters."""
        reg = self._make_registry()
        adapted = reg.get_with_adapters()
        assert len(adapted) == 1
        assert adapted[0].component_type == "chart"

    def test_get_categories(self):
        """get_categories returns sorted list of all categories."""
        reg = self._make_registry()
        cats = reg.get_categories()
        assert cats == ["Content", "Viz"]

    def test_get_clients(self):
        """get_clients returns sorted list of all client IDs."""
        reg = self._make_registry()
        clients = reg.get_clients()
        assert clients == ["jotty", "justjot"]

    def test_list_types(self):
        """list_types returns all component type names."""
        reg = self._make_registry()
        types = reg.list_types()
        assert set(types) == {"text", "code", "chart"}


# =============================================================================
# UIRegistry Conversion Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryConversion:
    """Tests for A2UI/AGUI conversion methods."""

    def test_convert_to_a2ui_with_func(self):
        """convert_to_a2ui calls to_a2ui_func and returns result."""
        reg = UIRegistry()
        converter = MagicMock(return_value=[{"type": "block"}])
        reg.register(component_type="x", label="X", category="C", to_a2ui_func=converter)
        result = reg.convert_to_a2ui("x", {"data": "test"})
        converter.assert_called_once_with({"data": "test"})
        assert result == [{"type": "block"}]

    def test_convert_to_a2ui_no_component(self):
        """convert_to_a2ui returns None for unknown component."""
        reg = UIRegistry()
        assert reg.convert_to_a2ui("unknown", {}) is None

    def test_convert_to_a2ui_no_adapter(self):
        """convert_to_a2ui returns None when component has no to_a2ui_func."""
        reg = UIRegistry()
        reg.register(component_type="x", label="X", category="C")
        assert reg.convert_to_a2ui("x", {}) is None

    def test_convert_to_a2ui_exception(self):
        """convert_to_a2ui returns None when adapter raises exception."""
        reg = UIRegistry()
        converter = MagicMock(side_effect=RuntimeError("conversion error"))
        reg.register(component_type="x", label="X", category="C", to_a2ui_func=converter)
        assert reg.convert_to_a2ui("x", {}) is None

    def test_convert_to_agui_with_func(self):
        """convert_to_agui calls to_agui_func and returns result."""
        reg = UIRegistry()
        converter = MagicMock(return_value={"agui": "data"})
        reg.register(component_type="x", label="X", category="C", to_agui_func=converter)
        result = reg.convert_to_agui("x", "content")
        assert result == {"agui": "data"}

    def test_convert_to_agui_no_component(self):
        """convert_to_agui returns None for unknown component."""
        reg = UIRegistry()
        assert reg.convert_to_agui("unknown", {}) is None

    def test_convert_to_agui_no_adapter(self):
        """convert_to_agui returns None when no to_agui_func."""
        reg = UIRegistry()
        reg.register(component_type="x", label="X", category="C")
        assert reg.convert_to_agui("x", {}) is None

    def test_convert_to_agui_exception(self):
        """convert_to_agui returns None when adapter raises."""
        reg = UIRegistry()
        converter = MagicMock(side_effect=ValueError("bad"))
        reg.register(component_type="x", label="X", category="C", to_agui_func=converter)
        assert reg.convert_to_agui("x", {}) is None

    def test_convert_from_a2ui_with_func(self):
        """convert_from_a2ui calls from_a2ui_func."""
        reg = UIRegistry()
        converter = MagicMock(return_value="original content")
        reg.register(component_type="x", label="X", category="C", from_a2ui_func=converter)
        result = reg.convert_from_a2ui("x", [{"type": "block"}])
        assert result == "original content"

    def test_convert_from_a2ui_no_component(self):
        """convert_from_a2ui returns None for unknown component."""
        reg = UIRegistry()
        assert reg.convert_from_a2ui("unknown", []) is None

    def test_convert_from_a2ui_no_adapter(self):
        """convert_from_a2ui returns None when no from_a2ui_func."""
        reg = UIRegistry()
        reg.register(component_type="x", label="X", category="C")
        assert reg.convert_from_a2ui("x", []) is None

    def test_convert_from_a2ui_exception(self):
        """convert_from_a2ui returns None when adapter raises."""
        reg = UIRegistry()
        converter = MagicMock(side_effect=TypeError("oops"))
        reg.register(component_type="x", label="X", category="C", from_a2ui_func=converter)
        assert reg.convert_from_a2ui("x", []) is None

    def test_convert_from_agui_with_func(self):
        """convert_from_agui calls from_agui_func."""
        reg = UIRegistry()
        converter = MagicMock(return_value="back to content")
        reg.register(component_type="x", label="X", category="C", from_agui_func=converter)
        result = reg.convert_from_agui("x", {"agui": "data"})
        assert result == "back to content"

    def test_convert_from_agui_no_component(self):
        """convert_from_agui returns None for unknown component."""
        reg = UIRegistry()
        assert reg.convert_from_agui("unknown", {}) is None

    def test_convert_from_agui_no_adapter(self):
        """convert_from_agui returns None when no from_agui_func."""
        reg = UIRegistry()
        reg.register(component_type="x", label="X", category="C")
        assert reg.convert_from_agui("x", {}) is None

    def test_convert_from_agui_exception(self):
        """convert_from_agui returns None when adapter raises."""
        reg = UIRegistry()
        converter = MagicMock(side_effect=Exception("fail"))
        reg.register(component_type="x", label="X", category="C", from_agui_func=converter)
        assert reg.convert_from_agui("x", {}) is None


# =============================================================================
# UIRegistry API Response Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryApiResponse:
    """Tests for API response and export methods."""

    def _make_registry(self):
        reg = UIRegistry()
        reg.register(component_type="a", label="A", category="C1", to_a2ui_func=lambda x: x)
        reg.register(component_type="b", label="B", category="C2")
        return reg

    def test_to_api_response_structure(self):
        """to_api_response returns dict with expected keys."""
        reg = self._make_registry()
        resp = reg.to_api_response()
        assert 'components' in resp
        assert 'categories' in resp
        assert 'clients' in resp
        assert 'count' in resp
        assert 'with_adapters' in resp
        assert resp['count'] == 2
        assert resp['with_adapters'] == 1

    def test_to_api_response_components_are_dicts(self):
        """to_api_response components are dict-serialized."""
        reg = self._make_registry()
        resp = reg.to_api_response()
        for comp in resp['components']:
            assert isinstance(comp, dict)
            assert 'component_type' in comp

    def test_export_for_remote_agent_all(self):
        """export_for_remote_agent returns all components when no client_id."""
        reg = self._make_registry()
        export = reg.export_for_remote_agent()
        assert export['count'] == 2
        assert export['client_id'] is None
        assert len(export['components']) == 2

    def test_export_for_remote_agent_by_client(self):
        """export_for_remote_agent filters by client_id."""
        reg = UIRegistry()
        reg.register(component_type="a", label="A", category="C", client_id="app1")
        reg.register(component_type="b", label="B", category="C", client_id="app2")
        export = reg.export_for_remote_agent(client_id="app1")
        assert export['count'] == 1
        assert export['client_id'] == "app1"

    def test_export_for_remote_agent_uses_json_serializable(self):
        """export_for_remote_agent uses to_json_serializable format."""
        reg = UIRegistry()
        reg.register(component_type="x", label="X", category="C", to_a2ui="code_str")
        export = reg.export_for_remote_agent()
        comp = export['components'][0]
        assert comp['to_a2ui'] == "code_str"


# =============================================================================
# UIRegistry Utility Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryClear:
    """Tests for UIRegistry.clear method."""

    def test_clear_removes_all(self):
        """clear removes all components and indexes."""
        reg = UIRegistry()
        reg.register(component_type="a", label="A", category="C")
        reg.register(component_type="b", label="B", category="D")
        assert len(reg.get_all()) == 2
        reg.clear()
        assert len(reg.get_all()) == 0
        assert reg.get_categories() == []
        assert reg.get_clients() == []
        assert reg.list_types() == []


@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistryMerge:
    """Tests for UIRegistry merge methods (widget and AGUI registries)."""

    def test_merge_from_widget_registry(self):
        """merge_from_widget_registry imports widgets via register_from_widget."""
        reg = UIRegistry()

        # Use a simple object without to_dict to trigger the fallback path
        class FakeWidget:
            value = "chart"
            label = "Chart"
            icon = "C"
            description = "A chart"
            category = "Viz"
            hasOwnUI = False
            contentType = "json"
            contentSchema = "{}"

        mock_widget_registry = MagicMock()
        mock_widget_registry.get_all.return_value = [FakeWidget()]

        count = reg.merge_from_widget_registry(mock_widget_registry)
        assert count == 1
        assert reg.get("chart") is not None

    def test_merge_from_agui_registry(self):
        """merge_from_agui_registry imports AGUI adapters."""
        reg = UIRegistry()
        mock_adapter = MagicMock()
        mock_adapter.section_type = "mermaid"
        mock_adapter.label = "Mermaid"
        mock_adapter.category = "Diagrams"
        mock_adapter.description = "Mermaid diagrams"
        mock_adapter.content_type = "text"
        mock_adapter.to_a2ui = None
        mock_adapter.to_agui = None
        mock_adapter.from_a2ui = None
        mock_adapter.from_agui = None
        mock_adapter.to_a2ui_func = None
        mock_adapter.to_agui_func = None
        mock_adapter.from_a2ui_func = None
        mock_adapter.from_agui_func = None
        mock_adapter.bidirectional = False
        mock_adapter.client_id = "test"
        mock_adapter.version = "1.0.0"

        mock_agui_registry = MagicMock()
        mock_agui_registry.get_all.return_value = [mock_adapter]

        count = reg.merge_from_agui_registry(mock_agui_registry)
        assert count == 1
        assert reg.get("mermaid") is not None

    def test_merge_from_widget_registry_with_to_dict(self):
        """merge_from_widget_registry uses widget.to_dict() when available."""
        reg = UIRegistry()
        mock_widget = MagicMock()
        mock_widget.to_dict.return_value = {
            'value': 'table',
            'label': 'Table',
            'icon': 'T',
            'description': 'A table',
            'category': 'Data',
            'hasOwnUI': False,
            'contentType': 'json',
            'contentSchema': '{}',
        }

        mock_widget_registry = MagicMock()
        mock_widget_registry.get_all.return_value = [mock_widget]

        count = reg.merge_from_widget_registry(mock_widget_registry)
        assert count == 1
        comp = reg.get("table")
        assert comp is not None
        assert comp.label == "Table"


# =============================================================================
# Global Singleton Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestUIRegistrySingleton:
    """Tests for get_ui_registry, reset_ui_registry, and compatibility aliases."""

    def setup_method(self):
        reset_ui_registry()

    def teardown_method(self):
        reset_ui_registry()

    @patch("Jotty.core.registry.ui_registry._load_builtin_components")
    def test_get_ui_registry_returns_instance(self, mock_load):
        """get_ui_registry returns a UIRegistry instance."""
        reg = get_ui_registry()
        assert isinstance(reg, UIRegistry)

    @patch("Jotty.core.registry.ui_registry._load_builtin_components")
    def test_get_ui_registry_singleton(self, mock_load):
        """get_ui_registry returns same instance on subsequent calls."""
        r1 = get_ui_registry()
        r2 = get_ui_registry()
        assert r1 is r2

    @patch("Jotty.core.registry.ui_registry._load_builtin_components")
    def test_reset_ui_registry(self, mock_load):
        """reset_ui_registry clears singleton."""
        r1 = get_ui_registry()
        reset_ui_registry()
        r2 = get_ui_registry()
        assert r1 is not r2

    @patch("Jotty.core.registry.ui_registry._load_builtin_components")
    def test_widget_registry_compat(self, mock_load):
        """get_widget_registry_compat returns same UIRegistry instance."""
        reg = get_widget_registry_compat()
        assert isinstance(reg, UIRegistry)

    @patch("Jotty.core.registry.ui_registry._load_builtin_components")
    def test_agui_registry_compat(self, mock_load):
        """get_agui_registry_compat returns same UIRegistry instance."""
        reg = get_agui_registry_compat()
        assert isinstance(reg, UIRegistry)


# =============================================================================
# Builtin Components Loading Tests
# =============================================================================

@pytest.mark.skipif(not UI_REGISTRY_AVAILABLE, reason="UIRegistry not importable")
@pytest.mark.unit
class TestLoadBuiltinComponents:
    """Tests for _load_builtin_components function."""

    @patch("Jotty.core.registry.ui_registry.get_supervisor_widgets", create=True)
    def test_load_core_components(self, mock_supervisor):
        """_load_builtin_components registers core components."""
        # Mock the supervisor import to avoid import errors
        mock_supervisor.side_effect = ImportError("no supervisor")
        reg = UIRegistry()
        with patch("Jotty.core.registry.ui_registry.get_supervisor_widgets", side_effect=ImportError):
            _load_builtin_components(reg)
        # Core components should be loaded
        assert len(reg.get_all()) >= 12  # At least 12 core components
        assert reg.get("text") is not None
        assert reg.get("code") is not None
        assert reg.get("mermaid") is not None
        assert reg.get("chart") is not None
        assert reg.get("data-table") is not None

    def test_load_builtin_component_categories(self):
        """_load_builtin_components creates correct categories."""
        reg = UIRegistry()
        with patch.dict("sys.modules", {"Jotty.core.registry.builtin_widgets": None}):
            try:
                _load_builtin_components(reg)
            except Exception:
                pass
        # Check categories exist (even if supervisor load fails)
        cats = reg.get_categories()
        # At least Content, Diagrams, Visualization, Data should exist
        all_comps = reg.get_all()
        if len(all_comps) > 0:
            categories_present = set(c.category for c in all_comps)
            assert "Content" in categories_present

    def test_load_builtin_components_idempotent(self):
        """_load_builtin_components called twice does not duplicate."""
        reg = UIRegistry()
        with patch.dict("sys.modules", {"Jotty.core.registry.builtin_widgets": None}):
            try:
                _load_builtin_components(reg)
            except Exception:
                pass
            count1 = len(reg.get_all())
            try:
                _load_builtin_components(reg)
            except Exception:
                pass
            count2 = len(reg.get_all())
        # Components are overwritten (same keys), not duplicated
        assert count2 == count1
