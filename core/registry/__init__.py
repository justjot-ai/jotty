"""
Jotty Tools and Widgets Registry
=================================

Unified registry for tools and widgets that can be used across projects.
Provides a generic, extensible system for managing AI tools and UI widgets.

Features:
- Tool registry with metadata (category, description, MCP support)
- Widget registry (section types) with metadata
- AGUI component registry for client adapters
- API endpoints for discovery
- Generic enough to work across different projects
"""

from .widget_registry import WidgetRegistry, WidgetSchema, get_widget_registry
from .tools_registry import ToolsRegistry, ToolSchema, get_tools_registry
from .unified_registry import UnifiedRegistry, get_unified_registry
from .agui_component_registry import AGUIComponentRegistry, AGUIComponentAdapter, get_agui_registry
from .client_registration_helpers import (
    register_agui_adapter_from_registry,
    register_agui_adapters_from_module,
    register_generic_agui_adapter,
    get_registered_adapters_for_client,
    export_adapters_for_agent
)

__all__ = [
    # Core registries
    'WidgetRegistry',
    'WidgetSchema',
    'get_widget_registry',
    'ToolsRegistry',
    'ToolSchema',
    'get_tools_registry',
    'UnifiedRegistry',
    'get_unified_registry',
    # AGUI component registry
    'AGUIComponentRegistry',
    'AGUIComponentAdapter',
    'get_agui_registry',
    # Client registration helpers
    'register_agui_adapter_from_registry',
    'register_agui_adapters_from_module',
    'register_generic_agui_adapter',
    'get_registered_adapters_for_client',
    'export_adapters_for_agent',
]
