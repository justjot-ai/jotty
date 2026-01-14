"""
Jotty Tools and Widgets Registry
=================================

Unified registry for tools and widgets that can be used across projects.
Provides a generic, extensible system for managing AI tools and UI widgets.

Features:
- Tool registry with metadata (category, description, MCP support)
- Widget registry (section types) with metadata
- API endpoints for discovery
- Generic enough to work across different projects
"""

from .widget_registry import WidgetRegistry, WidgetSchema
from .tools_registry import ToolsRegistry, ToolSchema
from .unified_registry import UnifiedRegistry

__all__ = [
    'WidgetRegistry',
    'WidgetSchema',
    'ToolsRegistry',
    'ToolSchema',
    'UnifiedRegistry',
]
