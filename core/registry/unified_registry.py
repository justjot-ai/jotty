"""
Unified Registry - Combined Tools and Widgets
==============================================

Provides a unified interface for accessing both tools and widgets.
This is the main entry point for projects using Jotty's registry system.
"""

from typing import Dict, Any, Optional
import logging
from .widget_registry import WidgetRegistry, get_widget_registry
from .tools_registry import ToolsRegistry, get_tools_registry

logger = logging.getLogger(__name__)


class UnifiedRegistry:
    """
    Unified registry combining tools and widgets.
    
    This is the main interface that projects should use to access
    Jotty's tools and widgets registry.
    """
    
    def __init__(
        self,
        widget_registry: Optional[WidgetRegistry] = None,
        tools_registry: Optional[ToolsRegistry] = None,
    ):
        """
        Initialize unified registry.
        
        Args:
            widget_registry: Widget registry instance (defaults to global)
            tools_registry: Tools registry instance (defaults to global)
        """
        self.widgets = widget_registry or get_widget_registry()
        self.tools = tools_registry or get_tools_registry()
        logger.info("🌐 UnifiedRegistry initialized")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all tools and widgets in API format.
        
        Returns:
            Dict with 'tools' and 'widgets' keys
        """
        return {
            'tools': self.tools.to_api_response(),
            'widgets': self.widgets.to_api_response(),
        }
    
    def get_tools(self) -> Dict[str, Any]:
        """Get tools registry data."""
        return self.tools.to_api_response()
    
    def get_widgets(self) -> Dict[str, Any]:
        """Get widgets registry data."""
        return self.widgets.to_api_response()
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool."""
        tool = self.tools.get(name)
        return tool.to_dict() if tool else None
    
    def get_widget(self, value: str) -> Optional[Dict[str, Any]]:
        """Get a specific widget."""
        widget = self.widgets.get(value)
        return widget.to_dict() if widget else None
    
    def validate_tools(self, tool_names: list[str]) -> Dict[str, bool]:
        """
        Validate that tool names exist.
        
        Returns:
            Dict mapping tool_name -> exists
        """
        all_tools = set(self.tools.list_names())
        return {name: name in all_tools for name in tool_names}
    
    def validate_widgets(self, widget_values: list[str]) -> Dict[str, bool]:
        """
        Validate that widget values exist.
        
        Returns:
            Dict mapping widget_value -> exists
        """
        all_widgets = set(self.widgets.list_values())
        return {value: value in all_widgets for value in widget_values}
    
    def get_enabled_defaults(self) -> Dict[str, Any]:
        """
        Get default enabled tools and widgets.
        
        Projects can override this to provide their own defaults.
        """
        # Default: enable all tools, common widgets
        common_widgets = ['text', 'mermaid', 'code', 'todos', 'chart', 'kanban-board']
        available_widgets = self.widgets.list_values()
        default_widgets = [w for w in common_widgets if w in available_widgets]
        
        return {
            'tools': self.tools.list_names(),  # Enable all by default
            'widgets': default_widgets if default_widgets else available_widgets[:10],  # Top 10 if no matches
        }


# Global instance
_global_unified_registry = UnifiedRegistry()


def get_unified_registry() -> UnifiedRegistry:
    """Get the global unified registry instance."""
    return _global_unified_registry
