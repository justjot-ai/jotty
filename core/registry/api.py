"""
Jotty Registry API
==================

FastAPI/Flask-style API endpoints for accessing tools and widgets registry.
Can be used by any project to discover available tools and widgets.

This provides a generic HTTP API that projects can call.
"""

from typing import Dict, Any, Optional, List
from .unified_registry import UnifiedRegistry, get_unified_registry

logger = None  # Will be set by the web framework


class RegistryAPI:
    """
    API handler for registry endpoints.
    
    This can be integrated into FastAPI, Flask, or any web framework.
    """
    
    def __init__(self, registry: Optional[UnifiedRegistry] = None):
        self.registry = registry or get_unified_registry()
    
    def get_all(self) -> Dict[str, Any]:
        """
        GET /api/jotty/registry
        
        Returns all tools and widgets.
        """
        return {
            'success': True,
            'data': self.registry.get_all(),
        }
    
    def get_tools(self) -> Dict[str, Any]:
        """
        GET /api/jotty/registry/tools
        
        Returns all tools.
        """
        return {
            'success': True,
            'data': self.registry.get_tools(),
        }
    
    def get_widgets(self) -> Dict[str, Any]:
        """
        GET /api/jotty/registry/widgets
        
        Returns all widgets.
        """
        return {
            'success': True,
            'data': self.registry.get_widgets(),
        }
    
    def get_tool(self, name: str) -> Dict[str, Any]:
        """
        GET /api/jotty/registry/tools/{name}
        
        Returns a specific tool.
        """
        tool = self.registry.get_tool(name)
        if tool:
            return {
                'success': True,
                'data': tool,
            }
        return {
            'success': False,
            'error': f'Tool "{name}" not found',
        }
    
    def get_widget(self, value: str) -> Dict[str, Any]:
        """
        GET /api/jotty/registry/widgets/{value}
        
        Returns a specific widget.
        """
        widget = self.registry.get_widget(value)
        if widget:
            return {
                'success': True,
                'data': widget,
            }
        return {
            'success': False,
            'error': f'Widget "{value}" not found',
        }
    
    def validate_tools(self, tool_names: List[str]) -> Dict[str, Any]:
        """
        POST /api/jotty/registry/tools/validate
        
        Validates a list of tool names.
        """
        return {
            'success': True,
            'data': self.registry.validate_tools(tool_names),
        }
    
    def validate_widgets(self, widget_values: List[str]) -> Dict[str, Any]:
        """
        POST /api/jotty/registry/widgets/validate
        
        Validates a list of widget values.
        """
        return {
            'success': True,
            'data': self.registry.validate_widgets(widget_values),
        }
    
    def get_defaults(self) -> Dict[str, Any]:
        """
        GET /api/jotty/registry/defaults
        
        Returns default enabled tools and widgets.
        """
        return {
            'success': True,
            'data': self.registry.get_enabled_defaults(),
        }


# Global API instance
_global_api = RegistryAPI()


def get_registry_api() -> RegistryAPI:
    """Get the global registry API instance."""
    return _global_api
