"""
JustJot.ai Registry Loader
===========================

Loads tools and widgets from JustJot.ai into Jotty's registry.
This allows Jotty to be aware of JustJot.ai's tools and widgets.

This is a bridge between JustJot.ai and Jotty's generic registry system.
"""

import logging
from typing import Dict, Any, Optional
from .unified_registry import UnifiedRegistry, get_unified_registry

logger = logging.getLogger(__name__)


def load_justjot_tools_and_widgets(
    tools_data: Optional[Dict[str, Any]] = None,
    widgets_data: Optional[Dict[str, Any]] = None,
    registry: Optional[UnifiedRegistry] = None
) -> UnifiedRegistry:
    """
    Load JustJot.ai tools and widgets into Jotty registry.
    
    Args:
        tools_data: Dict with 'available' list of tool dicts
        widgets_data: Dict with 'available' list of widget dicts
        registry: Registry instance (defaults to global)
    
    Returns:
        The registry instance
    """
    reg = registry or get_unified_registry()
    
    # Load tools
    if tools_data and 'available' in tools_data:
        for tool_data in tools_data['available']:
            reg.tools.register(
                name=tool_data.get('name', ''),
                description=tool_data.get('description', ''),
                category=tool_data.get('category', 'general'),
                mcp_enabled=tool_data.get('mcp_enabled', False),
                parameters=tool_data.get('parameters', {}),
                returns=tool_data.get('returns'),
            )
        logger.info(f" Loaded {len(tools_data['available'])} tools from JustJot.ai")
    
    # Load widgets
    if widgets_data and 'available' in widgets_data:
        for widget_data in widgets_data['available']:
            reg.widgets.register(
                value=widget_data.get('value', ''),
                label=widget_data.get('label', ''),
                icon=widget_data.get('icon', ''),
                description=widget_data.get('description', ''),
                category=widget_data.get('category', 'General'),
                hasOwnUI=widget_data.get('hasOwnUI', False),
                contentType=widget_data.get('contentType', 'text'),
                contentSchema=widget_data.get('contentSchema', ''),
            )
        logger.info(f" Loaded {len(widgets_data['available'])} widgets from JustJot.ai")
    
    return reg
