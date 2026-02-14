"""
Tools Registry - AI Tools (MCP Tools)
=====================================

Manages available AI tools that can be called by agents.
Tools represent functions/capabilities that AI can use.

This integrates with MCP (Model Context Protocol) and can be extended by any project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegistryToolSchema:
    """Schema describing an AI tool."""
    name: str  # Unique tool name
    description: str  # What this tool does
    category: str  # Category grouping (e.g., 'idea', 'section', 'generation')
    mcp_enabled: bool = False  # Whether tool is available via MCP
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameter schema
    returns: Optional[str] = None  # Return type description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'mcp_enabled': self.mcp_enabled,
            'parameters': self.parameters,
            'returns': self.returns,
        }


class ToolsRegistry:
    """
    Registry for AI tools.
    
    Generic registry that can be populated by any project using Jotty.
    Integrates with MCP tools when available.
    """
    
    def __init__(self):
        self._tools: Dict[str, RegistryToolSchema] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._implementations: Dict[str, Callable] = {}  # Optional: store actual implementations
        logger.info(" ToolsRegistry initialized")
    
    def register(
        self,
        name: str,
        description: str,
        category: str,
        mcp_enabled: bool = False,
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[str] = None,
        implementation: Optional[Callable] = None,
    ):
        """Register a tool."""
        tool = RegistryToolSchema(
            name=name,
            description=description,
            category=category,
            mcp_enabled=mcp_enabled,
            parameters=parameters or {},
            returns=returns,
        )
        
        self._tools[name] = tool
        
        # Update category index
        if category not in self._by_category:
            self._by_category[category] = []
        if name not in self._by_category[category]:
            self._by_category[category].append(name)
        
        # Store implementation if provided
        if implementation:
            self._implementations[name] = implementation
        
        logger.debug(f" Registered tool: {name} ({category})")
    
    def register_batch(self, tools: List[Dict[str, Any]]) -> None:
        """Register multiple tools at once."""
        for tool_data in tools:
            self.register(**tool_data)
    
    def get(self, name: str) -> Optional[RegistryToolSchema]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> List[RegistryToolSchema]:
        """Get all tools."""
        return list(self._tools.values())
    
    def get_by_category(self, category: str) -> List[RegistryToolSchema]:
        """Get tools in a category."""
        names = self._by_category.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]
    
    def get_categories(self) -> List[str]:
        """Get all categories."""
        return sorted(self._by_category.keys())
    
    def list_names(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())
    
    def get_implementation(self, name: str) -> Optional[Callable]:
        """Get tool implementation if available."""
        return self._implementations.get(name)
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            'available': [t.to_dict() for t in self.get_all()],
            'categories': self.get_categories(),
            'count': len(self._tools),
            'mcp_enabled_count': sum(1 for t in self._tools.values() if t.mcp_enabled),
        }
    
    def clear(self) -> None:
        """Clear all tools (useful for testing)."""
        self._tools.clear()
        self._by_category.clear()
        self._implementations.clear()
        logger.info(" ToolsRegistry cleared")


# Global instance (can be extended by projects)
_global_tools_registry = ToolsRegistry()


def get_tools_registry() -> ToolsRegistry:
    """Get the global tools registry instance."""
    return _global_tools_registry
