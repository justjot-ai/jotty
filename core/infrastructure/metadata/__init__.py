"""
Metadata Layer - Tool Discovery & Metadata Management
=====================================================

Tool management, metadata fetching, and protocol implementations.

Modules:
--------
- metadata_protocol: Metadata protocol definitions
- metadata_fetcher: Fetch metadata from providers
- metadata_tool_registry: Register & discover tools
- base_metadata_provider: Base metadata provider implementation
- mcp_metadata_provider: MCP (Model Context Protocol) provider
- a2ui_widget_provider: A2UI (Agent-to-User Interface) widget provider
- tool_shed: Tool management & caching
- tool_interceptor: Tool call interception
"""

# Tool interceptor moved to integration (2026-02-16)
from Jotty.core.infrastructure.integration import (  # type: ignore[import]
    ToolCall,
    ToolCallRegistry,
    ToolInterceptor,
)

# Tool shed moved to skills/_tools/tool_selector.py (2026-02-16)
from Jotty.skills._tools import AgenticToolSelector, CapabilityIndex, ToolShed, ToolShedSchema

from .a2ui_standard_widgets import (
    get_standard_widget_catalog,
    get_widget_count,
    list_widget_categories,
)
from .a2ui_widget_provider import (
    A2UIComponent,
    A2UIMessage,
    A2UIWidgetProvider,
    WidgetDefinition,
    create_widget_from_dict,
    create_widget_provider,
)
from .base_metadata_provider import BaseMetadataProvider, create_metadata_provider
from .mcp_metadata_provider import MCPMetadataProvider, create_mcp_provider_from_functions
from .metadata_fetcher import CacheEntry, MetaDataFetcher, MetaDataFetcherSignature, ToolMetadata
from .metadata_protocol import (
    JottyMetadataBase,
    MetadataIntrospector,
    MetadataProtocol,
    MetadataToolWrapper,
    MetadataValidator,
    MethodMetadata,
    jotty_method,
)
from .metadata_tool_registry import MetadataToolRegistry

# ToolResult - check if this exists in tool_selector
try:
    from Jotty.skills._tools.tool_selector import ToolResult
except ImportError:
    # Define a minimal ToolResult if not in tool_selector
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class ToolResult:  # type: ignore[no-redef]
        """Result from tool execution."""

        success: bool
        result: Any = None
        error: str | None = None


__all__ = [
    # base_metadata_provider
    "BaseMetadataProvider",
    "create_metadata_provider",
    # mcp_metadata_provider
    "MCPMetadataProvider",
    "create_mcp_provider_from_functions",
    # a2ui_widget_provider
    "A2UIWidgetProvider",
    "A2UIComponent",
    "A2UIMessage",
    "WidgetDefinition",
    "create_widget_provider",
    "create_widget_from_dict",
    # a2ui_standard_widgets
    "get_standard_widget_catalog",
    "list_widget_categories",
    "get_widget_count",
    # metadata_fetcher
    "CacheEntry",
    "MetaDataFetcher",
    "MetaDataFetcherSignature",
    "ToolMetadata",
    # metadata_protocol
    "JottyMetadataBase",
    "MetadataIntrospector",
    "MetadataProtocol",
    "MetadataToolWrapper",
    "MetadataValidator",
    "MethodMetadata",
    "jotty_method",
    # metadata_tool_registry
    "MetadataToolRegistry",
    # tool_interceptor
    "ToolCall",
    "ToolCallRegistry",
    "ToolInterceptor",
    # tool_shed
    "AgenticToolSelector",
    "CapabilityIndex",
    "ToolResult",
    "ToolShedSchema",
    "ToolShed",
]
