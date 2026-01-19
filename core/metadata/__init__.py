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

from .base_metadata_provider import (
    BaseMetadataProvider,
    create_metadata_provider,
)
from .mcp_metadata_provider import (
    MCPMetadataProvider,
    create_mcp_provider_from_functions,
)
from .a2ui_widget_provider import (
    A2UIWidgetProvider,
    A2UIComponent,
    A2UIMessage,
    WidgetDefinition,
    create_widget_provider,
    create_widget_from_dict,
)
from .a2ui_standard_widgets import (
    get_standard_widget_catalog,
    list_widget_categories,
    get_widget_count,
)
from .metadata_fetcher import (
    CacheEntry,
    MetaDataFetcher,
    MetaDataFetcherSignature,
    ToolMetadata,
)
from .metadata_protocol import (
    JottyMetadataBase,
    MetadataIntrospector,
    MetadataProtocol,
    MetadataToolWrapper,
    MetadataValidator,
    MethodMetadata,
    jotty_method,
)
from .metadata_tool_registry import (
    MetadataToolRegistry,
)
from .tool_interceptor import (
    ToolCall,
    ToolCallRegistry,
    ToolInterceptor,
)
from .tool_shed import (
    AgenticToolSelector,
    CapabilityIndex,
    ToolResult,
    ToolSchema,
    ToolShed,
)

__all__ = [
    # base_metadata_provider
    'BaseMetadataProvider',
    'create_metadata_provider',
    # mcp_metadata_provider
    'MCPMetadataProvider',
    'create_mcp_provider_from_functions',
    # a2ui_widget_provider
    'A2UIWidgetProvider',
    'A2UIComponent',
    'A2UIMessage',
    'WidgetDefinition',
    'create_widget_provider',
    'create_widget_from_dict',
    # a2ui_standard_widgets
    'get_standard_widget_catalog',
    'list_widget_categories',
    'get_widget_count',
    # metadata_fetcher
    'CacheEntry',
    'MetaDataFetcher',
    'MetaDataFetcherSignature',
    'ToolMetadata',
    # metadata_protocol
    'JottyMetadataBase',
    'MetadataIntrospector',
    'MetadataProtocol',
    'MetadataToolWrapper',
    'MetadataValidator',
    'MethodMetadata',
    'jotty_method',
    # metadata_tool_registry
    'MetadataToolRegistry',
    # tool_interceptor
    'ToolCall',
    'ToolCallRegistry',
    'ToolInterceptor',
    # tool_shed
    'AgenticToolSelector',
    'CapabilityIndex',
    'ToolResult',
    'ToolSchema',
    'ToolShed',
]
