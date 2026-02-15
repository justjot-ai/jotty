from typing import Any

"""
MCP Metadata Provider for Jotty
================================

Generic MCP (Model Context Protocol) integration for Jotty SDK.

Enables Jotty agents to access MCP resources from any MCP server:
- Auto-discovers MCP resources dynamically
- Exposes resources as callable tools
- Works with ANY MCP server (JustJot, GitHub, Filesystem, etc.)
- Follows Jotty BaseMetadataProvider pattern
- Uses dependency injection for MCP tool functions

Usage:
    ```python
    from Jotty.core.infrastructure.metadata.mcp_metadata_provider import MCPMetadataProvider,create_mcp_tools_from_functions

    # Define MCP tool functions (provided by client)
    def list_resources_fn(server=None):
        # Call actual MCP ListMcpResourcesTool
        return [...]

    def read_resource_fn(server, uri):
        # Call actual MCP ReadMcpResourceTool
        return {...}

    # Create provider with injected functions
    mcp_provider = MCPMetadataProvider(
        server="justjot",
        list_resources_fn=list_resources_fn,
        read_resource_fn=read_resource_fn
    )

    # Or use helper function
    mcp_provider = create_mcp_tools_from_functions(
        list_fn=list_resources_fn,
        read_fn=read_resource_fn,
        server="justjot"
    )

    # Get tools for agents
    mcp_tools = mcp_provider.get_tools()
    ```

Architecture:
    - Extends BaseMetadataProvider (Jotty pattern)
    - Uses dependency injection for MCP tool functions (client provides)
    - Auto-discovers MCP resources from any server
    - Exposes resources as callable Python functions
    - Thread-safe, cached, token-budgeted (BaseMetadataProvider safeguards)

True SaaS Principle:
    - Jotty SDK provides the framework (this class)
    - Clients provide the implementation (MCP tool functions)
    - Works with ANY MCP server (JustJot, GitHub, Filesystem, etc.)
"""

import logging
from typing import Callable, Dict, List, Optional

from .base_metadata_provider import BaseMetadataProvider

logger = logging.getLogger(__name__)


class MCPMetadataProvider(BaseMetadataProvider):
    """
    Generic MCP integration for Jotty SDK.

    Discovers and exposes MCP resources from any MCP server as callable tools.

    Features:
    - **Dependency Injection**: Client provides MCP tool functions (true SaaS!)
    - **Dynamic tools**: Each resource becomes a callable tool
    - **Server-agnostic**: Works with any MCP server
    - **Type-safe**: Validates resource URIs and handles errors gracefully
    - **Cached**: BaseMetadataProvider caching prevents redundant MCP calls
    - **Budgeted**: Token budget tracking prevents context explosion

    Example:
        ```python
        # Client provides MCP tool functions
        def my_list_fn(server=None):
            # Use whatever MCP client you have
            return list_mcp_resources_tool(server=server)

        def my_read_fn(server, uri):
            return read_mcp_resource_tool(server=server, uri=uri)

        # Create provider
        mcp_provider = MCPMetadataProvider(
            server="justjot",
            list_resources_fn=my_list_fn,
            read_resource_fn=my_read_fn
        )

        # Use with Jotty ChatHandler
        handler = ChatHandler(
            enabled_tools=["list_tasks"],
            mcp_provider=mcp_provider
        )
        ```

    Dependency Injection:
        This provider requires two functions:
        - list_resources_fn(server): Returns list of MCP resources
        - read_resource_fn(server, uri): Returns resource content

        Client is responsible for providing these (e.g., via Claude Code MCP tools,
        HTTP API calls, or any other MCP client implementation).
    """

    def __init__(
        self,
        list_resources_fn: Optional[Callable] = None,
        read_resource_fn: Optional[Callable] = None,
        server: Optional[str] = None,
        token_budget: int = 100000,
        enable_caching: bool = True,
    ) -> None:
        """
        Initialize MCP metadata provider with injected MCP tool functions.

        Args:
            list_resources_fn: Function to list MCP resources
                              Signature: list_resources_fn(server: Optional[str]) -> List[Dict]
            read_resource_fn: Function to read MCP resource by URI
                             Signature: read_resource_fn(server: str, uri: str) -> Dict
            server: MCP server name to filter resources (e.g., "justjot", "github")
                   If None, uses all available MCP servers
            token_budget: Maximum tokens for MCP resource fetches (default: 100k)
            enable_caching: Whether to cache MCP resource reads (default: True)
        """
        self.server = server
        self._list_resources_fn = list_resources_fn
        self._read_resource_fn = read_resource_fn
        self._resources_cache: Optional[List[Dict[str, Any]]] = None

        # Initialize base class (auto-registers methods as tools)
        super().__init__(token_budget=token_budget, enable_caching=enable_caching)

        logger.info(" MCPMetadataProvider initialized")
        logger.info(f"   Server filter: {server or 'all servers'}")
        logger.info(f"   List function: {'provided' if list_resources_fn else 'not provided'}")
        logger.info(f"   Read function: {'provided' if read_resource_fn else 'not provided'}")

    # =========================================================================
    # MCP Resource Discovery (Auto-exposed as tools)
    # =========================================================================

    def list_mcp_resources(
        self, server: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List all available MCP resources.

        Discovers MCP resources from configured servers.
        Results are cached to prevent redundant MCP calls.

        Args:
            server: Filter by specific server (e.g., "justjot", "github")
                   If None, uses provider's configured server
            limit: Maximum number of resources to return

        Returns:
            List of resource dictionaries with:
            - name: Resource name
            - uri: Resource URI (e.g., "justjot://idea/123")
            - description: Resource description
            - mimeType: Resource MIME type
            - server: MCP server providing this resource

        Example:
            ```python
            resources = provider.list_mcp_resources(server="justjot")
            # [
            #     {
            #         "name": "Deep Learning",
            #         "uri": "justjot://idea/696cf701ee8defce7e1c0013",
            #         "description": "Free-form idea documentation",
            #         "mimeType": "application/json",
            #         "server": "justjot"
            #     },
            #     ...
            # ]
            ```
        """
        if not self._list_resources_fn:
            logger.warning(" No list_resources_fn provided - returning empty list")
            return []

        # Use cached resources if available
        if self._resources_cache is not None:
            resources = self._resources_cache
        else:
            # Call injected function
            try:
                server_filter = server or self.server
                resources = self._list_resources_fn(server=server_filter)
                self._resources_cache = resources
            except Exception as e:
                logger.error(f" Failed to list MCP resources: {e}")
                return []

        # Filter by server if specified
        if server:
            resources = [r for r in resources if r.get("server") == server]

        # Limit results
        resources = resources[:limit]

        logger.info(
            f" Listed {len(resources)} MCP resources (server={server or self.server or 'all'})"
        )
        return resources

    def read_mcp_resource(self, uri: str, server: Optional[str] = None) -> Dict[str, Any]:
        """
        Read a specific MCP resource by URI.

        Args:
            uri: Resource URI (e.g., "justjot://idea/696cf701ee8defce7e1c0013")
            server: MCP server name (optional, uses provider's server if not specified)

        Returns:
            Resource content dictionary

        Example:
            ```python
            idea = provider.read_mcp_resource("justjot://idea/123")
            # {
            #     "id": "696cf701ee8defce7e1c0013",
            #     "title": "Deep Learning",
            #     "description": "Free-form idea documentation",
            #     "sections": [...]
            # }
            ```
        """
        if not self._read_resource_fn:
            raise RuntimeError("No read_resource_fn provided - cannot read MCP resources")

        # Call injected function
        try:
            server_name = server or self.server
            if not server_name:
                # Extract server from URI (e.g., "justjot://idea/123" → "justjot")
                if "://" in uri:
                    server_name = uri.split("://")[0]

            result = self._read_resource_fn(server=server_name, uri=uri)
            logger.info(f" Read MCP resource: {uri}")
            return result

        except Exception as e:
            logger.error(f" Failed to read MCP resource {uri}: {e}")
            raise

    def search_mcp_resources(
        self, query: str, server: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search MCP resources by query string.

        Searches resource names and descriptions.

        Args:
            query: Search query string
            server: Filter by specific server
            limit: Maximum number of results

        Returns:
            List of matching resources

        Example:
            ```python
            results = provider.search_mcp_resources("learning", server="justjot")
            # [
            #     {"name": "Deep Learning", ...},
            #     {"name": "Machine Learning Guide", ...}
            # ]
            ```
        """
        # Get all resources
        resources = self.list_mcp_resources(server=server, limit=1000)

        # Search by query (case-insensitive substring match)
        query_lower = query.lower()
        matching = []

        for resource in resources:
            name = resource.get("name", "").lower()
            description = resource.get("description", "").lower()

            if query_lower in name or query_lower in description:
                matching.append(resource)

        # Limit results
        matching = matching[:limit]

        logger.info(f" Found {len(matching)} resources matching '{query}'")
        return matching

    # =========================================================================
    # Protocol Methods (Required by BaseMetadataProvider)
    # =========================================================================

    def get_context_for_actor(
        self,
        actor_name: str,
        query: str,
        previous_outputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return MCP resources relevant to the actor's query.

        This is called by Jotty when an actor is about to execute.
        Provides relevant MCP resources based on the query.

        Args:
            actor_name: Name of the actor requesting context
            query: Current query/goal
            previous_outputs: Previous actor outputs (optional)
            **kwargs: Additional context

        Returns:
            Dictionary with:
            - available_mcp_resources: List of available resources
            - mcp_search_results: Resources matching the query (if applicable)

        Example:
            ```python
            context = provider.get_context_for_actor(
                actor_name="ChatAssistant",
                query="Show me my ideas about machine learning"
            )
            # {
            #     'available_mcp_resources': [...],
            #     'mcp_search_results': [...]  # Filtered by "machine learning"
            # }
            ```
        """
        context = {
            "current_query": query,
            "actor_name": actor_name,
        }

        # Add list of available resources (lightweight)
        if self._list_resources_fn:
            try:
                resources = self.list_mcp_resources(limit=50)
                context["available_mcp_resources"] = [
                    {
                        "name": r.get("name"),
                        "uri": r.get("uri"),
                        "description": r.get("description", "")[:100],  # Truncate
                    }
                    for r in resources[:10]  # Top 10 for context
                ]

                # If query mentions specific terms, add search results
                search_terms = ["idea", "template", "document", "note"]
                if any(term in query.lower() for term in search_terms):
                    # Extract search query from user query
                    # Simple heuristic: use words after common phrases
                    search_query = query.lower()
                    for prefix in ["show me", "find", "list", "get"]:
                        if prefix in search_query:
                            search_query = search_query.split(prefix, 1)[1].strip()
                            break

                    # Search resources
                    if search_query and len(search_query) > 3:
                        results = self.search_mcp_resources(search_query, limit=5)
                        if results:
                            context["mcp_search_results"] = results

            except Exception as e:
                logger.warning(f" Failed to fetch MCP resources: {e}")

        return context

    def get_swarm_context(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Return global MCP context for the entire swarm.

        This is called ONCE when swarm initializes.
        Provides high-level MCP server information.

        Returns:
            Dictionary with MCP server info
        """
        return {
            "mcp_server": self.server or "all",
            "mcp_enabled": self._list_resources_fn is not None,
        }


# =============================================================================
# Helper Function: Create MCP Provider from Python Functions
# =============================================================================


def create_mcp_provider_from_functions(
    list_fn: Callable, read_fn: Callable, server: Optional[str] = None, **kwargs: Any
) -> MCPMetadataProvider:
    """
    Create MCP provider from Python functions (SIMPLE INTERFACE).

    This is the recommended way for most clients to use MCP in Jotty.

    Args:
        list_fn: Function to list MCP resources
                Signature: list_fn(server: Optional[str]) -> List[Dict]
        read_fn: Function to read MCP resource
                Signature: read_fn(server: str, uri: str) -> Dict
        server: MCP server name filter
        **kwargs: Additional arguments for MCPMetadataProvider

    Returns:
        Configured MCPMetadataProvider ready to use

    Example:
        ```python
        # Wrapper functions for your MCP client
        def my_list(server=None):
            # Call your MCP client here
            return list_mcp_resources_tool(server=server)

        def my_read(server, uri):
            return read_mcp_resource_tool(server=server, uri=uri)

        # Create provider
        mcp_provider = create_mcp_provider_from_functions(
            list_fn=my_list,
            read_fn=my_read,
            server="justjot"
        )

        # Use with Jotty
        handler = ChatHandler(mcp_provider=mcp_provider)
        ```
    """
    return MCPMetadataProvider(
        list_resources_fn=list_fn, read_resource_fn=read_fn, server=server, **kwargs
    )
