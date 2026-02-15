"""
Tool Collection Framework

Loads collections of tools from:
- HuggingFace Hub collections
- MCP (Model Context Protocol) servers
- Local collections

Based on OAgents ToolCollection pattern.
"""

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Optional imports (will fail gracefully if not available)
try:
    from huggingface_hub import get_collection, hf_hub_download

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Hub integration disabled.")

try:
    # MCP integration (if available)
    import mcp

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.debug("MCP not available. MCP integration disabled.")


class ToolCollection:
    """
    Collection of tools that can be loaded from various sources.

    Based on OAgents ToolCollection pattern.

    Usage:
        # Load from Hub
        collection = ToolCollection.from_hub("collection-slug")

        # Load from MCP
        with ToolCollection.from_mcp(server_params) as collection:
            tools = collection.tools

        # Use with SkillsRegistry
        registry = get_skills_registry()
        registry.load_collection(collection)
    """

    def __init__(
        self, tools: List[Any], source: str = "local", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize tool collection.

        Args:
            tools: List of tools (can be SkillDefinition, Tool instances, or tool dicts)
            source: Source of collection ("hub", "mcp", "local")
            metadata: Optional metadata about the collection
        """
        self.tools = tools
        self.source = source
        self.metadata = metadata or {}
        self._converted_tools: Optional[List[Dict[str, Any]]] = None

    def __len__(self) -> int:
        """Return number of tools in collection."""
        return len(self.tools)

    def __iter__(self) -> Any:
        """Iterate over tools."""
        return iter(self.tools)

    def __getitem__(self, index: int) -> Any:
        """Get tool by index."""
        return self.tools[index]

    def to_skill_definitions(self) -> List[Any]:
        """
        Convert tools to SkillDefinition format for SkillsRegistry.

        Returns:
            List of SkillDefinition objects
        """
        from .skills_registry import SkillDefinition

        skill_definitions = []

        for tool in self.tools:
            # Convert tool to SkillDefinition
            if isinstance(tool, dict):
                # Tool dict format
                skill_def = self._dict_to_skill_definition(tool)
            elif hasattr(tool, "name") and hasattr(tool, "forward"):
                # Tool-like object (OAgents style)
                skill_def = self._tool_object_to_skill_definition(tool)
            else:
                logger.warning(f"Unknown tool format: {type(tool)}")
                continue

            if skill_def:
                skill_definitions.append(skill_def)

        return skill_definitions

    def _dict_to_skill_definition(self, tool_dict: Dict[str, Any]) -> Optional[Any]:
        """Convert tool dict to SkillDefinition."""
        from .skills_registry import SkillDefinition

        try:
            name = tool_dict.get("name", "unknown_tool")
            description = tool_dict.get("description", "")

            # Create a wrapper function for the tool
            def tool_executor(params: Dict[str, Any]) -> Dict[str, Any]:
                """Execute tool from dict."""
                try:
                    # If tool has a forward method or execute function
                    if "forward" in tool_dict:
                        result = tool_dict["forward"](**params)
                    elif "execute" in tool_dict:
                        result = tool_dict["execute"](params)
                    else:
                        return {"success": False, "error": "No execute method found"}

                    return {"success": True, "result": result}
                except Exception as e:
                    return {"success": False, "error": str(e)}

            tools = {name: tool_executor}

            return SkillDefinition(
                name=name,
                description=description,
                tools=tools,
                metadata={"source": self.source, **tool_dict.get("metadata", {})},
            )
        except Exception as e:
            logger.error(f"Failed to convert tool dict to SkillDefinition: {e}")
            return None

    def _tool_object_to_skill_definition(self, tool: Any) -> Optional[Any]:
        """Convert tool object (OAgents style) to SkillDefinition."""
        from .skills_registry import SkillDefinition

        try:
            name = getattr(tool, "name", "unknown_tool")
            description = getattr(tool, "description", "")

            # Create wrapper function
            def tool_executor(params: Dict[str, Any]) -> Dict[str, Any]:
                """Execute tool object."""
                try:
                    # Call tool's forward method
                    if hasattr(tool, "forward"):
                        result = tool.forward(**params)
                    elif hasattr(tool, "__call__"):
                        result = tool(**params)
                    else:
                        return {"success": False, "error": "No callable method found"}

                    return {"success": True, "result": result}
                except Exception as e:
                    return {"success": False, "error": str(e)}

            tools = {name: tool_executor}

            return SkillDefinition(
                name=name,
                description=description,
                tools=tools,
                metadata={"source": self.source, "tool_type": type(tool).__name__},
            )
        except Exception as e:
            logger.error(f"Failed to convert tool object to SkillDefinition: {e}")
            return None

    @classmethod
    def from_hub(
        cls, collection_slug: str, token: Optional[str] = None, trust_remote_code: bool = False
    ) -> "ToolCollection":
        """
        Load a tool collection from HuggingFace Hub.

        Args:
            collection_slug: Collection slug (e.g., "huggingface-tools/diffusion-tools")
            token: HuggingFace token (optional, uses cached token if not provided)
            trust_remote_code: Whether to trust remote code (required for loading tools)

        Returns:
            ToolCollection instance

        Example:
            collection = ToolCollection.from_hub("huggingface-tools/diffusion-tools")
        """
        if not HUGGINGFACE_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub not available. Install with: pip install huggingface_hub"
            )

        if not trust_remote_code:
            raise ValueError(
                "Loading tools from Hub requires trust_remote_code=True. "
                "Always inspect tools before loading them."
            )

        try:
            # Get collection
            collection = get_collection(collection_slug, token=token)

            # Extract space IDs (tools are typically in Spaces)
            hub_repo_ids = {item.item_id for item in collection.items if item.item_type == "space"}

            logger.info(f"Found {len(hub_repo_ids)} tools in collection {collection_slug}")

            # Load tools from each space
            tools = []
            for repo_id in hub_repo_ids:
                try:
                    tool = cls._load_tool_from_hub_space(repo_id, token, trust_remote_code)
                    if tool:
                        tools.append(tool)
                except Exception as e:
                    logger.warning(f"Failed to load tool from {repo_id}: {e}")

            return cls(
                tools=tools,
                source="hub",
                metadata={
                    "collection_slug": collection_slug,
                    "tool_count": len(tools),
                    "total_spaces": len(hub_repo_ids),
                },
            )

        except Exception as e:
            logger.error(f"Failed to load collection from Hub: {e}")
            raise

    @classmethod
    def _load_tool_from_hub_space(
        cls, repo_id: str, token: Optional[str], trust_remote_code: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Load a single tool from a Hub Space.

        Args:
            repo_id: Space repository ID
            token: HuggingFace token
            trust_remote_code: Whether to trust remote code

        Returns:
            Tool dict or None if failed
        """
        try:
            # Download tool.py from space
            tool_file = hf_hub_download(repo_id, "tool.py", token=token, repo_type="space")

            # Read and execute tool code
            tool_code = Path(tool_file).read_text()

            # Create a module and execute code
            import types

            module = types.ModuleType("dynamic_tool")
            exec(tool_code, module.__dict__)

            # Find Tool subclass
            import inspect

            tool_class = next(
                (
                    obj
                    for _, obj in inspect.getmembers(module, inspect.isclass)
                    if hasattr(obj, "name") and hasattr(obj, "forward")
                ),
                None,
            )

            if tool_class:
                tool_instance = tool_class()
                return {
                    "name": getattr(tool_instance, "name", repo_id.split("/")[-1]),
                    "description": getattr(tool_instance, "description", ""),
                    "forward": tool_instance.forward if hasattr(tool_instance, "forward") else None,
                    "tool_object": tool_instance,
                    "source": "hub",
                    "repo_id": repo_id,
                }

            return None

        except Exception as e:
            logger.warning(f"Failed to load tool from {repo_id}: {e}")
            return None

    @classmethod
    @contextmanager
    def from_mcp(
        cls, server_parameters: Any, trust_remote_code: bool = False
    ) -> ContextManager["ToolCollection"]:
        """
        Load a tool collection from an MCP server.

        Note: A separate thread will be spawned to run an asyncio event loop
        handling the MCP server.

        Args:
            server_parameters: MCP server parameters (StdioServerParameters)
            trust_remote_code: Whether to trust remote code

        Yields:
            ToolCollection instance

        Example:
            from mcp import StdioServerParameters

            server_params = StdioServerParameters(
                command="uv",
                args=["--quiet", "pubmedmcp@0.1.3"],
                env={"UV_PYTHON": "3.12", **os.environ}
            )

            with ToolCollection.from_mcp(server_params) as collection:
                tools = collection.tools
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP not available. Install with: pip install mcp")

        if not trust_remote_code:
            raise ValueError(
                "Loading tools from MCP requires trust_remote_code=True. "
                "Always inspect tools before loading them."
            )

        try:
            # Try to use MCP adapter if available
            try:
                from mcpadapt.core import MCPAdapt
                from mcpadapt.oagents_adapter import oagentsAdapter

                with MCPAdapt(server_parameters, oagentsAdapter()) as mcp_tools:
                    # Convert MCP tools to our format
                    tools = []
                    for tool in mcp_tools:
                        tools.append(
                            {
                                "name": getattr(tool, "name", "unknown"),
                                "description": getattr(tool, "description", ""),
                                "forward": tool.forward if hasattr(tool, "forward") else None,
                                "tool_object": tool,
                                "source": "mcp",
                            }
                        )

                    yield cls(tools=tools, source="mcp", metadata={"tool_count": len(tools)})

            except ImportError:
                # Fallback: manual MCP integration
                logger.warning("mcpadapt not available, using manual MCP integration")

                # Create MCP client
                import asyncio
                import threading

                from mcp import ClientSession, StdioServerParameters

                tools = []
                event_loop = None

                async def load_mcp_tools() -> Any:
                    """Load tools from MCP server."""
                    async with ClientSession(server_parameters) as session:
                        # List available tools
                        result = await session.list_tools()
                        for tool_info in result.tools:
                            tools.append(
                                {
                                    "name": tool_info.name,
                                    "description": tool_info.description or "",
                                    "source": "mcp",
                                    "mcp_tool": tool_info,
                                }
                            )

                def run_event_loop() -> None:
                    """Run asyncio event loop in thread."""
                    nonlocal event_loop
                    event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(event_loop)
                    event_loop.run_until_complete(load_mcp_tools())
                    event_loop.close()

                # Run in separate thread
                thread = threading.Thread(target=run_event_loop, daemon=True)
                thread.start()
                thread.join(timeout=30)  # Wait up to 30 seconds

                if thread.is_alive():
                    logger.error("MCP server connection timeout")
                    tools = []

                yield cls(tools=tools, source="mcp", metadata={"tool_count": len(tools)})

        except Exception as e:
            logger.error(f"Failed to load tools from MCP: {e}")
            raise

    @classmethod
    def from_local(cls, collection_path: Union[str, Path]) -> "ToolCollection":
        """
        Load a tool collection from local directory.

        Args:
            collection_path: Path to collection directory

        Returns:
            ToolCollection instance

        Example:
            collection = ToolCollection.from_local("./collections/my-tools")
        """
        collection_path = Path(collection_path)

        if not collection_path.exists():
            raise FileNotFoundError(f"Collection path not found: {collection_path}")

        tools = []

        # Look for tools in subdirectories or JSON files
        if collection_path.is_dir():
            # Check for collection.json
            collection_json = collection_path / "collection.json"
            if collection_json.exists():
                with open(collection_json) as f:
                    collection_data = json.load(f)
                    tools = collection_data.get("tools", [])
            else:
                # Load tools from subdirectories (like skills)
                from .skills_registry import SkillsRegistry

                registry = SkillsRegistry(skills_dir=str(collection_path))
                registry.init()

                # Convert skills to tools
                for skill_name, skill_def in registry.loaded_skills.items():
                    tools.append(
                        {
                            "name": skill_name,
                            "description": skill_def.description,
                            "tools": skill_def.tools,
                            "source": "local",
                        }
                    )

        return cls(
            tools=tools,
            source="local",
            metadata={"path": str(collection_path), "tool_count": len(tools)},
        )

    def save_to_local(self, output_path: Union[str, Path]) -> None:
        """
        Save collection to local directory.

        Args:
            output_path: Path to save collection
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save collection metadata
        collection_data = {"source": self.source, "metadata": self.metadata, "tools": []}

        # Convert tools to serializable format
        for tool in self.tools:
            if isinstance(tool, dict):
                # Remove non-serializable items (like functions)
                tool_dict = {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                    "source": tool.get("source", self.source),
                    "metadata": tool.get("metadata", {}),
                }
                # Don't include forward/execute functions (not serializable)
                collection_data["tools"].append(tool_dict)
            else:
                # Convert tool object to dict
                tool_dict = {
                    "name": getattr(tool, "name", "unknown"),
                    "description": getattr(tool, "description", ""),
                    "source": self.source,
                    "tool_type": type(tool).__name__,
                }
                collection_data["tools"].append(tool_dict)

        # Save to JSON
        collection_json = output_path / "collection.json"
        with open(collection_json, "w") as f:
            json.dump(collection_data, f, indent=2)

        logger.info(f"Saved collection metadata to {output_path}/collection.json")
        logger.info(
            "Note: Tool functions are not saved (not JSON serializable). "
            "Use collection.json as reference only."
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools in collection.

        Returns:
            List of tool metadata dicts
        """
        tools_list = []

        for tool in self.tools:
            if isinstance(tool, dict):
                tools_list.append(
                    {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "source": tool.get("source", self.source),
                    }
                )
            else:
                tools_list.append(
                    {
                        "name": getattr(tool, "name", "unknown"),
                        "description": getattr(tool, "description", ""),
                        "source": self.source,
                    }
                )

        return tools_list
