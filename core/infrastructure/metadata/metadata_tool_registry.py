"""
MetadataToolRegistry: Automatic discovery and registration of metadata tools.

# A-TEAM UNANIMOUS DESIGN (Vote: 5-0)
# LLM-driven tool selection
# Automatic discovery via @jotty_method decorator
# Rich metadata for LLM guidance
# Tool caching and monitoring

Design Philosophy:
- Discovers ALL @jotty_method decorated methods via introspection
- Extracts rich metadata (desc, when, params, returns)
- Creates LLM-friendly tool catalog
- Provides unified interface for tool calling
- Handles caching transparently
"""

import inspect
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetadataToolRegistry:
    """
        Discovers and manages metadata tools for Jotty actors.

    # A-TEAM DESIGN: Automatic discovery + LLM-driven selection

        Features:
        1. **Automatic Discovery**: Finds all @jotty_method decorated methods
        2. **Rich Metadata**: Extracts desc, when, params, returns for LLM
        3. **Tool Catalog**: Generates LLM-friendly documentation
        4. **Unified Calling**: Single interface for all tool calls
        5. **Smart Caching**: Caches results based on decorator config
        6. **Monitoring**: Tracks tool usage for optimization

        Usage:
            # User creates metadata
            metadata = MyMetadata(data_dir="/path/to/data")

            # ReVal creates registry (automatic!)
            registry = MetadataToolRegistry(metadata)

            # ReVal discovers tools
            print(registry.get_tool_catalog_for_llm())

            # Actors call tools
            result = registry.call_tool("get_business_terms")
    """

    def __init__(self, metadata_instance: Any) -> None:
        """
        Initialize tool registry with metadata instance.

        Automatically discovers all @jotty_method decorated methods.

        Args:
            metadata_instance: Instance of user's metadata class
                               (should inherit from BaseMetadataProvider)
        """
        self.metadata = metadata_instance
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, Any] = {}
        self._usage_stats: Dict[str, int] = {}

        logger.info(f" Discovering tools from {type(metadata_instance).__name__}...")
        self._discover_tools()
        logger.info(f" Discovered {len(self.tools)} ReVal tools")

    def _discover_tools(self) -> Any:
        """
                Discover all @jotty_method decorated methods via introspection.

        # A-TEAM DESIGN: Zero configuration, automatic discovery

                Process:
                1. Iterate through all attributes of metadata instance
                2. Check for _jotty_meta metadata (added by @jotty_method)
                3. Extract rich metadata (desc, when, params, returns, cache)
                4. Store as callable tool with metadata
        """
        discovered_count = 0

        for attr_name in dir(self.metadata):
            # Skip private/protected methods
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(self.metadata, attr_name)

                # A-TEAM FIX: Check for _jotty_meta (from metadata_protocol.py)
                if hasattr(attr, "_jotty_meta"):
                    tool_info = attr._jotty_meta
                    tool_name = attr_name  # Use the actual method name

                    # A-TEAM CRITICAL FIX: Extract for_architect/for_auditor flags!
                    for_architect = getattr(attr, "_jotty_for_architect", False)
                    for_auditor = getattr(attr, "_jotty_for_auditor", False)

                    # Store tool with rich metadata
                    self.tools[tool_name] = {
                        "name": tool_name,
                        "desc": tool_info["desc"],
                        "when": tool_info["when"],
                        "params": tool_info.get("params", {}),  # May not exist
                        "returns": tool_info.get("returns", "Data from metadata"),  # May not exist
                        "cache": tool_info["cache"],
                        "callable": attr,  # Bound method
                        "signature": self._extract_signature(attr),
                        "for_architect": for_architect,  # A-TEAM: Val agent flag
                        "for_auditor": for_auditor,  # A-TEAM: Val agent flag
                    }

                    logger.debug(
                        f" {tool_name}(): {tool_info['desc'][:60]}... (architect={for_architect}, auditor={for_auditor})"
                    )
                    discovered_count += 1

            except Exception as e:
                logger.warning(f" Failed to inspect {attr_name}: {e}")

        if discovered_count == 0:
            logger.warning(
                f" No @jotty_method decorated methods found in {type(self.metadata).__name__}. "
                f"Did you forget to add @jotty_method decorator?"
            )

    def _extract_signature(self, method: Callable) -> Dict[str, Any]:
        """
        Extract method signature for parameter validation.

        Args:
            method: Method to inspect

        Returns:
            Dict with parameter names, types, defaults
        """
        try:
            sig = inspect.signature(method)
            params = {}

            for param_name, param in sig.parameters.items():
                # Skip self, *args, and **kwargs - these are not named parameters
                if param_name == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    # *args and **kwargs are NOT required named parameters
                    continue

                params[param_name] = {
                    "annotation": (
                        param.annotation if param.annotation != inspect.Parameter.empty else Any
                    ),
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty,
                }

            return {
                "parameters": params,
                "return_annotation": (
                    sig.return_annotation
                    if sig.return_annotation != inspect.Signature.empty
                    else Any
                ),
            }

        except Exception as e:
            logger.warning(f" Failed to extract signature: {e}")
            return {"parameters": {}, "return_annotation": Any}

    def get_tool_catalog_for_llm(self) -> str:
        """
                Generate LLM-friendly tool catalog.

        # A-TEAM DESIGN: Rich context for LLM decision-making

                Returns:
                    Formatted string describing all available tools

                Example Output:
                    Available Metadata Tools:

                    Tool: get_business_terms()
                      Description: Get all business term definitions with SQL mappings
                      When to use: Agent needs to understand business terminology
                      Returns: Dict of term -> {definition, sql_column, examples}

                    Tool: get_table_schema(table_name)
                      Description: Get schema information for a specific table
                      When to use: Agent needs table structure, columns, types
                      Parameters:
                        - table_name: Name of the table to query
                      Returns: Dict with columns, types, primary_key, foreign_keys
        """
        if not self.tools:
            return "No metadata tools available."

        catalog = " Available Metadata Tools:\n\n"

        for tool_name, info in sorted(self.tools.items()):
            # Tool signature
            params_str = ""
            if info["signature"]["parameters"]:
                param_names = list(info["signature"]["parameters"].keys())
                params_str = f"({', '.join(param_names)})"

            catalog += f" Tool: {tool_name}{params_str}\n"
            catalog += f"   Description: {info['desc']}\n"
            catalog += f"   When to use: {info['when']}\n"

            # Parameters
            if info["params"]:
                catalog += f"   Parameters:\n"
                for param_name, param_desc in info["params"].items():
                    param_sig = info["signature"]["parameters"].get(param_name, {})
                    required = " (required)" if param_sig.get("required", False) else " (optional)"
                    catalog += f"     • {param_name}: {param_desc}{required}\n"

            # Return value
            catalog += f"   Returns: {info['returns']}\n"

            # Usage stats (if available)
            if tool_name in self._usage_stats:
                catalog += f"   Usage: Called {self._usage_stats[tool_name]} times\n"

            catalog += "\n"

        return catalog

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
                Call a tool by name with parameters.

        # A-TEAM DESIGN: Unified interface + smart caching

                Args:
                    tool_name: Name of the tool to call
                    **kwargs: Parameters to pass to the tool

                Returns:
                    Tool result (cached if cache=True in decorator)

                Raises:
                    ValueError: If tool not found
                    TypeError: If required parameters missing

                Example:
                    # No parameters
                    terms = registry.call_tool("get_business_terms")

                    # With parameters
                    schema = registry.call_tool("get_table_schema", table_name="users")
        """
        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            raise ValueError(f" Tool '{tool_name}' not found. " f"Available tools: {available}")

        tool = self.tools[tool_name]

        # Check for cached result
        if tool["cache"]:
            cache_key = self._make_cache_key(tool_name, kwargs)
            if cache_key in self._cache:
                logger.debug(f" Cache hit for {tool_name}()")
                self._usage_stats[tool_name] = self._usage_stats.get(tool_name, 0) + 1
                return self._cache[cache_key]

        # Validate parameters
        self._validate_parameters(tool, kwargs)

        # Call tool
        try:
            logger.debug(
                f" Calling {tool_name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
            )
            result = tool["callable"](**kwargs)

            # Cache if enabled
            if tool["cache"]:
                cache_key = self._make_cache_key(tool_name, kwargs)
                self._cache[cache_key] = result

            # Track usage
            self._usage_stats[tool_name] = self._usage_stats.get(tool_name, 0) + 1

            logger.debug(f" {tool_name}() returned {type(result).__name__}")
            return result

        except Exception as e:
            logger.error(f" Tool {tool_name}() failed: {e}")
            raise

    def _make_cache_key(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Create cache key from tool name and parameters."""
        # Simple string-based key (can be enhanced with hashing if needed)
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{tool_name}_{params_str}" if params_str else tool_name

    def _validate_parameters(self, tool: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        """
        Validate that required parameters are provided.

        Args:
            tool: Tool metadata
            kwargs: Provided parameters

        Raises:
            TypeError: If required parameters are missing
        """
        signature = tool["signature"]
        required_params = [
            name for name, info in signature["parameters"].items() if info["required"]
        ]

        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise TypeError(f" Tool {tool['name']}() missing required parameters: {missing}")

    def list_tools(self) -> List[str]:
        """
        List all available tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool metadata dict or None if not found
        """
        return self.tools.get(tool_name)

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics for all tools.

        Returns:
            Dict mapping tool_name -> call_count
        """
        return self._usage_stats.copy()

    def clear_cache(self) -> None:
        """Clear all cached tool results."""
        self._cache.clear()
        logger.info(f" Cleared tool cache")

    def __repr__(self) -> str:
        return (
            f"MetadataToolRegistry("
            f"tools={len(self.tools)}, "
            f"cached={len(self._cache)}, "
            f"total_calls={sum(self._usage_stats.values())}"
            f")"
        )


# Export public API
__all__ = [
    "MetadataToolRegistry",
]
