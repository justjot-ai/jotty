"""
ToolDiscoveryManager - Manages tool auto-discovery and registration.

Extracted from conductor.py to improve maintainability.
Handles discovery of @jotty_method decorated tools.
"""
import logging
from typing import List, Dict, Any
import dspy

logger = logging.getLogger(__name__)


class ToolDiscoveryManager:
    """
    Centralized tool discovery management.

    Responsibilities:
    - Auto-discovery of @jotty_method tools from metadata providers
    - DSPy tool wrapper creation
    - Tool filtering for Planner/Reviewer
    - Enhanced tool descriptions for LLM reasoning
    """

    def __init__(self, config, metadata_tool_registry=None):
        """
        Initialize tool discovery manager.

        Args:
            config: JottyConfig
            metadata_tool_registry: MetadataToolRegistry instance
        """
        self.config = config
        self.metadata_tool_registry = metadata_tool_registry
        self.discovered_tools = []

        logger.info("🔧 ToolDiscoveryManager initialized")

    def discover_tools(self, conductor_ref=None) -> List[Any]:
        """
        Discover all @jotty_method tools and wrap as DSPy tools.

        NOTE: This method delegates to conductor for now to use its context.
        Future enhancement: Make fully self-contained.

        Args:
            conductor_ref: Reference to conductor (for accessing shared context, io_manager, etc.)

        Returns:
            List of dspy.Tool objects
        """
        if not self.metadata_tool_registry:
            logger.warning("⚠️  No metadata_tool_registry found")
            return []

        if not conductor_ref:
            logger.warning("⚠️  No conductor reference provided")
            return []

        # Delegate to conductor's method for now
        # (Conductor has access to shared_context, io_manager, etc.)
        if hasattr(conductor_ref, '_get_auto_discovered_dspy_tools'):
            self.discovered_tools = conductor_ref._get_auto_discovered_dspy_tools()
            return self.discovered_tools

        return []

    def filter_tools_for_planner(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter tools appropriate for Planner (pre-execution exploration).

        Args:
            all_tools: All available tools

        Returns:
            Filtered list of tools for Planner
        """
        planner_tools = [
            t for t in all_tools
            if getattr(t, '_jotty_for_architect', False)
        ]
        logger.debug(f"Filtered {len(planner_tools)}/{len(all_tools)} tools for Planner")
        return planner_tools

    def filter_tools_for_reviewer(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter tools appropriate for Reviewer (post-execution validation).

        Args:
            all_tools: All available tools

        Returns:
            Filtered list of tools for Reviewer
        """
        reviewer_tools = [
            t for t in all_tools
            if getattr(t, '_jotty_for_auditor', False)
        ]
        logger.debug(f"Filtered {len(reviewer_tools)}/{len(all_tools)} tools for Reviewer")
        return reviewer_tools

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict with tool metadata
        """
        if self.metadata_tool_registry:
            return self.metadata_tool_registry.get_tool_info(tool_name)
        return {}

    def list_tools(self) -> List[str]:
        """
        List all discovered tool names.

        Returns:
            List of tool names
        """
        if self.metadata_tool_registry:
            return self.metadata_tool_registry.list_tools()
        return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tool discovery statistics.

        Returns:
            Dict with discovery metrics
        """
        return {
            "total_tools_discovered": len(self.discovered_tools),
            "planner_tools": len(self.filter_tools_for_planner(self.discovered_tools)),
            "reviewer_tools": len(self.filter_tools_for_reviewer(self.discovered_tools))
        }
