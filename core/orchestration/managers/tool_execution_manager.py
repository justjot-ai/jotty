"""
ToolExecutionManager - Manages tool execution and caching.

Extracted from conductor.py to improve maintainability.
Handles tool calls with caching via SharedScratchpad.
"""
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ToolExecutionManager:
    """
    Centralized tool execution management.

    Responsibilities:
    - Tool execution with caching
    - Error handling and helpful messages
    - Tool call statistics tracking
    """

    def __init__(self, config, metadata_tool_registry=None, shared_scratchpad=None):
        """
        Initialize tool execution manager.

        Args:
            config: JottyConfig
            metadata_tool_registry: MetadataToolRegistry instance
            shared_scratchpad: SharedScratchpad for caching
        """
        self.config = config
        self.metadata_tool_registry = metadata_tool_registry
        self.shared_scratchpad = shared_scratchpad or {}
        self.execution_count = 0
        self.cache_hits = 0

        logger.info("⚙️  ToolExecutionManager initialized")

    def call_tool_with_cache(self, tool_name: str, **kwargs) -> Any:
        """
        Call tool with caching via SharedScratchpad.

        Prevents duplicate tool calls across validation agents.

        Args:
            tool_name: Name of tool to call
            **kwargs: Parameters for tool

        Returns:
            Tool result (cached if available)
        """
        self.execution_count += 1

        # Create cache key
        cache_key = f"tool_call:{tool_name}:{json.dumps(kwargs, sort_keys=True)}"

        # Check cache
        if cache_key in self.shared_scratchpad:
            self.cache_hits += 1
            logger.debug(f"💾 Cache HIT: {tool_name}({list(kwargs.keys())})")
            return self.shared_scratchpad[cache_key]

        # Call actual tool
        logger.debug(f"📞 Calling {tool_name}({list(kwargs.keys())})")
        if self.metadata_tool_registry:
            result = self.metadata_tool_registry.call_tool(tool_name, **kwargs)
        else:
            result = {"error": "No metadata_tool_registry available"}

        # Store in cache
        self.shared_scratchpad[cache_key] = result

        return result

    def build_helpful_error_message(
        self,
        tool_name: str,
        tool_info: Dict,
        error: Exception,
        io_manager: Any = None
    ) -> str:
        """
        Build helpful error message when tool call fails.

        Shows available data and how to fix the issue.

        Args:
            tool_name: Name of tool that failed
            tool_info: Tool metadata
            error: Exception that occurred
            io_manager: Optional IOManager for showing available data

        Returns:
            Helpful error message string
        """
        error_str = str(error)
        msg_parts = [f"❌ {tool_name}() failed: {error_str}"]

        # If missing parameters, show available data
        if 'missing' in error_str.lower() or 'required' in error_str.lower():
            msg_parts.append("\n📦 AVAILABLE DATA (IOManager):")

            if io_manager:
                all_outputs = io_manager.get_all_outputs()
                for actor_name, output in all_outputs.items():
                    if hasattr(output, 'output_fields'):
                        fields = list(output.output_fields.keys()) if isinstance(output.output_fields, dict) else []
                        msg_parts.append(f"  • {actor_name}: {fields}")
            else:
                msg_parts.append("  (No IOManager available)")

            msg_parts.append("\n💡 TIP: You can provide parameters explicitly in your tool call.")
            msg_parts.append(f"   Example: {tool_name}(param_name=value)")

        return "\n".join(msg_parts)

    def build_enhanced_tool_description(self, tool_name: str, tool_info: Dict) -> str:
        """
        Build enhanced tool description for LLM reasoning.

        Includes parameters, when to use, auto-resolution hints.

        Args:
            tool_name: Name of the tool
            tool_info: Tool metadata

        Returns:
            Enhanced description string
        """
        parts = []

        # Base description
        if 'description' in tool_info:
            parts.append(tool_info['description'])

        # Parameters
        signature = tool_info.get('signature', {})
        params = signature.get('parameters', {})
        if params:
            parts.append("\n\nParameters:")
            for param_name, param_info in params.items():
                param_type = param_info.get('annotation', 'Any')
                required = param_info.get('required', True)
                req_str = "REQUIRED" if required else "optional"
                parts.append(f"  • {param_name} ({param_type}) - {req_str}")

        # Auto-resolution hint
        parts.append("\n💡 Parameters are auto-resolved from previous actor outputs when possible.")

        return " ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tool execution statistics.

        Returns:
            Dict with execution metrics
        """
        cache_hit_rate = (self.cache_hits / self.execution_count) if self.execution_count > 0 else 0.0
        return {
            "total_executions": self.execution_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.shared_scratchpad)
        }

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_count = 0
        self.cache_hits = 0
        logger.debug("ToolExecutionManager stats reset")

    def clear_cache(self):
        """Clear tool execution cache."""
        self.shared_scratchpad.clear()
        logger.debug("Tool execution cache cleared")
