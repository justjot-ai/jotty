"""
DSPy Agent with MCP Tool Support
Enables DSPy agents to call MCP tools from JustJot
"""
import dspy
import json
import logging
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Import MCP Tool Executor with fallback
try:
    from ..integration.mcp_tool_executor import MCPToolExecutor
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.integration.mcp_tool_executor import MCPToolExecutor


class ToolUseSignature(dspy.Signature):
    """Signature for tool-using agent with chain-of-thought reasoning"""

    query = dspy.InputField(desc="User query or task")
    available_tools = dspy.InputField(desc="Available MCP tools and their descriptions")
    conversation_history = dspy.InputField(desc="Previous messages in conversation", default="")

    reasoning = dspy.OutputField(desc="Step-by-step reasoning about which tools to use and why")
    tool_calls = dspy.OutputField(
        desc="List of tool calls in JSON array format: "
             '[{"name": "tool_name", "arguments": {"arg1": "value1"}}, ...]'
    )
    response = dspy.OutputField(desc="Final response to user incorporating tool results")


class ToolResultIntegrationSignature(dspy.Signature):
    """Signature for integrating tool results into response"""

    query = dspy.InputField(desc="Original user query")
    tool_calls = dspy.InputField(desc="Tools that were called")
    tool_results = dspy.InputField(desc="Results from tool execution")
    conversation_history = dspy.InputField(desc="Previous conversation context", default="")

    final_response = dspy.OutputField(
        desc="Final response integrating tool results to answer the user's query"
    )


class DSPyMCPAgent:
    """DSPy agent that can use MCP tools for enhanced capabilities"""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: Optional[str] = None,
        base_url: str = None
    ):
        """
        Initialize DSPy agent with MCP tool support

        Args:
            name: Agent name
            description: Agent description
            system_prompt: Custom system prompt (optional)
            base_url: Base URL for JustJot API
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt or f"You are {name}. {description}"
        self.mcp_executor = MCPToolExecutor(base_url=base_url)
        self.tool_planner = dspy.ChainOfThought(ToolUseSignature)
        self.result_integrator = dspy.ChainOfThought(ToolResultIntegrationSignature)
        self.initialized = False

    async def initialize(self):
        """Discover available MCP tools"""
        if self.initialized:
            return

        await self.mcp_executor.discover_tools()
        self.initialized = True
        logger.info(f"{self.name}: Discovered {len(self.mcp_executor.available_tools)} MCP tools")

    async def execute(
        self,
        query: str,
        conversation_history: str = "",
        max_tool_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Execute agent with MCP tool support

        Args:
            query: User query
            conversation_history: Previous conversation context
            max_tool_iterations: Maximum rounds of tool calling (prevents infinite loops)

        Returns:
            Agent response with tool execution details
        """
        if not self.initialized:
            await self.initialize()

        # Format available tools
        tools_description = self.mcp_executor.format_tools_for_dspy()

        all_tool_calls = []
        all_tool_results = []
        iteration = 0

        while iteration < max_tool_iterations:
            iteration += 1

            # Build context with previous tool results
            enhanced_query = query
            if all_tool_results:
                results_summary = self._format_tool_results(all_tool_results)
                enhanced_query = f"{query}\n\nPrevious tool results:\n{results_summary}"

            # Get tool plan from DSPy
            try:
                plan_result = self.tool_planner(
                    query=enhanced_query,
                    available_tools=tools_description,
                    conversation_history=conversation_history
                )
            except Exception as e:
                logger.warning(f"DSPy planning error: {e}")
                # Return error response
                return {
                    "reasoning": f"Error in planning: {str(e)}",
                    "tool_calls": [],
                    "tool_results": [],
                    "response": f"I encountered an error while planning: {str(e)}",
                    "error": str(e)
                }

            # Parse tool calls
            tool_calls = self._parse_tool_calls(plan_result.tool_calls)

            # If no more tools to call, break
            if not tool_calls:
                break

            # Execute tools
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})

                try:
                    result = await self.mcp_executor.execute_tool(tool_name, arguments)
                    tool_results.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    tool_results.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "error": str(e),
                        "success": False
                    })

            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(tool_results)

            # Check if we have what we need
            if self._has_sufficient_results(tool_results):
                break

        # Integrate tool results into final response
        if all_tool_results:
            try:
                final_result = self.result_integrator(
                    query=query,
                    tool_calls=json.dumps(all_tool_calls, indent=2),
                    tool_results=json.dumps(all_tool_results, indent=2),
                    conversation_history=conversation_history
                )
                final_response = final_result.final_response
            except Exception as e:
                logger.warning(f"Error integrating tool results: {e}")
                final_response = plan_result.response  # Fallback to initial response
        else:
            final_response = plan_result.response

        return {
            "reasoning": plan_result.reasoning,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "response": final_response,
            "iterations": iteration
        }

    def _parse_tool_calls(self, tool_calls_str: str) -> List[Dict[str, Any]]:
        """Parse tool calls from DSPy output"""
        try:
            # Try to parse as JSON
            tool_calls = json.loads(tool_calls_str)
            if isinstance(tool_calls, list):
                return tool_calls
            elif isinstance(tool_calls, dict):
                return [tool_calls]
            else:
                return []
        except json.JSONDecodeError:
            # If not JSON, try to extract tool calls from text
            # Look for patterns like: {"name": "...", "arguments": {...}}
            import re
            matches = re.findall(r'\{[^}]+\}', tool_calls_str)
            tool_calls = []
            for match in matches:
                try:
                    tool_call = json.loads(match)
                    if "name" in tool_call:
                        tool_calls.append(tool_call)
                except Exception:
                    continue
            return tool_calls

    def _format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool results for context"""
        lines = []
        for i, result in enumerate(tool_results, 1):
            if result.get("success"):
                lines.append(f"{i}. {result['tool']}: {result['result']}")
            else:
                lines.append(f"{i}. {result['tool']}: ERROR - {result.get('error', 'Unknown error')}")
        return "\n".join(lines)

    def _has_sufficient_results(self, tool_results: List[Dict[str, Any]]) -> bool:
        """Check if we have sufficient results to answer the query"""
        # Basic heuristic: if we have at least one successful result, we're good
        return any(r.get("success") for r in tool_results)
