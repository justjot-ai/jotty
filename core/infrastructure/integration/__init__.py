"""
Integration modules for external systems (LLM providers, MCP, Lotus).

Includes:
- LLM providers (Claude, OpenAI, Groq, etc.)
- MCP (Model Context Protocol) client and tool execution
- Tool interceptor for monitoring and learning
"""

from .tool_interceptor import ToolCall, ToolCallRegistry, ToolInterceptor

__all__ = [
    "ToolInterceptor",
    "ToolCall",
    "ToolCallRegistry",
]
