"""
MODES Layer - Execution Modes (Chat, API, Workflow)
"""
from Jotty.core.agents.chat_assistant import ChatAssistant
from Jotty.core.agents.auto_agent import AutoAgent
from Jotty.core.api import JottyAPI, ChatAPI, WorkflowAPI, generate_openapi_spec

try:
    from Jotty.core.integration.mcp_tool_executor import MCPToolExecutor
except ImportError:
    MCPToolExecutor = None

__all__ = ["ChatAssistant", "AutoAgent", "JottyAPI", "ChatAPI", "WorkflowAPI", "generate_openapi_spec", "MCPToolExecutor"]
