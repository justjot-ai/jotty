"""
Jotty HTTP Server

Production-ready HTTP server for Jotty agents.
Provides ready-to-use endpoints for chat, workflow, and agent management.

Minimal client integration:
    from Jotty.server import JottyServer
    
    server = JottyServer(
        agents=[...],
        config=JottyConfig(...)
    )
    
    # That's it! Server is ready with all endpoints
    server.run(port=8080)
"""

from .http_server import JottyHTTPServer, JottyServerConfig
from .middleware import AuthMiddleware, LoggingMiddleware, ErrorMiddleware
from .formats import SSEFormatter, useChatFormatter, OpenAIFormatter, AnthropicFormatter

__all__ = [
    "JottyHTTPServer",
    "JottyServerConfig",
    "AuthMiddleware",
    "LoggingMiddleware",
    "ErrorMiddleware",
    "SSEFormatter",
    "useChatFormatter",
    "OpenAIFormatter",
    "AnthropicFormatter",
]
