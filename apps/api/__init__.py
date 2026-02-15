"""
Jotty Web Package
=================

Web UI and API interface for Jotty, providing:
- FastAPI REST endpoints
- WebSocket streaming
- LibreChat-style chat interface
"""

from .api import create_app
from .jotty_api import JottyAPI
from .websocket import WebSocketManager

__all__ = [
    "create_app",
    "JottyAPI",
    "WebSocketManager",
]
