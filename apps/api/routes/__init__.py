"""Web API route modules."""

from .agui import register_agui_routes
from .chat import register_chat_routes
from .documents import register_document_routes
from .sessions import register_sessions_routes
from .sharing import register_sharing_routes
from .system import register_system_routes
from .tools import register_tools_routes
from .voice import register_voice_routes

__all__ = [
    "register_system_routes",
    "register_chat_routes",
    "register_sessions_routes",
    "register_document_routes",
    "register_agui_routes",
    "register_tools_routes",
    "register_sharing_routes",
    "register_voice_routes",
]
