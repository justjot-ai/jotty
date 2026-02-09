"""
API Layer for Jotty
===================

Primary entry point: ModeRouter - unified request routing with ExecutionContext.
All entry points (CLI, Gateway, Web, SDK) flow through ModeRouter.

Components:
- ModeRouter: Canonical execution path (chat + workflow + skill + agent)
- OpenAPI: Auto-generated spec from sdk_types.py dataclasses

Legacy (kept for backward compatibility, prefer ModeRouter):
- JottyAPI: Old unified API (uses SwarmManager/UseCases)
- ChatAPI: Old chat-specific API
- WorkflowAPI: Old workflow-specific API
"""

from .mode_router import ModeRouter, get_mode_router, RouteResult
from .openapi import generate_openapi_spec

# Legacy imports (backward compatibility)
from .unified import JottyAPI
from .chat_api import ChatAPI
from .workflow_api import WorkflowAPI

__all__ = [
    # Primary
    "ModeRouter",
    "get_mode_router",
    "RouteResult",
    "generate_openapi_spec",
    # Legacy
    "JottyAPI",
    "ChatAPI",
    "WorkflowAPI",
]
