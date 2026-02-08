"""
API Layer for Jotty
===================

Unified entry points for all use cases.

Components:
- JottyAPI: Main unified API (chat + workflow)
- ChatAPI: Chat-specific API
- WorkflowAPI: Workflow-specific API
- OpenAPI: OpenAPI 3.0 spec generator for SDK generation
"""

from .unified import JottyAPI
from .chat_api import ChatAPI
from .workflow_api import WorkflowAPI
from .openapi import generate_openapi_spec

__all__ = [
    "JottyAPI",
    "ChatAPI",
    "WorkflowAPI",
    "generate_openapi_spec",
]
