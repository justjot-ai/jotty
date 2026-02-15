"""
Jotty SDK - World-Class Python SDK
===================================

The official Python SDK for Jotty AI Framework.

Quick Start:
    from jotty import Jotty

    # Async usage
    client = Jotty()
    response = await client.chat("Hello!")

    # Sync usage
    from jotty import JottySync
    client = JottySync()
    response = client.chat("Hello!")

Features:
    - Chat mode: Conversational AI
    - Workflow mode: Multi-step autonomous execution
    - Streaming: Real-time response streaming
    - Events: Callback-based event handling
    - Skills: Direct skill execution
    - Agents: Direct agent invocation
    - Sessions: Cross-channel session management

For full documentation, see: https://docs.jotty.ai/sdk
"""

# Import types for convenience
from ..core.infrastructure.foundation.types.sdk_types import (
    ChannelType,
    ExecutionContext,
    ExecutionMode,
    ResponseFormat,
    SDKEvent,
    SDKEventType,
    SDKRequest,
    SDKResponse,
    SDKSession,
    SDKVoiceResponse,
)
from .client import (  # Main clients; Handles; Event emitter
    AgentHandle,
    Client,
    EventEmitter,
    Jotty,
    JottySync,
    SessionHandle,
    SkillHandle,
    SyncClient,
    VoiceHandle,
)

__all__ = [
    # Main clients
    "Jotty",
    "JottySync",
    "Client",
    "SyncClient",
    # Handles
    "SkillHandle",
    "AgentHandle",
    "SessionHandle",
    "VoiceHandle",
    # Event emitter
    "EventEmitter",
    # Types
    "ExecutionMode",
    "ChannelType",
    "SDKEventType",
    "ResponseFormat",
    "ExecutionContext",
    "SDKEvent",
    "SDKSession",
    "SDKResponse",
    "SDKVoiceResponse",
    "SDKRequest",
]

__version__ = "2.0.0"
