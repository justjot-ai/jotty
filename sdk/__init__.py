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

# ── Re-exports for apps/ layer (so apps never import from core directly) ──

# Routing & orchestration
from ..core.interface.api.mode_router import ModeRouter, get_mode_router
from ..core.capabilities.registry.unified_registry import get_unified_registry
from ..core.capabilities.registry.skills_registry import get_skills_registry
from ..core.intelligence.orchestration import Orchestrator

# API middleware
from ..core.infrastructure.utils.api_middleware import (
    RateLimiter,
    make_auth_middleware,
    make_rate_limit_middleware,
)

# MCP integration
from ..core.infrastructure.integration.mcp_client import (
    MCPClient,
    call_justjot_mcp_tool,
)

# Agent types used by apps
from ..core.intelligence.reasoning.agents.model_chat_agent import ModelChatAgent
from ..core.infrastructure.foundation.direct_anthropic_lm import (
    DirectAnthropicLM,
    get_anthropic_client_kwargs,
)

# Swarm types used by apps
from ..core.intelligence.orchestration.swarms.coding_swarm import CodingConfig, CodingSwarm

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
    # Routing & orchestration
    "ModeRouter",
    "get_mode_router",
    "get_unified_registry",
    "get_skills_registry",
    "Orchestrator",
    # API middleware
    "RateLimiter",
    "make_auth_middleware",
    "make_rate_limit_middleware",
    # MCP integration
    "MCPClient",
    "call_justjot_mcp_tool",
    # Agent types
    "ModelChatAgent",
    "DirectAnthropicLM",
    "get_anthropic_client_kwargs",
    # Swarm types
    "CodingConfig",
    "CodingSwarm",
]

__version__ = "2.0.0"
