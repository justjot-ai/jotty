"""
LLM Providers for ChatExecutor
==================================

Supports: Anthropic, OpenAI, OpenRouter, Groq, Google.
"""

from .anthropic import AnthropicProvider
from .base import LLMProvider
from .factory import auto_detect_provider, create_provider
from .google import GoogleProvider
from .openai import GroqProvider, OpenAIProvider, OpenRouterProvider
from .types import LLMExecutionResult, LLMResponse, StreamEvent, TextBlock, ToolResult, ToolUseBlock

__all__ = [
    # Types
    "ToolResult",
    "LLMExecutionResult",
    "StreamEvent",
    "LLMResponse",
    "ToolUseBlock",
    "TextBlock",
    # Base
    "LLMProvider",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "GoogleProvider",
    # Factory
    "create_provider",
    "auto_detect_provider",
]
