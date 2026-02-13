"""
LLM Providers for ChatExecutor
==================================

Supports: Anthropic, OpenAI, OpenRouter, Groq, Google, JottyClaude CLI.
"""

from .types import (
    ToolResult,
    ExecutionResult,
    StreamEvent,
    LLMResponse,
    ToolUseBlock,
    TextBlock,
)
from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider, OpenRouterProvider, GroqProvider
from .google import GoogleProvider
from .adapter import JottyClaudeProviderAdapter
from .factory import create_provider, auto_detect_provider

__all__ = [
    # Types
    'ToolResult', 'ExecutionResult', 'StreamEvent',
    'LLMResponse', 'ToolUseBlock', 'TextBlock',
    # Base
    'LLMProvider',
    # Providers
    'AnthropicProvider', 'OpenAIProvider', 'OpenRouterProvider',
    'GroqProvider', 'GoogleProvider', 'JottyClaudeProviderAdapter',
    # Factory
    'create_provider', 'auto_detect_provider',
]
