"""
Abstract base class for LLM providers with tool calling support.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List

from .types import LLMResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers with tool calling support."""

    @abstractmethod
    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert unified tool format to provider-specific format."""
        pass

    @abstractmethod
    async def call(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Call LLM API and return unified response."""
        pass

    @abstractmethod
    async def call_streaming(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        stream_callback: Callable[[str], Any],
        max_tokens: int = 4096
    ) -> tuple:
        """Call LLM API with streaming, return (response, streamed_content)."""
        pass

    def format_tool_result(self, tool_id: str, content: str) -> Dict:
        """Format tool result for next message."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content
        }
