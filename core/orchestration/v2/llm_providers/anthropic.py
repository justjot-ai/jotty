"""
Anthropic Claude LLM provider.
"""

import os
import asyncio
from typing import Dict, Any, Optional, Callable, List

from .base import LLMProvider, LLM_MAX_OUTPUT_TOKENS
from .types import LLMResponse, TextBlock, ToolUseBlock


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Anthropic uses input_schema format (already our default)."""
        return tools

    async def call(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=self.convert_tools(tools)
        )
        return self._parse_response(response)

    async def call_streaming(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        stream_callback: Callable[[str], Any],
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS
    ) -> tuple:
        full_content = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=self.convert_tools(tools)
        ) as stream:
            for text in stream.text_stream:
                if text:
                    result = stream_callback(text)
                    if asyncio.iscoroutine(result):
                        await result
                    full_content += text

            response = stream.get_final_message()

        return self._parse_response(response), full_content

    def _parse_response(self, response) -> LLMResponse:
        content = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(
                    id=block.id,
                    name=block.name,
                    input=block.input
                ))

        return LLMResponse(
            content=content,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            } if hasattr(response, 'usage') else None
        )
