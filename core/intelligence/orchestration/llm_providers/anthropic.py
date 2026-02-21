"""
Anthropic Claude LLM provider.

Respects ANTHROPIC_BASE_URL (e.g. claude-code-router) so all v2 orchestration
LLM calls can be routed through CCR when configured.
"""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

from .base import LLM_MAX_OUTPUT_TOKENS, LLMProvider
from .types import LLMResponse, TextBlock, ThinkingBlock, ToolUseBlock


def _get_client_kwargs(api_key: Optional[str] = None) -> Any:
    from Jotty.core.infrastructure.foundation.direct_anthropic_lm import (  # type: ignore[import-not-found, import]
        get_anthropic_client_kwargs,
    )

    return get_anthropic_client_kwargs(api_key=api_key)


_MODEL_MAX_OUTPUT: Dict[str, int] = {
    "claude-3-haiku-20240307": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-opus-20240229": 4096,
}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None
    ) -> None:
        self.model = model
        self._client_kwargs = _get_client_kwargs(api_key)
        self.api_key = self._client_kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[Anthropic] = None  # type: ignore[name-defined]

    def _cap_max_tokens(self, max_tokens: int) -> int:
        limit = _MODEL_MAX_OUTPUT.get(self.model)
        return min(max_tokens, limit) if limit else max_tokens

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(**self._client_kwargs)
        return self._client

    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Anthropic uses input_schema format (already our default)."""
        return tools

    async def call(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._cap_max_tokens(max_tokens),
            "system": system,
            "messages": messages,
            "tools": self.convert_tools(tools),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.messages.create(**kwargs)
        return self._parse_response(response)

    async def call_streaming(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        stream_callback: Callable[[str], Any],
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        thinking_callback: Optional[Callable[[str], Any]] = None,
    ) -> tuple:
        full_content = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self._cap_max_tokens(max_tokens),
            system=system,
            messages=messages,
            tools=self.convert_tools(tools),
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta" and delta.text:
                        result = stream_callback(delta.text)
                        if asyncio.iscoroutine(result):
                            await result
                        full_content += delta.text
                    elif delta.type == "thinking_delta" and thinking_callback:
                        thinking_text = getattr(delta, "thinking", "")
                        if thinking_text:
                            result = thinking_callback(thinking_text)
                            if asyncio.iscoroutine(result):
                                await result

            response = stream.get_final_message()

        return self._parse_response(response), full_content

    def _parse_response(self, response: Any) -> LLMResponse:
        content = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "thinking":
                content.append(ThinkingBlock(thinking=block.thinking))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))  # type: ignore[arg-type]

        return LLMResponse(
            content=content,
            stop_reason=response.stop_reason,
            usage=(
                {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                if hasattr(response, "usage")
                else None
            ),
        )
