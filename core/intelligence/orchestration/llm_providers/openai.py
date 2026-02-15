"""
OpenAI-compatible LLM providers: OpenAI, OpenRouter, Groq.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, Callable, List

from .base import LLMProvider, LLM_MAX_OUTPUT_TOKENS
from .types import LLMResponse, TextBlock, ToolUseBlock


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (also works for OpenRouter)."""

    def __init__(self, model: str = 'gpt-4o', api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert to OpenAI format (input_schema -> parameters)."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
                }
            })
        return openai_tools

    async def call(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS
    ) -> LLMResponse:
        # Add system message
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(self._convert_messages(messages))

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
            tools=self.convert_tools(tools),
            tool_choice="auto"
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
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(self._convert_messages(messages))

        full_content = ""
        tool_calls = []

        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
            tools=self.convert_tools(tools),
            tool_choice="auto",
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Handle text content
            if delta.content:
                result = stream_callback(delta.content)
                if asyncio.iscoroutine(result):
                    await result
                full_content += delta.content

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index is not None:
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc.index]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc.index]["arguments"] += tc.function.arguments

        # Parse tool calls
        content = []
        if full_content:
            content.append(TextBlock(text=full_content))

        for tc in tool_calls:
            if tc["id"] and tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                content.append(ToolUseBlock(
                    id=tc["id"],
                    name=tc["name"],
                    input=args
                ))

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(content=content, stop_reason=stop_reason), full_content

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert messages to OpenAI format."""
        converted = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                # Handle tool results
                if isinstance(content, list):
                    tool_results = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": str(item["content"])
                            })
                    if tool_results:
                        converted.extend(tool_results)
                        continue
                converted.append({"role": "user", "content": content if isinstance(content, str) else str(content)})
            elif msg["role"] == "assistant":
                content = msg["content"]
                if isinstance(content, list):
                    # Extract text and tool calls
                    text_parts = []
                    tool_calls = []
                    for item in content:
                        if hasattr(item, 'type'):
                            if item.type == "text":
                                text_parts.append(item.text)
                            elif item.type == "tool_use":
                                tool_calls.append({
                                    "id": item.id,
                                    "type": "function",
                                    "function": {
                                        "name": item.name,
                                        "arguments": json.dumps(item.input)
                                    }
                                })
                    converted.append({
                        "role": "assistant",
                        "content": " ".join(text_parts) if text_parts else None,
                        "tool_calls": tool_calls if tool_calls else None
                    })
                else:
                    converted.append({"role": "assistant", "content": str(content)})
        return converted

    def _parse_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        content = []

        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                content.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=args
                ))

        stop_reason = "tool_use" if choice.message.tool_calls else "end_turn"
        if choice.finish_reason == "stop":
            stop_reason = "end_turn"

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            } if response.usage else None
        )

    def format_tool_result(self, tool_id: str, content: str) -> Dict:
        """OpenAI uses 'tool' role for tool results."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content
        }


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider (OpenAI-compatible)."""

    def __init__(self, model: str = 'anthropic/claude-3.5-sonnet', api_key: Optional[str] = None) -> None:
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )


class GroqProvider(OpenAIProvider):
    """Groq provider (OpenAI-compatible)."""

    def __init__(self, model: str = 'llama-3.1-70b-versatile', api_key: Optional[str] = None) -> None:
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
