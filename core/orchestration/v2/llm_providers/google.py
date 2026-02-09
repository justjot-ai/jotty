"""
Google Gemini LLM provider.
"""

import os
import asyncio
from typing import Dict, Any, Optional, Callable, List

from .base import LLMProvider, LLM_MAX_OUTPUT_TOKENS
from .types import LLMResponse, TextBlock, ToolUseBlock


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert to Google format."""
        google_tools = []
        for tool in tools:
            google_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            })
        return google_tools

    async def call(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system: str,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS
    ) -> LLMResponse:
        import google.generativeai as genai

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, str):
                contents.append({"role": role, "parts": [content]})

        # Create tool config
        tool_config = genai.types.Tool(
            function_declarations=[
                genai.types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("input_schema", {})
                )
                for t in tools
            ]
        )

        response = self.client.generate_content(
            contents,
            tools=[tool_config],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens
            )
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
        # Google streaming is more complex, fall back to non-streaming
        response = await self.call(messages, tools, system, max_tokens)
        full_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                result = stream_callback(block.text)
                if asyncio.iscoroutine(result):
                    await result
                full_content += block.text
        return response, full_content

    def _parse_response(self, response) -> LLMResponse:
        content = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content.append(TextBlock(text=part.text))
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    content.append(ToolUseBlock(
                        id=f"fc_{fc.name}",
                        name=fc.name,
                        input=dict(fc.args)
                    ))

        has_tool_calls = any(isinstance(c, ToolUseBlock) for c in content)
        stop_reason = "tool_use" if has_tool_calls else "end_turn"

        return LLMResponse(content=content, stop_reason=stop_reason)
