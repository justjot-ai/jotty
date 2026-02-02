"""
UnifiedExecutor - World-Class Executor Combining LeanExecutor + ChatAssistant V2
=================================================================================

Uses native LLM tool calling for ALL decisions:
- Input decisions (web search, file read)
- Output decisions (save docx, send telegram)
- Visualization decisions (which section type)

Supports multiple providers:
- Anthropic (Claude)
- OpenAI (GPT-4)
- OpenRouter (any model)
- Google (Gemini)
- Groq (Llama, Mixtral)

Key advantages over DSPy signatures:
1. LLMs are specifically trained for tool use
2. First-class streaming support with tool events
3. Single LLM call with tool_choice="auto"
4. Lower latency (no DSPy overhead)

Architecture:
```
User Query
    ↓
┌─────────────────────────────────────────────┐
│  UNIFIED EXECUTOR                           │
│                                             │
│  LLM API call with ALL tools:               │
│  ├── Input Tools (web_search, file_read)    │
│  ├── Output Tools (save_docx, telegram)     │
│  ├── Visualization Tools (70+ sections)     │
│  └── Skills Tools (from SkillsRegistry)     │
│                                             │
│  tool_choice = "auto"  (LLM decides)        │
│  max_steps = 10  (multi-step reasoning)     │
└─────────────────────────────────────────────┘
    ↓
Streaming Response + Tool Executions + A2UI Sections
```
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from .tool_generator import UnifiedToolGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result from unified execution."""
    success: bool
    content: str
    tool_results: List[ToolResult] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    output_path: Optional[str] = None
    output_format: str = "text"
    error: Optional[str] = None
    steps_taken: List[str] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None


@dataclass
class StreamEvent:
    """Event emitted during streaming execution."""
    type: str  # 'text', 'tool_start', 'tool_end', 'section', 'complete', 'error'
    data: Any


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: List[Any]  # Content blocks (text, tool_use)
    stop_reason: str  # 'end_turn', 'tool_use', 'stop', etc.
    usage: Optional[Dict[str, int]] = None


@dataclass
class ToolUseBlock:
    """Unified tool use block."""
    id: str
    name: str
    input: Dict[str, Any]
    type: str = "tool_use"


@dataclass
class TextBlock:
    """Unified text block."""
    text: str
    type: str = "text"


# =============================================================================
# System Prompt
# =============================================================================

# Base output format mappings (non-visualization tools)
BASE_OUTPUT_FORMAT_TOOLS = {
    "auto": None,  # LLM decides
    "pdf": "save_pdf",
    "docx": "save_docx",
    "doc": "save_docx",
    "word": "save_docx",
    "slides": "save_slides",
    "pptx": "save_slides",
    "powerpoint": "save_slides",
    "presentation": "save_slides",
    "telegram": "send_telegram",
    "justjot": "save_to_justjot",
    "text": "return_text",
    "markdown": "return_text",
}


def get_output_format_tools() -> Dict[str, Optional[str]]:
    """
    Get output format to tool mapping, dynamically including JustJot section types.

    Base formats (pdf, docx, slides, telegram) are always available.
    Visualization formats are loaded from JustJot section registry.
    """
    # Start with base mappings
    tools = BASE_OUTPUT_FORMAT_TOOLS.copy()

    # Add visualization tools from JustJot section registry
    try:
        from ...ui.schema_validator import schema_registry

        for section_type in schema_registry.list_sections():
            # Convert section-type to format name (kebab-case to simple name)
            # e.g., "kanban-board" -> "kanban", "data-table" -> "table"
            format_name = section_type.replace('-board', '').replace('-', '_')

            # Also add the original section type as format
            tool_name = f"return_{section_type.replace('-', '_')}"

            # Add both variations
            if format_name not in tools:
                tools[format_name] = tool_name
            if section_type not in tools:
                tools[section_type] = tool_name

        logger.debug(f"Loaded {len(tools)} output format mappings")

    except Exception as e:
        logger.warning(f"Failed to load section types from registry: {e}")
        # Fallback to hardcoded visualization tools
        tools.update({
            "kanban": "return_kanban_board",
            "chart": "return_chart",
            "mermaid": "return_mermaid",
            "table": "return_data_table",
            "image": "return_image",
            "file_download": "return_file_download",
        })

    return tools


UNIFIED_SYSTEM_PROMPT = """You are Jotty, a world-class AI assistant with access to powerful tools.

## Available Capabilities

### Input Tools (gather information)
- web_search: Search the web for current information (news, prices, weather, recent events)
- file_read: Read content from local files
- fetch_url: Fetch and parse web pages

### Output Tools (save/send results)
- save_docx: Save as Word document
- save_pdf: Save as PDF document
- save_slides: Create PowerPoint presentation
- send_telegram: Send via Telegram
- save_to_justjot: Save as JustJot.ai idea

### Visualization Tools (display results)
- return_text: Plain text/markdown (DEFAULT for most responses)
- return_kanban: Kanban board for task tracking
- return_chart: Charts and graphs (bar, line, pie)
- return_mermaid: Diagrams and flowcharts
- return_data_table: Structured data tables
- ... and many more section types

## Guidelines

1. **Use web_search** when user asks about:
   - Current events, news, recent developments
   - Prices, weather, stock data
   - "Latest", "today", "recent", "current"
   - Anything requiring up-to-date information

2. **Choose visualization** based on content type:
   - Tasks/items → return_kanban
   - Data/metrics → return_chart
   - Processes/flows → return_mermaid
   - Structured data → return_data_table
   - Everything else → return_text

3. **Use output tools** when user explicitly asks to:
   - "Save as PDF/Word/document"
   - "Create a presentation"
   - "Send via Telegram"
   - "Save to JustJot"

4. **Multi-step reasoning**: You can use multiple tools in sequence.
   Example: web_search → generate content → save_docx

5. **Always be helpful** - use tools proactively to provide the best answer.
   Don't ask permission to search - just do it if it helps answer the question.
"""


# =============================================================================
# Provider Abstraction
# =============================================================================

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
        max_tokens: int = 4096
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
        max_tokens: int = 4096
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


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (also works for OpenRouter)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert to OpenAI format (input_schema → parameters)."""
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
        max_tokens: int = 4096
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
        max_tokens: int = 4096
    ) -> tuple:
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(self._convert_messages(messages))

        full_content = ""
        tool_calls = []
        current_tool_call = None

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
                import json
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
                                import json
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

    def _parse_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        content = []

        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))

        if choice.message.tool_calls:
            import json
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

    def __init__(self, model: str = "anthropic/claude-3.5-sonnet", api_key: Optional[str] = None):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )


class GroqProvider(OpenAIProvider):
    """Groq provider (OpenAI-compatible)."""

    def __init__(self, model: str = "llama-3.1-70b-versatile", api_key: Optional[str] = None):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )


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
        max_tokens: int = 4096
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
        max_tokens: int = 4096
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


# =============================================================================
# Provider Factory
# =============================================================================

def create_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMProvider:
    """
    Create LLM provider instance.

    Args:
        provider: Provider name ('anthropic', 'openai', 'openrouter', 'groq', 'google')
        model: Model name (uses provider default if not specified)
        api_key: API key (uses environment variable if not specified)

    Returns:
        LLMProvider instance
    """
    provider = provider.lower()

    default_models = {
        'anthropic': 'claude-sonnet-4-20250514',
        'openai': 'gpt-4o',
        'openrouter': 'anthropic/claude-3.5-sonnet',
        'groq': 'llama-3.1-70b-versatile',
        'google': 'gemini-2.0-flash-exp'
    }

    model = model or default_models.get(provider, 'gpt-4o')

    if provider == 'anthropic':
        return AnthropicProvider(model=model, api_key=api_key)
    elif provider == 'openai':
        return OpenAIProvider(model=model, api_key=api_key)
    elif provider == 'openrouter':
        return OpenRouterProvider(model=model, api_key=api_key)
    elif provider == 'groq':
        return GroqProvider(model=model, api_key=api_key)
    elif provider == 'google':
        return GoogleProvider(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: anthropic, openai, openrouter, groq, google")


def auto_detect_provider() -> tuple:
    """
    Auto-detect available provider based on environment variables.

    Returns:
        Tuple of (provider_name, LLMProvider instance)
    """
    providers_to_check = [
        ('anthropic', 'ANTHROPIC_API_KEY'),
        ('openai', 'OPENAI_API_KEY'),
        ('openrouter', 'OPENROUTER_API_KEY'),
        ('groq', 'GROQ_API_KEY'),
        ('google', 'GOOGLE_API_KEY'),
    ]

    for provider_name, env_key in providers_to_check:
        if os.environ.get(env_key):
            try:
                return provider_name, create_provider(provider_name)
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")
                continue

    raise RuntimeError("No LLM provider available. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY")


# =============================================================================
# Unified Executor
# =============================================================================

class UnifiedExecutor:
    """
    World-class unified executor combining LeanExecutor + ChatAssistant V2.

    Uses native LLM tool calling for ALL decisions:
    - Input decisions (web search, file read)
    - Output decisions (save docx, send telegram)
    - Visualization decisions (which section type)

    Supports multiple providers: Anthropic, OpenAI, OpenRouter, Groq, Google

    Example:
        # Auto-detect provider
        executor = UnifiedExecutor()

        # Specific provider
        executor = UnifiedExecutor(provider='openai', model='gpt-4o')

        result = await executor.execute("Research AI trends and create a presentation")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        enabled_tools: Optional[List[str]] = None,
        output_format: str = "auto",
        max_steps: int = 10,
        tool_timeout: float = 30.0
    ):
        """
        Initialize unified executor.

        Args:
            provider: LLM provider ('anthropic', 'openai', 'openrouter', 'groq', 'google')
                      If None, auto-detects based on available API keys
            model: Model name (uses provider default if not specified)
            api_key: API key (uses environment variable if not specified)
            status_callback: Called with (stage, detail) for progress updates
            stream_callback: Called with each token chunk for streaming output
            enabled_tools: Only enable these tools (None = all)
            output_format: Force output format ('auto', 'pdf', 'docx', 'slides', etc.)
                          Default 'auto' lets LLM decide based on user request
            max_steps: Maximum tool use iterations
            tool_timeout: Timeout for each tool execution (seconds)
        """
        self.status_callback = status_callback
        self.stream_callback = stream_callback
        self.enabled_tools = enabled_tools
        self.output_format = output_format.lower() if output_format else "auto"
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout

        # Initialize provider
        if provider:
            self.provider_name = provider
            self.llm_provider = create_provider(provider, model, api_key)
        else:
            self.provider_name, self.llm_provider = auto_detect_provider()

        logger.info(f"UnifiedExecutor initialized with provider: {self.provider_name}")

        # Initialize tool generator
        self.tool_generator = UnifiedToolGenerator()

    def _build_system_prompt(self) -> str:
        """Build system prompt with output format instructions if specified."""
        base_prompt = UNIFIED_SYSTEM_PROMPT

        # If output format is auto, use base prompt
        if self.output_format == "auto":
            return base_prompt

        # Get the tool name for this format (dynamic from registry)
        output_format_tools = get_output_format_tools()
        tool_name = output_format_tools.get(self.output_format)
        if not tool_name:
            return base_prompt

        # Add format-specific instructions
        format_instructions = {
            "save_pdf": """

## IMPORTANT: Output Format Requirement
The user has requested PDF output. You MUST:
1. Generate comprehensive content first
2. ALWAYS call the `save_pdf` tool at the end with the content
3. Do not skip the save_pdf tool - the user explicitly requested PDF format""",

            "save_docx": """

## IMPORTANT: Output Format Requirement
The user has requested Word document output. You MUST:
1. Generate comprehensive content first
2. ALWAYS call the `save_docx` tool at the end with the content
3. Do not skip the save_docx tool - the user explicitly requested DOCX format""",

            "save_slides": """

## IMPORTANT: Output Format Requirement
The user has requested a PowerPoint presentation. You MUST:
1. Organize content into clear slides with titles and bullet points
2. ALWAYS call the `save_slides` tool with structured slide data
3. Do not skip the save_slides tool - the user explicitly requested presentation format""",

            "send_telegram": """

## IMPORTANT: Output Format Requirement
The user has requested Telegram delivery. You MUST:
1. Generate a concise message (max 4000 characters)
2. ALWAYS call the `send_telegram` tool at the end
3. Do not skip the send_telegram tool - the user explicitly requested Telegram delivery""",

            "return_kanban": """

## IMPORTANT: Output Format Requirement
The user has requested Kanban board visualization. You MUST:
1. Organize content into columns and task items
2. ALWAYS call the `return_kanban` tool with structured kanban data
3. Do not use plain text - use the kanban board format""",

            "return_chart": """

## IMPORTANT: Output Format Requirement
The user has requested Chart visualization. You MUST:
1. Organize data into chart-compatible format
2. ALWAYS call the `return_chart` tool with chart data
3. Do not use plain text - use the chart format""",

            "return_mermaid": """

## IMPORTANT: Output Format Requirement
The user has requested a diagram. You MUST:
1. Create content in Mermaid diagram syntax
2. ALWAYS call the `return_mermaid` tool with the diagram
3. Do not use plain text - use the diagram format""",

            "return_data_table": """

## IMPORTANT: Output Format Requirement
The user has requested table format. You MUST:
1. Organize data in CSV format
2. ALWAYS call the `return_data_table` tool with the data
3. Do not use plain text - use the table format""",
        }

        instruction = format_instructions.get(tool_name)

        # If no explicit instruction, generate one for JustJot section types
        if instruction is None and tool_name.startswith("return_"):
            section_name = tool_name.replace("return_", "").replace("_", " ").title()
            instruction = f"""

## IMPORTANT: Output Format Requirement
The user has requested {section_name} visualization format. You MUST:
1. Organize content appropriately for {section_name} display
2. ALWAYS call the `{tool_name}` tool with the structured content
3. Do not use plain text - use the {section_name} format"""

        return base_prompt + (instruction or "")

    def _status(self, stage: str, detail: str = ""):
        """Report status update."""
        if self.status_callback:
            try:
                import inspect
                result = self.status_callback(stage, detail)
                if inspect.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        asyncio.run(result)
            except Exception as e:
                logger.debug(f"Status callback error: {e}")
        logger.info(f"📍 {stage}" + (f": {detail}" if detail else ""))

    async def _stream(self, chunk: str):
        """Stream a content chunk."""
        if self.stream_callback:
            try:
                result = self.stream_callback(chunk)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug(f"Stream callback error: {e}")

    async def execute(
        self,
        task: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute task using unified tool-calling approach.

        Args:
            task: User's task/request
            history: Optional conversation history (list of {role, content} dicts)

        Returns:
            ExecutionResult with content, tool results, and sections
        """
        steps = []
        tool_results = []
        sections = []
        output_path = None
        full_content = ""

        try:
            # 1. Generate all available tools
            self._status("Preparing", "loading tools")
            tools = self.tool_generator.generate_all_tools()

            # Filter tools if enabled_tools is set
            if self.enabled_tools:
                tools = [t for t in tools if t['name'] in self.enabled_tools]

            logger.info(f"Loaded {len(tools)} tools")
            steps.append("prepare_tools")

            # 2. Build messages (include history if provided)
            messages = []
            if history:
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if content:
                        messages.append({"role": role, "content": content})

            # Add current task as user message
            messages.append({"role": "user", "content": task})

            # 3. Multi-turn tool calling loop
            response = None
            for step in range(self.max_steps):
                self._status("Thinking", f"step {step + 1}")

                # Call LLM with streaming
                if self.stream_callback:
                    response, content_chunk = await self.llm_provider.call_streaming(
                        messages=messages,
                        tools=tools,
                        system=self._build_system_prompt(),
                        stream_callback=self._stream
                    )
                    full_content += content_chunk
                else:
                    response = await self.llm_provider.call(
                        messages=messages,
                        tools=tools,
                        system=self._build_system_prompt()
                    )
                    # Extract text content
                    for block in response.content:
                        if isinstance(block, TextBlock):
                            full_content += block.text

                # Check stop reason
                if response.stop_reason == "end_turn":
                    steps.append("complete")
                    break

                elif response.stop_reason == "tool_use":
                    # Execute tool calls
                    tool_use_blocks = [
                        block for block in response.content
                        if isinstance(block, ToolUseBlock)
                    ]

                    # Add assistant message with tool uses
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # Execute each tool and collect results
                    tool_results_for_message = []

                    for tool_use in tool_use_blocks:
                        tool_name = tool_use.name
                        tool_input = tool_use.input

                        self._status("Executing", tool_name)
                        steps.append(f"tool_{tool_name}")

                        # Execute tool
                        result = await self._execute_tool(tool_name, tool_input)
                        tool_results.append(result)

                        # Check for visualization sections
                        if tool_name.startswith("return_"):
                            if result.success and result.result:
                                sections.append(result.result)

                        # Check for output paths
                        if result.success and result.result:
                            if 'file_path' in result.result:
                                output_path = result.result['file_path']
                            elif 'output_path' in result.result:
                                output_path = result.result['output_path']

                        # Prepare tool result for next message
                        tool_results_for_message.append(
                            self.llm_provider.format_tool_result(
                                tool_use.id,
                                str(result.result) if result.success else f"Error: {result.error}"
                            )
                        )

                    # Add tool results message
                    messages.append({
                        "role": "user",
                        "content": tool_results_for_message
                    })

                else:
                    # Unknown stop reason
                    logger.warning(f"Unknown stop reason: {response.stop_reason}")
                    break

            # Extract output format from tool results
            output_format = "text"
            for result in tool_results:
                if result.tool_name.startswith("save_"):
                    output_format = result.tool_name.replace("save_", "")
                    break
                elif result.tool_name.startswith("return_"):
                    output_format = result.tool_name.replace("return_", "")

            return ExecutionResult(
                success=True,
                content=full_content,
                tool_results=tool_results,
                sections=sections,
                output_path=output_path,
                output_format=output_format,
                steps_taken=steps,
                usage=response.usage if response else None
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                content="",
                error=str(e),
                steps_taken=steps,
                tool_results=tool_results
            )

    async def execute_stream(
        self,
        task: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Execute task with full streaming support.

        Yields StreamEvents for real-time updates.

        Args:
            task: User's task/request
            history: Optional conversation history (list of {role, content} dicts)

        Yields:
            StreamEvent objects (text, tool_start, tool_end, section, complete, error)
        """
        steps = []
        tool_results = []
        sections = []

        try:
            # Generate tools
            tools = self.tool_generator.generate_all_tools()
            if self.enabled_tools:
                tools = [t for t in tools if t['name'] in self.enabled_tools]

            # Build messages (include history if provided)
            messages = []
            if history:
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if content:
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": task})

            for step in range(self.max_steps):
                # Create a queue for streaming events
                text_buffer = []

                async def stream_handler(chunk: str):
                    text_buffer.append(chunk)

                # Call LLM with streaming
                response, _ = await self.llm_provider.call_streaming(
                    messages=messages,
                    tools=tools,
                    system=UNIFIED_SYSTEM_PROMPT,
                    stream_callback=stream_handler
                )

                # Yield accumulated text
                if text_buffer:
                    yield StreamEvent(type="text", data="".join(text_buffer))

                # Process tool uses
                if response.stop_reason == "tool_use":
                    tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

                    messages.append({"role": "assistant", "content": response.content})
                    tool_results_for_message = []

                    for tool_use in tool_use_blocks:
                        tool_name = tool_use.name
                        tool_input = tool_use.input

                        yield StreamEvent(type="tool_start", data={"name": tool_name})
                        steps.append(f"tool_{tool_name}")

                        # Execute tool
                        result = await self._execute_tool(tool_name, tool_input)
                        tool_results.append(result)

                        yield StreamEvent(type="tool_end", data={
                            "name": tool_name,
                            "success": result.success,
                            "result": result.result
                        })

                        # Emit section if visualization
                        if tool_name.startswith("return_") and result.success:
                            sections.append(result.result)
                            yield StreamEvent(type="section", data=result.result)

                        tool_results_for_message.append(
                            self.llm_provider.format_tool_result(
                                tool_use.id,
                                str(result.result) if result.success else f"Error: {result.error}"
                            )
                        )

                    messages.append({"role": "user", "content": tool_results_for_message})

                elif response.stop_reason == "end_turn":
                    yield StreamEvent(type="complete", data={
                        "steps": steps,
                        "tool_results": [
                            {"name": r.tool_name, "success": r.success}
                            for r in tool_results
                        ],
                        "sections": sections
                    })
                    return

        except Exception as e:
            logger.error(f"Streaming execution failed: {e}", exc_info=True)
            yield StreamEvent(type="error", data={"error": str(e)})

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool with timeout and error handling."""
        try:
            executor = self.tool_generator.get_executor(tool_name)
            if not executor:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"No executor found for tool: {tool_name}"
                )

            # Execute with timeout
            result = await asyncio.wait_for(
                executor(tool_input),
                timeout=self.tool_timeout
            )

            # Determine success
            success = True
            if isinstance(result, dict):
                success = result.get('success', True)
                if 'error' in result and result['error']:
                    success = False

            return ToolResult(
                tool_name=tool_name,
                success=success,
                result=result
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool execution timed out after {self.tool_timeout}s"
            )

        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e)
            )


# =============================================================================
# Convenience Factory
# =============================================================================

def create_unified_executor(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    status_callback: Optional[Callable[[str, str], None]] = None,
    enabled_tools: Optional[List[str]] = None,
    output_format: str = "auto"
) -> UnifiedExecutor:
    """
    Factory function to create UnifiedExecutor.

    Args:
        provider: LLM provider ('anthropic', 'openai', 'openrouter', 'groq', 'google')
                  If None, auto-detects based on available API keys
        model: Model name (uses provider default if not specified)
        stream_callback: Callback for streaming text chunks
        status_callback: Callback for status updates
        enabled_tools: List of tool names to enable (None = all)
        output_format: Force output format ('auto', 'pdf', 'docx', 'slides', etc.)

    Returns:
        Configured UnifiedExecutor instance

    Example:
        # Auto-detect provider
        executor = create_unified_executor()

        # Specific provider with PDF output
        executor = create_unified_executor(
            provider='openai',
            model='gpt-4o',
            output_format='pdf',
            stream_callback=lambda chunk: print(chunk, end='')
        )

        result = await executor.execute("Research AI trends")
    """
    return UnifiedExecutor(
        provider=provider,
        model=model,
        stream_callback=stream_callback,
        status_callback=status_callback,
        enabled_tools=enabled_tools,
        output_format=output_format
    )
