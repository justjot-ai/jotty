"""
UnifiedExecutor - Native LLM Tool-Calling Executor
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

from .tool_generator import UnifiedToolGenerator

# Import types and providers from extracted llm_providers package
from .llm_providers import (
    ToolResult,
    ExecutionResult,
    StreamEvent,
    LLMResponse,
    ToolUseBlock,
    TextBlock,
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    GroqProvider,
    GoogleProvider,
    JottyClaudeProviderAdapter,
    create_provider,
    auto_detect_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DSPy Signatures (for JottyClaudeProvider fallback)
# =============================================================================

def _get_dspy_signatures():
    """Lazy import DSPy signatures to avoid import errors when DSPy not configured."""
    import dspy

    class TaskAnalysisSignature(dspy.Signature):
        """Analyze task requirements. You have web search capability."""
        task: str = dspy.InputField(desc="User's task/request")
        is_web_search: bool = dspy.OutputField(desc="TRUE for weather/news/prices/current events. FALSE for concepts/coding.")
        is_file_read: bool = dspy.OutputField(desc="True if task mentions a file path.")
        output_format: str = dspy.OutputField(desc="'text', 'docx', 'pdf', 'slides', 'telegram', 'justjot'")
        output_path_hint: str = dspy.OutputField(desc="Suggested filename or 'none'")
        reasoning: str = dspy.OutputField(desc="Brief explanation")

    class ContentGenerationSignature(dspy.Signature):
        """Generate content using provided context."""
        task: str = dspy.InputField(desc="User's request")
        context: str = dspy.InputField(desc="Data from web search/file. Use this as primary source.")
        output_format: str = dspy.InputField(desc="Desired format")
        content: str = dspy.OutputField(desc="Generated content")

    return TaskAnalysisSignature, ContentGenerationSignature


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
        from Jotty.core.ui.schema_validator import schema_registry

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
# Provider Abstraction (extracted to llm_providers/ subpackage)
# =============================================================================
# All provider classes now live in core/orchestration/v2/llm_providers/:
#   base.py      - LLMProvider ABC
#   anthropic.py - AnthropicProvider
#   openai.py    - OpenAIProvider, OpenRouterProvider, GroqProvider
#   google.py    - GoogleProvider
#   adapter.py   - JottyClaudeProviderAdapter
#   factory.py   - create_provider(), auto_detect_provider()
#   types.py     - ToolResult, ExecutionResult, StreamEvent, LLMResponse, etc.
# All are re-imported at the top of this file for backward compatibility.




# =============================================================================
# Unified Executor
# =============================================================================

class UnifiedExecutor:
    """
    Unified executor using native LLM tool calling for all decisions.

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

        logger.debug(f"UnifiedExecutor initialized with provider: {self.provider_name}")

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

    def _detect_web_search_needed(self, task: str) -> bool:
        """Keyword-based detection for when web search is needed."""
        task_lower = task.lower()

        # Keywords that ALWAYS need web search
        web_search_keywords = [
            'weather', 'temperature', 'forecast',
            'news', 'headlines', 'latest', 'recent', 'today', 'current',
            'stock price', 'market', 'crypto', 'bitcoin', 'price of',
            'search for', 'search the web', 'web search', 'look up', 'find out',
            'what is happening', 'what happened', 'score', 'results',
            'trending', 'update', 'live',
        ]

        for keyword in web_search_keywords:
            if keyword in task_lower:
                return True
        return False

    def _extract_content_from_error(self, error_msg: str) -> Optional[str]:
        """Try to extract usable content from a DSPy parsing error."""
        import re
        # DSPy errors often contain the raw LLM response which has good content
        # Pattern: "LM Response: {...content...}"
        match = re.search(r'LM Response:\s*\{?\s*(.+?)\s*\}?\s*\n\nExpected', error_msg, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Clean up escape sequences
            content = content.replace('\\n', '\n').replace('\\"', '"')
            if len(content) > 50:  # Only use if substantial
                return content
        return None

    async def _execute_dspy(
        self,
        task: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute task using DSPy signatures (for JottyClaudeProvider).

        This is used when native tool calling isn't available.
        Uses DSPy's ChainOfThought for task analysis and content generation.
        """
        import dspy
        TaskAnalysisSignature, ContentGenerationSignature = _get_dspy_signatures()

        steps = []
        tool_results = []
        max_retries = 2

        try:
            # Step 1: Keyword-based detection FIRST (override LLM decision)
            force_web_search = self._detect_web_search_needed(task)
            logger.info(f"Keyword detection: force_web_search={force_web_search} for task: {task[:50]}")

            # Step 2: Analyze task using DSPy (for output format, etc.)
            self._status("Analyzing", "understanding task requirements")
            steps.append("analyze")

            # Try analysis with retry
            analysis = None
            for attempt in range(max_retries):
                try:
                    analyzer = dspy.ChainOfThought(TaskAnalysisSignature)
                    analysis = analyzer(task=task)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"Analysis attempt {attempt + 1} failed, retrying...")
                        continue
                    # On final failure, use defaults
                    logger.warning(f"Analysis failed, using defaults: {e}")
                    analysis = None

            # Use keyword detection OR LLM decision (keyword takes priority)
            if analysis:
                is_web_search = force_web_search or bool(getattr(analysis, 'is_web_search', False))
                is_file_read = bool(getattr(analysis, 'is_file_read', False)) if not force_web_search else False
                output_format = str(getattr(analysis, 'output_format', 'text')).lower().strip()
            else:
                is_web_search = force_web_search
                is_file_read = False
                output_format = 'markdown'

            self._status("Processing", "preparing response")

            # Step 2: Get external data if needed
            context = "none"

            if is_web_search:
                self._status("Searching", "gathering external information")
                steps.append("web_search")
                context = await self._do_web_search_dspy(task)
                tool_results.append(ToolResult(
                    tool_name="web_search",
                    success=context != "Web search unavailable",
                    result={"context": context[:500] + "..." if len(context) > 500 else context}
                ))

            elif is_file_read:
                self._status("Reading", "loading file content")
                steps.append("file_read")
                context = await self._do_file_read_dspy(task)

            # Step 3: Generate content using DSPy with retry
            self._status("Generating", "creating content")
            steps.append("generate")

            content = None
            last_error = None

            for attempt in range(max_retries):
                try:
                    generator = dspy.ChainOfThought(ContentGenerationSignature)
                    generation = generator(task=task, context=context, output_format=output_format)
                    content = str(generation.content)
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e)

                    # Try to extract content from parsing error
                    if "AdapterParseError" in error_str or "failed to parse" in error_str.lower():
                        extracted = self._extract_content_from_error(error_str)
                        if extracted:
                            logger.info("Extracted content from parsing error response")
                            content = extracted
                            break

                    if attempt < max_retries - 1:
                        self._status("Retrying", "regenerating response")
                        logger.debug(f"Generation attempt {attempt + 1} failed, retrying...")
                        continue

            if content:
                return ExecutionResult(
                    success=True,
                    content=content,
                    tool_results=tool_results,
                    output_format=output_format,
                    steps_taken=steps
                )
            else:
                # Friendly error message, no tracebacks
                logger.error(f"DSPy execution failed after retries: {last_error}")
                return ExecutionResult(
                    success=False,
                    content="I encountered an issue generating the response. Please try again.",
                    error="Generation failed after retries",
                    steps_taken=steps,
                    tool_results=tool_results
                )

        except Exception as e:
            # Log full error for debugging, but return friendly message
            logger.error(f"DSPy execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                content="I encountered an unexpected issue. Please try again.",
                error="Execution error",
                steps_taken=steps,
                tool_results=tool_results
            )

    def _extract_search_query(self, task: str) -> str:
        """Extract search query from task, handling conversation context."""
        import re

        # If task contains "Current request:", extract just that
        if "Current request:" in task:
            match = re.search(r'Current request:\s*(.+?)(?:\n|$)', task, re.DOTALL)
            if match:
                query = match.group(1).strip()
                # Check if we need context from conversation (e.g., "next week" without location)
                if any(word in query.lower() for word in ['next', 'more', 'also', 'it', 'that', 'there']):
                    # Extract location/topic from conversation context
                    context_match = re.search(r'(?:weather|forecast|temperature).*?(?:in|for|at)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|\n|$)', task, re.IGNORECASE)
                    if context_match:
                        location = context_match.group(1).strip()
                        query = f"{query} {location}"
                return query[:200]

        # Remove common prefixes
        query = task
        prefixes = ["search for", "find", "look up", "what is", "what's", "tell me about"]
        for prefix in prefixes:
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break

        return query[:200]

    async def _do_web_search_dspy(self, task: str) -> str:
        """Execute web search for DSPy execution path."""
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            registry.init()

            skill = registry.get_skill('web-search')
            if not skill:
                return "Web search skill not found"

            search_tool = skill.tools.get('search_web_tool')
            if not search_tool:
                return "Search tool not found"

            # Extract search query intelligently
            query = self._extract_search_query(task)
            logger.info(f"Web search query: {query}")
            result = search_tool({'query': query, 'max_results': 10})

            if result.get('success'):
                results = result.get('results', [])
                if not results:
                    return "No search results found"

                from datetime import datetime
                now = datetime.now().strftime('%Y-%m-%d %H:%M')
                context_parts = [f"=== WEB SEARCH RESULTS (Retrieved: {now}) ===\n"]
                for i, r in enumerate(results[:8], 1):
                    title = r.get('title', 'Untitled')
                    snippet = r.get('snippet', '')
                    url = r.get('url', '')
                    context_parts.append(f"[{i}] {title}\n{snippet}\nSource: {url}\n")
                context_parts.append("=== END OF SEARCH RESULTS ===")
                return "\n".join(context_parts)
            else:
                return f"Search failed: {result.get('error', 'unknown')}"
        except Exception as e:
            return f"Search error: {e}"

    async def _do_file_read_dspy(self, task: str) -> str:
        """Read file content for DSPy execution path."""
        import re
        path_match = re.search(r'[\'"]?(/[^\s\'"]+|[A-Za-z]:\\[^\s\'"]+)[\'"]?', task)
        if not path_match:
            return "No file path found in task"

        file_path = path_match.group(1)
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"File read error: {e}"

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
        # If using JottyClaudeProvider (DSPy-based), use DSPy execution path
        if self.provider_name == 'jotty-claude':
            return await self._execute_dspy(task, history)
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
