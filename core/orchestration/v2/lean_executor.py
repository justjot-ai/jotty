"""
LeanExecutor - Clean LLM-first Architecture
============================================

Core Philosophy:
- LLM is the BRAIN, not a skill
- Skills are I/O TOOLS that work ON LLM content
- No hardcoded rules, intelligent decisions

Architecture:
```
User Task
    ↓
┌─────────────────────────────────┐
│  1. ANALYZE (LLM decides)       │
│     - Need external data?       │
│     - What output format?       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  2. INPUT (if needed)           │
│     Skills: web-search,         │
│     file-read, api-call         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  3. GENERATE (LLM always)       │
│     LLM creates/synthesizes     │
│     content from knowledge +    │
│     any input data              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  4. OUTPUT (if needed)          │
│     Skills: file-write,         │
│     docx-tools, telegram-send   │
└─────────────────────────────────┘
```

Skill Categories:
- INPUT: web-search, file-read, scrape-webpage
- OUTPUT: file-write, docx-tools, pdf-export, telegram-send
- TRANSFORM: document-converter
"""

import asyncio
import logging
import dspy
from typing import Dict, Any, Optional, Callable, List, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Skill Registry - I/O Tools Only
# =============================================================================

INPUT_SKILLS = {
    'web-search': 'Search the web for external information',
    'file-read': 'Read content from files',
    'file-operations': 'File system operations',
}

OUTPUT_SKILLS = {
    'file-write': 'Write content to file',
    'file-operations': 'File system operations',
    'docx-tools': 'Create Word documents',
    'document-converter': 'Convert to PDF/HTML/DOCX',
    'telegram-sender': 'Send via Telegram',
}

TRANSFORM_SKILLS = {
    'document-converter': 'Convert between document formats',
}


# =============================================================================
# Decision Signatures - Clean, No Hardcoded Rules
# =============================================================================

class TaskAnalysisSignature(dspy.Signature):
    """Analyze what a task needs. Be intelligent, not rule-based.

    You understand natural language. Decide:
    1. Can you answer this from your knowledge, or need external data?
    2. What output format does the user want?
    3. Is this a checklist/structured document task?

    Examples of your reasoning:
    - "Explain transformers" → You KNOW this, output=text (display inline)
    - "Create a checklist for X framework" → output=checklist (saved to DOCX)
    - "What's the latest news on X" → Need CURRENT info, search needed
    - "Summarize this file: /path" → Need to READ the file first
    - "Send report to telegram" → output=telegram
    - "Save as markdown/pdf/docx" → User explicitly wants file saved
    """

    task: str = dspy.InputField(desc="User's task/request")

    needs_external_data: bool = dspy.OutputField(
        desc="True ONLY if you need CURRENT/RECENT info (news, prices, live data). False for knowledge-based tasks like explaining concepts, creating checklists, writing documents."
    )
    input_type: str = dspy.OutputField(
        desc="If needs_external_data: 'web_search', 'file_read', or 'none'"
    )
    output_format: str = dspy.OutputField(
        desc="How to deliver output: 'text' (DEFAULT - display inline), 'checklist' (DOCX), 'docx', 'pdf', 'slides' (PowerPoint PPTX), 'slides_pdf' (slides as PDF), 'file', 'telegram', 'justjot' (save as idea to JustJot.ai). Use 'slides' for presentations, 'justjot' for JustJot ideas."
    )
    output_path_hint: str = dspy.OutputField(
        desc="If output is file/docx/pdf/checklist, suggest filename. Otherwise 'none'"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of your decision"
    )


class ContentGenerationSignature(dspy.Signature):
    """Generate content for the user's request.

    IMPORTANT: You have been provided with CURRENT information via the context field.
    The context contains REAL-TIME data from web searches performed just now.
    USE THIS CONTEXT as your primary source - it contains up-to-date information.

    Generate high-quality content based on:
    1. The provided context (PRIORITIZE THIS - it has current/recent data)
    2. Your knowledge (supplement where context is incomplete)

    DO NOT say you cannot access real-time data - the context already provides it.
    Be comprehensive, well-structured, and directly address the user's need.

    FOR CHECKLIST FORMAT - Use this structure:
    ```
    ## PART A: [SECTION NAME]

    ### 1. [Subsection Title]

    - [ ] Checklist item description | Legal Reference or Source
    - [ ] Another checklist item | Reference
    - [x] Completed item example | Reference

    ### 2. [Next Subsection]

    - [ ] Item with specific requirement | Regulation §X
    ```

    Include specific references (laws, regulations, standards) where applicable.
    Use PART A, PART B for major sections.
    Use ### for subsections with numbers.
    Use - [ ] for unchecked items, - [x] for checked.
    Add | Reference after each item where relevant.
    """

    task: str = dspy.InputField(desc="User's original request")
    context: str = dspy.InputField(desc="CURRENT DATA from web search or file. USE THIS as primary source. Contains real-time information. Value is 'none' only if no external data was needed.")
    output_format: str = dspy.InputField(desc="Desired format: checklist, text, markdown, slides, docx content, etc.")

    content: str = dspy.OutputField(
        desc="Generated content. For checklists: use ## PART headers, ### numbered sections, - [ ] items with | references. For other formats: use appropriate markdown."
    )


# =============================================================================
# LeanExecutor - The Clean Implementation
# =============================================================================

@dataclass
class ExecutionResult:
    """Result from lean execution."""
    success: bool
    content: str
    output_path: Optional[str] = None
    output_format: str = "text"
    error: Optional[str] = None
    steps_taken: List[str] = None

    def __post_init__(self):
        if self.steps_taken is None:
            self.steps_taken = []


class LeanExecutor:
    """
    Clean LLM-first executor.

    LLM is the brain, skills are I/O tools.
    No hardcoded rules, intelligent decisions.
    """

    def __init__(
        self,
        status_callback: Optional[Callable] = None,
        stream_callback: Optional[Callable[[str], None]] = None
    ):
        """Initialize executor.

        Args:
            status_callback: Called with (stage, detail) for progress updates
            stream_callback: Called with each token chunk for streaming output
        """
        self.status_callback = status_callback
        self.stream_callback = stream_callback
        self.analyzer = dspy.ChainOfThought(TaskAnalysisSignature)
        self.generator = dspy.ChainOfThought(ContentGenerationSignature)

        # Lazy-load skill registry
        self._registry = None

    def _status(self, stage: str, detail: str = ""):
        """Report status (supports async callbacks)."""
        if self.status_callback:
            try:
                import asyncio
                import inspect
                result = self.status_callback(stage, detail)
                # Handle async callbacks
                if inspect.iscoroutine(result):
                    # Schedule the coroutine to run
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop, run synchronously
                        asyncio.run(result)
            except Exception as e:
                logger.debug(f"Status callback error: {e}")
        logger.info(f"📍 {stage}" + (f": {detail}" if detail else ""))

    async def _stream(self, chunk: str):
        """Stream a content chunk (supports async callbacks)."""
        if self.stream_callback:
            try:
                result = self.stream_callback(chunk)
                # Handle async callbacks
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug(f"Stream callback error: {e}")

    async def _generate_streaming(self, task: str, context: str, output_format: str) -> str:
        """
        Generate content with streaming output.

        Attempts to stream tokens as they're generated.
        Falls back to non-streaming if LM doesn't support it.
        """
        # Check if current LM supports streaming
        lm = dspy.settings.lm
        supports_streaming = hasattr(lm, 'stream') or (
            hasattr(lm, 'kwargs') and lm.kwargs.get('stream', False)
        )

        if supports_streaming and self.stream_callback:
            # Try streaming generation
            try:
                # Build the prompt manually for streaming
                prompt = f"""Generate content for the user's request.

Task: {task}
Context: {context}
Output Format: {output_format}

Generate high-quality content based on your knowledge and the provided context.
For checklists: use ## PART headers, ### numbered sections, - [ ] items with | references.
For other formats: use appropriate markdown.

Content:"""

                full_content = ""

                # Try different streaming approaches based on LM type
                if hasattr(lm, 'stream'):
                    # Native streaming support
                    async for chunk in lm.stream(prompt):
                        if chunk:
                            await self._stream(chunk)
                            full_content += chunk
                elif hasattr(lm, '__call__') and asyncio.iscoroutinefunction(lm.__call__):
                    # Async LM with potential streaming
                    result = await lm(prompt, stream=True)
                    if hasattr(result, '__aiter__'):
                        async for chunk in result:
                            if isinstance(chunk, str):
                                await self._stream(chunk)
                                full_content += chunk
                            elif hasattr(chunk, 'text'):
                                await self._stream(chunk.text)
                                full_content += chunk.text
                    else:
                        # Not actually streaming
                        full_content = str(result)
                        await self._stream(full_content)
                else:
                    # Fall back to non-streaming
                    generation = self.generator(
                        task=task,
                        context=context,
                        output_format=output_format
                    )
                    full_content = str(generation.content)
                    # Stream in chunks for visual effect
                    chunk_size = 50
                    for i in range(0, len(full_content), chunk_size):
                        chunk = full_content[i:i+chunk_size]
                        await self._stream(chunk)
                        await asyncio.sleep(0.01)  # Small delay for visual effect

                return full_content

            except Exception as e:
                logger.debug(f"Streaming failed, falling back: {e}")
                # Fall through to non-streaming

        # Try real Anthropic streaming if available
        if self.stream_callback:
            try:
                content = await self._anthropic_stream(task, context, output_format)
                if content:
                    return content
            except Exception as e:
                logger.debug(f"Anthropic streaming failed: {e}")

        # Non-streaming fallback
        generation = self.generator(
            task=task,
            context=context,
            output_format=output_format
        )
        content = str(generation.content)

        # If stream callback exists, send content in chunks for visual effect
        if self.stream_callback:
            chunk_size = 100
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                await self._stream(chunk)
                await asyncio.sleep(0.005)

        return content

    async def _anthropic_stream(self, task: str, context: str, output_format: str) -> str:
        """
        Real streaming using Anthropic API directly.
        """
        import os

        # Try to get Anthropic client
        try:
            import anthropic
        except ImportError:
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Generate content for the user's request.

Task: {task}
Context: {context}
Output Format: {output_format}

Generate high-quality content based on your knowledge and the provided context.
For checklists: use ## PART headers, ### numbered sections, - [ ] items with | references.
For other formats: use appropriate markdown.

Content:"""

        full_content = ""

        # Use streaming API
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                if text:
                    await self._stream(text)
                    full_content += text

        return full_content

    def _get_registry(self):
        """Get skills registry (lazy load)."""
        if self._registry is None:
            try:
                from ...registry.skills_registry import get_skills_registry
                self._registry = get_skills_registry()
                self._registry.init()
            except Exception as e:
                logger.warning(f"Could not load skills registry: {e}")
        return self._registry

    async def execute(self, task: str) -> ExecutionResult:
        """
        Execute task with clean LLM-first approach.

        Flow:
        1. Analyze task (LLM decides what's needed)
        2. Input phase (if external data needed)
        3. Generate content (LLM always)
        4. Output phase (if specific format needed)
        """
        steps = []

        try:
            # =================================================================
            # Step 1: ANALYZE - Let LLM decide intelligently
            # =================================================================
            self._status("Analyzing", "understanding task requirements")
            steps.append("analyze")

            analysis = self.analyzer(task=task)

            needs_external = analysis.needs_external_data
            input_type = str(analysis.input_type).lower().strip()
            output_format = str(analysis.output_format).lower().strip()
            output_hint = str(analysis.output_path_hint).strip()

            self._status("Decision",
                f"external_data={needs_external}, input={input_type}, output={output_format}")
            logger.info(f"Analysis reasoning: {analysis.reasoning}")

            # =================================================================
            # Step 2: INPUT - Get external data if needed
            # =================================================================
            context = "none"

            # Flexible matching for web search (LLM may return variations)
            web_search_types = ["web_search", "web-search", "search", "research", "research_query", "internet", "online"]
            file_read_types = ["file_read", "file-read", "read_file", "file", "local_file"]

            if needs_external and input_type in web_search_types:
                self._status("Searching", "gathering external information")
                steps.append("web_search")
                context = await self._do_web_search(task)

            elif needs_external and input_type in file_read_types:
                self._status("Reading", "loading file content")
                steps.append("file_read")
                # Extract file path from task
                context = await self._do_file_read(task)

            # =================================================================
            # Step 3: GENERATE - LLM creates content (always happens)
            # =================================================================
            self._status("Generating", "creating content")
            steps.append("generate")

            # Use streaming if callback provided
            if self.stream_callback:
                content = await self._generate_streaming(task, context, output_format)
            else:
                generation = self.generator(
                    task=task,
                    context=context,
                    output_format=output_format
                )
                content = str(generation.content)

            self._status("Generated", f"{len(content)} chars")

            # =================================================================
            # Step 4: OUTPUT - Save/send if specific format needed
            # =================================================================
            output_path = None

            # Intelligent output decision
            # "text" and "markdown" = display inline (markdown is just formatted text)
            # Everything else (docx, pdf, checklist, file, telegram) = save/send
            inline_formats = ["text", "none", "", "markdown"]
            needs_file_output = output_format not in inline_formats
            needs_telegram = "telegram" in output_format.lower()

            needs_justjot = "justjot" in output_format.lower()

            if needs_telegram:
                self._status("Sending", "delivering via Telegram")
                steps.append("output_telegram")
                await self._do_telegram(content)

            elif needs_justjot:
                self._status("Saving", "creating idea on JustJot.ai")
                steps.append("output_justjot")
                justjot_result = await self._do_justjot(content, task)
                if justjot_result.get('success'):
                    output_path = justjot_result.get('url') or justjot_result.get('local_path')

            elif needs_file_output:
                self._status("Saving", f"creating {output_format} output")
                steps.append(f"output_{output_format}")

                output_path = await self._do_output(
                    content=content,
                    format=output_format,
                    hint=output_hint,
                    task=task
                )

                if output_path:
                    self._status("Saved", output_path)

            return ExecutionResult(
                success=True,
                content=content,
                output_path=output_path,
                output_format=output_format,
                steps_taken=steps
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                content="",
                error=str(e),
                steps_taken=steps
            )

    # =========================================================================
    # I/O Skill Implementations
    # =========================================================================

    async def _do_web_search(self, task: str) -> str:
        """Execute web search and return context."""
        registry = self._get_registry()
        if not registry:
            return "Web search unavailable"

        skill = registry.get_skill('web-search')
        if not skill:
            return "Web search skill not found"

        search_tool = skill.tools.get('search_web_tool')
        if not search_tool:
            return "Search tool not found"

        # Extract search query from task (simple approach)
        query = self._extract_search_query(task)

        try:
            result = search_tool({'query': query, 'max_results': 10})

            if result.get('success'):
                results = result.get('results', [])
                if not results:
                    return "No search results found"

                # Format results as context with clear header
                from datetime import datetime
                now = datetime.now().strftime('%Y-%m-%d %H:%M')

                context_parts = [f"=== WEB SEARCH RESULTS (Retrieved: {now}) ===\n"]
                for i, r in enumerate(results[:8], 1):
                    title = r.get('title', 'Untitled')
                    snippet = r.get('snippet', '')
                    url = r.get('url', '')
                    context_parts.append(f"[{i}] {title}\n{snippet}\nSource: {url}\n")

                context_parts.append("=== END OF SEARCH RESULTS ===")
                context_parts.append("\nUse the above current information to answer the user's request.")

                return "\n".join(context_parts)
            else:
                return f"Search failed: {result.get('error', 'unknown')}"
        except Exception as e:
            return f"Search error: {e}"

    async def _do_file_read(self, task: str) -> str:
        """Read file content."""
        # Extract file path from task
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

    async def _do_output(self, content: str, format: str, hint: str, task: str) -> Optional[str]:
        """Save content to file in specified format."""

        # Determine output path
        output_dir = Path.home() / "jotty" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate filename from hint or task
        if hint and hint != "none":
            base_name = hint.replace(" ", "_")[:50]
        else:
            # Extract key words from task
            base_name = "_".join(task.split()[:5]).replace("/", "_")[:50]

        if format == "checklist":
            return await self._save_professional_checklist(content, output_dir, base_name, timestamp, task)
        elif format == "docx":
            return await self._save_docx(content, output_dir, base_name, timestamp)
        elif format == "pdf":
            return await self._save_pdf(content, output_dir, base_name, timestamp)
        elif format == "slides":
            return await self._save_slides(content, output_dir, base_name, timestamp, task, export_pdf=False)
        elif format == "slides_pdf":
            return await self._save_slides(content, output_dir, base_name, timestamp, task, export_pdf=True)
        elif format in ["markdown", "file"]:
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        return None

    async def _save_professional_checklist(self, content: str, output_dir: Path, base_name: str, timestamp: str, task: str) -> Optional[str]:
        """Save as professional formatted checklist using docx-tools."""
        registry = self._get_registry()
        if not registry:
            return await self._save_docx(content, output_dir, base_name, timestamp)

        skill = registry.get_skill('docx-tools')
        if not skill:
            return await self._save_docx(content, output_dir, base_name, timestamp)

        # Try professional checklist tool first
        checklist_tool = skill.tools.get('create_professional_checklist_tool')
        if not checklist_tool:
            # Fall back to regular docx
            return await self._save_docx(content, output_dir, base_name, timestamp)

        output_path = output_dir / f"{base_name}_{timestamp}.docx"

        # Extract title and subtitle from task
        title = "CHECKLIST"
        subtitle = ""

        # Try to extract meaningful title from task
        task_lower = task.lower()
        if "checklist" in task_lower:
            # Extract what comes after "checklist for" or "checklist of"
            import re
            match = re.search(r'checklist\s+(?:for|of|on)\s+(.+?)(?:\s+under|\s+based|\s+prescribed|$)', task_lower)
            if match:
                subtitle = match.group(1).strip().title()
                title = f"{subtitle.upper()} CHECKLIST"
            else:
                # Use first few words after "checklist"
                parts = task_lower.split("checklist", 1)
                if len(parts) > 1 and parts[1].strip():
                    subtitle = parts[1].strip()[:50].title()
                    title = "COMPLIANCE CHECKLIST"

        try:
            result = checklist_tool({
                'content': content,
                'output_path': str(output_path),
                'title': title,
                'subtitle': subtitle,
                'include_form_fields': True
            })

            if result.get('success'):
                return str(result.get('file_path', output_path))
        except Exception as e:
            logger.warning(f"Professional checklist creation failed: {e}")

        # Fallback to regular docx
        return await self._save_docx(content, output_dir, base_name, timestamp)

    async def _save_docx(self, content: str, output_dir: Path, base_name: str, timestamp: str) -> Optional[str]:
        """Save as DOCX using docx-tools skill."""
        registry = self._get_registry()
        if not registry:
            # Fallback to markdown
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        skill = registry.get_skill('docx-tools')
        if not skill:
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        create_tool = skill.tools.get('create_docx_tool')
        if not create_tool:
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        output_path = output_dir / f"{base_name}_{timestamp}.docx"

        try:
            result = create_tool({
                'content': content,
                'output_path': str(output_path),
                'title': base_name.replace("_", " ").title()
            })

            if result.get('success'):
                return str(result.get('file_path', output_path))
        except Exception as e:
            logger.warning(f"DOCX creation failed: {e}")

        # Fallback
        return await self._save_markdown(content, output_dir, base_name, timestamp)

    async def _save_pdf(self, content: str, output_dir: Path, base_name: str, timestamp: str) -> Optional[str]:
        """Save as PDF using document-converter skill."""
        # First save as markdown
        md_path = await self._save_markdown(content, output_dir, base_name, timestamp)
        if not md_path:
            return None

        registry = self._get_registry()
        if not registry:
            return md_path  # Return markdown as fallback

        skill = registry.get_skill('document-converter')
        if not skill:
            return md_path

        convert_tool = skill.tools.get('convert_to_pdf_tool')
        if not convert_tool:
            return md_path

        pdf_path = output_dir / f"{base_name}_{timestamp}.pdf"

        try:
            result = convert_tool({
                'input_file': md_path,
                'output_file': str(pdf_path)
            })

            if result.get('success'):
                return str(result.get('output_path', pdf_path))
        except Exception as e:
            logger.warning(f"PDF conversion failed: {e}")

        return md_path

    async def _save_markdown(self, content: str, output_dir: Path, base_name: str, timestamp: str) -> Optional[str]:
        """Save as markdown file."""
        output_path = output_dir / f"{base_name}_{timestamp}.md"

        try:
            output_path.write_text(content)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save markdown: {e}")
            return None

    async def _save_slides(self, content: str, output_dir: Path, base_name: str, timestamp: str, task: str, export_pdf: bool = False) -> Optional[str]:
        """Save as PowerPoint slides using slide-generator skill."""
        registry = self._get_registry()
        if not registry:
            logger.warning("Cannot create slides: registry unavailable")
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        skill = registry.get_skill('slide-generator')
        if not skill:
            logger.warning("slide-generator skill not found")
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        # Try the AI-powered topic generator first
        topic_tool = skill.tools.get('generate_slides_from_topic_tool')

        if topic_tool:
            try:
                # Extract topic from task
                topic = task.replace("Research", "").replace("research", "").strip()
                topic = topic.split("-")[0].strip()  # Remove flags
                topic = topic.replace("'", "").replace('"', "").strip()

                import asyncio
                import inspect

                params = {
                    'topic': topic,
                    'n_slides': 8,
                    'template': 'dark',
                    'output_path': str(output_dir),
                    'export_as': 'pdf' if export_pdf else 'pptx',
                    'send_telegram': False  # We'll handle this separately
                }

                if inspect.iscoroutinefunction(topic_tool):
                    result = await topic_tool(params)
                else:
                    result = topic_tool(params)

                if result.get('success'):
                    # Return PDF path if requested, otherwise PPTX
                    if export_pdf and result.get('pdf_path'):
                        return result.get('pdf_path')
                    return result.get('file_path')
                else:
                    logger.warning(f"Slide generation failed: {result.get('error')}")

            except Exception as e:
                logger.warning(f"AI slide generation failed: {e}")

        # Fallback: Try to parse content and create slides manually
        slides_tool = skill.tools.get('generate_slides_tool')
        pdf_slides_tool = skill.tools.get('generate_slides_pdf_tool')

        if not slides_tool and not pdf_slides_tool:
            logger.warning("No slide generation tools available")
            return await self._save_markdown(content, output_dir, base_name, timestamp)

        try:
            # Parse content into slide structure
            slides_data = self._parse_content_to_slides(content, task)

            if export_pdf and pdf_slides_tool:
                result = pdf_slides_tool({
                    'title': slides_data['title'],
                    'subtitle': slides_data.get('subtitle', ''),
                    'slides': slides_data['slides'],
                    'template': 'dark',
                    'output_path': str(output_dir)
                })
            elif slides_tool:
                result = slides_tool({
                    'title': slides_data['title'],
                    'subtitle': slides_data.get('subtitle', ''),
                    'slides': slides_data['slides'],
                    'template': 'dark',
                    'output_path': str(output_dir)
                })
            else:
                return await self._save_markdown(content, output_dir, base_name, timestamp)

            if result.get('success'):
                return result.get('file_path')

        except Exception as e:
            logger.warning(f"Manual slide creation failed: {e}")

        return await self._save_markdown(content, output_dir, base_name, timestamp)

    def _parse_content_to_slides(self, content: str, task: str) -> dict:
        """Parse markdown content into slide structure."""
        import re

        lines = content.strip().split('\n')
        slides = []
        current_slide = None

        # Extract title from task or first heading
        title = "Research Findings"
        task_words = task.split()[:5]
        if task_words:
            title = " ".join(w for w in task_words if not w.startswith('-'))[:50]

        for line in lines:
            line = line.strip()

            # New slide on ## heading
            if line.startswith('## '):
                if current_slide and current_slide.get('bullets'):
                    slides.append(current_slide)
                current_slide = {
                    'title': line[3:].strip(),
                    'bullets': []
                }
            # Bullet point
            elif line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                if current_slide is None:
                    current_slide = {'title': 'Key Points', 'bullets': []}
                bullet = line[2:].strip()
                if bullet and len(current_slide['bullets']) < 6:
                    current_slide['bullets'].append(bullet[:100])
            # Numbered list
            elif re.match(r'^\d+\.\s', line):
                if current_slide is None:
                    current_slide = {'title': 'Key Points', 'bullets': []}
                bullet = re.sub(r'^\d+\.\s*', '', line).strip()
                if bullet and len(current_slide['bullets']) < 6:
                    current_slide['bullets'].append(bullet[:100])

        # Add last slide
        if current_slide and current_slide.get('bullets'):
            slides.append(current_slide)

        # If no slides parsed, create one from content
        if not slides:
            # Split content into chunks
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            bullets = []
            for p in paragraphs[:6]:
                # Take first sentence or first 100 chars
                sentence = p.split('.')[0][:100]
                if sentence:
                    bullets.append(sentence)
            if bullets:
                slides.append({'title': 'Key Findings', 'bullets': bullets})

        return {
            'title': title.title(),
            'subtitle': 'Research Summary',
            'slides': slides[:12]  # Max 12 slides
        }

    async def _do_telegram(self, content: str) -> bool:
        """Send via Telegram."""
        registry = self._get_registry()
        if not registry:
            logger.warning("Cannot send telegram: registry unavailable")
            return False

        skill = registry.get_skill('telegram-sender')
        if not skill:
            logger.warning("telegram-sender skill not found")
            return False

        send_tool = skill.tools.get('send_message_tool')
        if not send_tool:
            return False

        try:
            result = send_tool({'message': content[:4000]})  # Telegram limit
            return result.get('success', False)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    async def _do_justjot(self, content: str, task: str) -> dict:
        """Create idea on JustJot.ai via MCP HTTP."""
        registry = self._get_registry()
        if not registry:
            logger.warning("Cannot create JustJot idea: registry unavailable")
            return {'success': False, 'error': 'Registry unavailable'}

        # Try justjot-mcp-http skill first (full MCP integration)
        skill = registry.get_skill('justjot-mcp-http')
        if not skill:
            # Fallback to simple justjot skill
            skill = registry.get_skill('justjot')

        if not skill:
            logger.warning("No JustJot skill found (tried justjot-mcp-http and justjot)")
            return {'success': False, 'error': 'JustJot skill not found'}

        create_tool = skill.tools.get('create_idea_tool') or skill.tools.get('create_idea')
        if not create_tool:
            return {'success': False, 'error': 'create_idea_tool not found'}

        # Extract title from content (first heading or first line)
        lines = content.strip().split('\n')
        title = "Idea from Jotty"
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('#'):
                title = line.lstrip('#').strip()[:100]
                break
            elif line and len(line) > 10:
                title = line[:100]
                break

        try:
            # Use async if tool is async
            params = {
                'title': title,
                'description': content,
                'tags': ['jotty', 'ai-generated'],
            }

            if asyncio.iscoroutinefunction(create_tool):
                result = await create_tool(params)
            else:
                result = create_tool(params)

            if result.get('success'):
                idea_id = result.get('idea_id') or result.get('id')
                return {
                    'success': True,
                    'url': f"https://justjot.ai/ideas/{idea_id}" if idea_id else None,
                    'idea_id': idea_id
                }
            return result
        except Exception as e:
            logger.warning(f"JustJot create failed: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_search_query(self, task: str) -> str:
        """Extract search query from task."""
        # Remove common prefixes
        prefixes = [
            "search for", "find", "look up", "research",
            "what is", "what are", "tell me about",
            "get information on", "search"
        ]

        query = task.lower()
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
                break

        # Limit length
        return query[:100]
