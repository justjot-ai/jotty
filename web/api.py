"""
FastAPI Web API
===============

REST and WebSocket API for Jotty Web UI.

Endpoints:
- POST /api/chat - Send message, get response
- GET /api/sessions - List all sessions
- GET /api/sessions/{id} - Get session history
- DELETE /api/sessions/{id} - Clear session
- WS /ws/chat/{session_id} - Real-time streaming
"""

import asyncio
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# WEB SEARCH - LLM decides when to search via UnifiedExecutor
# =============================================================================
# The UnifiedExecutor uses native LLM tool calling where the LLM decides:
#   - Whether to call web_search, file_read, or other input tools
#   - What output format to use (docx, pdf, telegram, etc.)
# This is the AI-native way - no regex keyword matching.
# See: core/orchestration/v2/unified_executor.py


class JottyAPI:
    """
    Jotty API handler.

    Manages request processing and session integration.
    Uses shared JottyCLI instance for command execution to ensure
    all interfaces (CLI, Telegram, Web) have identical behavior.
    """

    def __init__(self):
        self._executor = None
        self._registry = None
        self._lm_configured = False
        self._cli = None  # Shared CLI instance for commands

    def _get_cli(self):
        """Get shared JottyCLI instance for command execution."""
        if self._cli is None:
            from ..cli.app import JottyCLI
            self._cli = JottyCLI(no_color=True)  # No color for web output
        return self._cli

    def _ensure_lm_configured(self):
        """Ensure DSPy LM is configured (same as CLI)."""
        if self._lm_configured:
            return True

        import dspy

        # Check if already configured
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
            self._lm_configured = True
            return True

        try:
            # Use the same unified_lm_provider as CLI
            from ..core.foundation.unified_lm_provider import configure_dspy_lm

            # Auto-detect: tries claude-cli first (free), then API providers
            lm = configure_dspy_lm()
            if lm:
                self._lm_configured = True
                model_name = getattr(lm, 'model', None) or getattr(lm, 'model_name', 'unknown')
                logger.info(f"LLM configured: {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")

        return False

    def _get_executor(self, status_callback=None, stream_callback=None):
        """Get UnifiedExecutor with callbacks (auto-detects provider)."""
        from ..core.orchestration.v2.unified_executor import UnifiedExecutor
        return UnifiedExecutor(
            status_callback=status_callback,
            stream_callback=stream_callback
        )

    def _get_session_registry(self):
        """Get session registry."""
        if self._registry is None:
            from ..cli.repl.session import get_session_registry
            self._registry = get_session_registry()
        return self._registry

    async def _execute_with_images(self, executor, task: str, images: List[str], status_cb=None):
        """
        Execute task with image attachments using multimodal LLM.

        Args:
            executor: The executor instance
            task: The task/prompt text
            images: List of base64 image data URLs
            status_cb: Status callback function

        Returns:
            ExecutorResult with LLM response
        """
        import os

        if status_cb:
            status_cb("analyzing", f"Processing {len(images)} image(s)...")

        # Create result dataclass
        from dataclasses import dataclass
        @dataclass
        class ImageResult:
            success: bool = True
            content: str = ""
            output_format: str = "markdown"
            output_path: str = None
            steps_taken: int = 1
            error: str = None

        try:
            # Build multimodal message content
            message_content = []

            # Add images first
            for i, img_data in enumerate(images):
                # Extract base64 data from data URL (data:image/png;base64,...)
                if img_data.startswith('data:'):
                    # Split off the header and get base64 data
                    header, b64_data = img_data.split(',', 1)
                    media_type = header.split(':')[1].split(';')[0]
                else:
                    b64_data = img_data
                    media_type = "image/jpeg"

                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data
                    }
                })

            # Add text content
            message_content.append({
                "type": "text",
                "text": task
            })

            if status_cb:
                status_cb("processing", "Analyzing with vision model...")

            logger.info(f"_execute_with_images: processing {len(images)} images")
            logger.info(f"  First image data starts with: {images[0][:60] if images else 'none'}...")

            # Try Anthropic SDK directly (best for vision)
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            logger.info(f"  ANTHROPIC_API_KEY present: {bool(api_key)}")
            if api_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": message_content}]
                    )

                    # Extract text from response
                    content = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            content += block.text

                    if status_cb:
                        status_cb("complete", "Image analysis complete")

                    return ImageResult(success=True, content=content)

                except Exception as e:
                    logger.warning(f"Anthropic vision failed: {e}")

            # Try OpenAI SDK (GPT-4V format)
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                try:
                    import openai
                    client = openai.OpenAI(api_key=openai_key)

                    # Convert to OpenAI format
                    openai_content = []
                    for item in message_content:
                        if item["type"] == "image":
                            # OpenAI uses URL format for base64
                            b64 = item["source"]["data"]
                            media = item["source"]["media_type"]
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media};base64,{b64}"}
                            })
                        else:
                            openai_content.append(item)

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": openai_content}]
                    )

                    content = response.choices[0].message.content

                    if status_cb:
                        status_cb("complete", "Image analysis complete")

                    return ImageResult(success=True, content=content)

                except Exception as e:
                    logger.warning(f"OpenAI vision failed: {e}")

            # Try OpenRouter (supports Claude vision via API)
            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            logger.info(f"Checking OpenRouter: key present = {bool(openrouter_key)}")
            if openrouter_key:
                try:
                    import openai
                    client = openai.OpenAI(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1"
                    )

                    # Convert to OpenAI format (OpenRouter uses same format)
                    openrouter_content = []
                    for item in message_content:
                        if item["type"] == "image":
                            b64 = item["source"]["data"]
                            media = item["source"]["media_type"]
                            openrouter_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media};base64,{b64}"}
                            })
                        else:
                            openrouter_content.append(item)

                    logger.info("Trying OpenRouter for vision...")
                    response = client.chat.completions.create(
                        model="anthropic/claude-sonnet-4",  # Vision-capable model
                        max_tokens=4096,
                        messages=[{"role": "user", "content": openrouter_content}]
                    )

                    content = response.choices[0].message.content

                    if status_cb:
                        status_cb("complete", "Image analysis complete")

                    return ImageResult(success=True, content=content)

                except Exception as e:
                    logger.warning(f"OpenRouter vision failed: {e}")

            # Fallback: describe that images were provided but couldn't be processed
            logger.warning("No vision-capable API available, falling back to text-only")
            enhanced_task = f"[Note: User attached {len(images)} image(s) but no vision API is configured. Please respond to: {task}]"
            return await executor.execute(enhanced_task)

        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            # Fallback to text-only execution
            return await executor.execute(task)

    async def process_message(
        self,
        message: str,
        session_id: str,
        user_id: str = "web_user",
        stream_callback=None,
        status_callback=None,
        attachments: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message with optional image attachments.

        Args:
            message: User message
            session_id: Session ID
            user_id: User identifier
            stream_callback: Optional callback for streaming
            status_callback: Optional callback for status updates
            attachments: List of image attachments [{type, name, data, size}]

        Returns:
            Response dict with content, output_path, etc.
        """
        from ..cli.repl.session import InterfaceType

        # Get session
        registry = self._get_session_registry()
        session = registry.get_session(
            session_id,
            create=True,
            interface=InterfaceType.WEB
        )

        # Process attachments: images (for vision) and documents (for text extraction)
        image_descriptions = []
        image_data_list = []
        document_contexts = []

        if attachments:
            for att in attachments:
                if att.get("type") == "image" and att.get("data"):
                    # Image: use for multimodal vision
                    image_data_list.append(att["data"])
                    image_descriptions.append(f"[Attached image: {att.get('name', 'image')}]")
                elif att.get("type") == "document" and att.get("docId"):
                    # Document: extract text for context
                    try:
                        from .documents import get_document_processor
                        processor = get_document_processor()
                        doc_text = processor.get_document_text(att["docId"])
                        if doc_text:
                            # Truncate if too long
                            max_len = 8000
                            if len(doc_text) > max_len:
                                doc_text = doc_text[:max_len] + f"\n\n[... truncated, {len(doc_text) - max_len} more chars ...]"
                            document_contexts.append(f"=== Content from {att.get('name', 'document')} ===\n{doc_text}")
                    except Exception as e:
                        logger.error(f"Failed to extract document text: {e}")

        # Build message content with attachments context
        # NOTE: Web search is handled by UnifiedExecutor via LLM tool calling (not regex)
        # The LLM uses web_search tool when it needs current info
        full_message = message
        context_parts = []

        if document_contexts:
            context_parts.append("ATTACHED DOCUMENTS:\n\n" + "\n\n".join(document_contexts))

        if image_descriptions:
            context_parts.append(' '.join(image_descriptions))

        if context_parts:
            full_message = "\n\n".join(context_parts) + f"\n\nUSER REQUEST: {message}" if message else "\n\n".join(context_parts)

        # Add user message
        message_id = str(uuid.uuid4())[:12]
        session.add_message(
            role="user",
            content=full_message,
            interface=InterfaceType.WEB,
            user_id=user_id,
            metadata={"message_id": message_id, "has_images": len(image_data_list) > 0}
        )

        # Create status callback that calls both logger and external callback
        def status_cb(stage, detail):
            logger.debug(f"Status: {stage} - {detail}")
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception as e:
                    logger.debug(f"Status callback error: {e}")

        executor = self._get_executor(
            status_callback=status_cb,
            stream_callback=stream_callback
        )

        # Build task with conversation context
        history = session.get_history()
        if len(history) > 1:
            # Include recent conversation for context (last 5 exchanges max)
            context_messages = history[-10:-1]  # Exclude current message
            if context_messages:
                context_str = "\n".join([
                    f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:500]}"
                    for m in context_messages
                ])
                task_with_context = f"""Previous conversation:
{context_str}

Current request: {full_message}"""
            else:
                task_with_context = full_message
        else:
            task_with_context = full_message

        try:
            # Execute with context and images
            # If we have images, try to use multimodal LLM
            if image_data_list:
                result = await self._execute_with_images(executor, task_with_context, image_data_list, status_cb)
            else:
                result = await executor.execute(task_with_context)

            response_id = str(uuid.uuid4())[:12]

            if result.success:
                # Add assistant response
                session.add_message(
                    role="assistant",
                    content=result.content,
                    interface=InterfaceType.WEB,
                    metadata={
                        "message_id": response_id,
                        "output_format": result.output_format,
                        "output_path": result.output_path,
                        "steps": result.steps_taken,
                    }
                )

                return {
                    "success": True,
                    "message_id": response_id,
                    "content": result.content,
                    "output_format": result.output_format,
                    "output_path": result.output_path,
                    "steps": result.steps_taken,
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Unknown error",
                    "steps": result.steps_taken,
                }

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def get_commands(self) -> List[Dict[str, Any]]:
        """Get available CLI commands."""
        from ..cli.commands import CommandRegistry
        from ..cli.commands import register_all_commands

        registry = CommandRegistry()
        register_all_commands(registry)

        commands = []
        for name, cmd in registry._commands.items():
            commands.append({
                "name": name,
                "description": getattr(cmd, "description", ""),
                "usage": getattr(cmd, "usage", f"/{name}"),
                "aliases": getattr(cmd, "aliases", []),
            })
        return commands

    async def execute_command(
        self,
        command: str,
        args: str = "",
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute a CLI command using the shared JottyCLI instance.

        This ensures all interfaces (CLI, Telegram, Web) have identical
        command behavior - they all use the same JottyCLI core.
        """
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        try:
            # Get shared CLI instance
            cli = self._get_cli()

            # Capture stdout/stderr for web display
            output_buffer = io.StringIO()

            # Execute command through CLI's command handler
            cmd_input = f"/{command} {args}".strip()

            # Temporarily replace renderer output capture
            original_print = cli.renderer.print
            captured_output = []

            def capture_print(text, *args, **kwargs):
                # Strip rich markup for web
                import re
                clean_text = re.sub(r'\[/?[^\]]+\]', '', str(text))
                captured_output.append(clean_text)

            # Monkey-patch renderer methods to capture output
            cli.renderer.print = capture_print
            cli.renderer.info = lambda t: captured_output.append(f"ℹ️ {t}")
            cli.renderer.success = lambda t: captured_output.append(f"✅ {t}")
            cli.renderer.warning = lambda t: captured_output.append(f"⚠️ {t}")
            cli.renderer.error = lambda t: captured_output.append(f"❌ {t}")

            # Capture panel output
            original_panel = getattr(cli.renderer, 'panel', None)
            cli.renderer.panel = lambda content, **kwargs: captured_output.append(f"📋 {kwargs.get('title', 'Panel')}:\n{content}")

            # Capture tree output
            original_tree = getattr(cli.renderer, 'tree', None)
            def capture_tree(data, **kwargs):
                title = kwargs.get('title', 'Data')
                if isinstance(data, dict):
                    lines = [f"🌳 {title}:"]
                    for k, v in data.items():
                        lines.append(f"  • {k}: {v}")
                    captured_output.append("\n".join(lines))
                else:
                    captured_output.append(f"🌳 {title}: {data}")
            cli.renderer.tree = capture_tree

            # Capture table output - patch the tables component
            original_print_table = cli.renderer.tables.print_table
            def capture_table(table):
                # Use Rich Console to render to string, then strip ANSI
                try:
                    from rich.console import Console
                    from io import StringIO
                    string_io = StringIO()
                    console = Console(file=string_io, force_terminal=False, no_color=True)
                    console.print(table)
                    table_text = string_io.getvalue()
                    captured_output.append(table_text)
                except Exception:
                    # Fallback: just convert to string
                    captured_output.append(str(table))
            cli.renderer.tables.print_table = capture_table

            try:
                # Execute via CLI's command handler
                result = await cli._handle_command(cmd_input)

                return {
                    "success": True,
                    "output": "\n".join(captured_output) if captured_output else "Command executed",
                    "data": None,
                }
            finally:
                # Restore original methods
                cli.renderer.print = original_print
                cli.renderer.tables.print_table = original_print_table
                if original_panel:
                    cli.renderer.panel = original_panel
                if original_tree:
                    cli.renderer.tree = original_tree

        except Exception as e:
            logger.error(f"Command execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions with metadata."""
        from ..cli.repl.session import SessionManager
        import json

        manager = SessionManager()
        sessions = manager.list_sessions()

        # Enhance sessions with metadata fields
        for session in sessions:
            session_file = manager.session_dir / f"{session.get('session_id')}.json"
            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    # Add metadata fields if they exist
                    for key in ['title', 'isPinned', 'isArchived', 'folderId']:
                        if key in data:
                            session[key] = data[key]
                except Exception:
                    pass
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        from ..cli.repl.session import InterfaceType

        registry = self._get_session_registry()
        session = registry.get_session(
            session_id,
            create=False,
            interface=InterfaceType.WEB
        )

        if session:
            return {
                **session.to_dict(),
                "history": session.get_history(),
            }
        return None

    def clear_session(self, session_id: str) -> bool:
        """Clear session history."""
        from ..cli.repl.session import InterfaceType

        registry = self._get_session_registry()
        session = registry.get_session(
            session_id,
            create=False,
            interface=InterfaceType.WEB
        )

        if session:
            session.clear_history()
            session.save()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        from ..cli.repl.session import SessionManager

        registry = self._get_session_registry()
        registry.remove_session(session_id)

        manager = SessionManager()
        manager.delete_session(session_id)
        return True

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session metadata (title, isPinned, isArchived, folderId)."""
        from ..cli.repl.session import SessionManager, InterfaceType

        manager = SessionManager()
        session_file = manager.session_dir / f"{session_id}.json"

        if not session_file.exists():
            return False

        try:
            import json
            with open(session_file, 'r') as f:
                data = json.load(f)

            # Update allowed fields
            for key in ['title', 'isPinned', 'isArchived', 'folderId']:
                if key in updates:
                    data[key] = updates[key]

            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False

    def get_folders(self) -> List[Dict[str, Any]]:
        """Get all folders."""
        folder_file = Path.home() / ".jotty" / "folders.json"
        if folder_file.exists():
            try:
                import json
                with open(folder_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def create_folder(self, folder: Dict[str, Any]) -> bool:
        """Create a new folder."""
        import json
        folder_file = Path.home() / ".jotty" / "folders.json"
        folder_file.parent.mkdir(parents=True, exist_ok=True)

        folders = self.get_folders()
        folders.append(folder)

        with open(folder_file, 'w') as f:
            json.dump(folders, f, indent=2)
        return True

    def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder."""
        import json
        folder_file = Path.home() / ".jotty" / "folders.json"

        folders = self.get_folders()
        folders = [f for f in folders if f.get('id') != folder_id]

        with open(folder_file, 'w') as f:
            json.dump(folders, f, indent=2)
        return True

    def save_folders(self, folders: List[Dict[str, Any]]) -> bool:
        """Save all folders (bulk update for reordering, renaming, color changes)."""
        import json
        folder_file = Path.home() / ".jotty" / "folders.json"
        folder_file.parent.mkdir(parents=True, exist_ok=True)

        with open(folder_file, 'w') as f:
            json.dump(folders, f, indent=2)
        return True


def create_app() -> "FastAPI":
    """
    Create FastAPI application.

    Returns:
        Configured FastAPI app
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(
        title="Jotty API",
        description="Jotty AI Assistant API",
        version="1.0.0"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API handler
    api = JottyAPI()

    # Request models
    class ChatRequest(BaseModel):
        message: str
        session_id: Optional[str] = None

    class ChatResponse(BaseModel):
        success: bool
        message_id: Optional[str] = None
        content: Optional[str] = None
        output_format: Optional[str] = None
        output_path: Optional[str] = None
        error: Optional[str] = None

    # Routes
    @app.get("/")
    async def root():
        """Serve chat UI."""
        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Jotty API", "docs": "/docs"}

    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    # ==========================================================================
    # DRY: Unified Registry APIs - Uses existing registries, no duplication
    # ==========================================================================

    @app.get("/api/widgets")
    async def get_widgets():
        """Get all widgets from unified registry."""
        try:
            from ..core.registry import get_unified_registry
            registry = get_unified_registry()
            return registry.get_widgets()
        except ImportError:
            # Fallback if registry not available
            return {"widgets": [], "categories": []}

    @app.get("/api/tools")
    async def get_tools():
        """Get all tools from unified registry."""
        try:
            from ..core.registry import get_unified_registry
            registry = get_unified_registry()
            return registry.get_tools()
        except ImportError:
            return {"tools": [], "categories": []}

    @app.get("/api/capabilities")
    async def get_capabilities():
        """Get unified tools + widgets + defaults (DRY single source of truth)."""
        # Fallback widgets and tools when registry is empty
        fallback_widgets = [
            {"value": "markdown", "label": "Markdown", "icon": "📝", "category": "Content"},
            {"value": "code", "label": "Code Block", "icon": "💻", "category": "Content"},
            {"value": "chart", "label": "Chart", "icon": "📊", "category": "Visualization"},
            {"value": "table", "label": "Table", "icon": "📋", "category": "Data"},
            {"value": "image", "label": "Image", "icon": "🖼️", "category": "Media"},
            {"value": "pdf", "label": "PDF Export", "icon": "📄", "category": "Export"},
            {"value": "slides", "label": "Slides", "icon": "🎯", "category": "Export"},
            {"value": "mermaid", "label": "Mermaid Diagram", "icon": "🔀", "category": "Visualization"},
            {"value": "latex", "label": "LaTeX Math", "icon": "∑", "category": "Content"},
            {"value": "json", "label": "JSON Viewer", "icon": "{ }", "category": "Data"},
        ]
        fallback_tools = [
            {"name": "web_search", "description": "Search the web", "category": "Research"},
            {"name": "web_browse", "description": "Browse and read web pages", "category": "Research"},
            {"name": "file_read", "description": "Read local files", "category": "Files"},
            {"name": "file_write", "description": "Write to local files", "category": "Files"},
            {"name": "code_execute", "description": "Execute code snippets", "category": "Development"},
            {"name": "shell", "description": "Run shell commands", "category": "System"},
            {"name": "image_generate", "description": "Generate images with AI", "category": "Media"},
            {"name": "pdf_generate", "description": "Generate PDF documents", "category": "Export"},
        ]

        try:
            from ..core.registry import get_unified_registry
            registry = get_unified_registry()
            all_data = registry.get_all()
            defaults = registry.get_enabled_defaults()

            # Use fallbacks if registry returns empty
            widgets = all_data.get("widgets", {}).get("available", [])
            tools = all_data.get("tools", {}).get("available", [])

            if not widgets:
                all_data["widgets"] = {"available": fallback_widgets, "categories": ["Content", "Visualization", "Data", "Media", "Export"]}
            if not tools:
                all_data["tools"] = {"available": fallback_tools, "categories": ["Research", "Files", "Development", "System", "Media", "Export"]}

            return {
                **all_data,
                "defaults": defaults if defaults.get("widgets") or defaults.get("tools") else {
                    "widgets": ["markdown", "code", "chart", "table"],
                    "tools": ["web_search", "web_browse"]
                }
            }
        except (ImportError, Exception):
            return {
                "widgets": {"available": fallback_widgets, "categories": ["Content", "Visualization", "Data", "Media", "Export"]},
                "tools": {"available": fallback_tools, "categories": ["Research", "Files", "Development", "System", "Media", "Export"]},
                "defaults": {"widgets": ["markdown", "code", "chart", "table"], "tools": ["web_search", "web_browse"]}
            }

    @app.get("/api/agents")
    async def list_agents():
        """Get agents from skills registry."""
        try:
            from ..core.registry import get_skills_registry
            registry = get_skills_registry()
            if hasattr(registry, 'list_agents_from_skills'):
                agents = registry.list_agents_from_skills()
                # Handle if it's a coroutine
                if hasattr(agents, '__await__'):
                    import asyncio
                    agents = await agents
                return {"agents": agents if agents else [], "count": len(agents) if agents else 0}
            return {"agents": [], "count": 0}
        except (ImportError, Exception) as e:
            # Fallback: Return basic agent list from CLI commands
            return {
                "agents": [
                    {"id": "research", "name": "Research Agent", "description": "Web research and synthesis", "category": "research"},
                    {"id": "code", "name": "Code Agent", "description": "Code analysis and generation", "category": "development"},
                    {"id": "ml", "name": "ML Agent", "description": "Machine learning pipeline", "category": "ml"},
                ],
                "count": 3
            }

    @app.get("/api/providers")
    async def list_providers():
        """Get LM providers status with detailed model info."""
        import os
        import shutil

        providers = {}

        # All available providers with their models
        provider_configs = {
            "anthropic": {
                "name": "Anthropic",
                "icon": "🅰️",
                "env_key": "ANTHROPIC_API_KEY",
                "models": [
                    {"id": "claude-sonnet-4-20250514", "name": "Claude 4 Sonnet", "context": "200K", "vision": True, "recommended": True},
                    {"id": "claude-opus-4-20250514", "name": "Claude 4 Opus", "context": "200K", "vision": True},
                    {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "context": "200K", "vision": True, "fast": True},
                ]
            },
            "openai": {
                "name": "OpenAI",
                "icon": "🤖",
                "env_key": "OPENAI_API_KEY",
                "models": [
                    {"id": "gpt-4o", "name": "GPT-4o", "context": "128K", "vision": True, "recommended": True},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "context": "128K", "vision": True},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context": "16K", "fast": True},
                ]
            },
            "google": {
                "name": "Google",
                "icon": "🔷",
                "env_key": "GOOGLE_API_KEY",
                "models": [
                    {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash", "context": "1M", "vision": True, "recommended": True},
                    {"id": "gemini-pro", "name": "Gemini Pro", "context": "32K"},
                    {"id": "gemini-pro-vision", "name": "Gemini Pro Vision", "context": "32K", "vision": True},
                ]
            },
            "groq": {
                "name": "Groq",
                "icon": "⚡",
                "env_key": "GROQ_API_KEY",
                "models": [
                    {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B", "context": "128K", "fast": True, "recommended": True},
                    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "context": "128K", "fast": True},
                    {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "context": "32K", "fast": True},
                ]
            },
            "openrouter": {
                "name": "OpenRouter",
                "icon": "🌐",
                "env_key": "OPENROUTER_API_KEY",
                "models": [
                    {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus (OR)", "context": "200K", "vision": True},
                    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B (Free)", "context": "128K", "free": True, "recommended": True},
                    {"id": "openai/gpt-4", "name": "GPT-4 (OR)", "context": "128K"},
                ]
            },
            "claude-cli": {
                "name": "Claude CLI",
                "icon": "💻",
                "env_key": None,  # Check for binary
                "models": [
                    {"id": "sonnet", "name": "Sonnet (via CLI)", "context": "200K", "vision": True, "local": True, "recommended": True},
                    {"id": "opus", "name": "Opus (via CLI)", "context": "200K", "vision": True, "local": True},
                    {"id": "haiku", "name": "Haiku (via CLI)", "context": "200K", "vision": True, "local": True, "fast": True},
                ]
            },
        }

        for provider_id, config in provider_configs.items():
            if config["env_key"]:
                configured = bool(os.environ.get(config["env_key"]))
            elif provider_id == "claude-cli":
                configured = shutil.which("claude") is not None
            else:
                configured = False

            providers[provider_id] = {
                "name": config["name"],
                "icon": config["icon"],
                "configured": configured,
                "models": config["models"] if configured else []
            }

        # Get current active model
        current_model = None
        try:
            import dspy
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                lm = dspy.settings.lm
                # Get model name from wrapped LM if ContextAwareLM
                if hasattr(lm, '_wrapped'):
                    lm = lm._wrapped
                current_model = getattr(lm, 'model', None) or getattr(lm, 'model_name', None)
        except Exception:
            pass

        return {
            "providers": providers,
            "current_model": current_model,
            "current_provider": None  # Will be inferred from model name
        }

    @app.get("/api/models")
    async def list_models():
        """Get flat list of all available models for model selector."""
        providers_response = await list_providers()
        providers = providers_response.get("providers", {})

        models = []
        for provider_id, provider in providers.items():
            if provider.get("configured"):
                for model in provider.get("models", []):
                    models.append({
                        "id": f"{provider_id}/{model['id']}",
                        "provider": provider_id,
                        "provider_name": provider["name"],
                        "provider_icon": provider["icon"],
                        "model_id": model["id"],
                        "name": model["name"],
                        "context": model.get("context", "N/A"),
                        "vision": model.get("vision", False),
                        "fast": model.get("fast", False),
                        "free": model.get("free", False),
                        "local": model.get("local", False),
                        "recommended": model.get("recommended", False),
                    })

        # Sort: recommended first, then by provider
        models.sort(key=lambda m: (not m["recommended"], m["provider"]))

        return {
            "models": models,
            "current": providers_response.get("current_model"),
            "count": len(models)
        }

    class SetModelRequest(BaseModel):
        provider: str
        model: str

    @app.post("/api/models/set")
    async def set_model(request: SetModelRequest):
        """Set the active LLM model by storing preference (avoids DSPy threading issues)."""
        try:
            # Store model preference - will be used when creating LLM instances
            api._current_provider = request.provider
            api._current_model = request.model

            # Also set environment variable so UnifiedLMProvider picks it up
            import os
            os.environ["JOTTY_LLM_PROVIDER"] = request.provider
            os.environ["JOTTY_LLM_MODEL"] = request.model

            # For vision calls, we use Anthropic SDK directly anyway
            # This preference is for text-only LLM calls

            return {
                "success": True,
                "provider": request.provider,
                "model": request.model,
                "message": f"Model preference set to {request.provider}/{request.model}"
            }
        except Exception as e:
            logger.error(f"Failed to set model: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    class SwarmRequest(BaseModel):
        task: str
        mode: str = "auto"  # auto | manual | workflow
        agents: Optional[List[str]] = None
        workflow: Optional[dict] = None
        session_id: Optional[str] = None

    @app.post("/api/agents/swarm")
    async def execute_swarm(request: SwarmRequest):
        """Execute multi-agent swarm."""
        try:
            from ..core.agents import SwarmManager
            manager = SwarmManager()
            result = await manager.execute(
                task=request.task,
                mode=request.mode,
                agents=request.agents,
                workflow=request.workflow
            )
            return {"success": True, "result": result}
        except ImportError:
            # Fallback: Execute through regular chat with swarm hint
            session_id = request.session_id or str(uuid.uuid4())[:8]
            enhanced_task = f"[Swarm Mode: {request.mode}] {request.task}"
            result = await api.process_message(
                message=enhanced_task,
                session_id=session_id
            )
            return {"success": result.get("success", False), "result": result}
        except Exception as e:
            logger.error(f"Swarm execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # CLI Commands endpoints
    @app.get("/api/commands")
    async def list_commands():
        """List available CLI commands."""
        commands = api.get_commands()
        return {"commands": commands}

    class CommandRequest(BaseModel):
        command: str
        args: str = ""
        session_id: Optional[str] = None

    @app.post("/api/commands/execute")
    async def execute_command(request: CommandRequest):
        """Execute a CLI command."""
        result = await api.execute_command(
            command=request.command,
            args=request.args,
            session_id=request.session_id
        )
        return result

    @app.get("/api/commands/stream")
    async def stream_command(command: str, args: str = "", session_id: Optional[str] = None):
        """
        SSE streaming command execution endpoint.

        Streams command output in real-time for long-running commands like /ml.
        """
        from starlette.responses import StreamingResponse
        import json
        import queue
        import threading
        import sys
        import io
        import re

        async def event_generator():
            # Thread-safe queue for output
            output_queue = queue.Queue()
            result_holder = {"done": False, "success": True, "error": None}

            # Padding to flush proxy buffers
            padding = " " * 16384  # 16KB padding to flush proxy buffers

            # Send initial event
            yield f"data: {json.dumps({'type': 'started', 'command': command})}\n\n{padding}"

            def clean_text(text):
                """Remove ANSI codes and Rich markup."""
                clean = re.sub(r'\x1b\[[0-9;]*m', '', str(text))
                clean = re.sub(r'\[/?[^\]]*\]', '', clean)
                return clean.strip()

            def add_output(text):
                """Add text to output queue."""
                cleaned = clean_text(text)
                if cleaned:
                    output_queue.put(cleaned)

            # Custom stdout wrapper
            class QueueWriter:
                def __init__(self, q, original):
                    self.queue = q
                    self.original = original
                    self.buffer = ""

                def write(self, text):
                    if text:
                        # Also write to original for logging
                        self.original.write(text)
                        # Process for queue
                        if '\n' in text or '\r' in text:
                            parts = re.split(r'[\n\r]+', self.buffer + text)
                            self.buffer = ""
                            for part in parts:
                                cleaned = clean_text(part)
                                if cleaned:
                                    self.queue.put(cleaned)
                        else:
                            self.buffer += text
                    return len(text) if text else 0

                def flush(self):
                    self.original.flush()
                    if self.buffer:
                        cleaned = clean_text(self.buffer)
                        if cleaned:
                            self.queue.put(cleaned)
                        self.buffer = ""

                def isatty(self):
                    return False

            try:
                # Get CLI instance (create fresh one for thread safety)
                from ..cli.app import JottyCLI
                cli = JottyCLI(no_color=True)

                def capture_print(text, *a, **kw):
                    add_output(text)

                # Monkey-patch renderer methods
                original_print = cli.renderer.print
                cli.renderer.print = capture_print
                cli.renderer.info = lambda t: add_output(f"ℹ️ {t}")
                cli.renderer.success = lambda t: add_output(f"✅ {t}")
                cli.renderer.warning = lambda t: add_output(f"⚠️ {t}")
                cli.renderer.error = lambda t: add_output(f"❌ {t}")

                # Capture panel output
                original_panel = getattr(cli.renderer, 'panel', None)
                cli.renderer.panel = lambda content, **kwargs: add_output(f"📋 {kwargs.get('title', 'Panel')}:\n{content}")

                # Capture tree output
                original_tree = getattr(cli.renderer, 'tree', None)
                def capture_tree(data, **kwargs):
                    title = kwargs.get('title', 'Data')
                    if isinstance(data, dict):
                        lines = [f"🌳 {title}:"]
                        for k, v in data.items():
                            lines.append(f"  • {k}: {v}")
                        add_output("\n".join(lines))
                    else:
                        add_output(f"🌳 {title}: {data}")
                cli.renderer.tree = capture_tree

                # Capture table output
                original_print_table = cli.renderer.tables.print_table
                def capture_table(table):
                    try:
                        from rich.console import Console
                        string_io = io.StringIO()
                        console = Console(file=string_io, force_terminal=False, no_color=True)
                        console.print(table)
                        add_output(string_io.getvalue())
                    except Exception:
                        add_output(str(table))
                cli.renderer.tables.print_table = capture_table

                # Wrap stdout to capture print statements
                original_stdout = sys.stdout
                queue_writer = QueueWriter(output_queue, original_stdout)

                # Run command in thread
                import concurrent.futures

                def run_command_sync():
                    """Run command synchronously in thread."""
                    # Redirect stdout in this thread
                    sys.stdout = queue_writer
                    try:
                        cmd_input = f"/{command} {args}".strip()
                        # Create new event loop for thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(cli._handle_command(cmd_input))
                        finally:
                            loop.close()
                    except Exception as e:
                        result_holder["success"] = False
                        result_holder["error"] = str(e)
                        add_output(f"❌ Error: {e}")
                    finally:
                        sys.stdout = original_stdout
                        queue_writer.flush()
                        result_holder["done"] = True

                # Start in thread pool
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                executor.submit(run_command_sync)

                # Stream output while command runs
                while not result_holder["done"]:
                    # Drain queue and send events
                    while True:
                        try:
                            line = output_queue.get_nowait()
                            yield f"data: {json.dumps({'type': 'output', 'line': line})}\n\n{padding}"
                        except queue.Empty:
                            break

                    await asyncio.sleep(0.15)
                    yield f": keepalive\n\n{padding}"

                # Drain remaining items from queue
                while True:
                    try:
                        line = output_queue.get_nowait()
                        yield f"data: {json.dumps({'type': 'output', 'line': line})}\n\n"
                    except queue.Empty:
                        break

                # Send completion
                yield f"data: {json.dumps({'type': 'complete', 'success': result_holder['success'], 'error': result_holder['error']})}\n\n"

            except Exception as e:
                logger.error(f"Stream command error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Send a chat message and get response.

        Non-streaming endpoint for simple integrations.
        """
        session_id = request.session_id or str(uuid.uuid4())[:8]

        result = await api.process_message(
            message=request.message,
            session_id=session_id
        )

        return ChatResponse(**result)

    @app.get("/api/chat/stream")
    async def chat_stream(message: str, session_id: Optional[str] = None):
        """
        SSE streaming chat endpoint.

        Returns Server-Sent Events for real-time streaming.
        Uses asyncio.Queue for proper async handling without blocking.
        """
        from starlette.responses import StreamingResponse
        import json

        session_id = session_id or str(uuid.uuid4())[:8]

        # Use asyncio.Queue for non-blocking async communication
        event_queue = asyncio.Queue()
        done_event = asyncio.Event()

        async def process_message_async():
            """Process message and put events in queue."""
            try:
                async def async_stream_cb(chunk: str):
                    """Async callback that puts chunks in queue."""
                    logger.info(f"STREAM_CB: '{chunk[:30]}...' queued")
                    await event_queue.put({"type": "stream", "chunk": chunk})

                async def async_status_cb(stage: str, detail: str):
                    """Async callback that puts status in queue."""
                    logger.info(f"STATUS_CB: {stage} - {detail}")
                    await event_queue.put({"type": "status", "stage": stage, "detail": detail})

                result = await api.process_message(
                    message=message,
                    session_id=session_id,
                    stream_callback=async_stream_cb,
                    status_callback=async_status_cb
                )
                await event_queue.put({"type": "complete", "result": result})
            except Exception as e:
                logger.error(f"Chat processing error: {e}", exc_info=True)
                await event_queue.put({"type": "complete", "result": {"success": False, "error": str(e)}})
            finally:
                done_event.set()

        async def event_generator():
            # Start processing as a background task
            task = asyncio.create_task(process_message_async())

            # Send initial connected event
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            try:
                while not done_event.is_set() or not event_queue.empty():
                    try:
                        # Wait for event with timeout
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        event_type = event.get("type")
                        logger.info(f"SENDING event: {event_type}")
                        yield f"data: {json.dumps(event)}\n\n"

                        # If complete event, we're done
                        if event_type == "complete":
                            break
                    except asyncio.TimeoutError:
                        # Send keepalive comment
                        yield ": keepalive\n\n"
            finally:
                # Ensure task is cleaned up
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, no-transform, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )

    @app.get("/api/sessions")
    async def list_sessions():
        """List all sessions."""
        sessions = api.get_sessions()
        return {"sessions": sessions}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session details and history."""
        session = api.get_session(session_id)
        if session:
            return session
        raise HTTPException(status_code=404, detail="Session not found")

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session."""
        success = api.delete_session(session_id)
        return {"success": success}

    @app.post("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        """Clear session history."""
        success = api.clear_session(session_id)
        return {"success": success}

    class SessionUpdateRequest(BaseModel):
        title: Optional[str] = None
        isPinned: Optional[bool] = None
        isArchived: Optional[bool] = None
        folderId: Optional[str] = None

    @app.patch("/api/sessions/{session_id}")
    async def update_session(session_id: str, request: SessionUpdateRequest):
        """Update session metadata (title, pin, archive, folder)."""
        updates = request.dict(exclude_none=True)
        success = api.update_session(session_id, updates)
        return {"success": success}

    # Folder management endpoints
    @app.get("/api/folders")
    async def list_folders():
        """List all folders."""
        folders = api.get_folders()
        return {"folders": folders}

    class FolderRequest(BaseModel):
        id: str
        name: str
        color: str = "#3b82f6"
        order: int = 0

    @app.post("/api/folders")
    async def create_folder(request: FolderRequest):
        """Create a new folder."""
        folder = request.dict()
        success = api.create_folder(folder)
        return {"success": success, "folder": folder}

    @app.delete("/api/folders/{folder_id}")
    async def delete_folder(folder_id: str):
        """Delete a folder."""
        success = api.delete_folder(folder_id)
        return {"success": success}

    class BulkFoldersRequest(BaseModel):
        folders: List[dict]

    @app.put("/api/folders/bulk")
    async def bulk_update_folders(request: BulkFoldersRequest):
        """Bulk update all folders (for reordering, renaming, etc.)."""
        success = api.save_folders(request.folders)
        return {"success": success}

    # ===== DOCUMENT UPLOAD & RAG ENDPOINTS =====

    @app.post("/api/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        folder_id: Optional[str] = Form(None)
    ):
        """
        Upload a document for RAG processing.

        Supported formats: PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML
        """
        from .documents import get_document_processor

        try:
            processor = get_document_processor()
            content = await file.read()

            doc_info = await processor.upload_document(
                file_content=content,
                filename=file.filename,
                folder_id=folder_id
            )

            return {
                "success": True,
                "document": doc_info
            }
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail=f"Document processing not available. Install dependencies: pip install chromadb sentence-transformers unstructured[all-docs]. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/documents")
    async def list_documents(folder_id: Optional[str] = None):
        """List all documents, optionally filtered by folder."""
        from .documents import get_document_processor

        try:
            processor = get_document_processor()

            if folder_id:
                docs = processor.get_folder_documents(folder_id)
            else:
                docs = list(processor._docs_index.get("documents", {}).values())

            return {"documents": docs}
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {"documents": []}

    @app.get("/api/documents/{doc_id}")
    async def get_document(doc_id: str, include_text: bool = False):
        """Get document info and optionally its text content."""
        from .documents import get_document_processor

        processor = get_document_processor()
        doc = processor.get_document(doc_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        result = {"document": doc}
        if include_text:
            result["text"] = processor.get_document_text(doc_id)

        return result

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: str):
        """Delete a document and its embeddings."""
        from .documents import get_document_processor

        processor = get_document_processor()
        success = processor.delete_document(doc_id)

        return {"success": success}

    @app.post("/api/documents/search")
    async def search_documents(request: dict):
        """
        Search documents using vector similarity.

        Args:
            query: Search text
            folder_id: Optional folder filter
            doc_ids: Optional list of document IDs to search
            n_results: Number of results (default 5)
        """
        from .documents import get_document_processor

        try:
            processor = get_document_processor()

            results = processor.search_documents(
                query=request.get("query", ""),
                folder_id=request.get("folder_id"),
                doc_ids=request.get("doc_ids"),
                n_results=request.get("n_results", 5)
            )

            return {"results": results}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== RAG CONFIGURATION ENDPOINTS =====

    @app.get("/api/rag/config")
    async def get_rag_config():
        """Get current RAG configuration."""
        from .documents import get_document_processor, RAGConfig

        processor = get_document_processor()
        return {
            "config": processor.config.to_dict(),
            "available_models": RAGConfig.EMBEDDING_MODELS
        }

    class RAGConfigUpdateRequest(BaseModel):
        chunk_size: Optional[int] = None
        overlap: Optional[int] = None
        embedding_model: Optional[str] = None

    @app.post("/api/rag/config")
    async def update_rag_config(request: RAGConfigUpdateRequest):
        """Update RAG configuration."""
        from .documents import get_document_processor, RAGConfig

        processor = get_document_processor()

        # Validate embedding model
        if request.embedding_model and request.embedding_model not in RAGConfig.EMBEDDING_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid embedding model. Available: {list(RAGConfig.EMBEDDING_MODELS.keys())}"
            )

        # Validate chunk_size
        if request.chunk_size is not None and (request.chunk_size < 100 or request.chunk_size > 4000):
            raise HTTPException(status_code=400, detail="chunk_size must be between 100 and 4000")

        # Validate overlap
        if request.overlap is not None and (request.overlap < 0 or request.overlap > 500):
            raise HTTPException(status_code=400, detail="overlap must be between 0 and 500")

        config = processor.update_config(
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            embedding_model=request.embedding_model
        )

        return {"success": True, "config": config.to_dict()}

    @app.post("/api/rag/reindex/{doc_id}")
    async def reindex_document(doc_id: str):
        """Re-index a document with current RAG settings."""
        from .documents import get_document_processor

        processor = get_document_processor()
        doc = processor.get_document(doc_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        try:
            # Get document text
            text = processor.get_document_text(doc_id)
            if not text:
                raise HTTPException(status_code=500, detail="Failed to read document")

            # Delete old embeddings
            chunk_count = doc.get("chunk_count", 100)
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(chunk_count)]
            try:
                processor.collection.delete(ids=chunk_ids)
            except Exception:
                pass

            # Re-chunk and re-embed
            chunks = processor.chunk_text(text)
            embeddings = processor.generate_embeddings(chunks)

            new_chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metas = [
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "folder_id": doc.get("folder_id") or "",
                    "filename": doc.get("filename", ""),
                    "embedding_model": processor.config.embedding_model,
                    "chunk_size": processor.config.chunk_size,
                }
                for i in range(len(chunks))
            ]

            processor.collection.add(
                ids=new_chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metas
            )

            # Update document info
            doc["chunk_count"] = len(chunks)
            doc["rag_config"] = processor.config.to_dict()
            doc["reindexed_at"] = datetime.now().isoformat()
            processor._save_docs_index()

            return {
                "success": True,
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "config": processor.config.to_dict()
            }

        except Exception as e:
            logger.error(f"Reindex failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    class ChatWithContextRequest(BaseModel):
        message: str
        context_type: str  # 'folder', 'document', 'chat'
        context_id: str
        session_id: Optional[str] = None

    @app.post("/api/chat/context")
    async def chat_with_context(request: ChatWithContextRequest):
        """
        Chat with document/folder context using RAG.

        The relevant context is retrieved and prepended to the message
        for the LLM to use in generating a response.
        """
        from .documents import get_document_processor

        try:
            processor = get_document_processor()

            # Get relevant context
            context = processor.get_context_for_chat(
                query=request.message,
                context_type=request.context_type,
                context_id=request.context_id
            )

            # Build enhanced message with context
            if context:
                enhanced_message = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so.

CONTEXT:
{context}

QUESTION: {request.message}"""
            else:
                enhanced_message = request.message

            # Process through normal chat flow
            session_id = request.session_id or str(uuid.uuid4())[:8]
            result = await api.process_message(
                message=enhanced_message,
                session_id=session_id
            )

            # Add context info to result
            result["context_used"] = bool(context)
            result["context_type"] = request.context_type
            result["context_id"] = request.context_id

            return result

        except Exception as e:
            logger.error(f"Context chat failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/chat/context/stream")
    async def chat_with_context_stream(
        message: str,
        context_type: str,
        context_id: str,
        session_id: Optional[str] = None
    ):
        """SSE streaming chat with document/folder context."""
        from starlette.responses import StreamingResponse
        from .documents import get_document_processor
        import queue
        import concurrent.futures
        import json

        session_id = session_id or str(uuid.uuid4())[:8]

        async def event_generator():
            # Get context
            try:
                processor = get_document_processor()
                context = processor.get_context_for_chat(
                    query=message,
                    context_type=context_type,
                    context_id=context_id
                )
            except Exception as e:
                logger.error(f"Failed to get context: {e}")
                context = ""

            # Build enhanced message
            if context:
                enhanced_message = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so.

CONTEXT:
{context}

QUESTION: {message}"""
                yield f"data: {json.dumps({'type': 'context', 'has_context': True, 'context_length': len(context)})}\n\n"
            else:
                enhanced_message = message
                yield f"data: {json.dumps({'type': 'context', 'has_context': False})}\n\n"

            # Thread-safe queue for events
            event_queue = queue.Queue()
            result_holder = {"result": None, "done": False}
            padding = " " * 16384

            def sync_stream_cb(chunk: str):
                event_queue.put({"type": "stream", "chunk": chunk})

            def sync_status_cb(stage: str, detail: str):
                event_queue.put({"type": "status", "stage": stage, "detail": detail})

            def process_sync():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            api.process_message(
                                message=enhanced_message,
                                session_id=session_id,
                                stream_callback=sync_stream_cb,
                                status_callback=sync_status_cb
                            )
                        )
                        result_holder["result"] = result
                    finally:
                        loop.close()
                except Exception as e:
                    result_holder["result"] = {"success": False, "error": str(e)}
                finally:
                    result_holder["done"] = True

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(process_sync)

            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n{padding}"

            while not result_holder["done"]:
                while True:
                    try:
                        event = event_queue.get_nowait()
                        yield f"data: {json.dumps(event)}\n\n{padding}"
                    except queue.Empty:
                        break
                await asyncio.sleep(0.05)
                yield f": keepalive\n\n{padding}"

            while True:
                try:
                    event = event_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    break

            yield f"data: {json.dumps({'type': 'complete', 'result': result_holder['result']})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, no-transform, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # ===== AG-UI PROTOCOL ENDPOINT =====
    # Implements the AG-UI (Agent-User Interaction) protocol
    # Compatible with CopilotKit and other AG-UI clients
    # Protocol spec: https://docs.ag-ui.com/

    class AGUIRunRequest(BaseModel):
        threadId: str
        runId: str
        messages: List[dict]
        state: Optional[dict] = None
        context: Optional[dict] = None

    @app.post("/api/agui/run")
    async def agui_run(request: AGUIRunRequest):
        """
        AG-UI Protocol streaming endpoint.

        Emits events following the AG-UI specification:
        - Lifecycle: RunStarted, RunFinished, RunError
        - Text: TextMessageStart, TextMessageContent, TextMessageEnd
        - Tool: ToolCallStart, ToolCallArgs, ToolCallEnd, ToolCallResult
        - State: StateSnapshot, StateDelta
        - Activity: ActivitySnapshot, ActivityDelta
        """
        from starlette.responses import StreamingResponse
        import queue
        import concurrent.futures
        import json

        thread_id = request.threadId
        run_id = request.runId

        async def agui_event_generator():
            # Emit RunStarted
            yield f"data: {json.dumps({'type': 'RunStarted', 'threadId': thread_id, 'runId': run_id, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Get the latest user message
            user_messages = [m for m in request.messages if m.get('role') == 'user']
            if not user_messages:
                yield f"data: {json.dumps({'type': 'RunError', 'message': 'No user message found', 'code': 'NO_MESSAGE'})}\n\n"
                return

            user_message = user_messages[-1].get('content', '')
            message_id = f"msg_{uuid.uuid4().hex[:12]}"

            # Thread-safe queue for events
            event_queue = queue.Queue()
            result_holder = {"result": None, "done": False}
            padding = " " * 8192

            # Emit TextMessageStart
            yield f"data: {json.dumps({'type': 'TextMessageStart', 'messageId': message_id, 'role': 'assistant', 'timestamp': datetime.now().isoformat()})}\n\n{padding}"

            def sync_stream_cb(chunk: str):
                """Emit TextMessageContent for each chunk."""
                event_queue.put({
                    'type': 'TextMessageContent',
                    'messageId': message_id,
                    'delta': chunk
                })

            def sync_status_cb(stage: str, detail: str):
                """Emit ActivitySnapshot for status updates."""
                event_queue.put({
                    'type': 'ActivitySnapshot',
                    'messageId': f"activity_{run_id}",
                    'activityType': 'PROGRESS',
                    'content': {'stage': stage, 'detail': detail, 'label': f"{stage}: {detail}"}
                })

            def process_sync():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            api.process_message(
                                message=user_message,
                                session_id=thread_id,
                                stream_callback=sync_stream_cb,
                                status_callback=sync_status_cb
                            )
                        )
                        result_holder["result"] = result
                    finally:
                        loop.close()
                except Exception as e:
                    logger.error(f"AG-UI processing error: {e}", exc_info=True)
                    result_holder["result"] = {"success": False, "error": str(e)}
                finally:
                    result_holder["done"] = True

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(process_sync)

            # Stream events
            while not result_holder["done"]:
                while True:
                    try:
                        event = event_queue.get_nowait()
                        yield f"data: {json.dumps(event)}\n\n{padding}"
                    except queue.Empty:
                        break
                await asyncio.sleep(0.05)
                yield f": keepalive\n\n"

            # Drain remaining events
            while True:
                try:
                    event = event_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    break

            # Emit TextMessageEnd
            yield f"data: {json.dumps({'type': 'TextMessageEnd', 'messageId': message_id, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Emit StateSnapshot with result
            result = result_holder["result"]
            if result:
                yield f"data: {json.dumps({'type': 'StateSnapshot', 'snapshot': {'lastResult': result}})}\n\n"

            # Emit RunFinished or RunError
            if result and result.get("success"):
                yield f"data: {json.dumps({'type': 'RunFinished', 'threadId': thread_id, 'runId': run_id, 'result': {'content': result.get('content', '')}, 'timestamp': datetime.now().isoformat()})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'RunError', 'message': result.get('error', 'Unknown error') if result else 'Processing failed', 'code': 'PROCESSING_ERROR'})}\n\n"

        return StreamingResponse(
            agui_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, no-transform, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            }
        )

    @app.get("/api/agui/info")
    async def agui_info():
        """AG-UI endpoint info."""
        return {
            "protocol": "AG-UI",
            "version": "1.0",
            "capabilities": [
                "text_streaming",
                "tool_calls",
                "state_management",
                "activity_tracking"
            ],
            "documentation": "https://docs.ag-ui.com/"
        }

    # Export endpoints
    @app.post("/api/export")
    async def export_content(request: dict):
        """
        Export content to various formats.

        Args:
            content: Markdown content to export
            format: Target format (md, pdf, docx, epub, html, slides)
            filename: Optional filename (without extension)
            title: Optional document title (defaults to filename)

        Features:
            - Multiple PDF engines with fallback (xelatex -> pdflatex -> weasyprint)
            - Custom LaTeX templates for professional styling
            - Code syntax highlighting
            - Math equation support
            - Detailed error messages
        """
        from starlette.responses import FileResponse
        from .export_utils import export_content as do_export, ExportError

        content = request.get("content", "")
        export_format = request.get("format", "md").lower()
        filename = request.get("filename", "export")
        title = request.get("title", filename)

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        try:
            # Use enhanced export utilities with fallbacks and templates
            output_file, media_type = do_export(
                content=content,
                format=export_format,
                filename=filename,
                title=title
            )

            if not output_file.exists():
                raise HTTPException(status_code=500, detail="Conversion failed - output file not created")

            # Add headers for inline viewing (especially for PDF)
            headers = {}
            if media_type == "application/pdf":
                # Content-Disposition: inline allows browser PDF viewer
                headers["Content-Disposition"] = f'inline; filename="{output_file.name}"'

            return FileResponse(
                path=str(output_file),
                filename=output_file.name,
                media_type=media_type,
                headers=headers,
                background=None  # Don't delete file immediately
            )

        except ExportError as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Export error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/preview")
    async def preview_content(request: dict):
        """
        Preview content in various formats (returns HTML/text for inline display).

        Args:
            content: Markdown content to preview
            format: Target format (html, docx-preview)
        """
        from starlette.responses import HTMLResponse, PlainTextResponse
        import tempfile
        import subprocess
        from pathlib import Path

        content = request.get("content", "")
        preview_format = request.get("format", "html").lower()

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        temp_dir = Path(tempfile.mkdtemp())
        md_file = temp_dir / "preview.md"
        md_file.write_text(content, encoding="utf-8")

        try:
            if preview_format == "html":
                # Convert to standalone HTML
                output_file = temp_dir / "preview.html"
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file),
                    "--standalone",
                    "--metadata", "title=Preview",
                    "--css", "data:text/css,body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:800px;margin:40px auto;padding:20px;line-height:1.6}pre{background:%23f5f5f5;padding:16px;border-radius:8px;overflow-x:auto}code{background:%23f0f0f0;padding:2px 6px;border-radius:4px}h1,h2,h3{margin-top:24px}"
                ], check=True)
                html_content = output_file.read_text(encoding="utf-8")
                return HTMLResponse(content=html_content)

            elif preview_format == "docx-preview":
                # Convert DOCX to HTML for preview
                # First create DOCX, then convert to HTML
                docx_file = temp_dir / "preview.docx"
                html_file = temp_dir / "preview.html"

                subprocess.run([
                    "pandoc", str(md_file), "-o", str(docx_file)
                ], check=True)

                subprocess.run([
                    "pandoc", str(docx_file), "-o", str(html_file),
                    "--standalone"
                ], check=True)

                html_content = html_file.read_text(encoding="utf-8")
                # Extract just the body content
                import re
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
                if body_match:
                    return HTMLResponse(content=body_match.group(1))
                return HTMLResponse(content=html_content)

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported preview format: {preview_format}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Preview conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
        except Exception as e:
            logger.error(f"Preview error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== VOICE CHAT ENDPOINTS =====

    @app.get("/api/voice/voices")
    async def list_voices():
        """Get available TTS voices."""
        from .voice import get_voice_processor
        processor = get_voice_processor()
        return {"voices": processor.get_available_voices()}

    @app.get("/api/voice/config")
    async def get_voice_config():
        """
        Get voice processing configuration.

        Returns current settings for:
        - Local Whisper (LOCAL_WHISPER env var)
        - Speculative TTS (SPECULATIVE_TTS env var)
        - WebSocket voice (WEBSOCKET_VOICE env var)
        - Whisper model size
        """
        import os
        use_local_whisper = os.environ.get("LOCAL_WHISPER", "0") == "1"
        return {
            "local_whisper": use_local_whisper,
            "whisper_model": os.environ.get("WHISPER_MODEL", "base") if use_local_whisper else None,
            "speculative_tts": os.environ.get("SPECULATIVE_TTS", "0") == "1",
            "websocket_voice": os.environ.get("WEBSOCKET_VOICE", "0") == "1",
            "websocket_url": "/ws/voice/{session_id}"
        }

    @app.post("/api/voice/config")
    async def set_voice_config(request: dict):
        """
        Update voice processing configuration at runtime.

        Request: {"local_whisper": true, "speculative_tts": true, "whisper_model": "small"}
        """
        import os

        if "local_whisper" in request:
            os.environ["LOCAL_WHISPER"] = "1" if request["local_whisper"] else "0"
        if "speculative_tts" in request:
            os.environ["SPECULATIVE_TTS"] = "1" if request["speculative_tts"] else "0"
        if "websocket_voice" in request:
            os.environ["WEBSOCKET_VOICE"] = "1" if request["websocket_voice"] else "0"
        if "whisper_model" in request:
            os.environ["WHISPER_MODEL"] = request["whisper_model"]

        # Return updated config
        return await get_voice_config()

    @app.post("/api/voice/tts")
    async def text_to_speech_endpoint(request: dict):
        """
        Convert text to speech.

        Request: {"text": "Hello", "voice": "en-US-AvaNeural"}
        Returns: Audio file (MP3)
        """
        from fastapi.responses import Response
        from .voice import get_voice_processor

        text = request.get("text", "")
        voice = request.get("voice")

        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        processor = get_voice_processor()
        audio_data = await processor.text_to_speech(text, voice)

        if not audio_data:
            raise HTTPException(status_code=500, detail="TTS failed")

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )

    @app.post("/api/voice/stt")
    async def speech_to_text_endpoint(audio: UploadFile = File(...)):
        """
        Convert speech to text.

        Accepts audio file (webm, wav, mp3, etc.)
        Returns: {"text": "transcribed text"}
        """
        from .voice import get_voice_processor

        content = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        processor = get_voice_processor()
        text, confidence = await processor.speech_to_text(content, mime_type)

        return {"text": text, "confidence": confidence, "success": bool(text)}

    @app.post("/api/voice/chat/audio")
    async def voice_chat_audio_endpoint(
        audio: UploadFile = File(...),
        session_id: Optional[str] = Form(None),
        voice: Optional[str] = Form(None)
    ):
        """
        Full voice-to-voice chat - returns raw audio.

        For direct audio playback in browser. Returns MP3 with metadata in headers.
        For JSON response with base64 audio, use /api/voice/chat instead.
        """
        from fastapi.responses import Response
        from .voice import get_voice_processor

        session_id = session_id or str(uuid.uuid4())[:8]

        # Read audio
        audio_content = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        processor = get_voice_processor()

        # 1. Speech to text
        user_text, _ = await processor.speech_to_text(audio_content, mime_type)
        if not user_text:
            # Return error audio
            error_audio = await processor.text_to_speech(
                "I couldn't understand that. Please try again.",
                voice
            )
            return Response(
                content=error_audio,
                media_type="audio/mpeg",
                headers={
                    "X-User-Text": "",
                    "X-Response-Text": "Could not transcribe",
                    "Content-Disposition": "inline; filename=response.mp3"
                }
            )

        # 2. Process with LLM
        result = await api.process_message(
            message=user_text,
            session_id=session_id
        )

        response_text = result.get("content", "I encountered an error processing your request.")

        # 3. Text to speech
        response_audio = await processor.text_to_speech(response_text, voice)

        if not response_audio:
            raise HTTPException(status_code=500, detail="TTS failed")

        # Return audio with metadata in headers
        import urllib.parse
        return Response(
            content=response_audio,
            media_type="audio/mpeg",
            headers={
                "X-User-Text": urllib.parse.quote(user_text[:500]),
                "X-Response-Text": urllib.parse.quote(response_text[:500]),
                "X-Session-Id": session_id,
                "Content-Disposition": "inline; filename=response.mp3"
            }
        )

    @app.websocket("/ws/voice/{session_id}")
    async def websocket_voice_chat(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for low-latency streaming voice chat.

        Features:
        - Real-time audio streaming (lower latency than HTTP)
        - Local Whisper support (if LOCAL_WHISPER=1)
        - Speculative TTS (if SPECULATIVE_TTS=1)
        - Confidence scores for transcription

        Client sends:
        - Binary audio chunks (accumulates until end_audio)
        - {"type": "config", "voice": "...", "speed": 1.0, "speculative": true}
        - {"type": "end_audio"} - triggers processing
        - {"type": "cancel"} - cancel current processing

        Server sends:
        - {"type": "config_ack", "voice": "...", "local_whisper": bool, "speculative_tts": bool}
        - {"type": "transcription", "text": "...", "confidence": 0.95}
        - {"type": "response_text", "text": "..."}
        - {"type": "audio_start"}
        - Binary audio chunks
        - {"type": "audio_end"}
        - {"type": "error", "message": "..."}
        """
        from .voice import get_voice_processor, VoiceConfig
        import os

        await websocket.accept()

        # Get config from environment
        use_local_whisper = os.environ.get("LOCAL_WHISPER", "0") == "1"
        use_speculative_tts = os.environ.get("SPECULATIVE_TTS", "0") == "1"

        processor = get_voice_processor()
        voice = None
        speed = 1.0
        audio_buffer = bytearray()
        cancelled = False

        try:
            # Send initial capabilities
            await websocket.send_json({
                "type": "capabilities",
                "local_whisper": use_local_whisper,
                "speculative_tts": use_speculative_tts,
                "whisper_model": os.environ.get("WHISPER_MODEL", "base") if use_local_whisper else None
            })

            while True:
                message = await websocket.receive()

                if "text" in message:
                    import json
                    data = json.loads(message["text"])

                    if data.get("type") == "config":
                        voice = data.get("voice")
                        speed = data.get("speed", 1.0)
                        # Allow client to override speculative TTS
                        if "speculative" in data:
                            use_speculative_tts = data["speculative"]
                        await websocket.send_json({
                            "type": "config_ack",
                            "voice": voice,
                            "speed": speed,
                            "local_whisper": use_local_whisper,
                            "speculative_tts": use_speculative_tts
                        })

                    elif data.get("type") == "cancel":
                        cancelled = True
                        audio_buffer.clear()
                        await websocket.send_json({"type": "cancelled"})

                    elif data.get("type") == "end_audio":
                        cancelled = False
                        if audio_buffer:
                            try:
                                # STT - with confidence
                                user_text, confidence = await processor.speech_to_text(
                                    bytes(audio_buffer), "audio/webm"
                                )
                                await websocket.send_json({
                                    "type": "transcription",
                                    "text": user_text,
                                    "confidence": confidence
                                })

                                if user_text and not cancelled:
                                    # LLM
                                    result = await api.process_message(
                                        message=user_text,
                                        session_id=session_id
                                    )
                                    response_text = result.get("content", "")

                                    if cancelled:
                                        continue

                                    await websocket.send_json({
                                        "type": "response_text",
                                        "text": response_text
                                    })

                                    # TTS - stream chunks
                                    await websocket.send_json({"type": "audio_start"})

                                    if use_speculative_tts and hasattr(processor, 'speculative_tts_stream'):
                                        # Use speculative TTS for even lower latency
                                        # Split into sentences for parallel generation
                                        import re
                                        sentences = re.split(r'(?<=[.!?])\s+', response_text)
                                        for sentence in sentences:
                                            if cancelled:
                                                break
                                            if sentence.strip():
                                                audio = await processor.text_to_speech(sentence, voice, speed)
                                                if audio:
                                                    await websocket.send_bytes(audio)
                                    else:
                                        # Standard streaming TTS
                                        async for chunk in processor.text_to_speech_stream(response_text, voice):
                                            if cancelled:
                                                break
                                            await websocket.send_bytes(chunk)

                                    await websocket.send_json({"type": "audio_end"})

                            except Exception as e:
                                logger.error(f"Voice processing error: {e}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": str(e)
                                })

                            audio_buffer.clear()

                elif "bytes" in message:
                    # Audio chunk - accumulate
                    audio_buffer.extend(message["bytes"])

        except Exception as e:
            logger.error(f"Voice WebSocket error: {e}")
        finally:
            audio_buffer.clear()

    # WebSocket endpoint
    @app.websocket("/ws/chat/{session_id}")
    async def websocket_chat(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for streaming chat.

        Messages:
        - Client sends: {"type": "message", "content": "..."}
        - Server sends: {"type": "stream", "chunk": "..."}
        - Server sends: {"type": "status", "stage": "...", "detail": "..."}
        - Server sends: {"type": "complete", "content": "...", "output_path": "..."}
        - Server sends: {"type": "error", "error": "..."}
        """
        from .websocket import get_websocket_manager

        await websocket.accept()
        ws_manager = get_websocket_manager()
        conn = await ws_manager.connect(websocket, session_id)

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                if data.get("type") == "message":
                    content = data.get("content", "")
                    attachments = data.get("attachments", [])

                    # Debug logging for attachments
                    logger.info(f"WS message received: content length={len(content)}, attachments={len(attachments)}")
                    if attachments:
                        for i, att in enumerate(attachments):
                            data_preview = att.get('data', '')[:50] if att.get('data') else 'None'
                            logger.info(f"  Attachment {i}: type={att.get('type')}, name={att.get('name')}, data_start={data_preview}")

                    if not content.strip() and not attachments:
                        continue

                    import queue
                    import concurrent.futures

                    # Thread-safe queue for events
                    event_queue = queue.Queue()
                    result_holder = {"result": None, "done": False}

                    # Sync callbacks that queue events
                    def sync_stream_cb(chunk: str):
                        event_queue.put({"type": "stream", "chunk": chunk})

                    def sync_status_cb(stage: str, detail: str):
                        event_queue.put({"type": "status", "stage": stage, "detail": detail})

                    # Send processing status
                    status_detail = "Starting..."
                    if attachments:
                        status_detail = f"Processing {len(attachments)} image(s)..."
                    await websocket.send_json({"type": "status", "stage": "processing", "detail": status_detail})

                    # Run processing in thread
                    def process_sync():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                result = loop.run_until_complete(
                                    api.process_message(
                                        message=content,
                                        session_id=session_id,
                                        stream_callback=sync_stream_cb,
                                        status_callback=sync_status_cb,
                                        attachments=attachments
                                    )
                                )
                                result_holder["result"] = result
                                logger.info(f"WS result: success={result.get('success')}, path={result.get('output_path')}")
                            finally:
                                loop.close()
                        except Exception as e:
                            logger.error(f"WS processing error: {e}", exc_info=True)
                            result_holder["result"] = {"success": False, "error": str(e)}
                        # Set done AFTER result is set (avoid race condition)
                        result_holder["done"] = True

                    # Start processing in thread
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    executor.submit(process_sync)

                    # Stream events while processing
                    while not result_holder["done"]:
                        # Drain queue and send events
                        while True:
                            try:
                                event = event_queue.get_nowait()
                                await websocket.send_json(event)
                            except queue.Empty:
                                break

                        await asyncio.sleep(0.05)

                    # Drain remaining events
                    while True:
                        try:
                            event = event_queue.get_nowait()
                            await websocket.send_json(event)
                        except queue.Empty:
                            break

                    # Small delay to ensure result is fully written by thread
                    await asyncio.sleep(0.1)

                    # Send result (match SSE format with nested 'result')
                    result = result_holder["result"]
                    logger.info(f"WS final result check: {result}")
                    if result and result.get("success"):
                        await websocket.send_json({
                            "type": "complete",
                            "result": {
                                "success": True,
                                "message_id": result.get("message_id", ""),
                                "content": result.get("content", ""),
                                "output_path": result.get("output_path")
                            }
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": result.get("error", "Unknown error") if result else "Processing failed"
                        })

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await ws_manager.disconnect(conn)

    # ===== URL PROXY ENDPOINT =====
    # Server-side proxy to load websites that block iframe embedding
    # Strips X-Frame-Options and Content-Security-Policy headers

    @app.get("/api/proxy")
    async def proxy_url(url: str):
        """
        Proxy a URL and strip headers that prevent iframe embedding.

        This allows loading external websites in the inline browser
        even if they set X-Frame-Options or CSP headers.
        """
        import httpx
        from starlette.responses import Response

        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
            ) as client:
                response = await client.get(url)

                # Get content type
                content_type = response.headers.get('content-type', 'text/html')

                # Build response headers - copy safe headers, skip restrictive ones
                safe_headers = {}
                skip_headers = {
                    'x-frame-options',
                    'content-security-policy',
                    'content-security-policy-report-only',
                    'x-content-type-options',
                    'strict-transport-security',
                    'transfer-encoding',
                    'content-encoding',
                    'content-length',  # Will be recalculated
                }

                for key, value in response.headers.items():
                    if key.lower() not in skip_headers:
                        safe_headers[key] = value

                # For HTML content, inject base tag to fix relative URLs
                content = response.content
                if 'text/html' in content_type:
                    try:
                        html = content.decode('utf-8', errors='replace')
                        # Parse the base URL
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        base_url = f"{parsed.scheme}://{parsed.netloc}"

                        # Inject base tag if not present
                        if '<base' not in html.lower():
                            # Insert base tag after <head>
                            if '<head>' in html:
                                html = html.replace('<head>', f'<head><base href="{base_url}/">', 1)
                            elif '<head ' in html:
                                html = html.replace('<head ', f'<base href="{base_url}/"><head ', 1)
                            elif '<HEAD>' in html:
                                html = html.replace('<HEAD>', f'<HEAD><base href="{base_url}/">', 1)

                        content = html.encode('utf-8')
                    except Exception as e:
                        logger.debug(f"Failed to inject base tag: {e}")

                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=safe_headers,
                    media_type=content_type.split(';')[0]  # Just the mime type, not charset
                )

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request timed out")
        except httpx.RequestError as e:
            logger.error(f"Proxy request error: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== LIBRECHAT-STYLE FEATURES =====

    # ===== MCP TOOLS ENDPOINTS =====

    @app.get("/api/mcp/tools")
    async def list_mcp_tools():
        """List available MCP tools with enable/disable status."""
        try:
            from ..core.integration.mcp_client import MCPClient

            # Try to get tools from MCP server
            try:
                client = MCPClient()
                await client.connect()
                tools = await client.list_tools()
                await client.disconnect()

                return {
                    "tools": [
                        {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "inputSchema": t.get("inputSchema", {}),
                            "enabled": True
                        }
                        for t in tools
                    ],
                    "count": len(tools),
                    "connected": True
                }
            except Exception as e:
                logger.warning(f"MCP connection failed: {e}")
                # Return fallback tools
                return {
                    "tools": [
                        {"name": "create_idea", "description": "Create a new idea/note", "enabled": True},
                        {"name": "list_ideas", "description": "List all ideas", "enabled": True},
                        {"name": "search_ideas", "description": "Search ideas by query", "enabled": True},
                    ],
                    "count": 3,
                    "connected": False,
                    "error": str(e)
                }
        except ImportError:
            return {"tools": [], "count": 0, "connected": False, "error": "MCP client not available"}

    class MCPExecuteRequest(BaseModel):
        tool_name: str
        arguments: dict = {}

    @app.post("/api/mcp/execute")
    async def execute_mcp_tool(request: MCPExecuteRequest):
        """Execute an MCP tool and return result."""
        import time
        start_time = time.time()

        try:
            from ..core.integration.mcp_client import call_justjot_mcp_tool

            result = await call_justjot_mcp_tool(
                tool_name=request.tool_name,
                arguments=request.arguments
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "tool_name": request.tool_name,
                "result": result,
                "duration_ms": duration_ms
            }
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return {
                "success": False,
                "tool_name": request.tool_name,
                "error": str(e),
                "duration_ms": int((time.time() - start_time) * 1000)
            }

    # ===== ARTIFACTS ENDPOINTS =====

    @app.post("/api/artifacts/extract")
    async def extract_artifacts(request: dict):
        """Extract artifacts (code blocks, diagrams, etc.) from text."""
        from .artifacts import extract_artifacts as do_extract

        text = request.get("text", "")
        if not text:
            return {"artifacts": [], "count": 0}

        artifacts = do_extract(text)
        return {"artifacts": artifacts, "count": len(artifacts)}

    @app.post("/api/artifacts/render")
    async def render_artifact(request: dict):
        """
        Render an artifact to displayable format.

        For HTML: returns sanitized HTML
        For Mermaid: returns SVG
        For code: returns syntax-highlighted HTML
        """
        artifact_type = request.get("type", "code")
        content = request.get("content", "")

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        if artifact_type == "mermaid":
            # For mermaid, return content as-is (frontend will use mermaid.js)
            return {
                "type": "mermaid",
                "content": content,
                "render_mode": "client"  # Render on client side
            }

        if artifact_type == "html":
            # Return HTML for sandboxed iframe
            return {
                "type": "html",
                "content": content,
                "render_mode": "iframe"
            }

        if artifact_type == "svg":
            return {
                "type": "svg",
                "content": content,
                "render_mode": "inline"
            }

        # Default: code block
        return {
            "type": "code",
            "content": content,
            "language": request.get("language", ""),
            "render_mode": "highlight"
        }

    # ===== CODE INTERPRETER ENDPOINTS =====

    class CodeExecuteRequest(BaseModel):
        code: str
        language: str = "python"
        timeout: int = 30

    @app.post("/api/code/execute")
    async def execute_code(request: CodeExecuteRequest):
        """Execute code in sandboxed environment."""
        from .code_interpreter import execute_code as do_execute

        if not request.code.strip():
            raise HTTPException(status_code=400, detail="Code is required")

        result = await do_execute(request.code, request.language)
        return result

    @app.get("/api/code/execute/stream")
    async def execute_code_stream(code: str, language: str = "python"):
        """SSE streaming code execution."""
        from starlette.responses import StreamingResponse
        from .code_interpreter import get_code_interpreter
        import json

        interpreter = get_code_interpreter()

        async def event_generator():
            async for event in interpreter.execute_streaming(code, language):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    @app.get("/api/code/languages")
    async def list_code_languages():
        """List supported programming languages for code execution."""
        return {
            "languages": [
                {"id": "python", "name": "Python", "extension": ".py", "available": True},
                {"id": "javascript", "name": "JavaScript", "extension": ".js", "available": True},
                {"id": "typescript", "name": "TypeScript", "extension": ".ts", "available": False},
                {"id": "bash", "name": "Bash", "extension": ".sh", "available": False},
            ]
        }

    # ===== CONVERSATION BRANCHING ENDPOINTS =====

    @app.get("/api/sessions/{session_id}/branches")
    async def list_branches(session_id: str):
        """Get all branches for a session."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "branches": session.get_branches(),
            "active_branch": getattr(session, 'active_branch', 'main'),
            "tree": session.get_branch_tree()
        }

    class CreateBranchRequest(BaseModel):
        from_message_id: str
        branch_name: Optional[str] = None

    @app.post("/api/sessions/{session_id}/branch")
    async def create_branch(session_id: str, request: CreateBranchRequest):
        """Create a new branch from a message."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            branch_id = session.create_branch(
                from_message_id=request.from_message_id,
                branch_name=request.branch_name
            )
            return {
                "success": True,
                "branch_id": branch_id,
                "branches": session.get_branches()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    class EditMessageRequest(BaseModel):
        new_content: str
        create_branch: bool = True

    @app.post("/api/sessions/{session_id}/messages/{message_id}/edit")
    async def edit_message(session_id: str, message_id: str, request: EditMessageRequest):
        """Edit a message, optionally creating a new branch."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            branch_id = session.edit_message(
                message_id=message_id,
                new_content=request.new_content,
                create_branch=request.create_branch
            )
            return {
                "success": True,
                "branch_id": branch_id,
                "branches": session.get_branches()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    class SwitchBranchRequest(BaseModel):
        branch_id: str

    @app.post("/api/sessions/{session_id}/branch/switch")
    async def switch_branch(session_id: str, request: SwitchBranchRequest):
        """Switch to a different branch."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            session.switch_branch(request.branch_id)
            return {
                "success": True,
                "active_branch": request.branch_id,
                "history": session.get_history()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/sessions/{session_id}/branches/{branch_id}")
    async def delete_branch(session_id: str, branch_id: str):
        """Delete a branch."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            session.delete_branch(branch_id)
            return {
                "success": True,
                "branches": session.get_branches()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ===== WEB SEARCH ENDPOINTS (Uses existing web-search skill) =====

    class WebSearchRequest(BaseModel):
        query: str
        max_results: int = 10

    @app.post("/api/search")
    async def web_search(request: WebSearchRequest):
        """
        Search the web using the existing web-search skill (DuckDuckGo).

        Args:
            query: Search query
            max_results: Maximum number of results (default 10)

        Returns:
            Search results with title, url, snippet
        """
        try:
            # Use existing skill directly
            import sys
            from pathlib import Path
            skills_path = Path(__file__).parent.parent / "skills" / "web-search"
            if str(skills_path) not in sys.path:
                sys.path.insert(0, str(skills_path))
            from tools import search_web_tool

            result = await asyncio.to_thread(
                search_web_tool,
                {"query": request.query, "max_results": request.max_results}
            )

            return result  # Already has success, results, count, query
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/search")
    async def web_search_get(query: str, max_results: int = 10):
        """GET endpoint for web search using existing skill."""
        try:
            import sys
            from pathlib import Path
            skills_path = Path(__file__).parent.parent / "skills" / "web-search"
            if str(skills_path) not in sys.path:
                sys.path.insert(0, str(skills_path))
            from tools import search_web_tool

            result = await asyncio.to_thread(
                search_web_tool,
                {"query": query, "max_results": max_results}
            )

            return result
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== SHAREABLE LINKS ENDPOINTS =====

    class CreateShareLinkRequest(BaseModel):
        session_id: str
        title: Optional[str] = None
        expires_in_days: Optional[int] = None
        branch_id: str = "main"

    @app.post("/api/share/create")
    async def create_share_link(request: CreateShareLinkRequest):
        """Create a shareable link for a conversation."""
        from cli.repl.session import get_share_link_registry, get_session_registry

        # Verify session exists
        registry = get_session_registry()
        session = registry.get_session(request.session_id, create=False)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create share link
        share_registry = get_share_link_registry()
        link = share_registry.create_link(
            session_id=request.session_id,
            title=request.title,
            expires_in_days=request.expires_in_days,
            branch_id=request.branch_id
        )

        return {
            "success": True,
            "link": link.to_dict(),
            "url": f"/share/{link.token}"
        }

    @app.get("/api/share/{token}")
    async def get_share_link_info(token: str):
        """Get information about a share link."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        return {"link": link.to_dict()}

    @app.get("/api/share/{token}/conversation")
    async def get_shared_conversation(token: str):
        """Get the shared conversation (public read-only view)."""
        from cli.repl.session import get_share_link_registry, get_session_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        # Record access
        share_registry.record_access(token)

        # Get session
        registry = get_session_registry()
        session = registry.get_session(link.session_id, create=False)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get messages for the branch
        messages = session.get_branch_messages(link.branch_id)

        return {
            "title": link.title or f"Shared Chat",
            "messages": [m.to_dict() for m in messages],
            "created_at": link.created_at.isoformat(),
            "access_count": link.access_count,
            "branch_id": link.branch_id
        }

    @app.get("/api/share/session/{session_id}")
    async def get_session_share_links(session_id: str):
        """Get all share links for a session."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        links = share_registry.get_session_links(session_id)

        return {"links": [link.to_dict() for link in links]}

    @app.post("/api/share/{token}/revoke")
    async def revoke_share_link(token: str):
        """Revoke a share link."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        success = share_registry.revoke_link(token)

        if not success:
            raise HTTPException(status_code=404, detail="Share link not found")

        return {"success": True}

    @app.post("/api/share/{token}/refresh")
    async def refresh_share_link(token: str, expires_in_days: int = 30):
        """Refresh a share link with new token and expiry."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        new_link = share_registry.refresh_link(token, expires_in_days)

        if not new_link:
            raise HTTPException(status_code=404, detail="Share link not found")

        return {
            "success": True,
            "link": new_link.to_dict(),
            "url": f"/share/{new_link.token}"
        }

    @app.get("/api/share/{token}/qrcode")
    async def get_share_qrcode(token: str, base_url: str = None):
        """Generate QR code for a share link."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        # Generate QR code as data URL
        try:
            import qrcode
            import io
            import base64

            share_url = f"{base_url or ''}/share/{token}"
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(share_url)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return {
                "qrcode": f"data:image/png;base64,{img_str}",
                "url": share_url
            }
        except ImportError:
            # qrcode library not installed, return URL only
            return {
                "qrcode": None,
                "url": f"{base_url or ''}/share/{token}",
                "error": "QR code library not installed"
            }

    # Public share page (no auth required)
    @app.get("/share/{token}")
    async def share_page(token: str):
        """Serve the public share page."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        # Return HTML page that will load the conversation
        return HTMLResponse(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shared Chat - Jotty</title>
    <link rel="stylesheet" href="/static/style.css">
    <style>
        .shared-container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        .shared-header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: var(--bg-secondary); border-radius: 10px; }}
        .shared-header h1 {{ margin: 0; font-size: 1.5rem; }}
        .shared-header p {{ color: var(--text-muted); margin: 10px 0 0; }}
        .shared-message {{ margin: 15px 0; padding: 15px; border-radius: 10px; }}
        .shared-message.user {{ background: var(--user-msg-bg); margin-left: 50px; }}
        .shared-message.assistant {{ background: var(--assistant-msg-bg); margin-right: 50px; }}
        .shared-footer {{ text-align: center; margin-top: 30px; color: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="shared-container">
        <div class="shared-header">
            <h1 id="share-title">Shared Chat</h1>
            <p id="share-info">Loading...</p>
        </div>
        <div id="messages"></div>
        <div class="shared-footer">
            <p>Shared via <a href="/">Jotty</a></p>
        </div>
    </div>
    <script>
        async function loadSharedConversation() {{
            try {{
                const response = await fetch('/api/share/{token}/conversation');
                if (!response.ok) throw new Error('Failed to load conversation');
                const data = await response.json();

                document.getElementById('share-title').textContent = data.title;
                document.getElementById('share-info').textContent = `Viewed ${{data.access_count}} times`;

                const messagesDiv = document.getElementById('messages');
                data.messages.forEach(msg => {{
                    const div = document.createElement('div');
                    div.className = `shared-message ${{msg.role}}`;
                    div.innerHTML = `<strong>${{msg.role === 'user' ? 'You' : 'Assistant'}}</strong><div>${{msg.content}}</div>`;
                    messagesDiv.appendChild(div);
                }});
            }} catch (error) {{
                document.getElementById('messages').innerHTML = '<p style="color: red;">Failed to load conversation</p>';
            }}
        }}
        loadSharedConversation();
    </script>
</body>
</html>
        """)

    # ===== TEMPORARY CHAT ENDPOINTS =====

    class CreateTempSessionRequest(BaseModel):
        expiry_days: Optional[int] = 30

    @app.post("/api/sessions/temporary")
    async def create_temporary_session(request: CreateTempSessionRequest = None):
        """Create a temporary (ephemeral) chat session."""
        from cli.repl.session import SessionManager, InterfaceType

        expiry_days = (request.expiry_days if request else None) or 30
        session = SessionManager(
            interface=InterfaceType.WEB,
            is_temporary=True
        )
        session.set_temporary(True, expiry_days)

        return {
            "session_id": session.session_id,
            "is_temporary": True,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None
        }

    @app.post("/api/sessions/{session_id}/temporary")
    async def toggle_session_temporary(session_id: str, is_temporary: bool, expiry_days: Optional[int] = 30):
        """Toggle temporary mode for a session."""
        from cli.repl.session import get_session_registry

        registry = get_session_registry()
        session = registry.get_session(session_id, create=False)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.set_temporary(is_temporary, expiry_days)

        if not is_temporary:
            # Save the session now that it's permanent
            session.auto_save = True
            session.save()

        return {
            "session_id": session_id,
            "is_temporary": session.is_temporary,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None
        }

    @app.post("/api/sessions/cleanup")
    async def cleanup_expired_sessions():
        """Clean up expired temporary sessions."""
        from cli.repl.session import SessionManager

        deleted = SessionManager.cleanup_expired_sessions()

        return {
            "success": True,
            "deleted_count": len(deleted),
            "deleted_sessions": deleted
        }

    @app.get("/api/sessions")
    async def list_all_sessions(include_temporary: bool = False, include_expired: bool = False):
        """List all available sessions with filters."""
        from cli.repl.session import SessionManager

        session_manager = SessionManager()
        sessions = session_manager.list_sessions(
            include_temporary=include_temporary,
            include_expired=include_expired
        )

        return {"sessions": sessions}

    # ===== UPDATED CAPABILITIES ENDPOINT =====
    # Override the earlier one to add feature flags

    @app.get("/api/features")
    async def get_feature_flags():
        """Get feature flags for UI capabilities."""
        return {
            "features": {
                "mcp_tools": True,
                "artifacts": True,
                "code_interpreter": True,
                "web_search": True,
                "branching": True,
                "voice": True,
                "documents": True,
                "shareable_links": True,
                "temporary_chat": True
            }
        }

    # ===== VOICE ENDPOINTS =====
    # Speech-to-Text, Text-to-Speech, and Voice-to-Voice pipelines

    @app.post("/api/voice/stt")
    async def speech_to_text(audio: UploadFile):
        """
        Convert speech audio to text using Groq Whisper (primary) or Deepgram (fallback).

        Accepts audio files (webm, wav, mp3, ogg, flac, m4a).
        Returns transcribed text.
        """
        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        transcript, confidence = await processor.speech_to_text(audio_data, mime_type)

        return {
            "success": bool(transcript),
            "transcript": transcript,
            "confidence": confidence,
            "mime_type": mime_type
        }

    @app.post("/api/voice/tts")
    async def text_to_speech(
        text: str = Form(...),
        voice: Optional[str] = Form(None)
    ):
        """
        Convert text to speech using edge-tts (Microsoft neural voices).

        Args:
            text: Text to convert to speech (form field)
            voice: Optional voice ID (default: en-US-AvaNeural)

        Returns audio/mpeg stream.
        """
        from .voice import get_voice_processor
        from fastapi.responses import Response

        processor = get_voice_processor()
        audio_bytes = await processor.text_to_speech(text, voice)

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS generation failed")

        return Response(content=audio_bytes, media_type="audio/mpeg")

    @app.get("/api/voice/voices")
    async def list_voices():
        """List available TTS voices."""
        from .voice import VoiceProcessor

        return {
            "voices": VoiceProcessor.get_available_voices(),
            "default": "en-US-AvaNeural"
        }

    @app.post("/api/voice/chat")
    async def voice_chat(audio: UploadFile, session_id: Optional[str] = None):
        """
        Full voice-to-voice pipeline: STT -> LLM -> TTS.

        Processes audio input through:
        1. Speech-to-Text (Groq Whisper)
        2. LLM processing (via chat endpoint logic)
        3. Text-to-Speech (edge-tts)

        Returns JSON with text and base64-encoded audio.
        """
        import base64
        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        # Define LLM processing function
        async def process_with_llm(user_text: str) -> str:
            result = await api.process_message(
                message=user_text,
                session_id=session_id or str(uuid.uuid4())[:8]
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)

        user_text, response_text, response_audio = await processor.process_voice_message(
            audio_data, mime_type, process_with_llm
        )

        return {
            "success": True,
            "user_text": user_text,
            "response_text": response_text,
            "response_audio_base64": base64.b64encode(response_audio).decode() if response_audio else None,
            "audio_format": "audio/mpeg"
        }

    @app.post("/api/voice/chat/fast")
    async def voice_chat_fast(
        audio: UploadFile,
        session_id: Optional[str] = None,
        max_chars: int = 200
    ):
        """
        Optimized voice pipeline for minimum latency.

        Optimizations:
        - Truncates response at sentence boundary (max 200 chars default)
        - Uses 15% faster speech rate
        - Reduces overall latency by ~40%

        Returns JSON with text and base64-encoded audio.
        """
        import base64
        from .voice import get_voice_processor

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        async def process_with_llm(user_text: str) -> str:
            result = await api.process_message(
                message=user_text,
                session_id=session_id or str(uuid.uuid4())[:8]
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)

        user_text, response_text, response_audio = await processor.process_voice_fast(
            audio_data, mime_type, process_with_llm, max_response_chars=max_chars
        )

        return {
            "success": True,
            "user_text": user_text,
            "response_text": response_text,
            "response_audio_base64": base64.b64encode(response_audio).decode() if response_audio else None,
            "audio_format": "audio/mpeg",
            "mode": "fast"
        }

    @app.post("/api/voice/chat/turbo")
    async def voice_chat_turbo(audio: UploadFile, session_id: Optional[str] = None):
        """
        Ultra-fast voice pipeline using Groq LLM (~2s total latency).

        Uses Groq for both STT (Whisper) and LLM (llama-3.1-8b-instant).
        Optimized for conversational voice chat where latency is critical.

        Latency breakdown:
        - STT (Groq Whisper): ~250ms
        - LLM (Groq llama-3.1-8b): ~180ms
        - TTS (edge-tts): ~700ms
        - Total: ~1.1-2.3s

        Returns raw audio/mpeg with X-User-Text and X-Response-Text headers.
        """
        import os
        import httpx
        from urllib.parse import quote
        from .voice import get_voice_processor
        from fastapi.responses import Response

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        # Ultra-fast LLM using Groq's llama-3.1-8b-instant
        async def process_with_groq_llm(user_text: str) -> str:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                return "I'm sorry, the fast response service is unavailable."

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_key}"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief and conversational (1-3 sentences)."},
                            {"role": "user", "content": user_text}
                        ],
                        "max_tokens": 150,
                        "temperature": 0.7
                    }
                )
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "I couldn't process that.")

        user_text, response_text, response_audio = await processor.process_voice_fast(
            audio_data, mime_type, process_with_groq_llm, max_response_chars=200
        )

        # Return raw audio with text in headers (for UI compatibility)
        return Response(
            content=response_audio or b"",
            media_type="audio/mpeg",
            headers={
                "X-User-Text": quote(user_text or ""),
                "X-Response-Text": quote(response_text or ""),
                "X-Mode": "turbo",
                "X-LLM": "groq/llama-3.1-8b-instant"
            }
        )

    @app.post("/api/voice/chat/stream")
    async def voice_chat_streaming(audio: UploadFile, session_id: Optional[str] = None):
        """
        Streaming voice pipeline for lower perceived latency.

        Returns audio sentence-by-sentence as Server-Sent Events.
        First event includes user_text, then response chunks follow.
        """
        import re
        import base64
        import json
        from .voice import get_voice_processor
        from fastapi.responses import StreamingResponse

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"

        async def process_with_llm(user_text: str) -> str:
            result = await api.process_message(
                message=user_text,
                session_id=session_id or str(uuid.uuid4())[:8]
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)

        async def generate_sse():
            # 1. Speech to text
            user_text, confidence = await processor.speech_to_text(audio_data, mime_type)

            if not user_text:
                error_audio = await processor.text_to_speech("I couldn't understand that.")
                data = {
                    "user_text": "",
                    "confidence": 0.0,
                    "text": "I couldn't understand that.",
                    "audio_base64": base64.b64encode(error_audio).decode() if error_audio else None
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Send first event with user_text immediately
            yield f"data: {json.dumps({'user_text': user_text, 'confidence': confidence, 'text': '', 'audio_base64': None})}\n\n"

            # 3. Process with LLM
            response_text = await process_with_llm(user_text)

            # 4. Split into sentences and stream TTS for each
            sentences = re.split(r'(?<=[.!?])\s+', response_text)

            for sentence in sentences:
                if sentence.strip():
                    audio_chunk = await processor.text_to_speech(sentence)
                    data = {
                        "text": sentence + " ",
                        "audio_base64": base64.b64encode(audio_chunk).decode() if audio_chunk else None
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream"
        )

    @app.post("/api/voice/chat/stream/turbo")
    async def voice_chat_streaming_turbo(
        audio: UploadFile,
        session_id: Optional[str] = None,
        voice: Optional[str] = None
    ):
        """
        Ultra-fast streaming voice pipeline using Groq LLM.

        Combines Groq Whisper STT + Groq LLM + parallel TTS.
        First sentence plays in ~1s, subsequent sentences ready immediately.

        Optimization: TTS for all sentences generated in parallel.
        """
        import re
        import os
        import asyncio
        import base64
        import json
        import httpx
        from .voice import get_voice_processor
        from fastapi.responses import StreamingResponse

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"
        tts_voice = voice or "en-US-AvaNeural"

        async def process_with_groq_llm(user_text: str) -> str:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                return "I'm sorry, the fast response service is unavailable."

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_key}"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief and conversational (2-4 sentences)."},
                            {"role": "user", "content": user_text}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.7
                    }
                )
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "I couldn't process that.")

        async def generate_sse():
            # 1. Speech to text (Groq Whisper - ~250ms, or local Whisper)
            user_text, confidence = await processor.speech_to_text(audio_data, mime_type)

            if not user_text:
                error_audio = await processor.text_to_speech("I couldn't understand that.")
                data = {
                    "user_text": "",
                    "confidence": 0.0,
                    "text": "I couldn't understand that.",
                    "audio_base64": base64.b64encode(error_audio).decode() if error_audio else None
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Send user_text immediately with confidence
            yield f"data: {json.dumps({'user_text': user_text, 'confidence': confidence, 'text': '', 'audio_base64': None})}\n\n"

            # 3. Process with Groq LLM (~180ms)
            response_text = await process_with_groq_llm(user_text)

            # 4. Split into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response_text) if s.strip()]

            if not sentences:
                yield "data: [DONE]\n\n"
                return

            # 5. PARALLEL TTS: Generate audio for ALL sentences concurrently
            # This reduces total TTS time from 700ms * N to ~700ms (parallel)
            async def generate_tts(sentence: str) -> tuple:
                audio = await processor.text_to_speech(sentence, tts_voice)
                return (sentence, audio)

            # Start all TTS tasks in parallel
            tts_tasks = [generate_tts(s) for s in sentences]

            # Stream results as they complete, but maintain order
            # Use asyncio.gather to run in parallel, preserving order
            results = await asyncio.gather(*tts_tasks)

            # Yield each result in order
            for sentence, audio_chunk in results:
                data = {
                    "text": sentence + " ",
                    "audio_base64": base64.b64encode(audio_chunk).decode() if audio_chunk else None
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream"
        )

    @app.post("/api/voice/chat/stream/ultra")
    async def voice_chat_streaming_ultra(
        audio: UploadFile,
        session_id: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ):
        """
        Ultra-low-latency streaming: TTS starts BEFORE LLM finishes.

        Streams LLM tokens, generates TTS as soon as each sentence completes.
        First audio plays ~500ms after first sentence is generated.
        """
        import re
        import os
        import base64
        import json
        import httpx
        from .voice import get_voice_processor
        from fastapi.responses import StreamingResponse

        processor = get_voice_processor()
        audio_data = await audio.read()
        mime_type = audio.content_type or "audio/webm"
        tts_voice = voice or "en-US-AvaNeural"
        tts_speed = speed or 1.0

        async def generate_sse():
            # 1. Speech to text (now returns tuple with confidence)
            user_text, confidence = await processor.speech_to_text(audio_data, mime_type)

            if not user_text:
                error_audio = await processor.text_to_speech("I couldn't understand that.", tts_voice, tts_speed)
                yield f"data: {json.dumps({'user_text': '', 'text': 'Error', 'confidence': 0.0, 'audio_base64': base64.b64encode(error_audio).decode() if error_audio else None})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Send user text immediately with confidence
            yield f"data: {json.dumps({'user_text': user_text, 'confidence': confidence, 'text': '', 'audio_base64': None})}\n\n"

            # 2. Stream from Groq LLM
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                error_audio = await processor.text_to_speech("Fast response service unavailable.", tts_voice, tts_speed)
                yield f"data: {json.dumps({'text': 'Error', 'audio_base64': base64.b64encode(error_audio).decode() if error_audio else None})}\n\n"
                yield "data: [DONE]\n\n"
                return

            sentence_buffer = ""
            full_response = ""

            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_key}"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief (2-3 sentences). IMPORTANT: Always respond in the SAME LANGUAGE the user speaks. If they speak Spanish, respond in Spanish. If French, respond in French. Match their language exactly."},
                            {"role": "user", "content": user_text}
                        ],
                        "max_tokens": 150,
                        "temperature": 0.7,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                sentence_buffer += content
                                full_response += content

                                # Check for sentence boundary
                                sentence_match = re.match(r'^(.*?[.!?])\s*(.*)$', sentence_buffer, re.DOTALL)
                                if sentence_match:
                                    complete_sentence = sentence_match.group(1).strip()
                                    sentence_buffer = sentence_match.group(2)

                                    if complete_sentence:
                                        # Generate TTS immediately for this sentence
                                        audio_chunk = await processor.text_to_speech(complete_sentence, tts_voice, tts_speed)
                                        yield f"data: {json.dumps({'text': complete_sentence + ' ', 'audio_base64': base64.b64encode(audio_chunk).decode() if audio_chunk else None})}\n\n"

                        except json.JSONDecodeError:
                            continue

            # Send any remaining text
            if sentence_buffer.strip():
                audio_chunk = await processor.text_to_speech(sentence_buffer.strip(), tts_voice, tts_speed)
                yield f"data: {json.dumps({'text': sentence_buffer.strip(), 'audio_base64': base64.b64encode(audio_chunk).decode() if audio_chunk else None})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream"
        )

    # Explicit routes for static files (ensure they work)
    static_dir = Path(__file__).parent / "static"

    @app.get("/static/style.css")
    async def get_css():
        css_file = static_dir / "style.css"
        if css_file.exists():
            return FileResponse(css_file, media_type="text/css")
        raise HTTPException(status_code=404, detail="CSS not found")

    @app.get("/static/app.js")
    async def get_js():
        js_file = static_dir / "app.js"
        if js_file.exists():
            return FileResponse(js_file, media_type="application/javascript")
        raise HTTPException(status_code=404, detail="JS not found")

    # Mount remaining static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Create app instance for uvicorn
app = create_app()
