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
        """Get LeanExecutor with callbacks."""
        # Ensure LM is configured before creating executor
        self._ensure_lm_configured()

        from ..core.orchestration.v2.lean_executor import LeanExecutor
        return LeanExecutor(
            status_callback=status_callback,
            stream_callback=stream_callback
        )

    def _get_session_registry(self):
        """Get session registry."""
        if self._registry is None:
            from ..cli.repl.session import get_session_registry
            self._registry = get_session_registry()
        return self._registry

    async def process_message(
        self,
        message: str,
        session_id: str,
        user_id: str = "web_user",
        stream_callback=None,
        status_callback=None
    ) -> Dict[str, Any]:
        """
        Process a chat message.

        Args:
            message: User message
            session_id: Session ID
            user_id: User identifier
            stream_callback: Optional callback for streaming
            status_callback: Optional callback for status updates

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

        # Add user message
        message_id = str(uuid.uuid4())[:12]
        session.add_message(
            role="user",
            content=message,
            interface=InterfaceType.WEB,
            user_id=user_id,
            metadata={"message_id": message_id}
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

Current request: {message}"""
            else:
                task_with_context = message
        else:
            task_with_context = message

        try:
            # Execute with context
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
        """Get LM providers status."""
        providers = {}

        # Check Anthropic
        import os
        providers["anthropic"] = {
            "configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        }

        # Check OpenAI
        providers["openai"] = {
            "configured": bool(os.environ.get("OPENAI_API_KEY")),
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        }

        # Check Groq
        providers["groq"] = {
            "configured": bool(os.environ.get("GROQ_API_KEY")),
            "models": ["llama-3.1-70b", "llama-3.1-8b", "mixtral-8x7b"]
        }

        # Check Google
        providers["google"] = {
            "configured": bool(os.environ.get("GOOGLE_API_KEY")),
            "models": ["gemini-pro", "gemini-pro-vision"]
        }

        # Check OpenRouter
        providers["openrouter"] = {
            "configured": bool(os.environ.get("OPENROUTER_API_KEY")),
            "models": ["anthropic/claude-3-opus", "openai/gpt-4"]
        }

        # Check Claude CLI (always available in container)
        try:
            import subprocess
            result = subprocess.run(["which", "claude"], capture_output=True)
            providers["claude-cli"] = {
                "configured": result.returncode == 0,
                "models": ["claude-cli"]
            }
        except Exception:
            providers["claude-cli"] = {"configured": False, "models": []}

        return {"providers": providers}

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
        """
        from starlette.responses import FileResponse
        import tempfile
        import subprocess
        from pathlib import Path

        content = request.get("content", "")
        export_format = request.get("format", "md").lower()
        filename = request.get("filename", "export")

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        # Create temp directory for conversion
        temp_dir = Path(tempfile.mkdtemp())
        md_file = temp_dir / f"{filename}.md"

        # Write markdown content
        md_file.write_text(content, encoding="utf-8")

        try:
            if export_format == "md":
                output_file = md_file
                media_type = "text/markdown"

            elif export_format == "html":
                output_file = temp_dir / f"{filename}.html"
                # Convert with pandoc
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file),
                    "--standalone", "--metadata", f"title={filename}"
                ], check=True)
                media_type = "text/html"

            elif export_format == "pdf":
                output_file = temp_dir / f"{filename}.pdf"
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file),
                    "--pdf-engine=xelatex",
                    "-V", "geometry:margin=1in",
                    "-V", "fontsize=11pt"
                ], check=True, capture_output=True)
                media_type = "application/pdf"

            elif export_format == "docx":
                output_file = temp_dir / f"{filename}.docx"
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file)
                ], check=True)
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            elif export_format == "epub":
                output_file = temp_dir / f"{filename}.epub"
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file),
                    "--metadata", f"title={filename}"
                ], check=True)
                media_type = "application/epub+zip"

            elif export_format == "slides":
                output_file = temp_dir / f"{filename}_slides.pdf"
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file),
                    "-t", "beamer",
                    "--pdf-engine=xelatex",
                    "-V", "theme:Madrid"
                ], check=True, capture_output=True)
                media_type = "application/pdf"

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {export_format}")

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

        except subprocess.CalledProcessError as e:
            logger.error(f"Export conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
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

                    if not content.strip():
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
                    await websocket.send_json({"type": "status", "stage": "processing", "detail": "Starting..."})

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
                                        status_callback=sync_status_cb
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
