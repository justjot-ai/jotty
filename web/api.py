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
        """Get all sessions."""
        registry = self._get_session_registry()
        # Use a temporary session manager to list sessions
        from ..cli.repl.session import SessionManager
        manager = SessionManager()
        return manager.list_sessions()

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


def create_app() -> "FastAPI":
    """
    Create FastAPI application.

    Returns:
        Configured FastAPI app
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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
        """
        from starlette.responses import StreamingResponse
        import json
        import queue
        import threading
        import concurrent.futures

        session_id = session_id or str(uuid.uuid4())[:8]

        async def event_generator():
            # Thread-safe queue for chunks and statuses
            event_queue = queue.Queue()
            result_holder = {"result": None, "done": False}

            def sync_stream_cb(chunk: str):
                """Thread-safe callback that queues chunks."""
                logger.info(f"STREAM_CB: '{chunk[:30]}...' queued")
                event_queue.put({"type": "stream", "chunk": chunk})

            def sync_status_cb(stage: str, detail: str):
                """Thread-safe callback that queues statuses."""
                logger.info(f"STATUS_CB: {stage} - {detail}")
                event_queue.put({"type": "status", "stage": stage, "detail": detail})

            # Run processing in thread pool to allow event loop to yield
            def process_sync():
                try:
                    # Create event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            api.process_message(
                                message=message,
                                session_id=session_id,
                                stream_callback=sync_stream_cb,
                                status_callback=sync_status_cb
                            )
                        )
                        result_holder["result"] = result
                    finally:
                        loop.close()
                except Exception as e:
                    logger.error(f"Chat processing error: {e}", exc_info=True)
                    result_holder["result"] = {"success": False, "error": str(e)}
                finally:
                    result_holder["done"] = True

            # Start in thread pool
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(process_sync)

            # Padding to flush proxy buffers - need to exceed proxy buffer size
            # nginx default is 4k/8k, but code-server proxy may have larger buffers
            padding = " " * 16384  # 16KB padding

            # Send initial event with padding
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n{padding}"

            # Poll for events until done
            event_count = 0
            while not result_holder["done"]:
                # Drain queue and send events
                while True:
                    try:
                        event = event_queue.get_nowait()
                        event_count += 1
                        logger.info(f"SENDING event #{event_count}: {event.get('type')}")
                        yield f"data: {json.dumps(event)}\n\n{padding}"
                    except queue.Empty:
                        break

                # Small sleep to prevent busy loop
                await asyncio.sleep(0.05)  # Faster polling
                yield f": keepalive\n\n{padding}"

            # Drain any remaining events
            while True:
                try:
                    event = event_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    break

            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'result': result_holder['result']})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, no-transform, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "X-Content-Type-Options": "nosniff",
                "Transfer-Encoding": "chunked",
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
