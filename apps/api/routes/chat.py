"""
Chat routes - messaging, streaming, WebSocket, command streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def register_chat_routes(app, api):
    from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel

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
            command=request.command, args=request.args, session_id=request.session_id
        )
        return result

    @app.get("/api/commands/stream")
    async def stream_command(command: str, args: str = "", session_id: Optional[str] = None):
        """
        SSE streaming command execution endpoint.

        Streams command output in real-time for long-running commands like /ml.
        """
        import io
        import json
        import queue
        import re
        import sys
        import threading

        from starlette.responses import StreamingResponse

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
                clean = re.sub(r"\x1b\[[0-9;]*m", "", str(text))
                clean = re.sub(r"\[/?[^\]]*\]", "", clean)
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
                        if "\n" in text or "\r" in text:
                            parts = re.split(r"[\n\r]+", self.buffer + text)
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
                original_panel = getattr(cli.renderer, "panel", None)
                cli.renderer.panel = lambda content, **kwargs: add_output(
                    f"📋 {kwargs.get('title', 'Panel')}:\n{content}"
                )

                # Capture tree output
                original_tree = getattr(cli.renderer, "tree", None)

                def capture_tree(data, **kwargs):
                    title = kwargs.get("title", "Data")
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
            },
        )

    class ChatRequest(BaseModel):
        message: str
        session_id: Optional[str] = None

    class ChatResponse(BaseModel):
        success: bool = True
        message_id: Optional[str] = None
        content: Optional[str] = None
        output_format: Optional[str] = None
        output_path: Optional[str] = None
        steps: Optional[int] = None
        error: Optional[str] = None

    class ChatWithContextRequest(BaseModel):
        message: str
        session_id: Optional[str] = None
        context_type: str = "document"  # "document" or "folder"
        context_id: Optional[str] = None

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Send a chat message and get response.

        Non-streaming endpoint for simple integrations.
        """
        import uuid

        session_id = request.session_id or str(uuid.uuid4())[:8]

        result = await api.process_message(message=request.message, session_id=session_id)

        return ChatResponse(**result)

    @app.get("/api/chat/stream")
    async def chat_stream(message: str, session_id: Optional[str] = None):
        """
        SSE streaming chat endpoint.

        Returns Server-Sent Events for real-time streaming.
        Uses asyncio.Queue for proper async handling without blocking.
        """
        import json

        from starlette.responses import StreamingResponse

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
                    status_callback=async_status_cb,
                )
                await event_queue.put({"type": "complete", "result": result})
            except Exception as e:
                logger.error(f"Chat processing error: {e}", exc_info=True)
                await event_queue.put(
                    {"type": "complete", "result": {"success": False, "error": str(e)}}
                )
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
            },
        )

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
                context_id=request.context_id,
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
            result = await api.process_message(message=enhanced_message, session_id=session_id)

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
        message: str, context_type: str, context_id: str, session_id: Optional[str] = None
    ):
        """SSE streaming chat with document/folder context."""
        import concurrent.futures
        import json
        import queue

        from starlette.responses import StreamingResponse

        from .documents import get_document_processor

        session_id = session_id or str(uuid.uuid4())[:8]

        async def event_generator():
            # Get context
            try:
                processor = get_document_processor()
                context = processor.get_context_for_chat(
                    query=message, context_type=context_type, context_id=context_id
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
                                status_callback=sync_status_cb,
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
            },
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
                    logger.info(
                        f"WS message received: content length={len(content)}, attachments={len(attachments)}"
                    )
                    if attachments:
                        for i, att in enumerate(attachments):
                            data_preview = att.get("data", "")[:50] if att.get("data") else "None"
                            logger.info(
                                f"  Attachment {i}: type={att.get('type')}, name={att.get('name')}, data_start={data_preview}"
                            )

                    if not content.strip() and not attachments:
                        continue

                    import concurrent.futures
                    import queue

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
                    await websocket.send_json(
                        {"type": "status", "stage": "processing", "detail": status_detail}
                    )

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
                                        attachments=attachments,
                                    )
                                )
                                result_holder["result"] = result
                                logger.info(
                                    f"WS result: success={result.get('success')}, path={result.get('output_path')}"
                                )
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
                        await websocket.send_json(
                            {
                                "type": "complete",
                                "result": {
                                    "success": True,
                                    "message_id": result.get("message_id", ""),
                                    "content": result.get("content", ""),
                                    "output_path": result.get("output_path"),
                                },
                            }
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": (
                                    result.get("error", "Unknown error")
                                    if result
                                    else "Processing failed"
                                ),
                            }
                        )

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
