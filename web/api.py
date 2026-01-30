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
    """

    def __init__(self):
        self._executor = None
        self._registry = None
        self._lm_configured = False

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

        try:
            # Execute
            result = await executor.execute(message)

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

        session_id = session_id or str(uuid.uuid4())[:8]

        async def event_generator():
            # Shared state for streaming
            chunks = []
            statuses = []
            result_holder = {"result": None, "done": False}

            def sync_stream_cb(chunk: str):
                """Synchronous callback that collects chunks."""
                chunks.append(chunk)

            def sync_status_cb(stage: str, detail: str):
                """Synchronous callback that collects statuses."""
                statuses.append({"stage": stage, "detail": detail})

            # Start processing in background task
            async def process():
                try:
                    result = await api.process_message(
                        message=message,
                        session_id=session_id,
                        stream_callback=sync_stream_cb,
                        status_callback=sync_status_cb
                    )
                    result_holder["result"] = result
                except Exception as e:
                    result_holder["result"] = {"success": False, "error": str(e)}
                finally:
                    result_holder["done"] = True

            process_task = asyncio.create_task(process())

            # Send initial event
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            last_chunk_idx = 0
            last_status_idx = 0

            # Poll for events until done
            while not result_holder["done"]:
                # Send any new status updates
                while last_status_idx < len(statuses):
                    status = statuses[last_status_idx]
                    yield f"data: {json.dumps({'type': 'status', 'stage': status['stage'], 'detail': status['detail']})}\n\n"
                    last_status_idx += 1

                # Send any new chunks
                while last_chunk_idx < len(chunks):
                    chunk = chunks[last_chunk_idx]
                    yield f"data: {json.dumps({'type': 'stream', 'chunk': chunk})}\n\n"
                    last_chunk_idx += 1

                # Small sleep to prevent busy loop, also sends keep-alive
                await asyncio.sleep(0.1)
                yield ": keepalive\n\n"

            # Send any remaining events
            while last_status_idx < len(statuses):
                status = statuses[last_status_idx]
                yield f"data: {json.dumps({'type': 'status', 'stage': status['stage'], 'detail': status['detail']})}\n\n"
                last_status_idx += 1

            while last_chunk_idx < len(chunks):
                chunk = chunks[last_chunk_idx]
                yield f"data: {json.dumps({'type': 'stream', 'chunk': chunk})}\n\n"
                last_chunk_idx += 1

            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'result': result_holder['result']})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
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

                    # Create stream callback
                    async def stream_cb(chunk: str):
                        await ws_manager.stream_chunk(session_id, chunk)

                    # Create status callback that sends to WebSocket
                    async def status_cb(stage: str, detail: str):
                        await ws_manager.send_status(session_id, stage, detail)

                    # Send processing status
                    await ws_manager.send_status(session_id, "processing", "Starting...")

                    # Keep-alive task to prevent proxy timeout
                    # Use a mutable container for the flag (closure issue)
                    keep_alive_state = {"running": True}

                    async def keep_alive():
                        count = 0
                        while keep_alive_state["running"]:
                            await asyncio.sleep(2)  # Send ping every 2 seconds
                            if keep_alive_state["running"]:
                                count += 1
                                try:
                                    await ws_manager.send_status(session_id, "thinking", f"Working... ({count * 2}s)")
                                    logger.debug(f"Keep-alive sent: {count * 2}s")
                                except Exception as e:
                                    logger.debug(f"Keep-alive failed: {e}")
                                    break

                    keep_alive_task = asyncio.create_task(keep_alive())

                    try:
                        # Process message with status callback
                        result = await api.process_message(
                            message=content,
                            session_id=session_id,
                            stream_callback=stream_cb,
                            status_callback=status_cb
                        )
                    finally:
                        keep_alive_state["running"] = False
                        keep_alive_task.cancel()
                        try:
                            await keep_alive_task
                        except asyncio.CancelledError:
                            pass

                    if result.get("success"):
                        await ws_manager.send_complete(
                            session_id,
                            result.get("message_id", ""),
                            result.get("content", ""),
                            result.get("output_path")
                        )
                    else:
                        await ws_manager.send_error(
                            session_id,
                            result.get("error", "Unknown error")
                        )

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
