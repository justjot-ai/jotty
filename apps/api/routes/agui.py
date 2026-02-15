"""
AG-UI routes - Agent UI protocol endpoints.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def register_agui_routes(app, api):
    import uuid

    from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel

    class AGUIRunRequest(BaseModel):
        threadId: str
        runId: str
        messages: List[Dict[str, Any]] = []

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
        import concurrent.futures
        import json
        import queue

        from starlette.responses import StreamingResponse

        thread_id = request.threadId
        run_id = request.runId

        async def agui_event_generator():
            # Emit RunStarted
            yield f"data: {json.dumps({'type': 'RunStarted', 'threadId': thread_id, 'runId': run_id, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Get the latest user message
            user_messages = [m for m in request.messages if m.get("role") == "user"]
            if not user_messages:
                yield f"data: {json.dumps({'type': 'RunError', 'message': 'No user message found', 'code': 'NO_MESSAGE'})}\n\n"
                return

            user_message = user_messages[-1].get("content", "")
            message_id = f"msg_{uuid.uuid4().hex[:12]}"

            # Thread-safe queue for events
            event_queue = queue.Queue()
            result_holder = {"result": None, "done": False}
            padding = " " * 8192

            # Emit TextMessageStart
            yield f"data: {json.dumps({'type': 'TextMessageStart', 'messageId': message_id, 'role': 'assistant', 'timestamp': datetime.now().isoformat()})}\n\n{padding}"

            def sync_stream_cb(chunk: str):
                """Emit TextMessageContent for each chunk."""
                event_queue.put(
                    {"type": "TextMessageContent", "messageId": message_id, "delta": chunk}
                )

            def sync_status_cb(stage: str, detail: str):
                """Emit ActivitySnapshot for status updates."""
                event_queue.put(
                    {
                        "type": "ActivitySnapshot",
                        "messageId": f"activity_{run_id}",
                        "activityType": "PROGRESS",
                        "content": {
                            "stage": stage,
                            "detail": detail,
                            "label": f"{stage}: {detail}",
                        },
                    }
                )

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
                                status_callback=sync_status_cb,
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
            },
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
                "activity_tracking",
            ],
            "documentation": "https://docs.ag-ui.com/",
        }

    # Export endpoints
