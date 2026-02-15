"""
Web Backend Server (FastAPI + WebSocket)
=========================================

Serves the web app and provides WebSocket API for chat.
Uses shared components for consistent behavior.
"""

import logging
import os

# Add parent to path
import sys
from typing import Dict
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# Import command registry for all 36 commands
from apps.cli.commands import register_all_commands
from apps.cli.commands.base import CommandRegistry, ParsedArgs
from apps.shared import ChatInterface
from apps.shared.events import EventProcessor
from apps.shared.models import Message
from apps.shared.renderers import TelegramMessageRenderer  # Reuse for JSON output
from Jotty.sdk import Jotty

logger = logging.getLogger(__name__)

app = FastAPI(title="Jotty Web API", version="1.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebChatSession:
    """Web chat session with shared components."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.sdk = Jotty()

        # Message queue for WebSocket
        self.message_queue = []

        # Create send callback that queues messages
        def queue_message(text: str):
            """Queue message to send via WebSocket."""
            self.message_queue.append(
                {
                    "type": "message",
                    "content": text,
                    "timestamp": Message.timestamp.default_factory().isoformat(),
                }
            )

        # Create chat interface with JSON renderer
        self.chat = ChatInterface(
            message_renderer=TelegramMessageRenderer(queue_message),  # Reuse for JSON
            status_renderer=None,  # Will send status via WebSocket directly
            input_handler=None,  # Not needed for web
        )

        # Event processor
        self.event_processor = EventProcessor(self.chat)

        # Command registry
        self.command_registry = CommandRegistry()
        try:
            register_all_commands(self.command_registry)
            logger.info(
                f"Session {session_id}: Registered {len(self.command_registry._commands)} commands"
            )
        except Exception as e:
            logger.warning(f"Failed to register commands: {e}")

    async def send_json(self, data: dict):
        """Send JSON data via WebSocket."""
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")

    async def send_queued_messages(self):
        """Send all queued messages."""
        while self.message_queue:
            msg = self.message_queue.pop(0)
            await self.send_json(msg)

    async def handle_message(self, data: dict):
        """Handle incoming message from client."""
        message_text = data.get("content", "")
        message_type = data.get("type", "chat")

        if message_type == "command" or message_text.startswith("/"):
            await self.handle_command(message_text)
        else:
            await self.handle_chat(message_text)

    async def handle_chat(self, message_text: str):
        """Handle regular chat message."""
        # Send user message back
        await self.send_json(
            {"type": "message", "role": "user", "content": message_text, "format": "text"}
        )

        # Add to chat
        user_msg = Message(role="user", content=message_text)
        self.chat.add_message(user_msg)

        # Send thinking status
        await self.send_json(
            {"type": "status", "state": "thinking", "message": "Processing your request..."}
        )

        try:
            # Process via SDK with streaming
            async for event in self.sdk.chat_stream(message_text, session_id=self.session_id):
                # Process event
                await self.event_processor.process_event(event)

                # Send queued messages
                await self.send_queued_messages()

                # Send event to client
                await self.send_json(
                    {
                        "type": "event",
                        "event_type": str(event.type) if hasattr(event, "type") else "unknown",
                        "data": str(event),
                    }
                )

            # Send completion status
            await self.send_json({"type": "status", "state": "idle", "message": "Done"})

        except Exception as e:
            logger.error(f"Error processing chat: {e}", exc_info=True)
            await self.send_json(
                {"type": "error", "message": str(e), "error_type": type(e).__name__}
            )

    async def handle_command(self, command: str):
        """Handle slash command."""
        # Parse command
        parts = command[1:].split() if command.startswith("/") else command.split()
        cmd_name = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        # Special commands
        if cmd_name == "clear":
            self.chat.clear()
            await self.send_json(
                {
                    "type": "command_result",
                    "command": "clear",
                    "success": True,
                    "message": "Chat cleared",
                }
            )
            return

        # Look up in registry
        cmd_obj = self.command_registry.get(cmd_name)

        if not cmd_obj:
            await self.send_json(
                {
                    "type": "error",
                    "message": f"Unknown command: /{cmd_name}",
                    "error_type": "CommandNotFound",
                }
            )
            return

        try:
            # Execute command
            parsed_args = ParsedArgs(command=cmd_name, args=args, raw_input=command)
            result = await cmd_obj.execute(parsed_args)

            # Send result
            await self.send_json(
                {
                    "type": "command_result",
                    "command": cmd_name,
                    "success": result.success if result else False,
                    "output": result.output if result else None,
                    "error": result.error if result and not result.success else None,
                }
            )

        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            await self.send_json(
                {"type": "error", "message": str(e), "error_type": type(e).__name__}
            )


# Active sessions
sessions: Dict[str, WebChatSession] = {}


@app.get("/")
async def read_root():
    """Serve the React app."""
    return FileResponse("apps/web/frontend/build/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "sessions": len(sessions), "version": "1.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat."""
    await websocket.accept()

    # Create session
    session_id = str(uuid4())
    session = WebChatSession(session_id, websocket)
    sessions[session_id] = session

    logger.info(f"New WebSocket connection: {session_id}")

    # Send welcome message
    await session.send_json(
        {
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to Jotty AI",
            "commands_available": len(session.command_registry._commands),
        }
    )

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Handle message
            await session.handle_message(data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        del sessions[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await session.send_json(
                {"type": "error", "message": str(e), "error_type": type(e).__name__}
            )
        except Exception:
            pass
        finally:
            if session_id in sessions:
                del sessions[session_id]


def main():
    """Start the web server."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting Jotty Web Server...")
    logger.info("WebSocket: ws://localhost:8000/ws")
    logger.info("Web UI: http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
