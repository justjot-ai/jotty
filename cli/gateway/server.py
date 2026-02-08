"""
Unified Gateway Server
======================

WebSocket + HTTP server for receiving messages from:
- Telegram (webhook)
- Slack (Events API)
- Discord (webhook)
- WhatsApp (webhook)
- Direct WebSocket connections

Inspired by OpenClaw's gateway architecture.
"""

import os
import asyncio
import json
import hmac
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Try imports
try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, RedirectResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn")

from .channels import ChannelRouter, ChannelType, MessageEvent, ResponseEvent
from .trust import TrustManager
from .responders import get_responder_registry, ChannelResponderRegistry
from .responders import ResponseEvent as ResponderResponseEvent


class UnifiedGateway:
    """
    Unified Gateway for all message channels.

    Provides:
    - WebSocket server for real-time connections
    - HTTP webhooks for Telegram, Slack, Discord
    - Message routing to Jotty agents
    - Response delivery back to channels
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766, enable_trust: bool = True):
        self.host = host
        self.port = port
        self.router = ChannelRouter()
        self.trust = TrustManager() if enable_trust else None
        self._app = None
        self._cli = None
        self._websocket_clients: Set[WebSocket] = set()

        # Channel tokens from environment
        self._telegram_token = os.getenv("TELEGRAM_TOKEN")
        self._slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        self._discord_public_key = os.getenv("DISCORD_PUBLIC_KEY")

        # Wire trust manager to router
        if self.trust:
            self.router.set_trust_manager(self.trust)

    def set_cli(self, cli):
        """Set JottyCLI instance."""
        self._cli = cli
        self.router.set_cli(cli)
        self._setup_responders()

    def _setup_responders(self):
        """Setup response handlers for each channel using the responder registry."""
        # Get the responder registry (uses registry-based skill discovery)
        responder_registry = get_responder_registry()

        # WebSocket responder needs access to _websocket_clients, so we keep it custom
        async def websocket_responder(response: ResponseEvent):
            message = json.dumps({
                "type": "response",
                "channel_id": response.channel_id,
                "content": response.content
            })
            # Broadcast to all connected clients
            for ws in list(self._websocket_clients):
                try:
                    await ws.send_text(message)
                except Exception:
                    self._websocket_clients.discard(ws)

        # Create wrapper responders that use the registry
        async def telegram_responder(response: ResponseEvent):
            resp_event = ResponderResponseEvent(
                channel=ChannelType.TELEGRAM,
                channel_id=response.channel_id,
                content=response.content,
                reply_to=response.reply_to
            )
            await responder_registry.send(resp_event)

        async def slack_responder(response: ResponseEvent):
            resp_event = ResponderResponseEvent(
                channel=ChannelType.SLACK,
                channel_id=response.channel_id,
                content=response.content,
                reply_to=response.reply_to
            )
            await responder_registry.send(resp_event)

        async def discord_responder(response: ResponseEvent):
            resp_event = ResponderResponseEvent(
                channel=ChannelType.DISCORD,
                channel_id=response.channel_id,
                content=response.content,
                reply_to=response.reply_to
            )
            await responder_registry.send(resp_event)

        async def whatsapp_responder(response: ResponseEvent):
            resp_event = ResponderResponseEvent(
                channel=ChannelType.WHATSAPP,
                channel_id=response.channel_id,
                content=response.content,
                reply_to=response.reply_to
            )
            await responder_registry.send(resp_event)

        self.router.register_responder(ChannelType.TELEGRAM, telegram_responder)
        self.router.register_responder(ChannelType.SLACK, slack_responder)
        self.router.register_responder(ChannelType.DISCORD, discord_responder)
        self.router.register_responder(ChannelType.WEBSOCKET, websocket_responder)
        self.router.register_responder(ChannelType.WHATSAPP, whatsapp_responder)

    def create_app(self) -> "FastAPI":
        """Create FastAPI application with all endpoints."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        app = FastAPI(
            title="Jotty Gateway",
            description="Unified message gateway for Jotty AI",
            version="1.0.0",
            docs_url="/docs"
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static files for PWA
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Root redirect to PWA
        @app.get("/")
        async def root():
            return RedirectResponse(url="/static/index.html")

        # PWA app route
        @app.get("/app")
        async def pwa_app():
            return FileResponse(static_dir / "index.html")

        # Health check
        @app.get("/health")
        async def health():
            result = {
                "status": "healthy",
                "service": "jotty-gateway",
                "active_sessions": self.router.active_sessions,
                "websocket_clients": len(self._websocket_clients)
            }
            if self.trust:
                result["trust"] = self.trust.stats
            return result

        # Gateway stats
        @app.get("/stats")
        async def stats():
            return {
                **self.router.stats,
                "websocket_clients": len(self._websocket_clients)
            }

        # ============ TELEGRAM WEBHOOK ============
        @app.post("/webhook/telegram")
        async def telegram_webhook(request: Request):
            """Handle Telegram webhook updates."""
            try:
                data = await request.json()
                logger.debug(f"Telegram webhook: {data}")

                # Extract message
                message = data.get("message") or data.get("edited_message")
                if not message:
                    return {"ok": True}

                text = message.get("text", "")
                if not text:
                    return {"ok": True}

                chat = message.get("chat", {})
                user = message.get("from", {})

                event = MessageEvent(
                    channel=ChannelType.TELEGRAM,
                    channel_id=str(chat.get("id")),
                    user_id=str(user.get("id")),
                    user_name=user.get("first_name", "Unknown"),
                    content=text,
                    message_id=str(message.get("message_id")),
                    raw_data=data
                )

                # Process async
                asyncio.create_task(self.router.handle_message(event))

                return {"ok": True}

            except Exception as e:
                logger.error(f"Telegram webhook error: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        # ============ SLACK EVENTS API ============
        @app.post("/webhook/slack")
        async def slack_webhook(request: Request):
            """Handle Slack Events API."""
            try:
                data = await request.json()

                # URL verification challenge
                if data.get("type") == "url_verification":
                    return {"challenge": data.get("challenge")}

                # Verify signature if secret available
                if self._slack_signing_secret:
                    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
                    sig_header = request.headers.get("X-Slack-Signature", "")
                    body = await request.body()

                    sig_basestring = f"v0:{timestamp}:{body.decode()}"
                    my_sig = "v0=" + hmac.new(
                        self._slack_signing_secret.encode(),
                        sig_basestring.encode(),
                        hashlib.sha256
                    ).hexdigest()

                    if not hmac.compare_digest(my_sig, sig_header):
                        raise HTTPException(status_code=401, detail="Invalid signature")

                # Handle event
                event = data.get("event", {})
                event_type = event.get("type")

                if event_type == "message" and not event.get("bot_id"):
                    msg_event = MessageEvent(
                        channel=ChannelType.SLACK,
                        channel_id=event.get("channel"),
                        user_id=event.get("user"),
                        user_name=event.get("user"),  # Will be resolved later
                        content=event.get("text", ""),
                        message_id=event.get("ts"),
                        reply_to=event.get("thread_ts"),
                        raw_data=data
                    )

                    asyncio.create_task(self.router.handle_message(msg_event))

                return {"ok": True}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Slack webhook error: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        # ============ DISCORD INTERACTIONS ============
        @app.post("/webhook/discord")
        async def discord_webhook(request: Request):
            """Handle Discord interactions/webhooks."""
            try:
                data = await request.json()

                # Ping verification
                if data.get("type") == 1:
                    return {"type": 1}

                # Message component or command
                if data.get("type") in [2, 3]:
                    user = data.get("member", {}).get("user", {}) or data.get("user", {})
                    channel_id = data.get("channel_id")

                    # Get content from interaction
                    content = ""
                    if data.get("type") == 2:  # Slash command
                        options = data.get("data", {}).get("options", [])
                        content = " ".join(opt.get("value", "") for opt in options)
                    elif data.get("type") == 3:  # Message component
                        content = data.get("data", {}).get("custom_id", "")

                    if content:
                        event = MessageEvent(
                            channel=ChannelType.DISCORD,
                            channel_id=channel_id,
                            user_id=user.get("id"),
                            user_name=user.get("username", "Unknown"),
                            content=content,
                            message_id=data.get("id"),
                            raw_data=data
                        )

                        asyncio.create_task(self.router.handle_message(event))

                    # Acknowledge interaction
                    return {
                        "type": 5,  # Deferred response
                    }

                return {"ok": True}

            except Exception as e:
                logger.error(f"Discord webhook error: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        # ============ WHATSAPP WEBHOOK ============
        @app.post("/webhook/whatsapp")
        async def whatsapp_webhook(request: Request):
            """Handle WhatsApp Business API webhooks."""
            try:
                data = await request.json()
                logger.debug(f"WhatsApp webhook: {data}")

                # Extract message from WhatsApp Cloud API format
                entry = data.get("entry", [{}])[0]
                changes = entry.get("changes", [{}])[0]
                value = changes.get("value", {})
                messages = value.get("messages", [])

                for msg in messages:
                    if msg.get("type") == "text":
                        contact = value.get("contacts", [{}])[0]

                        event = MessageEvent(
                            channel=ChannelType.WHATSAPP,
                            channel_id=msg.get("from"),
                            user_id=msg.get("from"),
                            user_name=contact.get("profile", {}).get("name", "Unknown"),
                            content=msg.get("text", {}).get("body", ""),
                            message_id=msg.get("id"),
                            raw_data=data
                        )

                        asyncio.create_task(self.router.handle_message(event))

                return {"status": "ok"}

            except Exception as e:
                logger.error(f"WhatsApp webhook error: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        # WhatsApp verification
        @app.get("/webhook/whatsapp")
        async def whatsapp_verify(request: Request):
            """WhatsApp webhook verification."""
            mode = request.query_params.get("hub.mode")
            token = request.query_params.get("hub.verify_token")
            challenge = request.query_params.get("hub.challenge")

            verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "jotty")

            if mode == "subscribe" and token == verify_token:
                return int(challenge)

            raise HTTPException(status_code=403, detail="Verification failed")

        # ============ WEBSOCKET ============
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time bidirectional communication."""
            await websocket.accept()
            self._websocket_clients.add(websocket)
            logger.info(f"WebSocket client connected. Total: {len(self._websocket_clients)}")

            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    event = MessageEvent(
                        channel=ChannelType.WEBSOCKET,
                        channel_id=message.get("channel_id", "default"),
                        user_id=message.get("user_id", "ws_user"),
                        user_name=message.get("user_name", "WebSocket User"),
                        content=message.get("content", ""),
                        raw_data=message
                    )

                    response = await self.router.handle_message(event)

                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    }))

            except WebSocketDisconnect:
                self._websocket_clients.discard(websocket)
                logger.info(f"WebSocket client disconnected. Total: {len(self._websocket_clients)}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                self._websocket_clients.discard(websocket)

        # ============ GENERIC HTTP ============
        @app.post("/message")
        async def http_message(request: Request):
            """Generic HTTP endpoint for sending messages."""
            try:
                data = await request.json()

                event = MessageEvent(
                    channel=ChannelType.HTTP,
                    channel_id=data.get("channel_id", "http"),
                    user_id=data.get("user_id", "http_user"),
                    user_name=data.get("user_name", "HTTP User"),
                    content=data.get("content", ""),
                    raw_data=data
                )

                response = await self.router.handle_message(event)

                return {
                    "success": True,
                    "response": response
                }

            except Exception as e:
                logger.error(f"HTTP message error: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

        self._app = app
        return app

    async def run_async(self):
        """Run gateway server asynchronously."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed")

        app = self.create_app()
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    def run(self):
        """Run gateway server (blocking)."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed")

        app = self.create_app()
        uvicorn.run(app, host=self.host, port=self.port)


def start_gateway(host: str = "0.0.0.0", port: int = 8766, cli=None):
    """Start the unified gateway server."""
    gateway = UnifiedGateway(host, port)
    if cli:
        gateway.set_cli(cli)
    gateway.run()


if __name__ == "__main__":
    start_gateway()
