"""
WhatsApp Web Client
===================

Python wrapper for whatsapp-web.js bridge.
Manages Node.js subprocess and provides async API.
"""

import asyncio
import json
import logging
import subprocess
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WhatsAppWebMessage:
    """Incoming WhatsApp message."""
    id: str
    from_number: str
    to_number: str
    body: str
    type: str
    timestamp: int
    is_group: bool
    chat_name: str
    sender_name: str
    has_media: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)


class WhatsAppWebClient:
    """
    WhatsApp Web client using whatsapp-web.js.

    Connects to WhatsApp via QR code pairing (like OpenClaw).
    No business account needed - uses your personal WhatsApp.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._connected = False
        self._qr_code: Optional[str] = None
        self._info: Optional[Dict] = None
        self._message_handlers: List[Callable] = []
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None

        # Path to bridge.js
        self._bridge_path = Path(__file__).parent / "bridge.js"

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def qr_code(self) -> Optional[str]:
        return self._qr_code

    def on_message(self, handler: Callable):
        """Register message handler."""
        self._message_handlers.append(handler)

    async def start(self) -> bool:
        """
        Start WhatsApp Web client.

        Returns True if started successfully.
        Will emit QR code event when ready to scan.
        """
        if self._process:
            logger.warning("WhatsApp client already running")
            return False

        # Check if Node.js is available
        try:
            result = subprocess.run(["node", "--version"], capture_output=True)
            if result.returncode != 0:
                raise FileNotFoundError()
        except FileNotFoundError:
            logger.error("Node.js not found. Install with: apt install nodejs")
            return False

        # Check if dependencies installed
        package_json = self._bridge_path.parent / "package.json"
        node_modules = self._bridge_path.parent / "node_modules"

        if not node_modules.exists():
            logger.info("Installing WhatsApp Web dependencies...")
            await self._install_dependencies()

        # Start Node.js process
        try:
            self._process = subprocess.Popen(
                ["node", str(self._bridge_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Start reading output
            self._read_task = asyncio.create_task(self._read_output())

            logger.info("WhatsApp Web client started")
            return True

        except Exception as e:
            logger.error(f"Failed to start WhatsApp client: {e}")
            return False

    async def _install_dependencies(self):
        """Install npm dependencies."""
        package_dir = self._bridge_path.parent

        # Create package.json if not exists
        package_json = package_dir / "package.json"
        if not package_json.exists():
            package_json.write_text(json.dumps({
                "name": "jotty-whatsapp-bridge",
                "version": "1.0.0",
                "dependencies": {
                    "whatsapp-web.js": "^1.26.0",
                    "qrcode-terminal": "^0.12.0"
                }
            }, indent=2))

        # Run npm install
        process = await asyncio.create_subprocess_exec(
            "npm", "install",
            cwd=str(package_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()

    async def _read_output(self):
        """Read and process output from Node.js bridge."""
        while self._process and self._process.poll() is None:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.readline
                )

                if not line:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    data = json.loads(line.strip())
                    await self._handle_event(data)
                except json.JSONDecodeError:
                    # QR code ASCII art or other non-JSON output
                    if "█" in line or "▄" in line:
                        print(line, end="")

            except Exception as e:
                logger.error(f"Error reading output: {e}")
                await asyncio.sleep(0.1)

    async def _handle_event(self, data: Dict[str, Any]):
        """Handle event from Node.js bridge."""
        event_type = data.get("type")

        if event_type == "qr":
            self._qr_code = data.get("qr_code")
            logger.info("QR code received - scan with WhatsApp")

        elif event_type == "ready":
            self._connected = True
            self._info = data.get("info")
            logger.info(f"WhatsApp connected: {self._info}")

        elif event_type == "authenticated":
            logger.info("WhatsApp authenticated")

        elif event_type == "disconnected":
            self._connected = False
            logger.warning(f"WhatsApp disconnected: {data.get('reason')}")

        elif event_type == "message":
            msg = WhatsAppWebMessage(
                id=data.get("id"),
                from_number=data.get("from"),
                to_number=data.get("to"),
                body=data.get("body", ""),
                type=data.get("type"),
                timestamp=data.get("timestamp"),
                is_group=data.get("is_group", False),
                chat_name=data.get("chat_name", ""),
                sender_name=data.get("sender_name", ""),
                has_media=data.get("has_media", False),
                raw_data=data
            )

            # Call message handlers
            for handler in self._message_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(msg)
                    else:
                        handler(msg)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")

        elif event_type == "sent":
            request_id = data.get("request_id")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                future.set_result(data)

        elif event_type == "chats":
            request_id = data.get("request_id")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                future.set_result(data.get("chats", []))

        elif event_type == "contacts":
            request_id = data.get("request_id")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                future.set_result(data.get("contacts", []))

        elif event_type == "error":
            request_id = data.get("request_id")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                future.set_exception(Exception(data.get("error")))

    def _send_command(self, command: Dict[str, Any]):
        """Send command to Node.js bridge."""
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("WhatsApp client not running")

        self._process.stdin.write(json.dumps(command) + "\n")
        self._process.stdin.flush()

    async def send_message(self, to: str, message: str) -> Dict[str, Any]:
        """
        Send a text message.

        Args:
            to: Phone number with country code (e.g., '14155238886')
            message: Message text

        Returns:
            Dict with success status and message_id
        """
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        self._send_command({
            "action": "send_message",
            "request_id": request_id,
            "to": to,
            "message": message
        })

        try:
            result = await asyncio.wait_for(future, timeout=30)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            return {"success": False, "error": "Timeout"}

    async def send_media(
        self,
        to: str,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        caption: str = ""
    ) -> Dict[str, Any]:
        """
        Send media (image, document, etc).

        Args:
            to: Phone number
            file_path: Local file path
            url: URL to media
            caption: Optional caption

        Returns:
            Dict with success status
        """
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        cmd = {
            "action": "send_media",
            "request_id": request_id,
            "to": to,
            "caption": caption
        }

        if file_path:
            cmd["file_path"] = file_path
        elif url:
            cmd["url"] = url

        self._send_command(cmd)

        try:
            result = await asyncio.wait_for(future, timeout=60)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            return {"success": False, "error": "Timeout"}

    async def get_chats(self, limit: int = 50) -> List[Dict]:
        """Get recent chats."""
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        self._send_command({
            "action": "get_chats",
            "request_id": request_id,
            "limit": limit
        })

        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            return []

    async def get_contacts(self, limit: int = 100) -> List[Dict]:
        """Get contacts."""
        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        self._send_command({
            "action": "get_contacts",
            "request_id": request_id,
            "limit": limit
        })

        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            return []

    async def logout(self):
        """Logout from WhatsApp."""
        if self._process:
            self._send_command({"action": "logout"})
            await asyncio.sleep(2)

    async def stop(self):
        """Stop the WhatsApp client."""
        if self._read_task:
            self._read_task.cancel()

        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None

        self._connected = False
        logger.info("WhatsApp client stopped")
