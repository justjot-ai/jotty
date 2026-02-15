"""
WhatsApp Command
================

/whatsapp - Connect to WhatsApp using QR code (like OpenClaw)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)

# Same path as bridge.js (Baileys useMultiFileAuthState)
WHATSAPP_SESSION_DIR = Path(os.environ.get("HOME", "/tmp")) / ".jotty" / "whatsapp_session"


def _whatsapp_session_has_creds() -> bool:
    """Return True if saved WhatsApp credentials exist (no QR needed)."""
    creds_file = WHATSAPP_SESSION_DIR / "creds.json"
    return creds_file.is_file()


class WhatsAppCommand(BaseCommand):
    """
    /whatsapp - Personal WhatsApp integration.

    Connect using your personal WhatsApp account via QR code.
    No business account needed - just like OpenClaw/Moltbot.
    """

    name = "whatsapp"
    aliases = ["wa"]
    description = "Personal WhatsApp via QR code (like OpenClaw)"
    usage = "/whatsapp [login|check|send|chats|logout] [args]"
    category = "messaging"

    _client = None  # Singleton client

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute WhatsApp command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "login":
            return await self._login(cli)
        elif subcommand == "check":
            return await self._check(cli)
        elif subcommand == "send":
            to = args.flags.get("to") or (args.positional[1] if len(args.positional) > 1 else None)
            message = (
                args.flags.get("message") or " ".join(args.positional[2:])
                if len(args.positional) > 2
                else None
            )
            return await self._send_message(cli, to, message)
        elif subcommand == "chats":
            return await self._list_chats(cli)
        elif subcommand == "contacts":
            return await self._list_contacts(cli)
        elif subcommand == "logout":
            return await self._logout(cli)
        elif subcommand == "status":
            return await self._status(cli)
        elif subcommand == "stop":
            return await self._stop(cli)
        else:
            # Assume it's a phone number and rest is message
            to = subcommand
            message = " ".join(args.positional[1:])
            if message:
                return await self._send_message(cli, to, message)
            else:
                return await self._status(cli)

    async def _get_client(self) -> Any:
        """Get or create WhatsApp client."""
        if WhatsAppCommand._client is None:
            from ..channels.whatsapp_web import WhatsAppWebClient, set_global_whatsapp_client

            WhatsAppCommand._client = WhatsAppWebClient()
            set_global_whatsapp_client(WhatsAppCommand._client)
        return WhatsAppCommand._client

    async def _login(self, cli: "JottyCLI") -> CommandResult:
        """Login via QR code."""
        cli.renderer.header("WhatsApp Login")
        cli.renderer.info("Starting WhatsApp Web connection...")
        cli.renderer.info("A QR code will appear - scan it with your phone's WhatsApp")
        cli.renderer.newline()

        try:
            client = await self._get_client()

            # Check if already connected
            if client.connected:
                cli.renderer.success("Already connected to WhatsApp!")
                return CommandResult.ok()

            # Register message handler to route to gateway
            async def handle_message(msg: Any) -> Any:
                cli.renderer.print(f"[green]WhatsApp[/green] {msg.sender_name}: {msg.body[:100]}")
                # Route to gateway if available
                if hasattr(cli, "_gateway_router"):
                    from ..gateway.channels import ChannelType, MessageEvent

                    event = MessageEvent(
                        channel=ChannelType.WHATSAPP,
                        channel_id=msg.from_number,
                        user_id=msg.from_number,
                        user_name=msg.sender_name,
                        content=msg.body,
                        message_id=msg.id,
                    )
                    await cli._gateway_router.handle_message(event)

            client.on_message(handle_message)

            # Start client
            started = await client.start()
            if not started:
                cli.renderer.error("Failed to start WhatsApp client")
                cli.renderer.info("Make sure Node.js is installed: apt install nodejs npm")
                return CommandResult.fail("Failed to start")

            # Show appropriate message: saved session = no QR needed
            if _whatsapp_session_has_creds():
                cli.renderer.info("Restoring session from saved credentials...")
            else:
                cli.renderer.info("Waiting for QR code...")
                cli.renderer.info(
                    "(The QR code will appear below - scan with WhatsApp on your phone)"
                )
            cli.renderer.newline()

            # Wait for connection (saved session or after QR scan)
            for i in range(60):  # Wait up to 60 seconds
                await asyncio.sleep(1)
                if client.connected:
                    cli.renderer.newline()
                    cli.renderer.success("WhatsApp connected successfully!")
                    cli.renderer.info("You can now receive and send messages")
                    cli.renderer.info("Messages will appear in this terminal")
                    cli.renderer.newline()
                    cli.renderer.info("Commands:")
                    cli.renderer.print("  /whatsapp send --to 14155238886 --message 'Hello!'")
                    cli.renderer.print("  /whatsapp chats")
                    cli.renderer.print("  /whatsapp logout")
                    return CommandResult.ok()

            cli.renderer.warning("Connection timeout - please try again")
            return CommandResult.fail("Timeout")

        except Exception as e:
            logger.error(f"WhatsApp login error: {e}", exc_info=True)
            cli.renderer.error(f"Login failed: {e}")
            return CommandResult.fail(str(e))

    async def _send_message(self, cli: "JottyCLI", to: str, message: str) -> CommandResult:
        """Send a message."""
        if not to:
            cli.renderer.error("Recipient required")
            cli.renderer.info("Usage: /whatsapp send --to 14155238886 --message 'Hello!'")
            cli.renderer.info("   or: /whatsapp 14155238886 Hello!")
            return CommandResult.fail("Recipient required")

        if not message:
            cli.renderer.error("Message required")
            return CommandResult.fail("Message required")

        try:
            client = await self._get_client()

            if not client.connected:
                cli.renderer.warning("Not connected to WhatsApp")
                cli.renderer.info("Run /whatsapp login first")
                return CommandResult.fail("Not connected")

            cli.renderer.info(f"Sending to {to}...")
            result = await client.send_message(to, message)

            if result.get("success"):
                cli.renderer.success(f"Message sent to {to}")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Send failed: {result.get('error')}")
                return CommandResult.fail(result.get("error"))

        except Exception as e:
            logger.error(f"Send error: {e}", exc_info=True)
            cli.renderer.error(f"Send failed: {e}")
            return CommandResult.fail(str(e))

    async def _list_chats(self, cli: "JottyCLI") -> CommandResult:
        """List recent chats."""
        try:
            client = await self._get_client()

            if not client.connected:
                cli.renderer.warning("Not connected to WhatsApp")
                cli.renderer.info("Run /whatsapp login first")
                return CommandResult.fail("Not connected")

            cli.renderer.info("Loading chats...")
            chats = await client.get_chats()

            cli.renderer.header("WhatsApp Chats")
            for chat in chats[:20]:
                unread = f" ({chat['unread_count']} unread)" if chat.get("unread_count") else ""
                group_icon = "" if chat.get("is_group") else ""
                cli.renderer.print(f"  {group_icon} {chat['name']}{unread}")

            return CommandResult.ok(data={"chats": chats})

        except Exception as e:
            cli.renderer.error(f"Failed to load chats: {e}")
            return CommandResult.fail(str(e))

    async def _list_contacts(self, cli: "JottyCLI") -> CommandResult:
        """List contacts."""
        try:
            client = await self._get_client()

            if not client.connected:
                cli.renderer.warning("Not connected to WhatsApp")
                return CommandResult.fail("Not connected")

            cli.renderer.info("Loading contacts...")
            contacts = await client.get_contacts()

            cli.renderer.header("WhatsApp Contacts")
            for contact in contacts[:30]:
                cli.renderer.print(f"  {contact['name'] or 'Unknown'}: {contact['number']}")

            return CommandResult.ok(data={"contacts": contacts})

        except Exception as e:
            cli.renderer.error(f"Failed to load contacts: {e}")
            return CommandResult.fail(str(e))

    async def _logout(self, cli: "JottyCLI") -> CommandResult:
        """Logout from WhatsApp."""
        try:
            client = await self._get_client()
            await client.logout()
            cli.renderer.success("Logged out from WhatsApp")
            return CommandResult.ok()
        except Exception as e:
            cli.renderer.error(f"Logout failed: {e}")
            return CommandResult.fail(str(e))

    async def _stop(self, cli: "JottyCLI") -> CommandResult:
        """Stop WhatsApp client."""
        try:
            client = await self._get_client()
            await client.stop()
            WhatsAppCommand._client = None
            from ..channels.whatsapp_web import set_global_whatsapp_client

            set_global_whatsapp_client(None)
            cli.renderer.success("WhatsApp client stopped")
            return CommandResult.ok()
        except Exception as e:
            cli.renderer.error(f"Stop failed: {e}")
            return CommandResult.fail(str(e))

    async def _check(self, cli: "JottyCLI") -> CommandResult:
        """Verify saved WhatsApp credentials (session path and creds.json)."""
        cli.renderer.header("WhatsApp session check")
        cli.renderer.print(f"  Session path: [cyan]{WHATSAPP_SESSION_DIR}[/cyan]")
        if not WHATSAPP_SESSION_DIR.exists():
            cli.renderer.warning("Session directory does not exist.")
            cli.renderer.info("Run /whatsapp login and scan the QR code to create it.")
            return CommandResult.fail("No session directory")
        creds = WHATSAPP_SESSION_DIR / "creds.json"
        if creds.is_file():
            cli.renderer.success("Saved credentials found (creds.json)")
            cli.renderer.info(
                "Next /whatsapp login will restore session without QR (if still valid)."
            )
        else:
            cli.renderer.warning("No creds.json found in session directory.")
            cli.renderer.info("Run /whatsapp login and scan the QR code once to save credentials.")
        others = [f.name for f in WHATSAPP_SESSION_DIR.iterdir() if f.is_file()]
        if others:
            cli.renderer.print(
                f"  Other files: [dim]{', '.join(others[:10])}{'...' if len(others) > 10 else ''}[/dim]"
            )
        return CommandResult.ok(
            data={"session_dir": str(WHATSAPP_SESSION_DIR), "has_creds": creds.is_file()}
        )

    async def _status(self, cli: "JottyCLI") -> CommandResult:
        """Show connection status."""
        try:
            client = await self._get_client()

            if client.connected:
                cli.renderer.success("WhatsApp: Connected")
                if client._info:
                    cli.renderer.print(
                        f"  Phone: {client._info.get('wid', {}).get('user', 'Unknown')}"
                    )
            else:
                cli.renderer.warning("WhatsApp: Not connected")
                if _whatsapp_session_has_creds():
                    cli.renderer.info(
                        "Saved credentials found. Run /whatsapp login to restore session (no QR)."
                    )
                else:
                    cli.renderer.info(
                        "Run /whatsapp login to connect (scan QR once to save credentials)."
                    )

            return CommandResult.ok()

        except Exception:
            cli.renderer.warning("WhatsApp: Not initialized")
            cli.renderer.info("Run /whatsapp login to connect")
            return CommandResult.ok()

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        subcommands = ["login", "check", "send", "chats", "contacts", "logout", "status", "stop"]
        flags = ["--to", "--message"]
        return [s for s in subcommands + flags if s.startswith(partial)]
