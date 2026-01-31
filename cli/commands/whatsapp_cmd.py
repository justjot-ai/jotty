"""
WhatsApp Command
================

/whatsapp - Connect to WhatsApp using QR code (like OpenClaw)
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class WhatsAppCommand(BaseCommand):
    """
    /whatsapp - Personal WhatsApp integration.

    Connect using your personal WhatsApp account via QR code.
    No business account needed - just like OpenClaw/Moltbot.
    """

    name = "whatsapp"
    aliases = ["wa"]
    description = "Personal WhatsApp via QR code (like OpenClaw)"
    usage = "/whatsapp [login|send|chats|logout] [args]"
    category = "messaging"

    _client = None  # Singleton client

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute WhatsApp command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "login":
            return await self._login(cli)
        elif subcommand == "send":
            to = args.flags.get("to") or (args.positional[1] if len(args.positional) > 1 else None)
            message = args.flags.get("message") or " ".join(args.positional[2:]) if len(args.positional) > 2 else None
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

    async def _get_client(self):
        """Get or create WhatsApp client."""
        if WhatsAppCommand._client is None:
            from ..channels.whatsapp_web import WhatsAppWebClient
            WhatsAppCommand._client = WhatsAppWebClient()
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
            async def handle_message(msg):
                cli.renderer.print(f"[green]WhatsApp[/green] {msg.sender_name}: {msg.body[:100]}")
                # Route to gateway if available
                if hasattr(cli, '_gateway_router'):
                    from ..gateway.channels import MessageEvent, ChannelType
                    event = MessageEvent(
                        channel=ChannelType.WHATSAPP,
                        channel_id=msg.from_number,
                        user_id=msg.from_number,
                        user_name=msg.sender_name,
                        content=msg.body,
                        message_id=msg.id
                    )
                    await cli._gateway_router.handle_message(event)

            client.on_message(handle_message)

            # Start client
            started = await client.start()
            if not started:
                cli.renderer.error("Failed to start WhatsApp client")
                cli.renderer.info("Make sure Node.js is installed: apt install nodejs npm")
                return CommandResult.fail("Failed to start")

            cli.renderer.info("Waiting for QR code...")
            cli.renderer.info("(The QR code will appear below - scan with WhatsApp on your phone)")
            cli.renderer.newline()

            # Wait for connection (QR code will be printed by the bridge)
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
                return CommandResult.fail(result.get('error'))

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
                unread = f" ({chat['unread_count']} unread)" if chat.get('unread_count') else ""
                group_icon = "👥" if chat.get('is_group') else "👤"
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
            cli.renderer.success("WhatsApp client stopped")
            return CommandResult.ok()
        except Exception as e:
            cli.renderer.error(f"Stop failed: {e}")
            return CommandResult.fail(str(e))

    async def _status(self, cli: "JottyCLI") -> CommandResult:
        """Show connection status."""
        try:
            client = await self._get_client()

            if client.connected:
                cli.renderer.success("WhatsApp: Connected")
                if client._info:
                    cli.renderer.print(f"  Phone: {client._info.get('wid', {}).get('user', 'Unknown')}")
            else:
                cli.renderer.warning("WhatsApp: Not connected")
                cli.renderer.info("Run /whatsapp login to connect")

            return CommandResult.ok()

        except Exception as e:
            cli.renderer.warning("WhatsApp: Not initialized")
            cli.renderer.info("Run /whatsapp login to connect")
            return CommandResult.ok()

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        subcommands = ["login", "send", "chats", "contacts", "logout", "status", "stop"]
        flags = ["--to", "--message"]
        return [s for s in subcommands + flags if s.startswith(partial)]
