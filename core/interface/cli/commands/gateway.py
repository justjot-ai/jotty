"""
Gateway Command
===============

/gateway - Start unified message gateway
"""

import logging
import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class GatewayCommand(BaseCommand):
    """
    /gateway - Start the unified message gateway.

    Receives messages from Telegram, Slack, Discord, WhatsApp
    and routes them to Jotty agents.
    """

    name = "gateway"
    aliases = ["gw", "inbox"]
    description = "Start unified message gateway"
    usage = "/gateway [start|stop|status] [--port PORT]"
    category = "automation"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute gateway command."""
        import socket

        subcommand = args.positional[0] if args.positional else "start"
        port = int(args.flags.get("port", 8766))
        host = args.flags.get("host", "0.0.0.0")
        foreground = args.flags.get("fg", args.flags.get("foreground", False))

        if subcommand == "start":
            return await self._start_gateway(cli, host, port, foreground)
        elif subcommand == "stop":
            return await self._stop_gateway(cli)
        elif subcommand == "status":
            return await self._gateway_status(cli, port)
        elif subcommand == "webhooks":
            return await self._show_webhooks(cli, port)
        else:
            cli.renderer.error(f"Unknown subcommand: {subcommand}")
            return CommandResult.fail("Unknown subcommand")

    async def _start_gateway(self, cli: "JottyCLI", host: str, port: int, foreground: bool) -> CommandResult:
        """Start the gateway server."""
        import socket

        # Check if port is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            sock.close()
        except OSError:
            cli.renderer.error(f"Port {port} is already in use!")
            cli.renderer.info("Stop existing: /gateway stop")
            cli.renderer.info(f"Or use different port: /gateway start --port {port + 1}")
            return CommandResult.fail(f"Port {port} in use")

        if foreground:
            return await self._run_gateway_foreground(cli, host, port)
        else:
            return await self._run_gateway_background(cli, host, port)

    async def _run_gateway_background(self, cli: "JottyCLI", host: str, port: int) -> CommandResult:
        """Start gateway in background."""
        cli.renderer.info(f"Starting gateway on port {port}...")

        # Get Jotty path
        jotty_path = str(Path(__file__).parent.parent.parent.resolve())

        # Create server script
        server_script = f'''
import sys
sys.path.insert(0, "{jotty_path}")
from Jotty.core.interface.cli.gateway import UnifiedGateway
gateway = UnifiedGateway("{host}", {port})
gateway.run()
'''

        # Write to temp file
        script_path = Path.home() / ".jotty" / "gateway_server.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(server_script)

        # PID file
        pid_file = Path.home() / ".jotty" / "gateway.pid"

        # Start in background
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        # Save PID
        pid_file.write_text(str(process.pid))

        # Wait and check
        await asyncio.sleep(1)

        if process.poll() is None:
            cli.renderer.success(f"Gateway running in background (PID: {process.pid})")
            cli.renderer.newline()
            cli.renderer.print("[bold green]PWA Chat App:[/bold green]")
            cli.renderer.print(f"  http://localhost:{port}")
            cli.renderer.print(f"  http://localhost:{port}/app")
            cli.renderer.newline()
            cli.renderer.print("[bold]API:[/bold]")
            cli.renderer.print(f"  Docs: http://localhost:{port}/docs")
            cli.renderer.print(f"  Health: http://localhost:{port}/health")
            cli.renderer.newline()
            cli.renderer.print("[bold]Webhook URLs:[/bold]")
            cli.renderer.print(f"  Telegram: http://YOUR_DOMAIN:{port}/webhook/telegram")
            cli.renderer.print(f"  Slack:    http://YOUR_DOMAIN:{port}/webhook/slack")
            cli.renderer.print(f"  Discord:  http://YOUR_DOMAIN:{port}/webhook/discord")
            cli.renderer.print(f"  WhatsApp: http://YOUR_DOMAIN:{port}/webhook/whatsapp")
            cli.renderer.print(f"  WebSocket: ws://YOUR_DOMAIN:{port}/ws")
            cli.renderer.newline()
            cli.renderer.info("Open browser: http://localhost:{port}")
            cli.renderer.info("Stop with: /gateway stop")
        else:
            cli.renderer.error("Gateway failed to start")
            return CommandResult.fail("Gateway failed")

        return CommandResult.ok()

    async def _run_gateway_foreground(self, cli: "JottyCLI", host: str, port: int) -> CommandResult:
        """Run gateway in foreground (blocking)."""
        cli.renderer.header("Starting Jotty Gateway (foreground)")
        cli.renderer.info(f"Host: {host}")
        cli.renderer.info(f"Port: {port}")
        cli.renderer.newline()
        cli.renderer.print("[bold green]PWA Chat App: http://localhost:{port}[/bold green]")
        cli.renderer.print(f"API Docs: http://localhost:{port}/docs")
        cli.renderer.newline()
        cli.renderer.info("Press Ctrl+C to stop")

        try:
            from ..gateway import UnifiedGateway
            import uvicorn

            gateway = UnifiedGateway(host, port)
            gateway.set_cli(cli)
            app = gateway.create_app()

            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

        except ImportError as e:
            cli.renderer.error(f"Missing dependency: {e}")
            cli.renderer.info("Install with: pip install fastapi uvicorn")
            return CommandResult.fail(str(e))
        except KeyboardInterrupt:
            cli.renderer.info("Gateway stopped.")

        return CommandResult.ok()

    async def _stop_gateway(self, cli: "JottyCLI") -> CommandResult:
        """Stop the background gateway."""
        import signal

        pid_file = Path.home() / ".jotty" / "gateway.pid"

        if not pid_file.exists():
            cli.renderer.warning("No gateway PID file found")
            return CommandResult.ok()

        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink()
            cli.renderer.success(f"Gateway stopped (PID: {pid})")
        except ProcessLookupError:
            pid_file.unlink()
            cli.renderer.info("Gateway was not running")
        except Exception as e:
            cli.renderer.error(f"Failed to stop gateway: {e}")
            return CommandResult.fail(str(e))

        return CommandResult.ok()

    async def _gateway_status(self, cli: "JottyCLI", port: int) -> CommandResult:
        """Check gateway status."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        cli.renderer.success("Gateway is running")
                        cli.renderer.print(f"  Active sessions: {data.get('active_sessions', 0)}")
                        cli.renderer.print(f"  WebSocket clients: {data.get('websocket_clients', 0)}")
                        return CommandResult.ok(data=data)
                    else:
                        cli.renderer.warning(f"Gateway returned {resp.status}")
        except Exception as e:
            cli.renderer.warning(f"Gateway not accessible: {e}")
            cli.renderer.info("Start with: /gateway start")

        return CommandResult.ok()

    async def _show_webhooks(self, cli: "JottyCLI", port: int) -> CommandResult:
        """Show webhook configuration URLs."""
        cli.renderer.header("Gateway Webhook URLs")

        # Get public URL if available
        public_url = os.getenv("PUBLIC_URL", f"http://YOUR_DOMAIN:{port}")

        cli.renderer.print("\n[bold]Configure these URLs in each platform:[/bold]\n")

        cli.renderer.print("[cyan]Telegram:[/cyan]")
        cli.renderer.print(f"  1. Talk to @BotFather, use /setwebhook")
        cli.renderer.print(f"  2. URL: {public_url}/webhook/telegram")
        cli.renderer.print("")

        cli.renderer.print("[cyan]Slack:[/cyan]")
        cli.renderer.print(f"  1. Go to api.slack.com/apps > Event Subscriptions")
        cli.renderer.print(f"  2. Request URL: {public_url}/webhook/slack")
        cli.renderer.print(f"  3. Subscribe to: message.channels, message.im")
        cli.renderer.print("")

        cli.renderer.print("[cyan]Discord:[/cyan]")
        cli.renderer.print(f"  1. Go to discord.com/developers/applications")
        cli.renderer.print(f"  2. Interactions Endpoint: {public_url}/webhook/discord")
        cli.renderer.print("")

        cli.renderer.print("[cyan]WhatsApp:[/cyan]")
        cli.renderer.print(f"  1. Go to developers.facebook.com > WhatsApp")
        cli.renderer.print(f"  2. Webhook URL: {public_url}/webhook/whatsapp")
        cli.renderer.print(f"  3. Verify Token: jotty (or set WHATSAPP_VERIFY_TOKEN)")
        cli.renderer.print("")

        cli.renderer.print("[cyan]WebSocket:[/cyan]")
        cli.renderer.print(f"  Connect to: ws://{public_url.replace('http://', '').replace('https://', '')}/ws")
        cli.renderer.print("")

        return CommandResult.ok()

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        subcommands = ["start", "stop", "status", "webhooks"]
        flags = ["--port", "--host", "--fg"]
        return [s for s in subcommands + flags if s.startswith(partial)]
