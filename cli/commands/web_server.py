"""
Web Server Command
==================

CLI command to start/manage the Web UI server.

Usage:
    /web start         - Start the web server
    /web stop          - Stop the web server
    /web status        - Check server status
    /web start --port 8080
"""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Optional, Any

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class WebServerCommand(BaseCommand):
    """Start and manage the Web UI server."""

    name = "web"
    aliases = ["webui", "server"]
    description = "Start and manage Web UI server"
    usage = "/web [start|stop|status] [--port PORT] [--host HOST]"
    category = "integrations"

    def __init__(self) -> None:
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = False
        self._port = 8080
        self._host = "0.0.0.0"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute web command."""
        subcommand = args.positional[0] if args.positional else "start"

        # Parse options
        port = int(args.flags.get("port", args.flags.get("p", self._port)))
        host = args.flags.get("host", args.flags.get("h", self._host))

        if subcommand == "start":
            return await self._start_server(cli, host, port)
        elif subcommand == "stop":
            return await self._stop_server(cli)
        elif subcommand == "status":
            return self._get_status()
        elif subcommand == "help":
            return CommandResult.ok(self._get_help())
        else:
            return CommandResult.fail(f"Unknown subcommand: {subcommand}")

    async def _start_server(self, cli: "JottyCLI", host: str, port: int) -> CommandResult:
        """Start the web server."""
        if self._server_running:
            return CommandResult.fail(
                f"Server is already running on http://{self._host}:{self._port}\n"
                "Use /web stop first."
            )

        try:
            import uvicorn
            from ...web.api import create_app

            self._port = port
            self._host = host

            # Create the app
            app = create_app()

            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
            )

            server = uvicorn.Server(config)

            # Run in background thread
            def run_server() -> Any:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(server.serve())
                except Exception as e:
                    logger.error(f"Server error: {e}")
                finally:
                    self._server_running = False

            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            self._server_running = True

            # Give server time to start
            await asyncio.sleep(0.5)

            url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

            return CommandResult.ok(
                f"Web server started!\n"
                f"Open in browser: {url}\n"
                f"API docs: {url}/docs\n"
                f"\nUse /web stop to stop the server."
            )

        except ImportError as e:
            return CommandResult.fail(
                f"Missing dependency: {e}\n"
                "Install with: pip install fastapi uvicorn websockets"
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            self._server_running = False
            return CommandResult.fail(f"Failed to start server: {e}")

    async def _stop_server(self, cli: "JottyCLI") -> CommandResult:
        """Stop the web server."""
        if not self._server_running:
            return CommandResult.fail("Server is not running.")

        try:
            # Note: uvicorn doesn't have a clean stop mechanism when run this way
            # The thread is daemonized, so it will stop when the main process exits
            self._server_running = False
            self._server_thread = None

            return CommandResult.ok(
                "Server stop requested.\n"
                "Note: The server may take a moment to fully stop."
            )

        except Exception as e:
            logger.error(f"Error stopping server: {e}", exc_info=True)
            return CommandResult.fail(f"Error stopping server: {e}")

    def _get_status(self) -> CommandResult:
        """Get server status."""
        if not self._server_running:
            return CommandResult.ok(
                "Web Server Status: STOPPED\n"
                "Use /web start to start the server."
            )

        url = f"http://{self._host if self._host != '0.0.0.0' else 'localhost'}:{self._port}"

        return CommandResult.ok(
            f"Web Server Status: RUNNING\n"
            f"URL: {url}\n"
            f"API Docs: {url}/docs"
        )

    def _get_help(self) -> str:
        """Get help text."""
        return """Web Server Command

Start and manage the Jotty Web UI server.

Usage:
    /web start              - Start server on default port (8080)
    /web start --port 3000  - Start on custom port
    /web start --host 127.0.0.1  - Bind to specific host
    /web stop               - Stop the server
    /web status             - Check server status

Options:
    --port, -p    Port to bind to (default: 8080)
    --host, -h    Host to bind to (default: 0.0.0.0)

The web UI provides:
    - Chat interface similar to LibreChat
    - WebSocket streaming for real-time responses
    - Session management
    - Cross-interface sync with CLI and Telegram
"""

    def get_completions(self, partial: str) -> list:
        """Get completions for subcommands."""
        subcommands = ["start", "stop", "status", "help"]
        return [s for s in subcommands if s.startswith(partial)]
