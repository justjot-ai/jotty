from typing import Any

"""
Jotty Unified Gateway
=====================

WebSocket gateway for receiving messages from multiple channels
(Telegram, Slack, Discord, WhatsApp) and routing to Jotty agents.

Features:
- Multi-channel support (Telegram, Slack, Discord, WhatsApp, WebSocket, HTTP)
- Persistent sessions across channels and restarts
- ExecutionContext integration for unified processing
- Registry-based channel responders (no hardcoded imports)

Inspired by OpenClaw's unified inbox architecture.

Usage (jotty.justjot.ai):
    python -m Jotty.cli.gateway              # Run gateway on port 8766
    python -m Jotty.cli.gateway --port 8080  # Custom port
"""

from .channels import ChannelRouter, ChannelType, MessageEvent, ResponseEvent
from .responders import ChannelResponderRegistry, get_responder_registry
from .server import UnifiedGateway, start_gateway
from .sessions import PersistentSessionManager, get_session_manager

__all__ = [
    # Server
    "UnifiedGateway",
    "start_gateway",
    # Channels
    "ChannelRouter",
    "MessageEvent",
    "ResponseEvent",
    "ChannelType",
    # Responders
    "ChannelResponderRegistry",
    "get_responder_registry",
    # Sessions
    "PersistentSessionManager",
    "get_session_manager",
]


def main() -> Any:
    """Main entry point for running the gateway server."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="jotty-gateway", description="Jotty Unified Gateway - WebSocket + HTTP API server"
    )
    parser.add_argument(
        "--host",
        "-H",
        default=os.getenv("JOTTY_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.getenv("JOTTY_PORT", "8766")),
        help="Port to bind to (default: 8766)",
    )
    parser.add_argument("--no-cli", action="store_true", help="Run without JottyCLI (echo mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Initialize CLI if requested
    cli = None
    if not args.no_cli:
        try:
            from ..app import JottyCLI

            cli = JottyCLI(no_color=True)
            logger.info("JottyCLI initialized for message processing")
        except Exception as e:
            logger.warning(f"Could not initialize JottyCLI: {e}. Running in echo mode.")

    # Start gateway
    logger.info(f"Starting Jotty Gateway on {args.host}:{args.port}")
    logger.info(f"PWA: http://{args.host}:{args.port}/")
    logger.info(f"WebSocket: ws://{args.host}:{args.port}/ws")
    logger.info(f"Health: http://{args.host}:{args.port}/health")

    start_gateway(host=args.host, port=args.port, cli=cli)


if __name__ == "__main__":
    main()
