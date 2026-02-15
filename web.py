#!/usr/bin/env python3
"""
Jotty Web Server for jotty.justjot.ai
=====================================

Standalone web server entry point for cmd.dev deployment.

Usage:
    python web.py                        # Run on default port 8766
    python web.py --port 8080            # Custom port
    JOTTY_PORT=8080 python web.py        # Via environment variable
    
    # For production (with process manager):
    nohup python web.py --port 8766 > jotty.log 2>&1 &
    
    # With screen:
    screen -dmS jotty python web.py --port 8766

Environment Variables:
    JOTTY_HOST         - Host to bind (default: 0.0.0.0)
    JOTTY_PORT         - Port to bind (default: 8766)
    ANTHROPIC_API_KEY  - For Claude API access
    OPENAI_API_KEY     - For OpenAI access  
    GROQ_API_KEY       - For Groq access
"""

import os
import sys
import argparse
import logging

logger = logging.getLogger(__name__)

# Suppress warnings before imports
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import warnings
warnings.filterwarnings('ignore', message='.*unauthenticated.*')
warnings.filterwarnings('ignore', message='.*huggingface.*')
warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(
        prog="jotty-web",
        description="Jotty Web Server - WebSocket + HTTP API for jotty.justjot.ai"
    )
    parser.add_argument(
        "--host", "-H",
        default=os.getenv("JOTTY_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("JOTTY_PORT", "8766")),
        help="Port to bind to (default: 8766)"
    )
    parser.add_argument(
        "--no-cli",
        action="store_true",
        help="Run without JottyCLI (echo mode)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Import gateway components (CLI now in apps/)
    try:
        from Jotty.apps.cli.gateway import UnifiedGateway, start_gateway
        from Jotty.apps.cli.app import JottyCLI
    except ImportError as e:
        logger.error(f"Failed to import CLI components: {e}")
        logger.info("Make sure Jotty.apps.cli is accessible")
        raise

    # Initialize CLI
    cli = None
    if not args.no_cli:
        try:
            cli = JottyCLI(no_color=True)
            logger.info("JottyCLI initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize JottyCLI: {e}")
            logger.info("Running in echo mode (messages will be echoed back)")

    # Log startup info
    logger.info("=" * 60)
    logger.info("  JOTTY WEB SERVER")
    logger.info("=" * 60)
    logger.info("  Host:      %s", args.host)
    logger.info("  Port:      %s", args.port)
    logger.info("  Mode:      %s", 'Full CLI' if cli else 'Echo')
    logger.info("=" * 60)
    logger.info("  PWA Chat:   http://%s:%s/", args.host, args.port)
    logger.info("  WebSocket:  ws://%s:%s/ws", args.host, args.port)
    logger.info("  Health:     http://%s:%s/health", args.host, args.port)
    logger.info("  API Docs:   http://%s:%s/docs", args.host, args.port)
    logger.info("=" * 60)
    logger.info("  Webhooks:")
    logger.info("    Telegram:  POST /webhook/telegram")
    logger.info("    Slack:     POST /webhook/slack")
    logger.info("    Discord:   POST /webhook/discord")
    logger.info("    WhatsApp:  POST /webhook/whatsapp")
    logger.info("    Generic:   POST /message")
    logger.info("=" * 60)

    # Start server
    start_gateway(host=args.host, port=args.port, cli=cli)


if __name__ == "__main__":
    main()
