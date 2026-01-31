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
    logger = logging.getLogger(__name__)

    # Import gateway components
    try:
        from cli.gateway import UnifiedGateway, start_gateway
        from cli.app import JottyCLI
    except ImportError:
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cli.gateway import UnifiedGateway, start_gateway
        from cli.app import JottyCLI

    # Initialize CLI
    cli = None
    if not args.no_cli:
        try:
            cli = JottyCLI(no_color=True)
            logger.info("JottyCLI initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize JottyCLI: {e}")
            logger.info("Running in echo mode (messages will be echoed back)")

    # Print startup info
    print("\n" + "=" * 60)
    print("  JOTTY WEB SERVER")
    print("=" * 60)
    print(f"  Host:      {args.host}")
    print(f"  Port:      {args.port}")
    print(f"  Mode:      {'Full CLI' if cli else 'Echo'}")
    print("=" * 60)
    print(f"\n  PWA Chat:   http://{args.host}:{args.port}/")
    print(f"  WebSocket:  ws://{args.host}:{args.port}/ws")
    print(f"  Health:     http://{args.host}:{args.port}/health")
    print(f"  API Docs:   http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print("\n  Webhooks:")
    print(f"    Telegram:  POST /webhook/telegram")
    print(f"    Slack:     POST /webhook/slack")
    print(f"    Discord:   POST /webhook/discord")
    print(f"    WhatsApp:  POST /webhook/whatsapp")
    print(f"    Generic:   POST /message")
    print("=" * 60 + "\n")

    # Start server
    start_gateway(host=args.host, port=args.port, cli=cli)


if __name__ == "__main__":
    main()
